"""Single-user, loopback-only upload application. Not a public multi-tenant service."""
import json
import shutil
import threading
import uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from .preprocess import prepare


def create_app(engine, work_dir):
    app=FastAPI(title='scMOBA-Chat — Analyze your own data')
    root=Path(work_dir).resolve();root.mkdir(parents=True,exist_ok=True)
    sessions={}; lock=threading.Lock(); job_lock=threading.Lock()
    class ChatRequest(BaseModel):
        dataset_id:str
        cell_id:str
        question:str|None=Field(default=None,max_length=16000)
        task:str='free'
        session_id:str|None=None
    def dataset_path(uid):
        try:
            if str(uuid.UUID(uid))!=uid:raise ValueError()
        except ValueError:raise HTTPException(400,'Invalid dataset ID')
        p=root/uid
        if not (p/'prepared/tokens').is_dir():raise HTTPException(404,'Dataset not found')
        return p
    @app.get('/',response_class=HTMLResponse)
    def index(): return (Path(__file__).parent/'static/index.html').read_text(encoding='utf-8')
    @app.get('/health')
    def health():return {'status':'ready','mode':'local-single-user','mock':False}
    @app.get('/tasks')
    def tasks():
        from .prompts import catalog
        return catalog()
    @app.post('/datasets')
    def upload(file:UploadFile=File(...),input_scale:str=Form(...),gene_type:str=Form('ensembl'),gene_column:str=Form('index'),layer:str=Form('X')):
        if not (file.filename or '').lower().endswith('.h5ad'):raise HTTPException(400,'Upload an h5ad file')
        if not job_lock.acquire(blocking=False):raise HTTPException(409,'Preprocessing is busy')
        uid=str(uuid.uuid4());p=root/uid;p.mkdir()
        try:
            if sum(1 for x in root.iterdir() if x.is_dir())>20:raise ValueError('Local limit: delete old datasets before uploading more')
            total=0
            with (p/'input.h5ad').open('wb') as out:
                while b:=file.file.read(1024*1024):
                    total+=len(b)
                    if total>512*1024**2:raise ValueError('Upload limit is 512 MiB; use CLI/subset for larger data')
                    out.write(b)
            qc=prepare(p/'input.h5ad',p/'prepared',engine.resources_path,input_scale=input_scale,gene_type=gene_type,gene_column=gene_column,layer=layer)
            from datasets import load_from_disk
            ds=load_from_disk(str(p/'prepared/tokens'))
            return {'dataset_id':uid,'qc':qc,'cell_ids':list(ds['cell_id'])}
        except (ValueError,KeyError,OSError) as e:
            shutil.rmtree(p)  # Newly-created UUID directory owned by this request only.
            raise HTTPException(400,str(e))
        except Exception:
            shutil.rmtree(p)
            raise
        finally:
            file.file.close();job_lock.release()
    @app.post('/chat')
    def chat(req:ChatRequest):
        from datasets import load_from_disk
        p=dataset_path(req.dataset_id)
        ds=load_from_disk(str(p/'prepared/tokens'))
        ids=list(ds['cell_id'])
        if req.cell_id not in ids:raise HTTPException(404,'Cell not found')
        # Serialize state changes and generation to avoid interleaving conversation history.
        with lock:
            from .prompts import task_question
            try:task_question(req.task,req.question)
            except ValueError as e:raise HTTPException(400,str(e))
            if req.task!='free' and req.session_id:raise HTTPException(400,'Template predictions must start a new conversation')
            sid=req.session_id or str(uuid.uuid4())
            if req.session_id and sid not in sessions:raise HTTPException(404,'Chat session expired/not found')
            old=sessions.get(sid,{'dataset_id':req.dataset_id,'cell_id':req.cell_id,'history':[]})
            if old['dataset_id']!=req.dataset_id or old['cell_id']!=req.cell_id:raise HTTPException(400,'Start a new chat when changing cells')
            if len(sessions)>=100 and sid not in sessions:raise HTTPException(429,'Session limit reached; restart local server')
            try:
                if req.task=='free':result=engine.chat(ds[ids.index(req.cell_id)]['input_ids'],req.question,old['history'])
                else:result=engine.chat(ds[ids.index(req.cell_id)]['input_ids'],req.question,old['history'],task=req.task)
            except ValueError as e:raise HTTPException(400,str(e))
            sessions[sid]={**old,'history':result['history']}
            with (p/'conversations.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps({'session_id':sid,'cell_id':req.cell_id,**result},ensure_ascii=False)+'\n')
            return {'session_id':sid,**result}
    return app
