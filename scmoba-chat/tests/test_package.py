import json
import numpy as np
import pandas as pd
import pytest
import anndata as ad
from scipy import sparse
from scmoba_chat.preprocess import prepare

@pytest.fixture
def resources(tmp_path):
    p=tmp_path/'resources.json'
    p.write_text(json.dumps({'tokens':{'ENSG1':4,'ENSG2':7},'medians':{'ENSG1':2.,'ENSG2':1.},'symbols':{'A':'ENSG1','B':'ENSG2'}}))
    return p

def h5(tmp_path, x, names=('ENSG1','ENSG2','UNKNOWN')):
    p=tmp_path/'input.h5ad';a=ad.AnnData(sparse.csr_matrix(x,dtype=float),obs=pd.DataFrame(index=[f'c{i}' for i in range(len(x))]),var=pd.DataFrame(index=list(names)));a.write_h5ad(p);return p

def test_counts_rank_and_qc(tmp_path,resources):
    from datasets import load_from_disk
    p=h5(tmp_path,[[8,2,0],[0,0,10]])
    qc=prepare(p,tmp_path/'prepared',resources,input_scale='counts',min_genes=1)
    ds=load_from_disk(str(tmp_path/'prepared/tokens'))
    assert ds[0]['input_ids']==[7,4] # log1p divided by median, not count rank.
    assert qc['kept_cells']==1 and qc['rejected_cells']==1
    with pytest.raises(FileExistsError):prepare(p,tmp_path/'prepared',resources,input_scale='counts')

def test_explicit_log_no_transform(tmp_path,resources):
    from datasets import load_from_disk
    p=h5(tmp_path,[[8,2,0]])
    prepare(p,tmp_path/'prepared',resources,input_scale='log1p',min_genes=1)
    assert load_from_disk(str(tmp_path/'prepared/tokens'))[0]['input_ids']==[4,7]

@pytest.mark.parametrize('x',[[[1.1,2,0]], [[-1,2,0]], [[float('nan'),2,0]]])
def test_bad_counts(tmp_path,resources,x):
    with pytest.raises(ValueError):prepare(h5(tmp_path,x),tmp_path/'prepared',resources,input_scale='counts',min_genes=1)

def test_duplicate_log_rejected(tmp_path,resources):
    p=h5(tmp_path,[[1,2,3]],names=('ENSG1.1','ENSG1.2','ENSG2'))
    with pytest.raises(ValueError,match='Duplicate'):prepare(p,tmp_path/'prepared',resources,input_scale='log1p')

def test_attention_eval_deterministic():
    import torch
    from scmoba_chat._model.attention import CrossMultiHeadAttention
    m=CrossMultiHeadAttention(8,2,.5).eval();a=torch.randn(1,2,8);b=torch.randn(1,3,8)
    assert torch.equal(m(a,b),m(a,b))

def test_template_prompt_and_free_unchanged():
    from scmoba_chat.prompts import task_question,TASKS
    for task,t in TASKS.items():
        q,prompt=task_question(task)
        assert q==t['question'] and prompt==t['instruction']+'\n\n'+q
    assert task_question('free','Explain this cell')==('Explain this cell','Explain this cell')
    with pytest.raises(ValueError):task_question('unknown')
    with pytest.raises(ValueError):task_question('subclass',TASKS['subclass']['instruction'])

def test_upload_and_history(tmp_path,resources):
    from fastapi.testclient import TestClient
    from scmoba_chat.server import create_app
    class Fake:
        resources_path=resources
        def chat(self,ids,q,history=None,task='free'):
            from scmoba_chat.prompts import task_question
            q,model_q=task_question(task,q)
            h=(history or [])+[{'role':'user','content':q},{'role':'assistant','content':'TEST ONLY'}]
            return {'answer':'TEST ONLY','history':h,'task':task,'model_question':model_q}
    c=TestClient(create_app(Fake(),tmp_path/'uploads'))
    # default min_genes=10 requires >=10 valid genes
    ref=json.loads(resources.read_text());ref['tokens']={f'ENSG{i}':i+3 for i in range(12)};ref['medians']={f'ENSG{i}':1 for i in range(12)};resources.write_text(json.dumps(ref))
    p=h5(tmp_path,[[2]*12],names=tuple(ref['tokens']))
    with p.open('rb') as f:r=c.post('/datasets',files={'file':('test.h5ad',f)},data={'input_scale':'counts'})
    assert r.status_code==200,r.text
    d=r.json();a=c.post('/chat',json={'dataset_id':d['dataset_id'],'cell_id':'c0','question':'Describe'}).json()
    b=c.post('/chat',json={'dataset_id':d['dataset_id'],'cell_id':'c0','question':'More','session_id':a['session_id']}).json()
    assert len(b['history'])==4
    t=c.post('/chat',json={'dataset_id':d['dataset_id'],'cell_id':'c0','task':'subclass'}).json()
    assert t['task']=='subclass' and t['model_question'].startswith('Classify this expression profile into its fine-grained cell subclass')
    assert c.post('/chat',json={'dataset_id':d['dataset_id'],'cell_id':'c0','task':'celltype','session_id':t['session_id']}).status_code==400
    assert c.get('/tasks').status_code==200
    assert c.post('/chat',json={'dataset_id':'../outside','cell_id':'c0','question':'x'}).status_code==400
    assert c.post('/datasets',files={'file':('evil.pkl',b'no')},data={'input_scale':'counts'}).status_code==400
