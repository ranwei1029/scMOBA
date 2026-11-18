import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(prog='scmoba-chat');sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--input',required=True);q.add_argument('--output',required=True);q.add_argument('--resources',required=True)
    q.add_argument('--input-scale',choices=['counts','log1p'],required=True);q.add_argument('--gene-column',default='index');q.add_argument('--gene-type',choices=['ensembl','symbol'],default='ensembl');q.add_argument('--layer',default='X');q.add_argument('--min-genes',type=int,default=10);q.add_argument('--max-cells',type=int,default=20000)
    from .prompts import TASKS
    q=sub.add_parser('infer');q.add_argument('--config',required=True);q.add_argument('--dataset',required=True);q.add_argument('--cell-id',required=True);q.add_argument('--question');q.add_argument('--task',choices=['free',*TASKS],default='free');q.add_argument('--output',required=True);q.add_argument('--device',default='cuda:0')
    q=sub.add_parser('serve');q.add_argument('--config',required=True);q.add_argument('--work-dir',required=True);q.add_argument('--port',type=int,default=7860);q.add_argument('--device',default='cuda:0')
    q=sub.add_parser('doctor');q.add_argument('--config');
    a=p.parse_args()
    if a.command=='prepare':
        from .preprocess import prepare
        print(json.dumps(prepare(a.input,a.output,a.resources,input_scale=a.input_scale,gene_column=a.gene_column,gene_type=a.gene_type,layer=a.layer,min_genes=a.min_genes,max_cells=a.max_cells),indent=2))
    elif a.command=='doctor':
        import importlib.metadata as md,torch,sys,site
        report={'python':sys.executable,'user_site_enabled':site.ENABLE_USER_SITE,'versions':{n:md.version(n) for n in ['torch','transformers','peft','anndata','datasets','fastapi']},'cuda_available':torch.cuda.is_available()}
        if report['cuda_available']:report['gpu']=torch.cuda.get_device_name(0)
        if a.config:
            c=Path(a.config).resolve();cfg=json.loads(c.read_text())
            report['assets']={k:k in cfg and (Path(cfg[k]) if Path(cfg[k]).is_absolute() else c.parent/cfg[k]).exists() for k in ['geneformer','qwen','checkpoint','resources']}
        print(json.dumps(report,indent=2))
        if a.config and not all(report['assets'].values()):raise SystemExit(1)
    else:
        from .inference import Engine
        if a.command=='infer':
            from .prompts import task_question
            task_question(a.task,a.question)
        if a.command=='infer' and Path(a.output).exists():raise FileExistsError(a.output)
        engine=Engine(a.config,a.device)
        if a.command=='serve':
            import uvicorn
            from .server import create_app
            uvicorn.run(create_app(engine,a.work_dir),host='127.0.0.1',port=a.port)
        else:
            from datasets import load_from_disk
            ds=load_from_disk(str(Path(a.dataset)/'tokens'));ids=list(ds['cell_id'])
            if a.cell_id not in ids:raise ValueError('Unknown cell ID')
            result=engine.chat(ds[ids.index(a.cell_id)]['input_ids'],a.question,task=a.task)
            Path(a.output).write_text(json.dumps({'cell_id':a.cell_id,**result},indent=2),encoding='utf-8')
            print(result['answer'])

if __name__=='__main__':main()
