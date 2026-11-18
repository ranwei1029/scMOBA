"""Real localhost HTTP integration test. Stops its own server on exit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import httpx
from datasets import load_from_disk

p = argparse.ArgumentParser()
p.add_argument('--config', required=True)
p.add_argument('--input', required=True)
p.add_argument('--output', required=True)
p.add_argument('--input-scale', choices=['counts','log1p'], default='log1p')
p.add_argument('--gene-type', choices=['symbol','ensembl'], default='symbol')
a = p.parse_args()
out = Path(a.output).resolve()
out.mkdir(parents=True, exist_ok=False)
config = Path(a.config).resolve()
cfg = json.loads(config.read_text())
resources = Path(cfg['resources'])
if not resources.is_absolute():
    resources = config.parent/resources
env = dict(os.environ, PYTHONNOUSERSITE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
env.pop('PYTHONPATH', None)
subprocess.run([sys.executable, '-m', 'scmoba_chat.cli', 'prepare', '--input', a.input,
    '--output', str(out/'cli-prepared'), '--resources', str(resources),
    '--input-scale', a.input_scale, '--gene-type', a.gene_type], check=True, env=env)
with socket.socket() as s:
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
log = (out/'server.log').open('w')
start = time.time()
server = subprocess.Popen([sys.executable, '-m', 'scmoba_chat.cli', 'serve',
    '--config', str(config), '--work-dir', str(out/'uploads'), '--port', str(port)],
    env=env, stdout=log, stderr=subprocess.STDOUT)
try:
    with httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=180, trust_env=False) as client:
        deadline = time.time()+300
        while True:
            if server.poll() is not None:
                raise RuntimeError('Server exited: see server.log')
            try:
                r = client.get('/health')
                r.raise_for_status()
                break
            except httpx.ConnectError:
                if time.time() > deadline:
                    raise TimeoutError('Server not ready')
                time.sleep(1)
        health = r.json()
        assert health['mock'] is False
        page = client.get('/')
        assert page.status_code == 200 and '<html' in page.text.lower()
        tasks = client.get('/tasks')
        tasks.raise_for_status()
        with open(a.input, 'rb') as f:
            r = client.post('/datasets', files={'file':('real.h5ad', f)},
                            data={'input_scale':a.input_scale, 'gene_type':a.gene_type})
        r.raise_for_status()
        upload = r.json()
        cli = load_from_disk(str(out/'cli-prepared/tokens'))
        api = load_from_disk(str(out/'uploads'/upload['dataset_id']/'prepared/tokens'))
        assert list(cli['input_ids']) == list(api['input_ids'])
        base = {'dataset_id':upload['dataset_id'], 'cell_id':upload['cell_ids'][0]}
        answers = []
        def ask(**kwargs):
            r = client.post('/chat', json={**base, **kwargs})
            r.raise_for_status()
            answer = r.json()
            assert answer['answer'].strip()
            answers.append(answer)
            return answer
        ask(task='subclass')
        first = ask(question="Provide a detailed overview of this cell's characteristics and potential molecular context.")
        second = ask(question='Which biological pathways are prominent in this cell?', session_id=first['session_id'])
        assert len(first['history']) == 2 and len(second['history']) == 4
        ask(question='Can this expression profile establish a clinical diagnosis and determine which drug the donor should take?')
        invalid = client.post('/chat', json={**base, 'cell_id':'nonexistent', 'question':'Describe this cell.'})
        assert invalid.status_code == 404
        mismatch = client.post('/chat', json={**base, 'cell_id':upload['cell_ids'][1],
            'session_id':first['session_id'], 'question':'Describe this cell.'})
        assert mismatch.status_code == 400
        journal = out/'uploads'/upload['dataset_id']/'conversations.jsonl'
        assert len(journal.read_text().splitlines()) == 4
        report = {'python':sys.executable, 'package_version':__import__('scmoba_chat').__version__,
            'input':a.input, 'input_sha256':hashlib.sha256(Path(a.input).read_bytes()).hexdigest(),
            'health':health, 'html_served':True, 'tasks':tasks.json(), 'upload':upload,
            'cli_api_tokens_identical':True, 'multiturn_history_preserved':True,
            'unknown_cell_rejected':True, 'cross_cell_history_rejected':True,
            'conversation_log_count':4, 'seconds':time.time()-start, 'answers':answers,
            'note':'Functional reproduction, not a biological accuracy evaluation; no interactive browser automation.'}
        (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2))
finally:
    server.terminate()
    try:
        server.wait(timeout=20)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait()
    log.close()
