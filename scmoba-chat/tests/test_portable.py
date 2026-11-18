import hashlib
import json
from pathlib import Path
import subprocess
import sys


def test_portable_export(tmp_path):
    script = Path(__file__).resolve().parents[1]/'scripts/export_portable_assets.py'
    source = tmp_path/'source'
    source.mkdir()
    config = {}
    for key in ('geneformer', 'qwen', 'checkpoint'):
        (source/key).mkdir()
        (source/key/'dummy.bin').write_bytes(b'test fixture, not model weights')
        config[key] = key
    (source/'resources.json').write_text('{}')
    config['resources'] = 'resources.json'
    (source/'config.json').write_text(json.dumps(config))
    output = tmp_path/'export'
    cmd = [sys.executable,str(script),'--config',str(source/'config.json'),'--output',str(output)]
    subprocess.run(cmd,check=True,capture_output=True)
    assert not output.exists()
    subprocess.run(cmd+['--copy'],check=True,capture_output=True)
    manifest = json.loads((output/'COMPLETE.json').read_text())
    for file in manifest['files']:
        assert hashlib.sha256((output/file['path']).read_bytes()).hexdigest() == file['sha256']
    assert all(not Path(v).is_absolute() for v in json.loads((output/'config.json').read_text()).values())
    assert subprocess.run(cmd+['--copy'],capture_output=True).returncode != 0
