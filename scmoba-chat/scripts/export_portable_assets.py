"""Copy authorized assets into a self-contained directory; never overwrite.

No network access or publication. Check redistribution permissions separately.
This can copy tens of GB. Symlinks are dereferenced to make the result portable.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--copy', action='store_true', help='Actually copy; default is a dry run')
    a = p.parse_args()
    config = Path(a.config).resolve()
    cfg = json.loads(config.read_text(encoding='utf-8'))
    destination = Path(a.output).resolve()
    paths = {}
    names = {'geneformer': 'geneformer', 'qwen': 'qwen3_8b',
             'checkpoint': 'checkpoint', 'resources': 'resources.json'}
    total = 0
    for key in names:
        source = Path(cfg[key])
        source = (config.parent / source).resolve() if not source.is_absolute() else source.resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        if destination == source or source in destination.parents:
            raise ValueError('Destination must not be inside an asset source')
        paths[key] = source
        total += sum(f.stat().st_size for f in source.rglob('*') if f.is_file()) if source.is_dir() else source.stat().st_size
    if destination.exists():
        raise FileExistsError(destination)
    print(json.dumps({'output': str(destination), 'bytes_to_copy': total,
                      'sources': {k: str(v) for k, v in paths.items()}, 'copy': a.copy}, indent=2))
    if not a.copy:
        return
    destination.mkdir(parents=True)
    # Incomplete exports have no COMPLETE.json and must not be deployed.
    for key, source in paths.items():
        target = destination / names[key]
        if source.is_dir():
            shutil.copytree(source, target, symlinks=False)
        else:
            shutil.copy2(source, target)
    (destination / 'config.json').write_text(json.dumps(names, indent=2), encoding='utf-8')
    manifest = []
    for file in sorted(destination.rglob('*')):
        if file.is_file():
            h = hashlib.sha256()
            with file.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    h.update(chunk)
            manifest.append({'path': file.relative_to(destination).as_posix(),
                             'bytes': file.stat().st_size, 'sha256': h.hexdigest()})
    (destination / 'COMPLETE.json').write_text(json.dumps({'files': manifest}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
