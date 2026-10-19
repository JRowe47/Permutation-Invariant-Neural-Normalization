"""Reproduce input downloads using immutable URLs and SHA256 checks."""
import argparse
import hashlib
import json
import pathlib
import shutil
import urllib.request

p = argparse.ArgumentParser()
p.add_argument('--manifest', type=pathlib.Path, required=True)
p.add_argument('--cache', type=pathlib.Path, required=True)
args = p.parse_args()
for model in json.loads(args.manifest.read_text()):
    d = args.cache / str(model['step']); d.mkdir(parents=True, exist_ok=True)
    for item in model['files']:
        target = d / item['filename']
        if not target.exists():
            temp = target.with_suffix(target.suffix + '.partial')
            with urllib.request.urlopen(item['url'], timeout=180) as source, temp.open('wb') as out:
                shutil.copyfileobj(source, out, 1024 * 1024)
            temp.rename(target)
        assert hashlib.sha256(target.read_bytes()).hexdigest() == item['sha256'], str(target)
        print('verified', model['step'], item['filename'], flush=True)
