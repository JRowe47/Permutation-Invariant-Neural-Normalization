"""Download public Pythia inputs, pinned by commit, outside the Git checkout."""
import argparse
import hashlib
import json
import pathlib
import urllib.request
from concurrent.futures import ThreadPoolExecutor

MODEL = 'EleutherAI/pythia-70m'
FINAL_SHA = 'de3e4e2d6cbb3b1a51f90fe153ea51fcd8c7c852'

def download(root, step):
    info = json.load(urllib.request.urlopen(f'https://huggingface.co/api/models/{MODEL}/revision/step{step}', timeout=90))
    sha = info['sha']
    if step == 143000:
        assert sha == FINAL_SHA
    folder = root / str(step)
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'model.safetensors']:
        dst = folder / name
        url = f'https://huggingface.co/{MODEL}/resolve/{sha}/{name}'
        if not dst.exists():
            tmp = dst.with_suffix(dst.suffix + '.partial')
            with urllib.request.urlopen(url, timeout=180) as src, tmp.open('wb') as out:
                while chunk := src.read(1024 * 1024):
                    out.write(chunk)
            tmp.rename(dst)
        digest = hashlib.sha256(dst.read_bytes()).hexdigest()
        rows.append({'filename': name, 'bytes': dst.stat().st_size, 'sha256': digest, 'url': url})
        print(step, name, dst.stat().st_size, flush=True)
    return {'model': MODEL, 'step': step, 'revision': sha, 'files': rows}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=pathlib.Path, required=True)
    parser.add_argument('--manifest', type=pathlib.Path, required=True)
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(lambda step: download(args.cache, step), [143000, 64000]))
    args.manifest.write_text(json.dumps(rows, indent=2) + '\n')
