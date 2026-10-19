"""Lossless repository packaging for the retained derivative NPZ archive."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import numpy as np


def derivative_bytes(folder,expected_sha=None):
    folder=Path(folder);manifest_path=folder/'DERIVATIVE_ARCHIVE.json'
    if manifest_path.exists():
        m=json.loads(manifest_path.read_text())
        if m['artifact']!='derivatives.npz' or not 0<m['bytes']<=64*1024*1024:
            raise ValueError('Unexpected archive identity or size')
        chunks=[];offset=0
        for part in m['parts']:
            if Path(part['path']).name!=part['path'] or part['offset']!=offset:
                raise ValueError('Invalid archive part path or offset')
            data=(folder/part['path']).read_bytes()
            if len(data)!=part['bytes'] or hashlib.sha256(data).hexdigest()!=part['sha256']:
                raise ValueError('Archive part changed')
            chunks.append(data);offset+=len(data)
        data=b''.join(chunks)
        if offset!=m['bytes'] or hashlib.sha256(data).hexdigest()!=m['sha256']:
            raise ValueError('Reassembled archive differs')
    else:
        data=(folder/'derivatives.npz').read_bytes()
    if expected_sha and hashlib.sha256(data).hexdigest()!=expected_sha:
        raise ValueError('Archive differs from the measurement receipt')
    return data


def load_derivatives(folder,expected_sha=None):
    return np.load(io.BytesIO(derivative_bytes(folder,expected_sha)),allow_pickle=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path)
    parser.add_argument('--unpack',action='store_true');args=parser.parse_args()
    data=derivative_bytes(args.folder)
    if args.unpack:
        target=args.folder/'derivatives.npz'
        if target.exists() and target.read_bytes()!=data:raise ValueError('Existing unpacked archive differs')
        if not target.exists():target.write_bytes(data)
    print(json.dumps({'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'unpacked':args.unpack},indent=2))
