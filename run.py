#!/usr/bin/env python3
"""
KLA PS1 - AI-Based Restoration of Degraded Images
Required submission entry point.

    python run.py <input-dir> <output-dir>

Reads every .npy in <input-dir> (recursively), restores it, and writes one
.npy per input to <output-dir> under the same filename. Creates <output-dir>
if it does not exist. Outputs are float32, shape (H, W), values in [0, 1],
free of NaN and Inf.

Model logic lives in evaluate.py alongside this file; this script is the thin
entry wrapper required by the submission spec. Weights are read from
models/model.pt, falling back to weights/model.pt. No download, no API key,
no network access, no user interaction.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from evaluate import Restorer, load_gray, find_inputs, restore_chunk


def resolve_weights(explicit=None):
    """models/model.pt is the submission location; weights/model.pt is the
    original repo location. Accept either so both layouts work."""
    if explicit:
        return explicit
    for rel in (('models', 'model.pt'), ('weights', 'model.pt')):
        p = os.path.join(HERE, *rel)
        if os.path.exists(p):
            return p
    return os.path.join(HERE, 'models', 'model.pt')


def main():
    p = argparse.ArgumentParser(
        description='Restore degraded semiconductor inspection images.',
        usage='python run.py <input-dir> <output-dir>')
    p.add_argument('input_dir', nargs='?', default=None,
                   help='directory of degraded .npy files (searched recursively)')
    p.add_argument('output_dir', nargs='?', default=None,
                   help='directory for restored .npy files (created if absent)')
    # Flag forms kept so existing commands and scripts continue to work.
    p.add_argument('--input-dir', dest='input_flag', default=None)
    p.add_argument('--output-dir', dest='output_flag', default=None)
    p.add_argument('--weights', default=None)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--ensemble', action='store_true',
                   help='8x self-ensemble: ~+0.2 dB, 8x slower. Off by default.')
    p.add_argument('--device', default=None, help='cuda | cpu (default: auto)')
    args = p.parse_args()

    in_dir = args.input_dir or args.input_flag
    out_dir = args.output_dir or args.output_flag
    if not in_dir or not out_dir:
        p.error('usage: python run.py <input-dir> <output-dir>')

    weights = resolve_weights(args.weights)
    if not os.path.exists(weights):
        sys.exit(f'ERROR: weights not found at {weights}')
    if not os.path.exists(in_dir):
        sys.exit(f'ERROR: input path does not exist: {in_dir}')

    dev = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(out_dir, exist_ok=True)

    sd = torch.load(weights, map_location=dev)
    net = Restorer(dim=sd['inp.weight'].shape[0]).to(dev)
    net.load_state_dict(sd)
    net.eval()

    root, files = find_inputs(in_dir)
    if not files:
        sys.exit(f'ERROR: no .npy files found under {in_dir}')

    # KLA's Test_NoisyLR.zip unpacks to a NoisyLR/ subfolder. If the grader
    # points at the parent, mirroring that subfolder would hide every output
    # from a flat glob of the output directory. So: flatten to bare filenames
    # whenever they are unique, and only mirror structure if flattening would
    # collide. Either way each output keeps its input's filename.
    names = [os.path.basename(f) for f in files]
    flatten = len(set(names)) == len(names)
    out_names = names if flatten else files

    print(f'device : {dev}')
    print(f'weights: {weights}')
    print(f'inputs : {len(files)} files from {in_dir}')
    print(f'outputs: {out_dir}')
    print(f'layout : {"flat" if flatten else "mirroring input subdirectories"}',
          flush=True)

    # group by shape so each batch is homogeneous
    name_of = dict(zip(files, out_names))
    shapes = {}
    for f in files:
        s = tuple(d for d in np.load(os.path.join(root, f), mmap_mode='r').shape
                  if d != 1)
        shapes.setdefault(s, []).append(f)

    t0, done = time.time(), 0
    with torch.no_grad():
        for _, group in shapes.items():
            for i in range(0, len(group), args.batch_size):
                chunk = group[i:i + args.batch_size]
                arr = np.stack([load_gray(os.path.join(root, f)) for f in chunk])
                out = restore_chunk(net, arr, dev, ensemble=args.ensemble)
                # Contract enforcement: finite, float32, in [0, 1], shape (H, W).
                out = np.nan_to_num(out.astype(np.float32),
                                    nan=0.0, posinf=1.0, neginf=0.0)
                out = np.clip(out, 0.0, 1.0)
                for f, a in zip(chunk, out):
                    dst = os.path.join(out_dir, name_of[f])
                    d = os.path.dirname(dst)
                    if d:
                        os.makedirs(d, exist_ok=True)
                    np.save(dst, a.astype(np.float32))
                done += len(chunk)
                print(f'  {done}/{len(files)}', flush=True)

    dt = time.time() - t0
    print(f'\ndone: {len(files)} images in {dt:.2f} s '
          f'({dt / len(files) * 1000:.2f} ms/image)')


if __name__ == '__main__':
    main()
