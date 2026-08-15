#!/usr/bin/env python3
"""
Render .npy images as viewable 8-bit grayscale PNGs.

    python preview.py --input-dir restored --output-dir preview
    python preview.py --input-dir samples  --output-dir preview_in --stretch

The .npy files remain the scored artefact; these PNGs exist so a reviewer can
look at the results without writing any code. Subdirectory structure is
mirrored, same basenames, .png extension.

Depends only on numpy and the standard library, so it runs under the same
minimal requirements.txt as evaluate.py.
"""

import argparse
import os
import struct
import sys
import zlib

import numpy as np


def write_png_gray(path, img_u8):
    """Write a 2-D uint8 array as an 8-bit grayscale PNG."""
    h, w = img_u8.shape
    raw = b''.join(b'\x00' + img_u8[r].tobytes() for r in range(h))

    def chunk(tag, data):
        c = struct.pack('>I', len(data)) + tag + data
        return c + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)

    with open(path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n')
        f.write(chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 0, 0, 0, 0)))
        f.write(chunk(b'IDAT', zlib.compress(raw, 6)))
        f.write(chunk(b'IEND', b''))


def to_u8(a, stretch):
    """Map a float array to 0-255.

    Restored outputs are already in [0,1], so clipping is exact. Degraded
    inputs are NOT: speckle pushes pixels past 1.0, and clipping those would
    misrepresent the input the model actually receives. --stretch rescales by
    the observed range instead, so nothing is silently discarded.
    """
    a = np.squeeze(a).astype(np.float32)
    if a.ndim != 2:
        raise ValueError(f"expected 2-D grayscale, got shape {a.shape}")
    if stretch:
        lo, hi = float(a.min()), float(a.max())
        a = (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)
    else:
        a = np.clip(a, 0.0, 1.0)
    return np.rint(a * 255.0).astype(np.uint8)


def main():
    p = argparse.ArgumentParser(
        description="Render .npy images as viewable grayscale PNGs.")
    p.add_argument('--input-dir', required=True,
                   help='.npy file, or directory (searched recursively)')
    p.add_argument('--output-dir', required=True, help='directory for .png files')
    p.add_argument('--stretch', action='store_true',
                   help='rescale by observed min/max instead of clipping to '
                        '[0,1]; use for degraded inputs, which exceed 1.0')
    p.add_argument('--limit', type=int, default=0,
                   help='render only the first N images (0 = all)')
    args = p.parse_args()

    if not os.path.exists(args.input_dir):
        sys.exit(f"ERROR: input path does not exist: {args.input_dir}")

    if os.path.isfile(args.input_dir):
        root = os.path.dirname(os.path.abspath(args.input_dir))
        files = [os.path.basename(args.input_dir)]
    else:
        root, files = args.input_dir, []
        for dirpath, _, names in os.walk(args.input_dir):
            for f in names:
                if f.endswith('.npy') and not f.startswith('._'):
                    files.append(os.path.relpath(os.path.join(dirpath, f),
                                                 args.input_dir))
        files.sort()

    if not files:
        sys.exit(f"ERROR: no .npy files found under {args.input_dir}")
    if args.limit:
        files = files[:args.limit]

    os.makedirs(args.output_dir, exist_ok=True)
    lo = hi = None
    for f in files:
        a = np.load(os.path.join(root, f))
        lo = min(lo, float(a.min())) if lo is not None else float(a.min())
        hi = max(hi, float(a.max())) if hi is not None else float(a.max())
        dst = os.path.join(args.output_dir, os.path.splitext(f)[0] + '.png')
        d = os.path.dirname(dst)
        if d:
            os.makedirs(d, exist_ok=True)
        write_png_gray(dst, to_u8(a, args.stretch))

    print(f"wrote {len(files)} PNG(s) to {args.output_dir}")
    print(f"source value range: [{lo:.3f}, {hi:.3f}]  "
          f"({'min/max stretched' if args.stretch else 'clipped to [0,1]'})")
    if not args.stretch and hi > 1.0:
        print("note: input exceeds 1.0 and was clipped for display; "
              "re-run with --stretch to see the full range")


if __name__ == '__main__':
    main()
