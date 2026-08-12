#!/usr/bin/env python3
"""
Visual comparison: degraded input, bicubic baseline, and our restoration.

    python compare.py --input 000374.npy
    python compare.py --input path/to/folder --max 8
    python compare.py --input folder --indices 17,195,257,374
    python compare.py --input 000374.npy --zoom 40,30,48

Writes a PNG panel. Bicubic is included because it is what a reviewer gets for
free; the meaningful comparison is against that, not against imagination.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evaluate import Restorer, estimate_nlf, restore


def load_model(weights, dev):
    sd = torch.load(weights, map_location=dev)
    dim = sd['inp.weight'].shape[0]
    net = Restorer(dim=dim).to(dev)
    net.load_state_dict(sd)
    net.eval()
    return net


def gather(path, indices=None, limit=6):
    """Return a list of (name, array) from a file or a directory."""
    if os.path.isfile(path):
        return [(os.path.basename(path), np.load(path).astype(np.float32))]

    files = sorted(f for f in os.listdir(path)
                   if f.endswith('.npy') and not f.startswith('._'))
    if indices:
        want = {f'{int(i):06d}.npy' for i in indices}
        files = [f for f in files if f in want]
        missing = want - set(files)
        if missing:
            print(f"not found: {sorted(missing)}")
    else:
        files = files[:limit]
    return [(f, np.load(os.path.join(path, f)).astype(np.float32)) for f in files]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='.npy file or directory')
    p.add_argument('--out', default='comparison.png')
    p.add_argument('--weights', default=None)
    p.add_argument('--indices', default=None,
                   help='comma-separated, e.g. 17,195,257,374')
    p.add_argument('--max', type=int, default=6, help='max images from a directory')
    p.add_argument('--zoom', default=None,
                   help='y,x,size — crop region in INPUT coordinates')
    p.add_argument('--no-ensemble', action='store_true')
    p.add_argument('--device', default=None)
    args = p.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    weights = args.weights or os.path.join(here, 'weights', 'model.pt')
    dev = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    net = load_model(weights, dev)

    idx = [s.strip() for s in args.indices.split(',')] if args.indices else None
    items = gather(args.input, idx, args.max)
    if not items:
        raise SystemExit(f"no .npy files found in {args.input}")
    print(f"{len(items)} image(s), device {dev}, "
          f"ensemble {'off' if args.no_ensemble else 'on'}")

    zoom = None
    if args.zoom:
        y0, x0, sz = (int(v) for v in args.zoom.split(','))
        zoom = (y0, x0, sz)

    rows = []
    for name, inp in items:
        y = torch.from_numpy(inp[None, None]).to(dev)      # never clipped
        a, b = estimate_nlf(y, clamp=True)
        a_raw, b_raw = estimate_nlf(y, clamp=False)
        with torch.no_grad():
            out = restore(net, y, ensemble=not args.no_ensemble)
        bic = F.interpolate(y, scale_factor=2, mode='bicubic',
                            align_corners=False).clamp(0, 1)
        rows.append(dict(
            name=name, inp=inp,
            bic=bic.cpu().numpy()[0, 0], out=out.cpu().numpy()[0, 0],
            a=a.item(), s=b.item() ** 0.5,
            a_raw=a_raw.item(), s_raw=b_raw.item() ** 0.5,
            lo=float(inp.min()), hi=float(inp.max()),
            over=float((inp > 1).mean() * 100)))
        flag = '  <-- clamped' if a_raw.item() > a.item() + 1e-6 else ''
        print(f"  {name}: a={a.item():.4f} sigma={b.item()**0.5:.4f} "
              f"(raw a={a_raw.item():.4f}){flag}")

    ncol = 3 if zoom is None else 5
    fig, ax = plt.subplots(len(rows), ncol, figsize=(4.3 * ncol, 4.6 * len(rows)),
                           squeeze=False)

    for r, d in enumerate(rows):
        panels = [
            (d['inp'], f"{d['name']}  input {d['inp'].shape[0]}²\n"
                       f"range [{d['lo']:.2f}, {d['hi']:.2f}]  ·  {d['over']:.1f}% > 1.0"),
            (d['bic'], f"bicubic ×2  {d['bic'].shape[0]}²\n(what you get for free)"),
            (d['out'], f"restored {d['out'].shape[0]}²\n"
                       f"est a={d['a']:.3f}  σ={d['s']:.3f}"),
        ]
        if zoom:
            y0, x0, sz = zoom
            panels += [
                (d['bic'][y0*2:y0*2+sz*2, x0*2:x0*2+sz*2], 'bicubic — zoom'),
                (d['out'][y0*2:y0*2+sz*2, x0*2:x0*2+sz*2], 'restored — zoom'),
            ]
            panels[0] = (d['inp'][y0:y0+sz, x0:x0+sz],
                         f"{d['name']}  input — zoom")

        for c, (im, title) in enumerate(panels):
            ax[r, c].imshow(im, cmap='gray', vmin=0, vmax=1,
                            interpolation='nearest' if im.shape[0] <= 128 else 'antialiased')
            ax[r, c].set_title(title, fontsize=9)
            ax[r, c].axis('off')

    plt.tight_layout()
    plt.savefig(args.out, dpi=110, bbox_inches='tight')
    print(f"\nsaved {args.out}")


if __name__ == '__main__':
    main()
