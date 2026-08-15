#!/usr/bin/env python3
"""
KLA PS1 - AI-Based Restoration of Degraded Images
Inference script. Run as-is; no manual edits required.

    python evaluate.py --input-dir <dir of .npy> --output-dir <dir>

Accepts a single .npy file, a flat directory, or nested subdirectories. Writes
each restored image to --output-dir under the SAME filename, mirroring any
subdirectory structure.

Self-contained by design: the model definition is duplicated here rather than
imported from src/, so a broken PYTHONPATH or a missing package cannot stop the
benchmark from running. src/ holds the same code in modular form for reading.

Tested: Python 3.10-3.12, PyTorch 2.0+, CUDA and CPU.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# Constants recovered by measurement (see docs/ablations.md)
# --------------------------------------------------------------------------
A_FLOOR = 5e-3        # below this, the Gaussian limit of the VST is exact
Y_LO, Y_HI = -1.0, 3.0  # observed input range across 3600 supplied images
A_MAX, S_MAX = 0.22, 0.19  # training envelope; estimates are clamped to it


# --------------------------------------------------------------------------
# Variance-stabilising transform
#
# The degradation operator was recovered from 3200 matched pairs as
#     mu = AreaAvg_2x2(x);   y = mu * s + n,   s ~ Gamma(L,1/L),  n ~ N(0,sigma^2)
# giving   Var(y | mu) = a*mu^2 + b   with a = 1/L, b = sigma^2.
#
# For that variance form the stabilising transform has a closed form. Applying
# it makes the noise approximately homoscedastic, so a network trained at one
# severity generalises to any other. (a, b) are estimated per image at
# inference, so no ground truth or prior knowledge is required.
# --------------------------------------------------------------------------
def vst(y, a, b, eps=1e-6):
    a = a.float().view(-1, 1, 1, 1)
    b = b.float().clamp(min=eps).view(-1, 1, 1, 1)
    a_s = a.clamp(min=A_FLOOR)
    z_gauss = y.float() / b.sqrt()                                   # a -> 0 limit
    z_speck = torch.asinh(y.float() * (a_s / b).sqrt()) / a_s.sqrt()
    return torch.where(a < A_FLOOR, z_gauss, z_speck)


def ivst(z, a, b, eps=1e-6):
    a = a.float().view(-1, 1, 1, 1)
    b = b.float().clamp(min=eps).view(-1, 1, 1, 1)
    a_s = a.clamp(min=A_FLOOR)
    y_gauss = z.float() * b.sqrt()
    y_speck = (b / a_s).sqrt() * torch.sinh((z.float() * a_s.sqrt()).clamp(-6, 6))
    return torch.where(a < A_FLOOR, y_gauss, y_speck).clamp(Y_LO, Y_HI)


def estimate_nlf(y, clamp=True):
    """Blind per-image estimation of (a, b). No labels, no priors.

    An Immerkaer high-pass kernel annihilates image structure up to linear, so
    the residual is dominated by noise. Binning by local brightness and
    regressing robust residual variance on brightness-squared recovers a and b.

    Validated at 0.978 correlation with true total variance (8.4% median error).

    The clamp bounds estimates to the training envelope. On high-contrast
    periodic structure the high-pass reads texture as speckle and can
    over-estimate a (worst observed: 1.21 vs a training max of 0.22), which
    causes severe over-denoising. 3 of 400 supplied test images trigger it.
    """
    B = y.shape[0]
    k = torch.tensor([[1., -2., 1.], [-2., 4., -2.], [1., -2., 1.]],
                     device=y.device, dtype=torch.float32).view(1, 1, 3, 3) / 6.
    hp = F.conv2d(F.pad(y.float(), (1, 1, 1, 1), mode='reflect'), k)
    lm = F.avg_pool2d(y.float(), 9, stride=1, padding=4)

    A, Bb = [], []
    for i in range(B):
        h, m = hp[i].flatten(), lm[i].flatten()
        q = torch.quantile(m, torch.linspace(0, 1, 11, device=y.device))
        M, V = [], []
        for j in range(10):
            s = (m >= q[j]) & (m < q[j + 1])
            if s.sum() >= 200:
                M.append(m[s].mean())
                V.append((h[s].abs().median() * 1.4826) ** 2)   # robust std
        if len(M) < 4:                                          # degenerate image
            A.append(torch.tensor(0.03, device=y.device))
            Bb.append(torch.tensor(1e-4, device=y.device))
            continue
        M, V = torch.stack(M), torch.stack(V)
        X = torch.stack([M ** 2, torch.ones_like(M)], 1)
        c = torch.linalg.lstsq(X, V.unsqueeze(1)).solution.squeeze()
        A.append(c[0].clamp(min=0.0))
        Bb.append(c[1].clamp(min=1e-6))
    a, b = torch.stack(A), torch.stack(Bb)
    if clamp:
        a = a.clamp(max=A_MAX)
        b = b.clamp(max=S_MAX ** 2)
    return a, b


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class FiLM(nn.Module):
    """Per-channel scale and shift conditioned on the degradation embedding."""
    def __init__(self, dim, cdim=64):
        super().__init__()
        self.fc = nn.Linear(cdim, dim * 2)

    def forward(self, x, c):
        g, b = self.fc(c).chunk(2, 1)
        return x * (1 + g[:, :, None, None]) + b[:, :, None, None]


class Block(nn.Module):
    """NAFNet-style block: depthwise conv, SimpleGate, channel attention, FiLM."""
    def __init__(self, dim, cdim=64):
        super().__init__()
        self.n1 = nn.GroupNorm(1, dim)
        self.c1 = nn.Conv2d(dim, dim * 2, 1)
        self.dw = nn.Conv2d(dim * 2, dim * 2, 3, padding=1, groups=dim * 2)
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dim, dim, 1))
        self.c2 = nn.Conv2d(dim, dim, 1)
        self.film = FiLM(dim, cdim)
        self.n2 = nn.GroupNorm(1, dim)
        self.c3 = nn.Conv2d(dim, dim * 2, 1)
        self.c4 = nn.Conv2d(dim, dim, 1)
        self.b1 = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.b2 = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x, c):
        r = x
        x = self.dw(self.c1(self.n1(x)))
        a, b = x.chunk(2, 1)
        x = a * b                                   # SimpleGate
        x = x * self.ca(x)
        x = self.film(self.c2(x), c)
        x = r + x * self.b1
        r = x
        a, b = self.c3(self.n2(x)).chunk(2, 1)
        x = self.c4(a * b)
        return r + x * self.b2


class Restorer(nn.Module):
    """Degradation-conditioned restoration network.

    Path: VST -> encoder/decoder -> gated denoise -> inverse VST -> 2x SR head.

    dgate lets the network modulate denoising strength from the measured noise
    level, so a nearly clean input passes through nearly unchanged. Without it
    the model applies fixed denoising and damages low-noise images (measured:
    -4.3 dB on clean input before the gate was added; 30.28 dB with it).
    """
    def __init__(self, dim=64, cdim=64):
        super().__init__()
        self.cond = nn.Sequential(nn.Linear(4, cdim), nn.GELU(),
                                  nn.Linear(cdim, cdim), nn.GELU())
        self.inp = nn.Conv2d(1, dim, 3, padding=1)
        self.e1 = nn.ModuleList([Block(dim, cdim) for _ in range(2)])
        self.d1 = nn.Conv2d(dim, dim * 2, 2, stride=2)
        self.e2 = nn.ModuleList([Block(dim * 2, cdim) for _ in range(2)])
        self.d2 = nn.Conv2d(dim * 2, dim * 4, 2, stride=2)
        self.mid = nn.ModuleList([Block(dim * 4, cdim) for _ in range(3)])
        self.u2 = nn.Sequential(nn.Conv2d(dim * 4, dim * 8, 1), nn.PixelShuffle(2))
        self.f2 = nn.ModuleList([Block(dim * 2, cdim) for _ in range(2)])
        self.u1 = nn.Sequential(nn.Conv2d(dim * 2, dim * 4, 1), nn.PixelShuffle(2))
        self.f1 = nn.ModuleList([Block(dim, cdim) for _ in range(2)])
        self.head_mu = nn.Conv2d(dim, 1, 3, padding=1)
        self.dgate = nn.Sequential(nn.Linear(cdim, 1), nn.Sigmoid())
        self.hr_up = nn.Sequential(nn.Conv2d(dim, dim * 4, 3, padding=1),
                                   nn.PixelShuffle(2))
        self.hr_ref = nn.ModuleList([Block(dim, cdim) for _ in range(2)])
        self.hr_out = nn.Conv2d(dim, 1, 3, padding=1)

    def forward(self, y, a, b):
        av = a.view(-1).clamp(min=A_FLOOR)
        bv = b.view(-1).clamp(min=1e-6)
        c = self.cond(torch.stack([av, bv, av.sqrt(), bv.sqrt()], 1))

        z = vst(y, av, bv)
        zm = z.mean(dim=[1, 2, 3], keepdim=True)
        zs = z.std(dim=[1, 2, 3], keepdim=True).clamp(min=1e-6)
        x = self.inp((z - zm) / zs)

        for blk in self.e1: x = blk(x, c)
        s1 = x; x = self.d1(x)
        for blk in self.e2: x = blk(x, c)
        s2 = x; x = self.d2(x)
        for blk in self.mid: x = blk(x, c)
        x = self.u2(x) + s2
        for blk in self.f2: x = blk(x, c)
        x = self.u1(x) + s1
        for blk in self.f1: x = blk(x, c)

        dg = self.dgate(c).view(-1, 1, 1, 1)
        mu_hat = ivst(z + dg * self.head_mu(x) * zs, av, bv)

        h = self.hr_up(x)
        for blk in self.hr_ref: h = blk(h, c)
        base = F.interpolate(mu_hat, scale_factor=2, mode='bicubic',
                             align_corners=False)
        return base + self.hr_out(h), mu_hat


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------
def restore(net, y, ensemble=True):
    """y: (B,1,H,W) unclipped. Returns (B,1,2H,2W) in [0,1]."""
    a, b = estimate_nlf(y, clamp=True)
    if not ensemble:
        return net(y, a, b)[0].clamp(0, 1)
    acc = 0
    for k in range(4):                              # 4 rotations x 2 flips
        for flip in (False, True):
            t = torch.rot90(y, k, [2, 3])
            if flip:
                t = torch.flip(t, [3])
            o = net(t, a, b)[0]
            if flip:
                o = torch.flip(o, [3])
            acc = acc + torch.rot90(o, -k, [2, 3])
    return (acc / 8).clamp(0, 1)


def load_gray(path):
    """Load a 2-D grayscale .npy, tolerating a singleton channel axis."""
    a = np.squeeze(np.load(path)).astype(np.float32)
    if a.ndim != 2:
        raise ValueError(f"{path}: expected 2-D grayscale, got {np.load(path).shape}")
    return a


def restore_chunk(net, arr, dev, ensemble):
    """Restore a stack of images, halving the batch on OOM down to one at a time.

    Batched inference is the fast path and the default. If a batch will not fit
    in memory this falls back automatically rather than aborting the run, so the
    script still completes on a smaller card or at a larger --batch-size.
    """
    try:
        y = torch.from_numpy(arr)[:, None].to(dev)
        return restore(net, y, ensemble=ensemble).cpu().numpy()[:, 0]
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        if 'out of memory' not in str(e).lower() or len(arr) == 1:
            raise
        y = None
        if dev == 'cuda':
            torch.cuda.empty_cache()
        half = max(1, len(arr) // 2)
        print(f"  out of memory at batch {len(arr)}; retrying at {half}", flush=True)
        return np.concatenate(
            [restore_chunk(net, arr[i:i + half], dev, ensemble)
             for i in range(0, len(arr), half)], 0)


def find_inputs(path):
    """Return (root, [relative paths]). Accepts a file, a flat dir, or nested dirs."""
    if os.path.isfile(path):
        return os.path.dirname(os.path.abspath(path)), [os.path.basename(path)]
    files = []
    for dirpath, _, names in os.walk(path):
        for f in names:
            if f.endswith('.npy') and not f.startswith('._'):
                files.append(os.path.relpath(os.path.join(dirpath, f), path))
    return path, sorted(files)


def main():
    p = argparse.ArgumentParser(
        description="Restore degraded semiconductor inspection images.")
    p.add_argument('--input-dir', required=True,
                   help='.npy file, or directory (searched recursively)')
    p.add_argument('--output-dir', required=True, help='directory for restored .npy')
    p.add_argument('--weights', default=None,
                   help='path to weights (default: weights/model.pt beside this script)')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--ensemble', action='store_true',
                   help='enable 8x self-ensemble: ~+0.2 dB, 8x slower. '
                        'Off by default so results match the reported figures.')
    p.add_argument('--no-ensemble', action='store_true',
                   help=argparse.SUPPRESS)   # accepted for compatibility; now default
    p.add_argument('--device', default=None, help='cuda | cpu (default: auto)')
    args = p.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    weights = args.weights or os.path.join(here, 'weights', 'model.pt')
    if not os.path.exists(weights):
        sys.exit(f"ERROR: weights not found at {weights}\n"
                 f"See README.md - place the file at weights/model.pt or pass --weights")
    if not os.path.exists(args.input_dir):
        sys.exit(f"ERROR: input path does not exist: {args.input_dir}")

    dev = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    sd = torch.load(weights, map_location=dev)
    dim = sd['inp.weight'].shape[0]                 # infer width from checkpoint
    net = Restorer(dim=dim).to(dev)
    net.load_state_dict(sd)
    net.eval()

    root, files = find_inputs(args.input_dir)
    if not files:
        sys.exit(f"ERROR: no .npy files found under {args.input_dir}")

    n_par = sum(q.numel() for q in net.parameters())
    print(f"device       : {dev}"
          f"{' (' + torch.cuda.get_device_name(0) + ')' if dev == 'cuda' else ''}")
    print(f"parameters   : {n_par/1e6:.2f} M")
    print(f"self-ensemble: {'on (8x)' if args.ensemble else 'off'}")
    print(f"inputs       : {len(files)} files from {args.input_dir}")
    print(f"outputs      : {args.output_dir}", flush=True)

    # group by shape so batches are homogeneous (handles mixed 128 and 256 inputs)
    shapes = {}
    for f in files:
        s = tuple(d for d in np.load(os.path.join(root, f), mmap_mode='r').shape
                  if d != 1)
        shapes.setdefault(s, []).append(f)
    if len(shapes) > 1:
        print(f"note         : mixed input sizes {sorted(shapes.keys())}", flush=True)

    if dev == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    done = 0

    with torch.no_grad():
        for shape, group in shapes.items():
            for i in range(0, len(group), args.batch_size):
                chunk = group[i:i + args.batch_size]
                arr = np.stack([load_gray(os.path.join(root, f)) for f in chunk])
                # NOTE: input is deliberately NOT clipped. Speckle pushes up to
                # 12% of pixels beyond [0,1]; clipping destroys real signal.
                out = restore_chunk(net, arr, dev,
                                    ensemble=args.ensemble).astype(np.float32)
                for f, a in zip(chunk, out):
                    dst = os.path.join(args.output_dir, f)
                    d = os.path.dirname(dst)
                    if d:
                        os.makedirs(d, exist_ok=True)
                    np.save(dst, a)
                done += len(chunk)
                print(f"  {done}/{len(files)}", flush=True)

    if dev == 'cuda':
        torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"\ndone: {len(files)} images in {dt:.2f} s "
          f"({dt/len(files)*1000:.2f} ms/image, end-to-end incl. I/O)")


if __name__ == '__main__':
    main()