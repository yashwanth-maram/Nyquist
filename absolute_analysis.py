#!/usr/bin/env python3
"""
Absolute performance analysis.

Answers "where does this model sit on a scale that exists independently of our
own baseline", not "how much better is it than bicubic".

    python absolute_analysis.py --gt-dir <train/GT> --lr-dir <train/NoisyLR> \
                                --weights weights/model.pt --steps 7200

Three parts:
  TASK 1  GT noise-floor ceiling      (CPU, ~2 min, no training)
  TASK 2  matched external baseline   (GPU, ~30 min at 7200 steps)
  TASK 3  positioning statement       (generated from the measured numbers)

Every number printed is measured. Nothing is extrapolated.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluate import Restorer, estimate_nlf, vst, ivst

try:
    from pytorch_msssim import ssim as _ssim, ms_ssim
except ImportError:
    raise SystemExit("pip install pytorch-msssim scikit-learn lpips")
from sklearn.cluster import KMeans


# ==========================================================================
# TASK 1 — GT noise floor and the PSNR ceiling it implies
# ==========================================================================
def wavelet_mad(x):
    """Donoho's estimator: MAD of the finest diagonal wavelet subband.

    Chosen over a Laplacian-MAD because the HH subband is the standard noise
    estimator in the denoising literature and is less sensitive to smooth
    gradients. Both were computed during development and agree to within ~10%
    at the median.

    DIRECTION OF BIAS, stated because it matters: on textured images this
    OVER-estimates sigma, because fine texture appears in the HH subband. An
    over-estimated sigma yields an UNDER-estimated ceiling, which flatters the
    model under test. The true ceiling is therefore at least this high.
    """
    a, b = x[::2, ::2], x[::2, 1::2]
    c, d = x[1::2, ::2], x[1::2, 1::2]
    hh = (a - b - c + d) / 2.0
    return np.median(np.abs(hh)) / 0.6745


def task1_ceiling(GT, val_idx, mine_psnr, bicubic_psnr, out):
    print('=' * 72)
    print('TASK 1 — GT noise floor and the PSNR ceiling')
    print('=' * 72)

    t0 = time.time()
    sigma = np.array([wavelet_mad(GT[i].astype(np.float64)) for i in range(len(GT))])
    print(f'estimated over {len(GT)} GT images in {time.time()-t0:.0f}s\n')

    def report(sig, tag):
        ceil = 10 * np.log10(1.0 / np.maximum(sig, 1e-9) ** 2)
        print(f'  {tag}   n = {len(sig)}')
        print(f'    sigma    p10 {np.percentile(sig,10):.5f}   p50 {np.percentile(sig,50):.5f}'
              f'   p90 {np.percentile(sig,90):.5f}   max {sig.max():.5f}')
        print(f'    ceiling  p10 {np.percentile(ceil,10):6.2f}   p50 {np.percentile(ceil,50):6.2f}'
              f'   p90 {np.percentile(ceil,90):6.2f}   min {ceil.min():6.2f}  dB')
        print(f'    MEAN of per-image ceilings: {ceil.mean():.2f} dB'
              f'   <-- matches how PSNR is averaged\n')
        return ceil

    ceil_all = report(sigma, 'all GT')
    ceil_val = report(sigma[val_idx], 'held-out split')

    C = ceil_val.mean()
    pos = (mine_psnr - bicubic_psnr) / (C - bicubic_psnr) * 100

    print('  ABSOLUTE POSITION on the held-out split')
    print(f'    bicubic x2                {bicubic_psnr:6.3f} dB     0.0%')
    print(f'    this model                {mine_psnr:6.3f} dB   {pos:5.1f}%')
    print(f'    GT noise-floor ceiling    {C:6.3f} dB   100.0%')
    print(f'    headroom remaining        {C - mine_psnr:6.3f} dB\n')

    # sensitivity: how does the answer move with the estimator's assumption?
    print('  SENSITIVITY — position under different ceiling assumptions')
    for lbl, c in [('mean per-image (headline)', ceil_val.mean()),
                   ('median',                    np.median(ceil_val)),
                   ('conservative p10',          np.percentile(ceil_val, 10)),
                   ('worst single image',        ceil_val.min())]:
        p = (mine_psnr - bicubic_psnr) / (c - bicubic_psnr) * 100
        print(f'    {lbl:26s} ceiling {c:6.2f} dB  ->  {p:5.1f}%')
    print()

    r = dict(n_gt=len(GT), n_val=len(val_idx),
             sigma_p10=float(np.percentile(sigma, 10)),
             sigma_p50=float(np.percentile(sigma, 50)),
             sigma_p90=float(np.percentile(sigma, 90)),
             sigma_max=float(sigma.max()),
             ceiling_val_mean=float(ceil_val.mean()),
             ceiling_val_median=float(np.median(ceil_val)),
             ceiling_val_p10=float(np.percentile(ceil_val, 10)),
             ceiling_val_min=float(ceil_val.min()),
             position_pct=float(pos),
             headroom_db=float(ceil_val.mean() - mine_psnr))
    json.dump(r, open(os.path.join(out, 'task1_ceiling.json'), 'w'), indent=2)
    return r


# ==========================================================================
# TASK 2 — matched external baseline
# ==========================================================================
class NAFBlock(nn.Module):
    """NAFNet block as published (Chen et al., ECCV 2022).

    Identical to the block in our model MINUS the FiLM conditioning layer.
    That single difference is deliberate: it isolates the contribution of
    degradation conditioning.
    """
    def __init__(self, dim):
        super().__init__()
        self.n1 = nn.GroupNorm(1, dim)
        self.c1 = nn.Conv2d(dim, dim * 2, 1)
        self.dw = nn.Conv2d(dim * 2, dim * 2, 3, padding=1, groups=dim * 2)
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dim, dim, 1))
        self.c2 = nn.Conv2d(dim, dim, 1)
        self.n2 = nn.GroupNorm(1, dim)
        self.c3 = nn.Conv2d(dim, dim * 2, 1)
        self.c4 = nn.Conv2d(dim, dim, 1)
        self.b1 = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.b2 = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        r = x
        x = self.dw(self.c1(self.n1(x)))
        a, b = x.chunk(2, 1)
        x = a * b
        x = x * self.ca(x)
        x = self.c2(x)
        x = r + x * self.b1
        r = x
        a, b = self.c3(self.n2(x)).chunk(2, 1)
        x = self.c4(a * b)
        return r + x * self.b2


class NAFNetBaseline(nn.Module):
    """NAFNet for 2x restoration. Matched capacity to the model under test.

    FAIRNESS NOTES — read before quoting any comparison:

    * Same width (dim=64), same block counts, same U-Net topology.
    * Given the SAME bicubic-residual output head. This is standard SR practice,
      not a contribution of ours, and withholding it would cripple the baseline
      by design.
    * NOT given: the variance-stabilising transform, FiLM degradation
      conditioning, or the denoise gate. Those are exactly the three components
      we claim, so this comparison isolates them.

    The result is a slightly harder baseline than published NAFNet, which is
    the direction of error we want.
    """
    def __init__(self, dim=64):
        super().__init__()
        self.inp = nn.Conv2d(1, dim, 3, padding=1)
        self.e1 = nn.Sequential(*[NAFBlock(dim) for _ in range(2)])
        self.d1 = nn.Conv2d(dim, dim * 2, 2, stride=2)
        self.e2 = nn.Sequential(*[NAFBlock(dim * 2) for _ in range(2)])
        self.d2 = nn.Conv2d(dim * 2, dim * 4, 2, stride=2)
        self.mid = nn.Sequential(*[NAFBlock(dim * 4) for _ in range(3)])
        self.u2 = nn.Sequential(nn.Conv2d(dim * 4, dim * 8, 1), nn.PixelShuffle(2))
        self.f2 = nn.Sequential(*[NAFBlock(dim * 2) for _ in range(2)])
        self.u1 = nn.Sequential(nn.Conv2d(dim * 2, dim * 4, 1), nn.PixelShuffle(2))
        self.f1 = nn.Sequential(*[NAFBlock(dim) for _ in range(2)])
        self.hr_up = nn.Sequential(nn.Conv2d(dim, dim * 4, 3, padding=1),
                                   nn.PixelShuffle(2))
        self.hr_ref = nn.Sequential(*[NAFBlock(dim) for _ in range(2)])
        self.hr_out = nn.Conv2d(dim, 1, 3, padding=1)
        nn.init.zeros_(self.hr_out.weight)
        nn.init.zeros_(self.hr_out.bias)

    def forward(self, y, a=None, b=None):       # a, b accepted and ignored
        m = y.mean(dim=[1, 2, 3], keepdim=True)
        s = y.std(dim=[1, 2, 3], keepdim=True).clamp(min=1e-6)
        x = self.inp((y - m) / s)
        x = self.e1(x); s1 = x; x = self.d1(x)
        x = self.e2(x); s2 = x; x = self.d2(x)
        x = self.mid(x)
        x = self.u2(x) + s2; x = self.f2(x)
        x = self.u1(x) + s1; x = self.f1(x)
        h = self.hr_ref(self.hr_up(x))
        base = F.interpolate(y, scale_factor=2, mode='bicubic', align_corners=False)
        return base + self.hr_out(h), None


# ==========================================================================
def degrade_batch(gt, a_rng=(0.0, 0.22), s_rng=(0.0, 0.19),
                  p_speckle=0.85, p_gauss=0.80):
    B = gt.shape[0]; dev = gt.device
    mu = F.avg_pool2d(gt, 2)
    a = torch.rand(B, 1, 1, 1, device=dev) * (a_rng[1] - a_rng[0]) + a_rng[0]
    s = torch.rand(B, 1, 1, 1, device=dev) * (s_rng[1] - s_rng[0]) + s_rng[0]
    a = a * (torch.rand(B, 1, 1, 1, device=dev) < p_speckle).float()
    s = s * (torch.rand(B, 1, 1, 1, device=dev) < p_gauss).float()
    k = (1.0 / a.clamp(min=1e-6)).expand_as(mu)
    y = torch.where(a > 1e-6, mu * torch.distributions.Gamma(k, k).sample(), mu)
    y = y + torch.randn_like(mu) * s
    return y, mu, a.view(-1), (s.view(-1) ** 2)


def charb(x, y, eps=1e-3):
    return torch.sqrt((x - y) ** 2 + eps ** 2).mean()


def train_matched(net, GT, LR, trn_idx, dev, steps, bs=16, lr=6e-4, p_real=0.5,
                  tag='baseline'):
    """Identical recipe to the model under test: same data mixture, same loss,
    same optimiser, same schedule, same step count."""
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.10)
    sob = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                       device=dev).view(1, 1, 3, 3)
    k5 = (1 / 16) * torch.tensor([[1., 4., 6., 4., 1.]], device=dev)
    k5 = (k5.T @ k5).view(1, 1, 5, 5)
    ema = {k: v.detach().clone() for k, v in net.state_dict().items()}

    def sobel(x):
        return torch.cat([F.conv2d(x, sob, padding=1),
                          F.conv2d(x, sob.transpose(2, 3), padding=1)], 1)

    def hp(t):
        return t - F.conv2d(F.pad(t, (2, 2, 2, 2), mode='reflect'), k5)

    t0 = time.time(); done = 0
    while done < steps:
        perm = np.random.permutation(trn_idx)
        net.train()
        for i in range(0, len(perm) - bs + 1, bs):
            if done >= steps:
                break
            ids = np.sort(perm[i:i + bs])
            gt = torch.from_numpy(GT[ids][:, None].copy()).to(dev)
            if np.random.rand() < p_real:
                y = torch.from_numpy(LR[ids][:, None].copy()).to(dev)
                mu = F.avg_pool2d(gt, 2)
                with torch.no_grad():
                    a, b = estimate_nlf(y)
            else:
                if np.random.rand() < 0.5: gt = torch.flip(gt, [3])
                if np.random.rand() < 0.5: gt = torch.flip(gt, [2])
                k = np.random.randint(4)
                if k: gt = torch.rot90(gt, k, [2, 3])
                y, mu, a, b = degrade_batch(gt)

            out, mh = net(y, a, b)
            out = out.float()
            loss = (charb(out, gt)
                    + 0.3 * (1 - ms_ssim(out.clamp(0, 1), gt, data_range=1.0, win_size=7))
                    + 0.1 * charb(sobel(out), sobel(gt))
                    + 0.4 * charb(hp(out), hp(gt)))
            if mh is not None:                       # only our model has this head
                loss = loss + 0.5 * charb(mh.float(), mu)
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True); sch.step(); done += 1; continue
            opt.zero_grad(set_to_none=True); loss.backward()
            gn = nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            if torch.isfinite(gn): opt.step()
            sch.step(); done += 1
            with torch.no_grad():
                for k_, v in net.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema[k_].mul_(0.999).add_(v, alpha=0.001)
                    else:
                        ema[k_].copy_(v)
            if done % 1000 == 0:
                print(f'    {tag}: {done}/{steps} steps '
                      f'[{(time.time()-t0)/60:.1f}m] loss {loss.item():.4f}', flush=True)
    net.load_state_dict(ema); net.eval()
    return net


@torch.no_grad()
def measure(net, GT, LR, ids, dev, lp, tag, bs=16):
    P = S = L = 0.0; n = 0
    for i in range(0, len(ids), bs):
        b = ids[i:i + bs]
        y = torch.from_numpy(LR[b][:, None].copy()).float().to(dev)
        gt = torch.from_numpy(GT[b][:, None].copy()).float().to(dev)
        a_, b_ = estimate_nlf(y)
        o = net(y, a_, b_)[0].clamp(0, 1)
        mse = ((o - gt) ** 2).mean(dim=[1, 2, 3]).clamp(min=1e-12)
        P += (-10 * torch.log10(mse)).sum().item()
        S += _ssim(o, gt, data_range=1.0, size_average=False).sum().item()
        L += lp(o.repeat(1, 3, 1, 1) * 2 - 1, gt.repeat(1, 3, 1, 1) * 2 - 1).sum().item()
        n += len(b)
    return P / n, S / n, L / n


@torch.no_grad()
def latency(net, dev, n=32, reps=20):
    y = torch.rand(n, 1, 128, 128, device=dev)
    a, b = estimate_nlf(y)
    for _ in range(3):
        net(y, a, b)
    if dev == 'cuda':
        torch.cuda.synchronize()
    t = time.time()
    for _ in range(reps):
        net(y, a, b)
    if dev == 'cuda':
        torch.cuda.synchronize()
    return (time.time() - t) / reps / n * 1000


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gt-dir', required=True)
    p.add_argument('--lr-dir', required=True)
    p.add_argument('--weights', default='weights/model.pt')
    p.add_argument('--out', default='docs')
    p.add_argument('--steps', type=int, default=7200,
                   help='MUST match the step count of the model under test')
    p.add_argument('--val-limit', type=int, default=None,
                   help='evaluate on the first N val images only (label results as partial)')
    p.add_argument('--skip-task2', action='store_true')
    p.add_argument('--device', default=None)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dev = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def load(d):
        fs = sorted(f for f in os.listdir(d)
                    if f.endswith('.npy') and not f.startswith('._'))
        return np.stack([np.load(os.path.join(d, f)) for f in fs]).astype(np.float32)

    print('loading...')
    GT, LR = load(args.gt_dir), load(args.lr_dir)
    print(f'GT {GT.shape}  LR {LR.shape}\n')

    # exact reproduction of the held-out split
    sig = np.stack([F.adaptive_avg_pool2d(
        torch.from_numpy(GT[i][None, None].copy()), 16).view(-1).numpy()
        for i in range(len(GT))])
    sig = (sig - sig.mean(1, keepdims=True)) / (sig.std(1, keepdims=True) + 1e-8)
    lab = KMeans(n_clusters=64, n_init=4, random_state=0).fit_predict(sig)
    rng = np.random.default_rng(0); val_c, n = [], 0
    for c in rng.permutation(64):
        if n >= 300: break
        val_c.append(c); n += (lab == c).sum()
    val_idx = np.where(np.isin(lab, val_c))[0]
    trn_idx = np.where(~np.isin(lab, val_c))[0]
    eval_idx = val_idx[:args.val_limit] if args.val_limit else val_idx
    partial = args.val_limit is not None
    print(f'train {len(trn_idx)}  val {len(val_idx)}'
          + (f'  (evaluating on {len(eval_idx)} — PARTIAL)' if partial else '') + '\n')

    import lpips
    lp = lpips.LPIPS(net='alex').to(dev)

    # ---- the model under test ----
    sd = torch.load(args.weights, map_location=dev)
    mine = Restorer(dim=sd['inp.weight'].shape[0]).to(dev)
    mine.load_state_dict(sd); mine.eval()
    p_mine = sum(q.numel() for q in mine.parameters())

    print('measuring the model under test...')
    P_m, S_m, L_m = measure(mine, GT, LR, eval_idx, dev, lp, 'ours')
    ms_m = latency(mine, dev)

    # ---- bicubic ----
    class Bic(nn.Module):
        def forward(self, y, a=None, b=None):
            return F.interpolate(y, scale_factor=2, mode='bicubic',
                                 align_corners=False).clamp(0, 1), None
    P_b, S_b, L_b = measure(Bic().to(dev), GT, LR, eval_idx, dev, lp, 'bicubic')

    t1 = task1_ceiling(GT, eval_idx, P_m, P_b, args.out)

    rows = [('bicubic x2', P_b, S_b, L_b, 0, 0.0),
            ('ours', P_m, S_m, L_m, p_mine, ms_m)]

    # ---- TASK 2 ----
    if not args.skip_task2:
        print('=' * 72)
        print(f'TASK 2 — matched NAFNet baseline, {args.steps} steps')
        print('=' * 72)
        torch.manual_seed(0)
        base = NAFNetBaseline(dim=64).to(dev)
        p_base = sum(q.numel() for q in base.parameters())
        print(f'  baseline params {p_base/1e6:.2f} M   vs ours {p_mine/1e6:.2f} M\n')
        base = train_matched(base, GT, LR, trn_idx, dev, args.steps, tag='NAFNet')
        P_n, S_n, L_n = measure(base, GT, LR, eval_idx, dev, lp, 'nafnet')
        ms_n = latency(base, dev)
        torch.save(base.state_dict(), os.path.join(args.out, 'nafnet_baseline.pt'))
        rows.insert(1, ('NAFNet (matched)', P_n, S_n, L_n, p_base, ms_n))

    # ---- table ----
    C = t1['ceiling_val_mean']
    print('=' * 72)
    print('ABSOLUTE COMPARISON' + ('  [PARTIAL — subset of val]' if partial else ''))
    print('=' * 72)
    print(f'{"model":22s} {"PSNR":>8s} {"SSIM":>8s} {"LPIPS":>8s} {"params":>10s} {"ms/img":>9s}')
    print('-' * 72)
    for nm, P_, S_, L_, pr, ms in rows:
        print(f'{nm:22s} {P_:8.3f} {S_:8.4f} {L_:8.4f} '
              f'{(f"{pr/1e6:.2f} M" if pr else "-"):>10s} {(f"{ms:.2f}" if ms else "-"):>9s}')
    print(f'{"GT noise-floor ceiling":22s} {C:8.3f} {"-":>8s} {"-":>8s} {"-":>10s} {"-":>9s}')
    print()

    md = ['| model | PSNR | SSIM | LPIPS | params | ms/image |',
          '|:--|--:|--:|--:|--:|--:|']
    for nm, P_, S_, L_, pr, ms in rows:
        md.append(f'| {nm} | {P_:.3f} dB | {S_:.4f} | {L_:.4f} | '
                  f'{f"{pr/1e6:.2f} M" if pr else "—"} | {f"{ms:.2f}" if ms else "—"} |')
    md.append(f'| *GT noise-floor ceiling* | *{C:.3f} dB* | — | — | — | — |')

    # ---- TASK 3 ----
    pos = t1['position_pct']
    if len(rows) > 2:
        _, P_n, S_n, L_n, p_base, ms_n = rows[1]
        d = P_m - P_n
        vs = (f'It sits {abs(d):.3f} dB {"above" if d > 0 else "below"} a capacity-matched '
              f'NAFNet baseline trained on identical data for the same {args.steps} steps '
              f'({p_base/1e6:.2f} M parameters, {ms_n:.2f} ms/image against '
              f'{p_mine/1e6:.2f} M and {ms_m:.2f} ms/image). ')
    else:
        vs = ''
    para = (f'On a {len(eval_idx)}-image held-out split, using KLA\'s actual degraded files '
            f'and blind parameter estimation, the model scores {P_m:.3f} dB PSNR, '
            f'{S_m:.4f} SSIM and {L_m:.4f} LPIPS. Bicubic x2 upsampling scores {P_b:.3f} dB; '
            f'the ground truth\'s own noise floor caps any method at approximately '
            f'{C:.1f} dB. The model therefore stands about {pos:.0f}% of the way from '
            f'trivial interpolation to the information-theoretic limit of this data, '
            f'with {C - P_m:.1f} dB of headroom remaining. {vs}'
            f'All figures are measured, not extrapolated.')

    print('=' * 72); print('TASK 3 — positioning statement'); print('=' * 72)
    print(para); print(f'\n[{len(para.split())} words]\n')

    open(os.path.join(args.out, 'absolute_comparison.md'), 'w').write(
        '# Absolute performance\n\n' + '\n'.join(md) +
        '\n\n## Positioning\n\n' + para + '\n')
    json.dump({'rows': [{'model': r[0], 'psnr': r[1], 'ssim': r[2], 'lpips': r[3],
                         'params': r[4], 'ms_per_img': r[5]} for r in rows],
               'ceiling_db': C, 'position_pct': pos, 'partial': partial,
               'n_eval': len(eval_idx), 'steps': args.steps},
              open(os.path.join(args.out, 'absolute_analysis.json'), 'w'), indent=2)
    print(f'wrote {args.out}/absolute_comparison.md and absolute_analysis.json')


if __name__ == '__main__':
    main()
