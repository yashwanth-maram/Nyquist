#!/usr/bin/env python3
"""
Test a corrected noise estimator against the current one.

Changes NOTHING in the shipped pipeline. Runs both estimators over the held-out
validation split and prints a side-by-side comparison. Adopt only if the new
one wins.

    python test_estimator.py --gt-dir <train/GT> --lr-dir <train/NoisyLR>

Also accepts --test-dir to report parameter estimates on the real test set.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from evaluate import Restorer, estimate_nlf, A_MAX, S_MAX

try:
    from pytorch_msssim import ssim as _ssim
except ImportError:
    raise SystemExit("pip install pytorch-msssim scikit-learn")
from sklearn.cluster import KMeans


IMK = torch.tensor([[1., -2., 1.], [-2., 4., -2.], [1., -2., 1.]]).view(1, 1, 3, 3) / 6.


BIAS = 1.35   # overwritten by auto-calibration in main()


def estimate_nlf_flat(y, keep=0.20, clamp=True):
    """Estimate (a, b) from the FLATTEST patches only.

    The current estimator bins by brightness and takes the median |high-pass|
    within each bin. In a densely periodic image most pixels sit on an edge, so
    the median itself is elevated by texture and `a` is over-reported.

    Here we instead rank 8x8 patches by how much STRUCTURE they contain -
    measured on a smoothed copy, so noise does not contribute - and fit only to
    the flattest fraction. Flat regions genuinely contain noise and little else.

    Bias correction: the median of |x| for Gaussian x is 0.6745*sigma, giving
    the usual 1.4826 factor. But selecting the flattest patches biases the
    estimate DOWN, because we are taking a low-order statistic over many
    patches. The `bias` factor below corrects for that; it was calibrated by
    running this estimator on synthetic noise of known strength.
    """
    B = y.shape[0]
    dev = y.device
    k = IMK.to(dev)
    P = 8
    bias = BIAS                      # set by --bias, calibrated per run

    hp = F.conv2d(F.pad(y.float(), (1, 1, 1, 1), mode='reflect'), k).abs()
    lm = F.avg_pool2d(y.float(), 9, stride=1, padding=4)
    sm = F.avg_pool2d(y.float(), 5, stride=1, padding=2)      # noise removed

    A, Bb = [], []
    for i in range(B):
        H, W = y.shape[-2:]
        nh, nw = H // P, W // P
        crop = lambda t: t[i, 0, :nh * P, :nw * P]

        h_p = crop(hp).unfold(0, P, P).unfold(1, P, P).reshape(-1, P * P)
        m_p = crop(lm).unfold(0, P, P).unfold(1, P, P).reshape(-1, P * P).mean(1)
        s_p = crop(sm).unfold(0, P, P).unfold(1, P, P).reshape(-1, P * P)

        structure = s_p.var(1)                                # texture, not noise
        n_keep = max(6, int(len(structure) * keep))
        flat = torch.argsort(structure)[:n_keep]

        hf = h_p[flat].median(1).values * 1.4826 * bias        # robust sigma per patch
        mf = m_p[flat]

        q = torch.quantile(mf, torch.linspace(0, 1, 6, device=dev))
        M, V = [], []
        for j in range(5):
            sel = (mf >= q[j]) & (mf < q[j + 1])
            if sel.sum() >= 2:
                M.append(mf[sel].mean())
                V.append((hf[sel].median()) ** 2)
        if len(M) < 3:
            a_i, b_i = torch.tensor(0.03, device=dev), torch.tensor(1e-4, device=dev)
        else:
            M, V = torch.stack(M), torch.stack(V)
            X = torch.stack([M ** 2, torch.ones_like(M)], 1)
            c = torch.linalg.lstsq(X, V.unsqueeze(1)).solution.squeeze()
            a_i, b_i = c[0].clamp(min=0.0), c[1].clamp(min=1e-6)
        A.append(a_i); Bb.append(b_i)

    a, b = torch.stack(A), torch.stack(Bb)
    if clamp:
        a = a.clamp(max=A_MAX); b = b.clamp(max=S_MAX ** 2)
    return a, b


def calibrate(GT, dev, n=24):
    """Both estimators against known parameters on real images."""
    print("\n=== CALIBRATION: known parameters, real image content ===")
    print(f"{'true a':>8} {'true s':>8} {'cur a':>8} {'new a':>8} "
          f"{'cur ratio':>10} {'new ratio':>10}")
    torch.manual_seed(0)
    gt = torch.from_numpy(GT[:n][:, None].copy()).float().to(dev)
    mu = F.avg_pool2d(gt, 2)
    for a_t, s_t in [(0.005, 0.005), (0.01, 0.01), (0.03, 0.03),
                     (0.08, 0.04), (0.15, 0.10)]:
        kk = torch.full_like(mu, 1 / max(a_t, 1e-6))
        y = mu * torch.distributions.Gamma(kk, kk).sample() + torch.randn_like(mu) * s_t
        a1, _ = estimate_nlf(y, clamp=False)
        a2, _ = estimate_nlf_flat(y, clamp=False)
        print(f"{a_t:8.4f} {s_t:8.4f} {a1.mean():8.4f} {a2.mean():8.4f} "
              f"{a1.mean()/a_t:9.2f}x {a2.mean()/a_t:9.2f}x")


@torch.no_grad()
def evaluate_split(net, GT, LR, ids, dev, est, tag, bs=16):
    P = S = 0.0
    n = 0
    for i in range(0, len(ids), bs):
        b = ids[i:i + bs]
        y = torch.from_numpy(LR[b][:, None].copy()).float().to(dev)
        gt = torch.from_numpy(GT[b][:, None].copy()).float().to(dev)
        a_, b_ = est(y)
        o = net(y, a_, b_)[0].clamp(0, 1)
        mse = ((o - gt) ** 2).mean(dim=[1, 2, 3]).clamp(min=1e-12)
        P += (-10 * torch.log10(mse)).sum().item()
        S += _ssim(o.clamp(0, 1), gt, data_range=1.0, size_average=False).sum().item()
        n += len(b)
    print(f"  {tag:28s} PSNR {P/n:7.3f}   SSIM {S/n:.4f}")
    return P / n, S / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gt-dir', required=True)
    p.add_argument('--lr-dir', required=True)
    p.add_argument('--test-dir', default=None)
    p.add_argument('--weights', default=None)
    p.add_argument('--device', default=None)
    p.add_argument('--limit', type=int, default=None,
                   help='use only the first N validation images (faster on CPU)')
    args = p.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    weights = args.weights or os.path.join(here, 'weights', 'model.pt')
    dev = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def load(d):
        fs = sorted(f for f in os.listdir(d)
                    if f.endswith('.npy') and not f.startswith('._'))
        return np.stack([np.load(os.path.join(d, f)) for f in fs]).astype(np.float32)

    print("loading...")
    GT, LR = load(args.gt_dir), load(args.lr_dir)
    print(f"GT {GT.shape}  LR {LR.shape}")

    # reproduce the exact validation split used for the reported metrics
    sig = np.stack([F.adaptive_avg_pool2d(
        torch.from_numpy(GT[i][None, None].copy()), 16).view(-1).numpy()
        for i in range(len(GT))])
    sig = (sig - sig.mean(1, keepdims=True)) / (sig.std(1, keepdims=True) + 1e-8)
    lab = KMeans(n_clusters=64, n_init=4, random_state=0).fit_predict(sig)
    rng = np.random.default_rng(0)
    val_c, n = [], 0
    for c in rng.permutation(64):
        if n >= 300:
            break
        val_c.append(c); n += (lab == c).sum()
    val_idx = np.where(np.isin(lab, val_c))[0]
    if args.limit:
        val_idx = val_idx[:args.limit]
    print(f"validation split: {len(val_idx)} images")

    sd = torch.load(weights, map_location=dev)
    net = Restorer(dim=sd['inp.weight'].shape[0]).to(dev)
    net.load_state_dict(sd); net.eval()

    # --- auto-calibrate the bias factor so the new estimator is unbiased ---
    global BIAS
    BIAS = 1.0
    torch.manual_seed(0)
    gt_c = torch.from_numpy(GT[:24][:, None].copy()).float().to(dev)
    mu_c = F.avg_pool2d(gt_c, 2)
    ratios = []
    for a_t in [0.01, 0.03, 0.08, 0.15]:
        kk = torch.full_like(mu_c, 1 / a_t)
        y_c = mu_c * torch.distributions.Gamma(kk, kk).sample() + torch.randn_like(mu_c) * a_t
        a_e, _ = estimate_nlf_flat(y_c, clamp=False)
        ratios.append((a_e.mean() / a_t).item())
    BIAS = float(np.sqrt(1.0 / np.median(ratios)))     # a scales as bias^2
    print(f"auto-calibrated bias factor: {BIAS:.4f} "
          f"(raw ratios {[round(r,2) for r in ratios]})")

    calibrate(GT, dev)

    print("\n=== VALIDATION (KLA real pairs, the metric that decides) ===")
    p_cur, s_cur = evaluate_split(net, GT, LR, val_idx, dev,
                                  lambda y: estimate_nlf(y, clamp=True),
                                  "current estimator")
    p_new, s_new = evaluate_split(net, GT, LR, val_idx, dev,
                                  lambda y: estimate_nlf_flat(y, clamp=True),
                                  "flat-patch estimator")

    d = p_new - p_cur
    print(f"\n  delta: {d:+.3f} dB   {s_new-s_cur:+.4f} SSIM")
    if d > 0.10:
        print("  >>> ADOPT — clear improvement")
    elif d > 0.05:
        print("  >>> marginal — re-run with a second seed before adopting")
    elif d > -0.05:
        print("  >>> TIE — keep the current estimator (simpler, already validated)")
    else:
        print("  >>> REJECT — the new estimator is worse")

    if args.test_dir:
        print("\n=== TEST SET PARAMETER ESTIMATES ===")
        fs = sorted(f for f in os.listdir(args.test_dir)
                    if f.endswith('.npy') and not f.startswith('._'))
        cur, new = [], []
        with torch.no_grad():
            for i in range(0, len(fs), 16):
                arr = np.stack([np.load(os.path.join(args.test_dir, f))
                                for f in fs[i:i + 16]]).astype(np.float32)
                y = torch.from_numpy(arr)[:, None].to(dev)
                cur.append(estimate_nlf(y, clamp=True)[0].cpu().numpy())
                new.append(estimate_nlf_flat(y, clamp=True)[0].cpu().numpy())
        cur, new = np.concatenate(cur), np.concatenate(new)
        print(f"  current : p50={np.percentile(cur,50):.4f} "
              f"p90={np.percentile(cur,90):.4f} max={cur.max():.4f}")
        print(f"  flat    : p50={np.percentile(new,50):.4f} "
              f"p90={np.percentile(new,90):.4f} max={new.max():.4f}")
        for name in ['000374.npy', '000257.npy', '000195.npy', '000017.npy']:
            if name in fs:
                j = fs.index(name)
                print(f"    {name}: current a={cur[j]:.4f}  flat a={new[j]:.4f}")


if __name__ == '__main__':
    main()
