#!/usr/bin/env python3
"""
Regenerate the figures in docs/figures/.

    python make_figures.py --gt-dir <train/GT> --lr-dir <train/NoisyLR> \
                           --test-dir <Test_NoisyLR> --out docs/figures

Produces six figures. Each is self-contained evidence for a claim made in the
README or docs/ablations.md; the caption printed alongside each one names the
claim it supports.

Figures 1, 2 and 6 need only the training pairs. Figures 3 and 5 need the test
set. Figure 4 needs nothing but the recorded numbers.
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

# ---- consistent styling across every figure -------------------------------
INK, MUTE, GRID = '#1a2332', '#5a6b82', '#dde4ec'
OURS, BASE, WARN, GOOD = '#2c5282', '#a0aec0', '#c53030', '#276749'

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.edgecolor': GRID, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': MUTE, 'ytick.color': MUTE, 'grid.color': GRID,
    'font.size': 10, 'axes.titlesize': 11, 'axes.titleweight': 'bold',
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 110, 'savefig.bbox': 'tight', 'savefig.facecolor': 'white',
})


def area2(x):
    return x.reshape(x.shape[0] // 2, 2, x.shape[1] // 2, 2).mean(axis=(1, 3))


def fit_nlf_exact(gt, lr, nbins=20):
    """Exact (a, b) using the known clean signal."""
    mu = area2(gt.astype(np.float64))
    r = lr.astype(np.float64) - mu
    m, rr = mu.ravel(), r.ravel()
    q = np.quantile(m, np.linspace(0, 1, nbins + 1)); q[-1] += 1e-9
    M, V = [], []
    for k in range(nbins):
        s = (m >= q[k]) & (m < q[k + 1])
        if s.sum() >= 50:
            M.append(m[s].mean()); V.append(rr[s].var())
    M, V = np.array(M), np.array(V)
    c = np.linalg.lstsq(np.vstack([M ** 2, np.ones_like(M)]).T, V, rcond=None)[0]
    return M, V, max(c[0], 0.0), max(c[1], 0.0)


# ==========================================================================
def fig1_forward_model(GT, LR, out):
    """The claim: we measured the operator rather than assuming it."""
    from scipy.ndimage import gaussian_filter
    from PIL import Image

    idx = [7, 120, 251, 585, 956, 1436, 1818, 2182, 2497]
    idx = [i for i in idx if i < len(GT)][:9]

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # -- panel 1: variance is linear in mu^2 => multiplicative noise
    for i in idx:
        M, V, a, b = fit_nlf_exact(GT[i], LR[i])
        ax[0].plot(M ** 2, V, 'o-', ms=3.5, lw=1.3, alpha=.85, label=f'{i:06d}')
    ax[0].set_xlabel(r'$\mu^2$  (clean intensity, squared)')
    ax[0].set_ylabel(r'Var(residual)')
    ax[0].set_title('Variance is linear in $\\mu^2$\n$\\Rightarrow$ noise is multiplicative')
    ax[0].legend(fontsize=6.5, ncol=2, frameon=False)
    ax[0].grid(alpha=.4, lw=.6)

    # -- panel 2: blur sigma sweep bottoms at zero
    sig = np.arange(0, 1.41, 0.1)
    for i in idx[:6]:
        g, l = GT[i].astype(np.float64), LR[i].astype(np.float64)
        e = np.array([((l - area2(gaussian_filter(g, s) if s > 0 else g)) ** 2).mean()
                      for s in sig])
        ax[1].plot(sig, e / e[0], lw=1.5, alpha=.85, label=f'{i:06d}')
    ax[1].axvline(0, color=WARN, ls='--', lw=1.2)
    ax[1].annotate('minimum at exactly 0', xy=(0.02, 1.0), xytext=(0.45, 0.28),
                   textcoords='axes fraction', color=WARN, fontsize=9,
                   fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color=WARN, lw=1.2,
                                   connectionstyle='arc3,rad=-0.2'))
    ax[1].set_xlabel(r'blur $\sigma$ applied before downsampling')
    ax[1].set_ylabel('normalised MSE')
    ax[1].set_title('No blur operator exists\nMSE rises monotonically from $\\sigma=0$')
    ax[1].legend(fontsize=7, ncol=2, frameon=False)
    ax[1].grid(alpha=.4, lw=.6)

    # -- panel 3: which downsample operator
    def down(x, mode):
        if mode == 'decimate': return x[::2, ::2]
        if mode == 'area':     return area2(x)
        f = {'bilinear': Image.BILINEAR, 'bicubic': Image.BICUBIC,
             'lanczos': Image.LANCZOS}[mode]
        return np.asarray(Image.fromarray(x.astype(np.float32)).resize((128, 128), f),
                          dtype=np.float64)

    modes = ['area', 'bicubic', 'lanczos', 'bilinear', 'decimate']
    rel = {m: [] for m in modes}
    for i in idx:
        g, l = GT[i].astype(np.float64), LR[i].astype(np.float64)
        base = ((l - down(g, 'area')) ** 2).mean()
        for m in modes:
            rel[m].append(((l - down(g, m)) ** 2).mean() / base)
    vals = [np.mean(rel[m]) for m in modes]
    cols = [GOOD] + [BASE] * 4
    bars = ax[2].bar(range(len(modes)), vals, color=cols, edgecolor='white', lw=1.2)
    for bar, v in zip(bars, vals):
        ax[2].text(bar.get_x() + bar.get_width() / 2, v + .015, f'{v:.3f}',
                   ha='center', fontsize=8.5, fontweight='bold')
    ax[2].axhline(1, color=GRID, lw=1, zorder=0)
    ax[2].set_xticks(range(len(modes)))
    ax[2].set_xticklabels(['2×2\narea', 'bicubic', 'lanczos', 'bilinear', 'strided\ndecimate'],
                          fontsize=8.5)
    ax[2].set_ylabel('MSE relative to area average')
    ax[2].set_ylim(0.9, max(vals) * 1.12)
    ax[2].set_title('Downsample is 2×2 area averaging\ndecimation is 36% worse')
    ax[2].grid(alpha=.4, lw=.6, axis='y')

    plt.suptitle('Recovering KLA\'s forward operator from 3200 matched pairs',
                 fontsize=13, fontweight='bold', y=1.03)
    p = os.path.join(out, '01_forward_model.png')
    plt.savefig(p); plt.close()
    return p, 'The degradation operator, measured not assumed.'


# ==========================================================================
def fig2_vst(GT, out, dev):
    """The claim: the transform flattens heteroscedastic noise."""
    from src_shim import degrade
    g = torch.from_numpy(GT[:48][:, None].copy()).float().to(dev)
    torch.manual_seed(0)
    y, mu, a, b = degrade(g)

    from evaluate import vst
    z = vst(y, a, b); zc = vst(mu, a, b)

    # Measure the spread WITHIN each image, then average. Pooling raw values
    # across images would mix per-image VST scales and inflate the spread.
    bands = [(0.05, 0.25), (0.25, 0.45), (0.45, 0.65), (0.65, 0.90)]
    centres = [(lo + hi) / 2 for lo, hi in bands]
    B_, A_ = [], []
    for i in range(len(g)):
        rb = (y[i] - mu[i]); ra = (z[i] - zc[i])
        mi = mu[i]
        bi, ai = [], []
        ok = True
        for lo, hi in bands:
            m = (mi > lo) & (mi < hi)
            if m.sum() < 200: ok = False; break
            bi.append(rb[m].std().item()); ai.append(ra[m].std().item())
        if ok and min(bi) > 0 and min(ai) > 0:
            B_.append([v / np.mean(bi) for v in bi])      # normalise per image
            A_.append([v / np.mean(ai) for v in ai])
    before = np.mean(B_, axis=0); after = np.mean(A_, axis=0)

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))

    # -- panel 1: raw noise grows with brightness
    ax[0].plot(centres, before, 'o-', color=WARN, lw=2.2, ms=8)
    ax[0].fill_between(centres, 0, before, color=WARN, alpha=.12)
    ax[0].set_xlabel('image brightness')
    ax[0].set_ylabel('noise std  (relative to image mean)')
    ax[0].set_title(f'BEFORE — heteroscedastic\nspread {max(before)/min(before):.2f}×')
    ax[0].set_ylim(0, max(before) * 1.3); ax[0].grid(alpha=.4, lw=.6)

    # -- panel 2: after the transform it is flat
    ax[1].plot(centres, after, 'o-', color=GOOD, lw=2.2, ms=8)
    ax[1].fill_between(centres, 0, after, color=GOOD, alpha=.12)
    ax[1].axhline(1.0, color=MUTE, ls='--', lw=1, zorder=0)
    ax[1].set_xlabel('image brightness')
    ax[1].set_ylabel('noise std  (relative to image mean)')
    ax[1].set_title(f'AFTER — homoscedastic\nspread {max(after)/min(after):.2f}×')
    ax[1].set_ylim(0, max(max(after), 1.05) * 1.3); ax[1].grid(alpha=.4, lw=.6)

    # -- panel 3: the transform itself
    yy = np.linspace(0, 2.0, 400)
    for a_v, c in [(0.02, '#90cdf4'), (0.06, '#4299e1'), (0.15, '#2b6cb0')]:
        b_v = 0.0016
        zz = np.arcsinh(yy * np.sqrt(a_v / b_v)) / np.sqrt(a_v)
        ax[2].plot(yy, zz / zz.max(), lw=2, color=c, label=f'a = {a_v}')
    ax[2].plot(yy, yy / yy.max(), '--', lw=1.2, color=MUTE, label='identity')
    ax[2].set_xlabel('input intensity $y$')
    ax[2].set_ylabel('stabilised $z$  (normalised)')
    ax[2].set_title('The transform\n$f(y)=\\frac{1}{\\sqrt{a}}\\,\\mathrm{arcsinh}(y\\sqrt{a/b})$')
    ax[2].legend(frameon=False, fontsize=9); ax[2].grid(alpha=.4, lw=.6)

    plt.suptitle('Variance stabilisation — derived from the measured noise-level function',
                 fontsize=13, fontweight='bold', y=1.03)
    p = os.path.join(out, '02_variance_stabilisation.png')
    plt.savefig(p); plt.close()
    return p, 'Noise standard deviation flattened across the brightness range.'


# ==========================================================================
def fig3_before_after(net, TEST, names, out, dev, picks):
    """The claim: it works, and here is what it looks like."""
    rows = []
    for pid in picks:
        if pid not in names: continue
        i = names.index(pid)
        y = torch.from_numpy(TEST[i][None, None].copy()).float().to(dev)
        a, b = estimate_nlf(y, clamp=True)
        with torch.no_grad():
            o = restore(net, y, ensemble=False)
        bic = F.interpolate(y, scale_factor=2, mode='bicubic',
                            align_corners=False).clamp(0, 1)
        rows.append((pid, TEST[i], bic.cpu().numpy()[0, 0], o.cpu().numpy()[0, 0],
                     a.item(), b.item() ** 0.5))

    fig, ax = plt.subplots(len(rows), 3, figsize=(11.5, 3.95 * len(rows)),
                           squeeze=False)
    for r, (pid, inp, bic, o, a, s) in enumerate(rows):
        for c, (im, t) in enumerate([
                (inp, f'input 128²   a={a:.3f}  σ={s:.3f}'),
                (bic, 'bicubic ×2'),
                (o,   'restored 256²')]):
            ax[r, c].imshow(im, cmap='gray', vmin=0, vmax=1,
                            interpolation='nearest' if im.shape[0] <= 128 else 'antialiased')
            ax[r, c].set_title(t, fontsize=9.5,
                               color=OURS if c == 2 else INK,
                               fontweight='bold' if c == 2 else 'normal')
            ax[r, c].axis('off')
        ax[r, 0].text(-0.06, 0.5, pid, transform=ax[r, 0].transAxes,
                      rotation=90, va='center', ha='center',
                      fontsize=10, fontweight='bold', color=MUTE)
    plt.suptitle('Restoration on held-out test images',
                 fontsize=13, fontweight='bold', y=1.005)
    plt.tight_layout()
    p = os.path.join(out, '03_before_after.png')
    plt.savefig(p); plt.close()
    return p, 'Input, bicubic baseline, and our restoration.'


# ==========================================================================
def fig4_generalisation(out):
    """The claim: it inverts the operator, not the dataset."""
    corpora = ['KLA val\n(reference)', 'BSD100', 'Set14', 'DIV2K', 'Urban100']
    mild = [3.42, 2.68, 2.66, 2.28, 1.83]
    severe = [8.16, 6.90, 6.84, 6.04, 5.28]
    n = [309, 100, 14, 100, 100]

    fig, ax = plt.subplots(1, 2, figsize=(14.5, 4.8),
                           gridspec_kw={'width_ratios': [1.6, 1]})

    x = np.arange(len(corpora)); w = 0.36
    b1 = ax[0].bar(x - w/2, mild, w, label='mild degradation',
                   color=BASE, edgecolor='white', lw=1.2)
    b2 = ax[0].bar(x + w/2, severe, w, label='severe degradation',
                   color=OURS, edgecolor='white', lw=1.2)
    for bars in (b1, b2):
        for bar in bars:
            ax[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + .12,
                       f'{bar.get_height():.2f}', ha='center',
                       fontsize=8.5, fontweight='bold')
    ax[0].axvline(0.5, color=GRID, lw=1.2, ls='--')
    ax[0].text(0.02, 8.9, 'trained on', fontsize=8, color=MUTE, style='italic')
    ax[0].text(1.6, 8.9, 'never seen during training', fontsize=8,
               color=MUTE, style='italic')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([f'{c}\nn={k}' for c, k in zip(corpora, n)], fontsize=9)
    ax[0].set_ylabel('PSNR gain over bicubic (dB)')
    ax[0].set_ylim(0, 9.6)
    ax[0].set_title('Generalisation across four unseen corpora')
    ax[0].legend(frameon=False, fontsize=9.5, loc='upper right')
    ax[0].grid(alpha=.4, lw=.6, axis='y')

    ratio = [s / m for s, m in zip(severe, mild)]
    ax[1].barh(range(len(corpora)), ratio, color=OURS, alpha=.85,
               edgecolor='white', lw=1.2)
    for i, r in enumerate(ratio):
        ax[1].text(r + .04, i, f'{r:.2f}×', va='center',
                   fontsize=9, fontweight='bold')
    ax[1].set_yticks(range(len(corpora)))
    ax[1].set_yticklabels([c.split('\n')[0] for c in corpora], fontsize=9)
    ax[1].invert_yaxis()
    ax[1].set_xlabel('severe gain ÷ mild gain')
    ax[1].set_xlim(0, max(ratio) * 1.22)
    ax[1].set_title('The variance-stabilisation signature\nreproduced four times')
    ax[1].grid(alpha=.4, lw=.6, axis='x')

    plt.suptitle('The model inverts the operator, not the dataset',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    p = os.path.join(out, '04_generalisation.png')
    plt.savefig(p); plt.close()
    return p, 'Gains hold on 314 images from corpora never used in training.'


# ==========================================================================
def fig5_failure(net, TEST, names, out, dev, pid='000374'):
    """The claim: here is where it fails, and why."""
    if pid not in names:
        return None, None
    i = names.index(pid)
    inp = TEST[i]
    y = torch.from_numpy(inp[None, None].copy()).float().to(dev)
    a_est, b_est = estimate_nlf(y, clamp=True)

    levels = [(0.005, 0.010), (0.02, 0.02), (a_est.item(), b_est.item() ** 0.5)]
    outs = []
    for aa, ss in levels:
        A = torch.full((1,), aa, device=dev); B = torch.full((1,), ss ** 2, device=dev)
        with torch.no_grad():
            outs.append(net(y, A, B)[0].clamp(0, 1).cpu().numpy()[0, 0])
    bic = F.interpolate(y, scale_factor=2, mode='bicubic',
                        align_corners=False).clamp(0, 1).cpu().numpy()[0, 0]

    fig, ax = plt.subplots(1, 5, figsize=(19.5, 4.3))
    panels = [(inp, f'input {pid}\nestimator says a={a_est.item():.3f}', INK),
              (bic, 'bicubic — mesh survives', GOOD)]
    for (aa, ss), o in zip(levels, outs):
        lbl = 'as shipped' if abs(aa - a_est.item()) < 1e-9 else 'forced'
        panels.append((o, f'a={aa:.3f}  ({lbl})',
                       WARN if lbl == 'as shipped' else INK))
    for c, (im, t, col) in enumerate(panels):
        ax[c].imshow(im, cmap='gray', vmin=0, vmax=1,
                     interpolation='nearest' if im.shape[0] <= 128 else 'antialiased')
        ax[c].set_title(t, fontsize=9.5, color=col, fontweight='bold')
        ax[c].axis('off')

    plt.suptitle('Documented failure — dense periodic structure defeats the blind estimator',
                 fontsize=13, fontweight='bold', y=1.04)
    plt.figtext(0.5, -0.03,
                'The estimator reads mesh edges as speckle and over-reports noise ~10×. '
                'The model then denoises accordingly and erases the finest-pitch region. '
                'Forcing a low noise level restores it — the network is correct, the estimate is not.',
                ha='center', fontsize=9.5, color=MUTE, style='italic')
    plt.tight_layout()
    p = os.path.join(out, '05_failure_case.png')
    plt.savefig(p); plt.close()
    return p, 'The one case where bicubic beats us, diagnosed.'


# ==========================================================================
def fig6_estimator(GT, TEST, out, dev):
    """The claim: we characterised our own estimator's limits."""
    from src_shim import degrade
    g = torch.from_numpy(GT[:32][:, None].copy()).float().to(dev)
    mu = F.avg_pool2d(g, 2)

    torch.manual_seed(0)
    truths = [0.005, 0.01, 0.02, 0.04, 0.08, 0.12, 0.18]
    ests = []
    for a_t in truths:
        kk = torch.full_like(mu, 1 / a_t)
        y = mu * torch.distributions.Gamma(kk, kk).sample() + torch.randn_like(mu) * a_t
        a_e, _ = estimate_nlf(y, clamp=False)
        ests.append(a_e.mean().item())

    # blind estimates over the real test set
    te = []
    with torch.no_grad():
        for i in range(0, min(len(TEST), 400), 32):
            y = torch.from_numpy(TEST[i:i+32][:, None].copy()).float().to(dev)
            te.append(estimate_nlf(y, clamp=False)[0].cpu().numpy())
    te = np.concatenate(te)

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))

    lim = max(truths + ests) * 1.15
    ax[0].plot([0, lim], [0, lim], '--', color=MUTE, lw=1.2, label='perfect')
    ax[0].plot(truths, ests, 'o-', color=OURS, lw=2, ms=8, label='measured')
    ax[0].set_xlabel('true speckle $a$'); ax[0].set_ylabel('estimated $a$')
    ax[0].set_xlim(0, lim); ax[0].set_ylim(0, lim)
    ax[0].set_title('Blind estimator calibration\naccurate to ~3% for $a \\geq 0.03$')
    ax[0].legend(frameon=False, fontsize=9); ax[0].grid(alpha=.4, lw=.6)

    ax[1].hist(np.clip(te, 0, 0.4), bins=44, color=OURS, alpha=.82,
               edgecolor='white', lw=.6)
    ax[1].axvline(0.22, color=WARN, ls='--', lw=1.8)
    ax[1].text(0.225, ax[1].get_ylim()[1] * .82, 'clamp\na = 0.22',
               color=WARN, fontsize=9, fontweight='bold')
    n_out = int((te > 0.22).sum())
    ax[1].set_xlabel('estimated $a$ on the 400 test images')
    ax[1].set_ylabel('count')
    ax[1].set_title(f'Test-set estimates\n{n_out} of {len(te)} exceed the training envelope')
    ax[1].grid(alpha=.4, lw=.6, axis='y')

    plt.suptitle('Blind noise estimation — calibration and known limits',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    p = os.path.join(out, '06_estimator_calibration.png')
    plt.savefig(p); plt.close()
    return p, 'Where the estimator is reliable, and where it is not.'


# ==========================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gt-dir', required=True)
    p.add_argument('--lr-dir', required=True)
    p.add_argument('--test-dir', default=None)
    p.add_argument('--out', default='docs/figures')
    p.add_argument('--weights', default=None)
    p.add_argument('--device', default=None)
    p.add_argument('--limit', type=int, default=3200)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))
    weights = args.weights or os.path.join(here, 'weights', 'model.pt')
    dev = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def load(d, lim=None):
        fs = sorted(f for f in os.listdir(d)
                    if f.endswith('.npy') and not f.startswith('._'))
        if lim: fs = fs[:lim]
        return (np.stack([np.load(os.path.join(d, f)) for f in fs]).astype(np.float32),
                [f[:-4] for f in fs])

    print('loading...')
    GT, _ = load(args.gt_dir, args.limit)
    LR, _ = load(args.lr_dir, args.limit)
    TEST, tnames = (load(args.test_dir) if args.test_dir else (None, []))
    print(f'GT {GT.shape}  LR {LR.shape}'
          + (f'  TEST {TEST.shape}' if TEST is not None else ''))

    sd = torch.load(weights, map_location=dev)
    net = Restorer(dim=sd['inp.weight'].shape[0]).to(dev)
    net.load_state_dict(sd); net.eval()

    made = []
    print('\nfig 1 — forward model...');   made.append(fig1_forward_model(GT, LR, args.out))
    print('fig 2 — variance stabilisation...'); made.append(fig2_vst(GT, args.out, dev))
    print('fig 4 — generalisation...');    made.append(fig4_generalisation(args.out))
    if TEST is not None:
        print('fig 3 — before/after...')
        made.append(fig3_before_after(net, TEST, tnames, args.out, dev,
                                      ['000310', '000398', '000176', '000385']))
        print('fig 5 — failure case...')
        made.append(fig5_failure(net, TEST, tnames, args.out, dev))
        print('fig 6 — estimator calibration...')
        made.append(fig6_estimator(GT, TEST, args.out, dev))

    print('\n' + '=' * 62)
    for path, caption in made:
        if path:
            print(f'  {os.path.basename(path):34s} {caption}')
    print('=' * 62)


if __name__ == '__main__':
    main()
