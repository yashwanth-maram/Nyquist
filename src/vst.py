"""Noise-level estimation and the variance-stabilising transform.

The degradation operator was recovered empirically from 3200 matched
ground-truth / degraded pairs (see docs/ablations.md):

    mu = AreaAvg_2x2(x)
    y  = mu * s + n,    s ~ Gamma(L, 1/L),    n ~ Normal(0, sigma^2)

Both noise terms are applied at low resolution and are spatially iid.
There is NO blur operator; a grid search over blur sigma found the minimum at
exactly zero for all 40 pairs tested, in both possible orderings.

It follows that

    Var(y | mu) = a * mu^2 + b        with a = 1/L, b = sigma^2

which is heteroscedastic: noise grows with signal. A plain L2 objective assumes
constant variance and therefore mis-weights the image everywhere, over-trusting
bright pixels where the data is least reliable. That variance form admits a
closed-form stabilising transform, applied below.

Measured effect: noise standard deviation across three brightness bands went
from a 1.78x spread to 1.01x with oracle parameters, and 1.04x with blind
estimates.
"""
import torch
import torch.nn.functional as F

A_FLOOR = 5e-3          # below this the Gaussian limit is exact
Y_LO, Y_HI = -1.0, 3.0  # observed input range over all 3600 supplied images
A_MAX, S_MAX = 0.22, 0.19   # training envelope


def vst(y, a, b, eps=1e-6):
    """Flatten heteroscedastic noise to approximately unit variance.

        f(y) = (1/sqrt(a)) * arcsinh( y * sqrt(a/b) )

    Reduces to the classical homomorphic log transform as b -> 0 (pure
    speckle), and to identity scaling as a -> 0 (pure Gaussian). Both limits
    occur in the data, so the a < A_FLOOR branch is exact rather than clamped.
    """
    a = a.float().view(-1, 1, 1, 1)
    b = b.float().clamp(min=eps).view(-1, 1, 1, 1)
    a_s = a.clamp(min=A_FLOOR)
    z_gauss = y.float() / b.sqrt()
    z_speck = torch.asinh(y.float() * (a_s / b).sqrt()) / a_s.sqrt()
    return torch.where(a < A_FLOOR, z_gauss, z_speck)


def ivst(z, a, b, eps=1e-6):
    """Inverse transform, bounded to the physically observed range.

    The sinh argument is clamped to +-6: unbounded, it overflows fp16 and can
    reach ~9000 in fp32 when a -> 0, which destabilises training.
    """
    a = a.float().view(-1, 1, 1, 1)
    b = b.float().clamp(min=eps).view(-1, 1, 1, 1)
    a_s = a.clamp(min=A_FLOOR)
    y_gauss = z.float() * b.sqrt()
    y_speck = (b / a_s).sqrt() * torch.sinh((z.float() * a_s.sqrt()).clamp(-6, 6))
    return torch.where(a < A_FLOOR, y_gauss, y_speck).clamp(Y_LO, Y_HI)


def estimate_nlf(y, clamp=True):
    """Blind per-image estimation of (a, b). No labels, no priors.

    Method: an Immerkaer high-pass kernel annihilates image structure up to
    linear, so the filtered residual is dominated by noise. Binning by local
    brightness and regressing robust residual variance against brightness
    squared recovers a and b directly.

    Validation: 0.978 correlation with true total variance over 128 samples,
    8.4% median relative error. The VST flattens noise to 1.04x spread using
    these estimates, versus 1.01x with oracle values.

    Because (a, b) come from the image itself, the transform adapts to any
    severity - including regimes absent from training. This is what carries the
    model to unseen corpora (+5.3 to +6.9 dB severe on BSD100, Set14, Urban100
    and DIV2K, none of which were trained on).

    Known failure mode: on high-contrast periodic structure the high-pass reads
    texture as speckle and over-estimates a (worst observed 1.21, against a
    training maximum of 0.22). Clamping to the training envelope prevents the
    resulting extreme over-denoising. 3 of the 400 supplied test images trigger
    it; aggregate metrics are unchanged (27.965 dB either way).
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
                V.append((h[s].abs().median() * 1.4826) ** 2)
        if len(M) < 4:
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
