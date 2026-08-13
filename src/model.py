"""Restoration network.

Architecture notes, each traceable to a measured result in docs/ablations.md:

* FiLM conditioning on (a, b, sqrt(a), sqrt(b)) lets the network modulate its
  behaviour by the measured noise level rather than assuming a fixed severity.

* dgate is a learned scalar gate on the denoiser output. Without it the network
  applies fixed denoising and damages nearly clean input: measured -4.3 dB
  against bicubic on an undegraded image, because it removes detail that was
  never noise. With the gate, clean-input PSNR rose from 27.09 to 30.28 dB
  against a 31.40 dB bicubic reference.

* hr_up + hr_ref refine at full 256x256 resolution rather than projecting
  straight from the 128x128 feature map. A bare projection cannot synthesise
  detail; adding two refinement blocks at full resolution was the single
  largest architectural gain.

* The output is a residual on bicubic(mu_hat), so the network starts at the
  interpolation baseline and every update is an improvement on it.

NOT present, and deliberately so:
* No blur deconvolution - there is no blur in the forward operator.
* No data-consistency projection enforcing AreaAvg(x_hat) = mu_hat. It was
  implemented with a learnable gate, swept across six strengths, and removed:
  it degrades severe-degradation performance monotonically (-0.70 dB at full
  strength) because it stamps the denoiser's own residual error in as truth.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vst import vst, ivst, A_FLOOR


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