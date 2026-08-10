"""Evaluation metrics.

hf_error is included because PSNR alone is a misleading target for this
problem: a model can score well while destroying most of the fine detail, and
the problem statement is explicit that "Do not blur the image to remove noise,
that destroys useful information."

Note that measuring raw high-frequency ENERGY in the output is the wrong test -
residual noise counts as high-frequency energy and inflates the score. The
correct measurement is high-frequency ERROR against ground truth, below.
"""
import torch
import torch.nn.functional as F

try:
    from pytorch_msssim import ssim as _ssim
except ImportError:
    _ssim = None


def psnr(x, y):
    mse = ((x - y) ** 2).mean(dim=[1, 2, 3]).clamp(min=1e-12)
    return (-10 * torch.log10(mse)).mean().item()


def ssim_m(x, y):
    if _ssim is None:
        raise ImportError("pip install pytorch-msssim")
    return _ssim(x.clamp(0, 1), y.clamp(0, 1), data_range=1.0,
                 size_average=True).item()


def hf_error(pred, gt):
    """Relative L1 error in the high-frequency band. Lower is better.

    Reference values on the validation split:
        bicubic  132.3%
        ours      87.5%
    """
    k = (1 / 16) * torch.tensor([[1., 4., 6., 4., 1.]], device=pred.device)
    k5 = (k.T @ k).view(1, 1, 5, 5)
    hp = lambda t: t - F.conv2d(F.pad(t, (2, 2, 2, 2), mode='reflect'), k5)
    return (hp(pred) - hp(gt)).abs().mean().item() / (hp(gt).abs().mean().item() + 1e-9) * 100
