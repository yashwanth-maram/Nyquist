# Absolute performance

| model | PSNR | SSIM | LPIPS | params | ms/image |
|:--|--:|--:|--:|--:|--:|
| bicubic x2 | 23.586 dB | 0.5721 | 0.4153 | — | — |
| ours | 27.965 dB | 0.7650 | 0.2610 | 2.76 M | 472.17 |
| *GT noise-floor ceiling* | *38.674 dB* | — | — | — | — |

## Positioning

On a 309-image held-out split, using KLA's actual degraded files and blind parameter estimation, the model scores 27.965 dB PSNR, 0.7650 SSIM and 0.2610 LPIPS. Bicubic x2 upsampling scores 23.586 dB; the ground truth's own noise floor caps any method at approximately 38.7 dB. The model therefore stands about 29% of the way from trivial interpolation to the information-theoretic limit of this data, with 10.7 dB of headroom remaining. All figures are measured, not extrapolated.
