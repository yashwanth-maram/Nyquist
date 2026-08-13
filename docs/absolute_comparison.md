# Absolute performance

| model | PSNR | SSIM | LPIPS | params | ms/image |
|:--|--:|--:|--:|--:|--:|
| bicubic x2 | 23.586 dB | 0.5721 | 0.4153 | — | — |
| ours | 28.531 dB | 0.7776 | 0.2731 | 2.76 M | 3.66 |
| *GT noise-floor ceiling* | *38.674 dB* | — | — | — | — |

<sub>Latency measured on A100-SXM4-80GB. Quality metrics are hardware-independent.</sub>

## Positioning

On a 309-image held-out split, using KLA's actual degraded files and blind parameter estimation, the model scores 28.531 dB PSNR, 0.7776 SSIM and 0.2731 LPIPS. Bicubic x2 upsampling scores 23.586 dB; the ground truth's own noise floor caps any method at approximately 38.7 dB. The model therefore stands about 33% of the way from trivial interpolation to the information-theoretic limit of this data, with 10.1 dB of headroom remaining. All figures are measured, not extrapolated.
