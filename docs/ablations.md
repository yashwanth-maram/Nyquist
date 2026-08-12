<div align="center">

# 🧪 Ablation Study

**Seven rungs · Seven rejections · Every decision measured**

[![Rungs](https://img.shields.io/badge/rungs-7-blue)]()
[![Rejected](https://img.shields.io/badge/components%20rejected-7-red)]()
[![Champion](https://img.shields.io/badge/champion-27.965%20dB-success)]()

</div>

---

> [!IMPORTANT]
> **Primary metric** throughout is PSNR/SSIM on KLA's *actual* `NoisyLR` files over a held-out, content-clustered validation split of **309 images**, using **blind** parameter estimation.
>
> This is the only measurement made under exactly the conditions of the hidden test set.

### Decision rules — fixed *before* any result was seen

| Rule | |
|:--|:--|
| `Δ < 0.05 dB` | Noise. Declare a tie. |
| `Δ > 0.10 dB` | Real. Advance. |
| Tie | Break toward the **simpler and faster** option. |
| Regresses severe | **Does not advance**, whatever it does elsewhere. |

---

## 📋 The ladder at a glance

| # | Change | Real PSNR | Real SSIM | |
|:--|:--|--:|--:|:--|
| — | Bicubic ×2 *(floor)* | 23.586 | 0.5721 | `baseline` |
| 1 | VST + FiLM + NAFNet-32 | — | — | first working model |
| 1b | + min–max output renormalisation | — | — | ❌ **rejected** `−4.07 dB` |
| 1c | + data-consistency projection | — | — | ❌ **rejected** `−0.70 dB` |
| 2 | Band-weighted FFT loss | — | — | ❌ **rejected** `HF halved` |
| 3 | + dgate, wide SR head, clean samples | 23.840 | 0.5240 | ✅ advanced |
| 4 | 30-epoch extension | — | — | superseded |
| 5 | **+ 50% real KLA pairs in training** | 27.554 | 0.7505 | 🏆 **largest single gain** |
| 5b | Clean-GT retrain | — | — | ❌ **rejected** `+0.086 dB` |
| 6 | dim 32 → 64 at matched step count | 27.905 | 0.7638 | ✅ advanced |
| 6b | Two-model ensemble | 27.934 | 0.7639 | ❌ **rejected** `+0.029 dB` |
| 7 | **+ absolute high-pass loss fine-tune** | **27.965** | **0.7652** | 🏆 **champion** |
| 7b | + NLF clamp to training envelope | 27.965 | 0.7650 | ✅ adopted *(safety)* |
| 7c | Flat-patch noise estimator | 27.124 | 0.7556 | ❌ **rejected** `−0.244 dB` |

---

## Part 1 · Recovering the forward operator

Before any model was trained, the degradation was measured from **3200 matched pairs**.

![Forward model identification](figures/01_forward_model.png)
<sub><i>Left: residual variance is linear in μ² across nine independent images — the signature of multiplicative noise, established from data alone. Centre: MSE against blur σ, minimised at exactly zero for every pair. Right: the downsample operator comparison.</i></sub>

> [!TIP]
> **The enabling observation:** LR mean matches GT mean to three decimals in every pair. Since Gamma speckle has mean 1 and Gaussian noise has mean 0, `E[y] = D(B(x))` *independent of operator order* — so a least-squares fit recovers the operator without the noise biasing it.

### Downsample operator

Mean MSE relative to area averaging, 100 pairs:

| Operator | Relative MSE | |
|:--|--:|:--|
| **2×2 area average** | **1.000** | ✅ selected |
| Bicubic | 1.048 | |
| Lanczos | 1.052 | |
| Bilinear | 1.104 | |
| Strided decimation | 1.364 | 36% worse |

### Blur — there isn't any

Grid search over σ ∈ [0, 6] at 0.05 resolution, crossed with seven downsample operators:

| σ | 0.0 | 0.2 | 0.4 | 0.6 | 1.0 |
|:--|--:|--:|--:|--:|--:|
| Normalised MSE | **1.000** | 1.000 | 1.013 | 1.076 | 1.201 |

> [!WARNING]
> Minimum at exactly `0.0` for **every pair**, rising monotonically, in **both** possible orderings.
>
> The problem statement's summary table describes Gaussian noise as making the image *"soft and hazy"*, which reads like blur. But KLA's own technical webinar defines the transform as *"Speckle noise, down-sampling, additive gaussian noise"* — three operations, blur not among them.
>
> **We model no blur. Teams that do will be inverting an operation that was never applied.**

<details>
<summary><b>Speckle is multiplicative Gamma — the evidence</b></summary>

<br>

**Variance structure.** Exact residual `r = y − AreaAvg(x)`, binned by clean intensity, fitted to `Var = a·μ² + b`:

| Index | a | L = 1/a | σ_g | R² |
|:--|--:|--:|--:|--:|
| `000007` | 0.03766 | 26.6 | 0.000 | 0.784 |
| `000120` | 0.02055 | 48.7 | 0.027 | 0.968 |
| `000251` | 0.06283 | 15.9 | 0.025 | 0.941 |
| `000585` | 0.03527 | 28.3 | 0.006 | 0.932 |
| `000956` | 0.02972 | 33.7 | 0.031 | 0.973 |
| `001436` | 0.02859 | 35.0 | 0.042 | 0.962 |
| `001818` | 0.03035 | 32.9 | 0.009 | 0.993 |
| `002182` | 0.02173 | 46.0 | 0.032 | 0.984 |
| `002497` | 0.03658 | 27.3 | 0.000 | 0.986 |

R² of 0.93–0.99 on 8 of 9. **Variance linear in the square of the mean is the signature of multiplicative noise.**

**Distributional check.** For bright pixels, the ratio `s = y/μ`:

| Index | mean(s) | var(s) | measured skew | Gamma predicts 2√var |
|:--|--:|--:|--:|--:|
| `000120` | 1.0014 | 0.0218 | +0.283 | +0.295 |
| `000585` | 1.0020 | 0.0347 | +0.356 | +0.372 |
| `001818` | 0.9980 | 0.0301 | +0.331 | +0.347 |
| `002497` | 1.0015 | 0.0357 | +0.359 | +0.378 |

Mean is 1.00 in every image; skew tracks the Gamma prediction closely.

**Noise is iid at low resolution.** Autocorrelation of the exact residual: −0.06 to +0.02 at lag 1 in all directions. Applied *after* downsampling, independently per pixel.

</details>

### Round-trip validation

Synthesising degradation with the fitted parameters and re-measuring:

| | real | synthetic | error |
|:--|--:|--:|--:|
| speckle `a` (p50) | 0.0287 | 0.0288 | **2.3%** |
| Gaussian σ (p50) | 0.0214 | 0.0221 | — |

> [!NOTE]
> **Confirmed operator**
> ```
> mu = AreaAvg_2x2(x)
> y  = mu·s + n      s ~ Γ(L, 1/L)     n ~ N(0, σ²)
> ```
> Applied at low resolution, spatially iid, order **and presence** both randomised.

---

## Part 2 · The seven rejections

<details open>
<summary><h3>❌ 1b · Min–max output renormalisation — <code>−4.07 dB</code></h3></summary>

<br>

GT is per-image min–max normalised to exactly [0,1] — verified across all 3200 files, every one has precisely one pixel at 0.0 and one at 1.0. Matching that seemed free.

| | bicubic | + min–max |
|:--|--:|--:|
| mild | 24.277 | **20.205** |
| severe | 15.356 | **14.375** |

**Why it fails:** the input range is `[−0.279, 2.158]` and those extremes are *noise outliers*. Dividing by `(max − min) ≈ 2.44` crushes the real signal into a fraction of its proper range.

You end up **rescaling by the noise, not the signal.** GT has that property because KLA normalised a *clean* image; a prediction with residual noise does not.

</details>

<details>
<summary><h3>❌ 1c · Data-consistency projection — <code>−0.70 dB severe</code></h3></summary>

<br>

Since the operator is a known 2×2 average, `AreaAvg₂(x̂) = μ̂` can be enforced exactly by projection. Implemented with a learnable sigmoid gate so the model could choose its own strength.

Inference-time sweep over six gate values:

| gate | mild | severe |
|:--|--:|--:|
| 0.00 | 26.903 | **22.745** |
| 0.25 | **27.018** | 22.626 |
| 0.50 | 26.950 | 22.467 |
| 0.75 | 26.695 | 22.271 |
| 1.00 | 26.301 | 22.044 |

Mild peaks at gate 0.25 with `+0.115 dB` — inside the noise floor. **Severe degrades monotonically.**

**Why it fails:** the constraint is correct only if `μ̂` is exact. The denoiser is good but not perfect, so the projection stamps its residual error in as *hard truth* and removes the network's ability to correct downstream. The harder the degradation, the worse `μ̂` is — hence severe falling fastest.

> [!TIP]
> The learned gate converged to **0.126** over 12 epochs despite being free to open. **The model had already concluded the constraint didn't pay — we only listened after measuring.**

</details>

<details>
<summary><h3>❌ 2 · Band-weighted FFT loss — <code>spectral retention halved</code></h3></summary>

<br>

To push high-frequency reconstruction, the FFT loss was weighted by radius² and its weight raised 5×.

**Result: high-frequency retention fell from 12.6% to 6.3%** — the exact opposite of the intent.

**Why it fails:** the loss was *relative*, normalised by the weighted GT magnitude. In high bands that denominator is tiny, so the term explodes there — and the cheapest way for the optimiser to shrink it is to emit **less** high-frequency content.

> [!CAUTION]
> **We built a loss that rewarded smoothing.**

The corrected version (rung 7) uses an **absolute** L1 on the high-pass residual, with no denominator to game. That version improved every metric.

</details>

<details>
<summary><h3>❌ 5b · Clean ground-truth retrain — <code>+0.086 dB</code></h3></summary>

<br>

A structure-to-noise detector found **126 of 3200 GT images (3.9%) are structureless noise**, occurring in contiguous index blocks — `2537–2539`, `2637–2638`, `2981–2983`, `625–626` — suggesting one bad batch at source rather than random corruption.

| | PSNR | gain over bicubic |
|:--|--:|--:|
| all val (309) | 27.554 | +3.969 |
| clean val (298) | 27.867 | +4.055 |

Excluding them shifts the *gain* by only `+0.086 dB`, because bicubic improves on the cleaner set too and most of the difference cancels.

**Reported on the full set for comparability.**

</details>

<details>
<summary><h3>❌ 6b · Two-model ensemble — <code>+0.029 dB</code></h3></summary>

<br>

Weight sweep over rung 5 + rung 6 output averaging:

| w(rung5) | 0.00 | 0.15 | 0.25 | 0.35 | 0.50 |
|:--|--:|--:|--:|--:|--:|
| PSNR | 27.905 | **27.934** | 27.933 | 27.921 | 27.881 |

Best mixture beats the single model by `0.029 dB` — below the 0.05 dB noise floor — for **double the inference cost**. Rejected.

</details>

<details>
<summary><h3>❌ 7c · Flat-patch noise estimator — <code>−0.244 dB</code></h3></summary>

<br>

**The problem it was built to solve.** Image `000374` — a dense mesh grating — restores with the finest-pitch region erased to a flat grey panel. Bicubic preserves the mesh; our model does not.

![Failure case](figures/05_failure_case.png)
<sub><i>Forcing a low noise level restores the mesh — the network is correct, the estimate is not.</i></sub>

Traced to the blind estimator reporting `a = 0.084` where the flattest patches imply `a ≈ 0.008` — a **10× over-estimate** that falls *below* the 0.22 clamp and therefore passes through unmitigated. A parameter sweep confirmed the mesh is fully preserved at `a = 0.003–0.010`.

**The fix we tried.** Estimate from the flattest 20% of 8×8 patches, ranked by structure measured on a *smoothed* copy so noise doesn't contribute, with a runtime-calibrated bias correction for the order statistic.

**Calibration against known parameters on real images:**

| true a | current ratio | flat-patch ratio |
|--:|--:|--:|
| 0.005 | 0.70× | **0.99×** |
| 0.010 | 0.77× | **0.91×** |
| 0.030 | 0.86× | **0.99×** |
| 0.080 | 0.90× | 0.83× |
| 0.150 | 0.92× | 0.81× |

The new estimator is unbiased where the current one under-reports by 8–30%.

![Estimator calibration](figures/06_estimator_calibration.png)
<sub><i>Left: blind estimator accuracy against known parameters. Right: estimates across the 400 test images, with the training-envelope clamp marked.</i></sub>

**But on the validation split:**

| | PSNR | SSIM |
|:--|--:|--:|
| current estimator | **27.367** | **0.7573** |
| flat-patch estimator | 27.124 | 0.7556 |
| | `−0.244 dB` | `−0.0017` |

And on the target image it estimated *higher* speckle — `0.095` against `0.084`. It fixed `000195` (0.220 → 0.020) and made `000374` worse.

> [!NOTE]
> **Why:** on a dense mesh there aren't enough genuinely flat patches to estimate from. The "flattest 20%" are still mesh, just slightly less contrasty mesh.
>
> Dense periodic texture and multiplicative noise have overlapping statistical signatures at these scales. Separating them likely requires a **spectral** rather than spatial approach.

Reproduce with `python test_estimator.py --gt-dir <train/GT> --lr-dir <train/NoisyLR>`.

</details>

<details>
<summary><h3>⚠️ Superseded metric · HF <i>retention</i> → HF <i>error</i></h3></summary>

<br>

An early metric measured raw high-frequency **energy** in the output relative to GT energy. By that measure rung 5 (51.5%) beat rung 6 (31.1%) — **which nearly caused the better model to be discarded.**

**The metric was wrong.** Residual noise is broadband and counts as high-frequency energy, so a model that denoises *less* scores higher. Visual inspection contradicted it plainly: rung 6 resolved individual ballast stones and truss lines that rung 5 smeared into grey.

Corrected to high-frequency **error against ground truth**:

| | HF error ↓ |
|:--|--:|
| bicubic | 132.3% |
| rung 5 | 90.4% |
| rung 6 | 88.8% |
| **rung 7** | **87.5%** |

We record this because it is a genuine methodological error that we caught in ourselves, and it **changed a decision**.

</details>

---

## Part 3 · What worked, and why

### 🏆 Rung 3 · The dgate identity path

**Diagnostic:** feed the model a *perfectly clean* low-resolution image and ask only for 2× upsampling.

| | PSNR |
|:--|--:|
| plain bicubic | 31.40 dB |
| our model | 27.09 dB **(−4.31)** |

The model **damaged an image that needed no repair.** Isolating further:

```
bicubic(μ)              31.40 dB    ← what we should match
denoise → bicubic       26.59 dB    ← −4.81, the denoiser
+ head_hr               27.09 dB    ← +0.50, the SR head helps
```

The denoiser was responsible for the entire loss. On a clean input its output deviated by up to **0.4845 in a single pixel** — nearly half the intensity range, on an image with no noise at all.

**The cause:** every training sample was degraded, so the network learned *"always smooth"* as an unconditional operation. It had no mechanism to observe *this one is already clean, do nothing.*

**The fix:** a learned scalar gate from the degradation embedding, plus ~12% completely undegraded training samples so "do nothing" is a learnable answer.

| | rung 1 | rung 3 |
|:--|--:|--:|
| clean-input PSNR | 27.09 | **29.80** |
| denoiser damage | 29.45 | **36.34** |
| learned gate on clean input | — | **0.680** |

The gate converging to 0.68 rather than 1.0 confirms it learned to **modulate**, not to always denoise at full strength.

### 📐 The transform underneath it all

Every rung above sits on top of the variance-stabilising transform, derived from the noise-level function measured in Part 1.

![Variance stabilisation](figures/02_variance_stabilisation.png)
<sub><i>Left: raw noise grows with brightness — 2.24× spread across brightness bands. Middle: after the transform, flat at unit variance — 1.06×. Right: the transform itself at three severities.</i></sub>

Because `(a, b)` are estimated per image at inference, an image degraded far beyond anything in training arrives at the network **already standardised**. That is what produces the generalisation result in Part 4.

### 🏆 Rung 5 · Real pairs in training — the largest gain

Rungs 1–4 trained purely on self-synthesised degradation. That produced strong synthetic scores and **near-zero real-world benefit**:

| | rung 3 | rung 5 |
|:--|--:|--:|
| real PSNR gain | +0.255 dB | **+4.033 dB** |
| real SSIM | 0.5240 *(below bicubic)* | **0.7418** |
| synthetic mild | 27.925 | 28.069 |

> [!WARNING]
> Rung 3 had learned to invert **our** operator rather than KLA's. Its SSIM on real data was *worse than plain bicubic interpolation* — a clear signal that the synthetic metrics completely hid.

Mixing 50% real pairs fixed it — **and synthetic performance improved too**, so the two distributions share representation rather than competing.

Real batches use **no geometric augmentation** (the pair is fixed and aligned) and **blind** `(a,b)` estimation, matching test-time conditions exactly.

### ✅ Rung 6 · Capacity — and a methodology lesson

Three attempts:

| attempt | config | steps | result |
|:--|:--|--:|:--|
| 1 | bs=48, lr=1.4e-3 | 300 | **−4.57 dB** — diverged |
| 2 | bs=32, lr=6e-4 | 270 | −0.18 dB — aborted prematurely |
| 3 | bs=16, lr=6e-4 | **7,200** | **+0.351 dB** — success |

Attempts 1 and 2 received roughly **4% of the training rung 5 got**. Neither was a fair test.

> [!IMPORTANT]
> **Total optimiser steps, not epoch count, governs convergence.**
>
> With `bs=48`, 30 epochs is 1,800 steps. With `bs=8`, 20 epochs is 7,220. Comparing *"30 epochs"* against *"20 epochs"* is meaningless without holding steps fixed.

### 🏆 Rung 7 · Absolute high-pass loss

```
L += 0.4 · Charbonnier( highpass(out), highpass(gt) )
```

**Absolute**, unlike rung 2's relative version. Applied as a 12-epoch fine-tune at `lr = 1.5e-4`.

| | rung 6 | rung 7 |
|:--|--:|--:|
| PSNR | 27.905 | **27.965** |
| SSIM | 0.7638 | **0.7652** |
| HF error | 88.8% | **87.5%** |

All three improve, none regress. The gain is small — 1.3 percentage points of HF error against a predicted ~5 — which indicates the remaining spectral error is **structural**, not a loss-weighting problem.

### ✅ 7b · NLF clamp — a safeguard with a documented failure mode

On 3 of 400 test images the blind estimator returned `a` far outside the training envelope:

| Image | estimated a | training max |
|:--|--:|--:|
| `000017` | 0.231 | 0.22 |
| `000195` | 0.363 | 0.22 |
| `000257` | **1.212** | 0.22 |

An `a` of 1.212 implies `L = 0.83`, physically impossible for a Gamma speckle process. All three are sharp diagonal or grid patterns with almost no actual noise.

Clamping to `a ≤ 0.22, σ ≤ 0.19` fixes all three visually. **Aggregate impact: none** — 27.965 dB either way, because validation estimates were already inside the envelope.

> [!TIP]
> **Principle: never ask the model to operate outside the regime it was trained on.**

### 📷 The result

![Restoration examples](figures/03_before_after.png)
<sub><i>Held-out test images spanning the severity range. Left: degraded input with its estimated noise parameters. Middle: bicubic. Right: our restoration.</i></sub>

---

## Part 4 · Generalisation

Four public corpora, **none used in training**, degraded with the verified operator and restored with blind parameter estimation.

| Corpus | n | Mild gain | Severe gain | SSIM gain *(severe)* |
|:--|--:|--:|--:|--:|
| *KLA val (reference)* | 309 | +3.42 | **+8.16** | — |
| BSD100 | 100 | +2.68 | **+6.90** | +0.327 |
| Set14 | 14 | +2.66 | **+6.84** | +0.327 |
| DIV2K | 100 | +2.28 | **+6.04** | +0.292 |
| Urban100 | 100 | +1.83 | **+5.28** | +0.221 |

![Generalisation](figures/04_generalisation.png)

**Two things this establishes:**

The model inverts the **operator**, not the dataset — 5–7 dB on 314 unseen images is not memorisation.

The variance-stabilisation signature holds everywhere: **severe gains 2.5–3× more than mild on every single corpus**, reproduced four independent times.

> [!NOTE]
> **Urban100 is the weakest at +5.28 dB.** Building facades — repeating windows, balconies, railings — the closest natural analogue to periodic wafer geometry, and the same structure class that defeats the blind estimator (see 7c). **This is our documented weakness.**

---

## Part 5 · Training and machine hygiene

> [!CAUTION]
> **fp32 throughout.** Mixed precision produces NaN in this pipeline:
> - `sinh` in the inverse VST overflows fp16 above an argument of ~11
> - `ms_ssim` underflows in its five-scale product, giving NaN gradients even when the forward value prints finite
>
> Measured under autocast: **792 of 2891 batches produced non-finite loss**, and EMA weights froze entirely. A 30-step fp32 control run was clean (loss 0.495 → 0.301). Cost is ~2× wall-clock; correctness wins.

| Practice | Detail |
|:--|:--|
| **Content-clustered split** | KMeans on 16×16 thumbnails — near-duplicates cannot straddle train and val |
| **Checkpoint selection** | On real-pair PSNR, never on training loss |
| **EMA weights** | Decay 0.998, used for all evaluation and the exported model |
| **Gradient clipping** | 0.5, with a finiteness guard skipping bad batches |
| **Fixed seeds** | torch, numpy, and the clustering |

**Reference run:** A100-SXM4-80GB · 40 epochs · batch 16 · lr 6e-4 · 7,200 steps · **30.2 minutes** — followed by a 12-epoch fine-tune at lr 1.5e-4 with the high-pass loss.

---

## Part 6 · Data quality findings

| Finding | Detail |
|:--|:--|
| Exact duplicates | **0** across 3200 GT (MD5) |
| Structureless GT | **126 images (3.9%)** below a structure-to-noise threshold of 2.0, in contiguous index blocks |
| GT normalisation | All 3200 have exactly one pixel at 0.0 and one at 1.0 |
| GT is not 8-bit | ×255 gives mean fractional part 0.25 — float resampling preceded normalisation |
| GT sharpness spread | High-frequency energy varies **143×** across the corpus (0.015% → 2.147%) |
| GT noise floor | Up to σ ≈ 0.038 in some GT images — **the target is not clean** |
| Input range | `[−0.279, 2.158]`; up to 12% of pixels exceed 1.0. **Never clip.** |
| Test envelope | a p99 = 0.207, σ p99 = 0.166 — **99.2% / 99.5%** inside the trained range |

> [!IMPORTANT]
> The GT sharpness spread and the GT noise floor together **bound what any model can achieve**: on images whose ground truth is itself grainy, *perfect* denoising **reduces** PSNR, because the target retains the grain.

---

<div align="center">

**Every number above is reproducible from this repository.**

<sub>Figures: <code>python make_figures.py --gt-dir &lt;train/GT&gt; --lr-dir &lt;train/NoisyLR&gt; --test-dir &lt;Test_NoisyLR&gt;</code><br>
Rejection 7c: <code>python test_estimator.py --gt-dir &lt;train/GT&gt; --lr-dir &lt;train/NoisyLR&gt;</code></sub>

</div>