# Ablations

Every design decision below was measured, not assumed. Six components were built, tested and **removed** because the data said so; those are recorded here alongside the ones that survived.

**Primary metric** throughout is PSNR/SSIM on KLA's *actual* `NoisyLR` files over a held-out, content-clustered validation split of 309 images, using blind parameter estimation. This is the only measurement made under exactly the conditions of the hidden test set.

**Decision rules**, fixed before any result was seen:
- Δ < 0.05 dB is noise. Declare a tie.
- Δ > 0.10 dB is real. Advance.
- Ties break toward the simpler and faster option.
- A component that regresses severe-degradation performance does not advance, whatever it does elsewhere.

---

## Part 1 — Recovering the forward operator

Before any model was trained, the degradation was measured from 3200 matched pairs.

The enabling observation: LR mean matches GT mean to three decimals in every pair. Since Gamma speckle has mean 1 and Gaussian noise has mean 0, `E[y] = D(B(x))` independent of operator order — so a least-squares fit recovers the operator without the noise biasing it.

### Downsample operator

Mean MSE relative to area averaging, 100 pairs:

| Operator | Relative MSE |
|---|---|
| **2×2 area average** | **1.000** |
| Bicubic | 1.048 |
| Lanczos | 1.052 |
| Bilinear | 1.104 |
| Strided decimation | 1.364 |

### Blur: none

Grid search over σ ∈ [0, 6] at 0.05 resolution, crossed with seven downsample operators. Normalised MSE:

| σ | 0.0 | 0.2 | 0.4 | 0.6 | 1.0 |
|---|---|---|---|---|---|
| mean | **1.000** | 1.000 | 1.013 | 1.076 | 1.201 |

Minimum at exactly 0 for every pair, rising monotonically, in both possible orderings. The problem statement's summary table implies a blur; **the deck's own definition of the transform lists only three operations and blur is not among them.** We model no blur.

### Speckle is multiplicative Gamma

Exact residual `r = y − AreaAvg(x)`, binned by clean intensity, fitted to `Var = a·μ² + b`. R² of 0.93–0.99 on 8 of 9 pairs. Distributional check on bright pixels: `mean(y/μ) = 1.00` in every image, with measured skew tracking the Gamma prediction `2√var` closely (e.g. 0.356 vs 0.372 predicted).

### Noise is iid at low resolution

Autocorrelation of the exact residual: −0.06 to +0.02 at lag 1 in all directions. Noise is applied **after** downsampling, independently per pixel.

### Round-trip validation

Synthesising degradation with the fitted parameters and re-measuring gives **2.3% median error** in recovered `a` over 300 pairs (`a` real p50 = 0.0287, synthetic p50 = 0.0288).

> **Confirmed operator**
> `mu = AreaAvg_2x2(x)` · `y = mu·s + n` · `s ~ Γ(L,1/L)` · `n ~ N(0,σ²)`
> Applied at low resolution, spatially iid, order and presence both randomised.

---

## Part 2 — The rungs

| # | Change | Real PSNR | Real SSIM | Verdict |
|---|---|---|---|---|
| — | Bicubic ×2 (floor) | 23.586 | 0.5721 | baseline |
| 1 | VST + FiLM + NAFNet-32 | — | — | first working model |
| 1b | + min–max output renormalisation | — | — | ✗ **rejected** |
| 1c | + data-consistency projection | — | — | ✗ **rejected** |
| 2 | Band-weighted FFT loss | — | — | ✗ **rejected** |
| 3 | + dgate, wide SR head, clean samples | 23.840 | 0.5240 | advanced |
| 4 | 30-epoch extension | — | — | superseded |
| 5 | **+ 50% real KLA pairs in training** | 27.554 | 0.7505 | **largest single gain** |
| 5b | Clean-GT retrain | — | — | ✗ **rejected** |
| 6 | dim 32 → 64 at matched step count | 27.905 | 0.7638 | advanced |
| 6b | Model ensemble (rung5 + rung6) | 27.934 | 0.7639 | ✗ **rejected** |
| 7 | **+ absolute high-pass loss fine-tune** | **27.965** | **0.7652** | **champion** |
| 7b | + NLF clamp to training envelope | 27.965 | 0.7650 | adopted (safety) |

---

## Part 3 — The rejections

### 1b · Min–max output renormalisation — rejected, −4.07 dB

GT is per-image min–max normalised to exactly [0,1] (verified across all 3200: every file has precisely one pixel at 0.0 and one at 1.0). Renormalising the output to match seemed free.

| | bicubic | + min–max |
|---|---|---|
| mild | 24.277 | **20.205** |
| severe | 15.356 | **14.375** |

**Why it fails:** the input range is `[−0.279, 2.158]` and those extremes are *noise outliers*. Dividing by `(max − min) ≈ 2.44` crushes the real signal. You end up rescaling by the noise, not the signal. GT has that property because KLA normalised a *clean* image; a prediction with residual noise does not.

### 1c · Data-consistency projection — rejected, −0.70 dB severe

Since the operator is a known 2×2 average, `AreaAvg₂(x̂) = μ̂` can be enforced exactly by projection. Implemented with a learnable sigmoid gate so the model could choose its strength.

Inference-time sweep over six gate values:

| gate | mild | severe |
|---|---|---|
| 0.00 | 26.903 | **22.745** |
| 0.25 | **27.018** | 22.626 |
| 0.50 | 26.950 | 22.467 |
| 1.00 | 26.301 | 22.044 |

Mild peaks at gate 0.25 with +0.115 dB — inside the noise floor. **Severe degrades monotonically**, losing 0.70 dB at full strength. By the pre-registered rule, a component that regresses severe does not advance.

**Why it fails:** the constraint is only correct if `μ̂` is exact. The denoiser is good but not perfect, so the projection stamps its residual error in as hard truth and removes the network's ability to correct downstream. The harder the degradation, the worse `μ̂` is — hence severe falling fastest.

The learned gate converged to 0.126 over 12 epochs despite being free to open. **The model had already concluded the constraint didn't pay; we only listened after measuring.**

### 2 · Band-weighted FFT loss — rejected, spectral retention halved

To push high-frequency reconstruction, the FFT loss was weighted by radius² and its weight raised 5×.

Result: high-frequency retention fell from 12.6% to **6.3%** — the opposite of the intent.

**Why it fails:** the loss was *relative*, normalised by the weighted GT magnitude. In high bands that denominator is tiny, so the term explodes there, and the cheapest way for the optimiser to shrink it is to emit *less* high-frequency content. **We built a loss that rewarded smoothing.**

The corrected version (rung 7) uses an *absolute* L1 on the high-pass residual, with no denominator to game.

### 5b · Clean ground-truth retrain — rejected, +0.086 dB

A structure-to-noise detector found **126 of 3200 GT images (3.9%) are structureless noise**, occurring in contiguous index blocks — 2537–2539, 2637–2638, 2981–2983, 625–626 — which suggests one bad batch at source rather than random corruption.

| | PSNR | gain over bicubic |
|---|---|---|
| all val (309) | 27.554 | +3.969 |
| clean val (298) | 27.867 | +4.055 |

Excluding them shifts the *gain* by only +0.086 dB, because bicubic improves on the cleaner set too and most of the difference cancels. Not worth the retrain. **Reported on the full set for comparability.**

### 6b · Model ensemble — rejected, +0.029 dB

Weight sweep over rung 5 + rung 6 output averaging:

| w(rung5) | 0.00 | 0.15 | 0.25 | 0.35 | 0.50 |
|---|---|---|---|---|---|
| PSNR | 27.905 | **27.934** | 27.933 | 27.921 | 27.881 |

Best mixture beats pure rung 6 by 0.029 dB — below the 0.05 dB noise floor — for double the inference cost. Rejected.

### Superseded metric · HF *retention* → HF *error*

An early metric measured raw high-frequency **energy** in the output relative to GT energy. By that measure rung 5 (51.5%) beat rung 6 (31.1%), which nearly caused the better model to be discarded.

The metric was wrong. **Residual noise is broadband and counts as high-frequency energy**, so a model that denoises *less* scores higher. Visual inspection contradicted it plainly: rung 6 resolved individual ballast stones and truss lines that rung 5 smeared.

Corrected to high-frequency **error against ground truth**:

| | HF error |
|---|---|
| bicubic | 132.3% |
| rung 5 | 90.4% |
| rung 6 | 88.8% |
| **rung 7** | **87.5%** |

---

## Part 4 — What worked, and why

### Rung 3 · dgate identity path — the largest architectural fix

Diagnostic: feed the model a **perfectly clean** low-resolution image and ask only for 2× upsampling.

| | PSNR |
|---|---|
| bicubic | 31.40 |
| rung 1 | 27.09 (**−4.31**) |

The model *damaged* an image needing no repair, because it applied denoising unconditionally. Isolating further: `denoise → bicubic` scored 26.59 while `+ head_hr` recovered to 27.09 — so the denoiser was responsible for the whole loss.

Fix: a learned scalar gate on the denoiser output, conditioned on the measured noise level, plus ~12% completely undegraded training samples so "do nothing" is a learnable answer.

| | rung 1 | rung 3 |
|---|---|---|
| clean-input PSNR | 27.09 | **29.80** |
| denoiser damage | 29.45 | **36.34** |
| dgate on clean input | — | 0.680 |

The gate converging to 0.68 rather than 1.0 confirms the model learned to modulate rather than always denoise at full strength.

### Rung 5 · Real pairs in training — the largest gain overall

Rungs 1–4 trained purely on self-synthesised degradation. That produced strong synthetic scores and near-zero real-world benefit:

| | rung 3 | rung 5 |
|---|---|---|
| real PSNR gain | +0.255 dB | **+4.033 dB** |
| real SSIM | 0.5240 (**below bicubic**) | **0.7418** |
| synthetic mild | 27.925 | 28.069 |

Rung 3 had learned to invert *our* operator rather than KLA's. Mixing 50% real pairs into training fixed it — **and synthetic performance improved too**, so the two distributions share representation rather than competing.

Real batches use no geometric augmentation (the pair is fixed and aligned) and blind `(a,b)` estimation, matching test-time conditions exactly.

### Rung 6 · Capacity — and a methodology lesson

Three attempts:

| attempt | config | steps | result |
|---|---|---|---|
| 1 | bs=48, lr=1.4e-3 | 300 | **−4.57 dB** — diverged |
| 2 | bs=32, lr=6e-4 | 270 | −0.18 dB — aborted prematurely |
| 3 | bs=16, lr=6e-4 | **7,200** | **+0.351 dB** — success |

Attempts 1 and 2 received roughly **4% of the training rung 5 got**. Neither was a fair test.

**Total optimiser steps, not epoch count, governs convergence.** With `bs=48`, 30 epochs is 1,800 steps; with `bs=8`, 20 epochs is 7,220. Comparing "30 epochs" against "20 epochs" is meaningless without holding steps fixed.

### Rung 7 · Absolute high-pass loss

`L += 0.4 · Charbonnier(highpass(out), highpass(gt))` — absolute, unlike rung 2's relative version. Applied as a 12-epoch fine-tune at lr = 1.5e-4.

| | rung 6 | rung 7 |
|---|---|---|
| PSNR | 27.905 | **27.965** |
| SSIM | 0.7638 | **0.7652** |
| HF error | 88.8% | **87.5%** |

All three improve, none regress. Small — 1.3 percentage points of HF error against a predicted ~5 — which indicates the remaining spectral error is structural, not a loss-weighting problem.

### 7b · NLF clamp — a safeguard with a documented failure mode

On 3 of 400 test images the blind estimator returned `a` far outside the training envelope: 0.231, 0.363 and **1.212**, against a training maximum of 0.22. All three are high-contrast periodic patterns with almost no actual noise — the Immerkaer high-pass reads sharp repeating edges as speckle.

The consequence is visible: image `000257` restores with blown-out stripes and lost line separation. Clamping estimates to `a ≤ 0.22, σ ≤ 0.19` fixes all three.

Aggregate impact on validation: **none** (27.965 dB either way, SSIM −0.0002), because validation estimates were already inside the envelope. Pure safety net.

Principle: **never ask the model to operate outside the regime it was trained on.**

---

## Part 5 — Generalisation

Four public corpora, none used in training, degraded with the verified operator and restored with blind parameter estimation.

| Corpus | n | Mild gain | Severe gain | SSIM gain (severe) |
|---|---|---|---|---|
| KLA val *(reference)* | 309 | +3.42 | **+8.16** | — |
| BSD100 | 100 | +2.68 | **+6.90** | +0.327 |
| Set14 | 14 | +2.66 | **+6.84** | +0.327 |
| DIV2K | 100 | +2.28 | **+6.04** | +0.292 |
| Urban100 | 100 | +1.83 | **+5.28** | +0.221 |

Two things this establishes. The model inverts the **operator**, not the dataset — 5–7 dB on 314 unseen images is not memorisation. And the variance-stabilisation signature holds everywhere: severe gains 2.5–3× more than mild on every single corpus, reproduced four independent times.

**Urban100 is the weakest at +5.28 dB.** It is building facades — repeating windows, balconies, railings — the closest natural analogue to periodic wafer geometry, and the same structure class that defeats the blind NLF estimator. This is our documented weakness.

---

## Part 6 — Training and machine hygiene

**fp32 throughout.** Mixed precision produces NaN in this pipeline: `sinh` in the inverse VST overflows fp16 above an argument of ~11, and `ms_ssim` underflows in its five-scale product, giving NaN gradients even when the forward value prints finite. Measured under autocast: **792 of 2891 batches produced non-finite loss**, and EMA weights froze entirely. A 30-step fp32 control run was clean (loss 0.495 → 0.301). The cost is roughly 2× wall-clock; correctness wins.

**Validation split is content-clustered**, using KMeans on 16×16 thumbnails, so near-duplicate images cannot straddle train and val. A random split would leak.

**Checkpoint selection on real-pair PSNR**, never on training loss.

**EMA weights** (decay 0.998) used for all evaluation and for the exported model.

**Gradient clipping at 0.5**, with a finiteness guard that skips any batch producing a non-finite loss or gradient norm.

**Fixed seeds** for torch, numpy and the clustering.

**Reference run:** A100-SXM4-80GB, 40 epochs, batch 16, lr 6e-4, 7,200 steps, 30.2 minutes — followed by a 12-epoch fine-tune at lr 1.5e-4 with the high-pass loss.

---

## Part 7 — Data quality findings

| Finding | Detail |
|---|---|
| Exact duplicates | **0** across 3200 GT (MD5) |
| Structureless GT | **126 images (3.9%)** below a structure-to-noise threshold of 2.0; contiguous index blocks |
| GT normalisation | All 3200 have exactly one pixel at 0.0 and one at 1.0 |
| GT is not 8-bit | ×255 gives mean fractional part 0.25 — float resampling preceded normalisation |
| GT sharpness spread | High-frequency energy varies **143×** across the corpus (0.015% to 2.147%) |
| GT noise floor | Up to σ ≈ 0.038 in some GT images — **the target is not clean** |
| Input range | `[−0.279, 2.158]`; up to 12% of pixels exceed 1.0. **Never clip.** |
| Test envelope | a p99 = 0.207, σ p99 = 0.166 — 99.2% / 99.5% inside the trained range |

The GT sharpness spread and the GT noise floor together bound what any model can achieve: on images whose ground truth is itself grainy, perfect denoising *reduces* PSNR, because the target retains the grain.
