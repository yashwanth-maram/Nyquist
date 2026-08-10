# AI-Based Restoration of Degraded Images

**SEMICON India Hackathon 2026 · Problem Statement 1 (KLA)**

Restores speckled, noisy, half-resolution semiconductor inspection images to clean full resolution.

---

## Quick start

```bash
git clone <REPO_URL>
cd kla-ps1
pip install -r requirements.txt
python evaluate.py --input-dir /path/to/test --output-dir ./restored
```

That is the whole procedure. `weights/model.pt` (11 MB) is committed to the repository — no download step, no Git LFS, no configuration.

Reads every `.npy` in `--input-dir`, writes each restored image to `--output-dir` **under the same filename**. Runs on GPU if available, CPU otherwise. Handles 128→256 and 256→512 automatically, including mixed sizes in one directory.

### Options

| Flag | Default | |
|---|---|---|
| `--batch-size` | 16 | lower it if memory is tight |
| `--no-ensemble` | off | 8× faster, ~0.2 dB lower |
| `--device` | auto | force `cuda` or `cpu` |
| `--weights` | `weights/model.pt` | alternative checkpoint |

---

## Results

Measured on a held-out, **content-clustered** validation split of 309 images, using KLA's *actual* `NoisyLR` files with **blind** parameter estimation — no ground truth or prior knowledge used at inference.

| | PSNR | SSIM | HF error |
|---|---|---|---|
| Bicubic ×2 | 23.586 dB | 0.5721 | 132.3% |
| **This model** | **27.965 dB** | **0.7652** | **87.5%** |
| Gain | **+4.379 dB** | **+0.193** | **−44.8 pts** |

By degradation severity, gain over bicubic:

| Severity | Gain |
|---|---|
| very mild | +1.04 dB |
| mild | +0.63 dB |
| moderate | +0.84 dB |
| **severe** | **+8.16 dB** |

The asymmetry is the design working: heavy degradation is normalised into the regime the network already handles.

### Generalisation — four corpora, none used in training

| Corpus | n | Mild gain | Severe gain |
|---|---|---|---|
| KLA val *(reference)* | 309 | +3.42 | +8.16 |
| BSD100 | 100 | +2.68 | +6.90 |
| Set14 | 14 | +2.66 | +6.84 |
| DIV2K | 100 | +2.28 | +6.04 |
| Urban100 | 100 | +1.83 | +5.28 |

314 unseen images, 5.3–6.9 dB on severe degradation. The model inverts the **operator**, not the dataset.

### Performance

| | |
|---|---|
| Parameters | 2.76 M |
| Weights on disk | 11.1 MB |
| Single pass | **3.68 ms/image** (A100-80GB) |
| 8× self-ensemble | **29.4 ms/image** |
| Full 400-image test set | **11.8 s** end-to-end |

---

## Method

### 1. The degradation operator was measured, not assumed

From 3200 matched pairs:

```
mu = AreaAvg_2x2(x)
y  = mu * s + n      s ~ Gamma(L, 1/L),  n ~ Normal(0, sigma^2)
```

Both noise terms applied at **low resolution**, spatially iid, order and presence randomised per sample.

Three findings that differ from the obvious reading of the brief:

- **There is no blur operator.** MSE against blur σ is minimised at exactly 0.0 and rises monotonically, for every pair, in both orderings.
- **The downsample is 2×2 area averaging**, not decimation — which is 36% worse — nor bicubic, which is 5% worse.
- **Round-trip validated:** synthesising with the fitted parameters reproduces KLA's noise-level function to **2.3% median error** over 300 pairs.

### 2. A variance-stabilising transform, derived from that operator

The measured operator implies `Var(y|μ) = a·μ² + b`, which is heteroscedastic — noise grows with signal. A plain L2 objective assumes constant variance and mis-weights the image everywhere, over-trusting bright pixels where the data is least reliable.

That variance form admits a closed-form stabilising transform:

```
f(y) = (1/sqrt(a)) * arcsinh( y * sqrt(a/b) )
```

**Measured effect:** noise standard deviation across three brightness bands went from a **1.78× spread to 1.01×**.

Crucially, `(a, b)` are estimated **per image at inference**, unsupervised, from the image alone — 0.978 correlation with true total variance, 8.4% median error. So the transform adapts to any severity, including regimes absent from training. This is what carries the model to unseen corpora.

### 3. A gated identity path

Early versions applied denoising unconditionally and **damaged clean input by 4.3 dB** — removing detail that was never noise. A learned gate conditioned on the measured noise level lets the network modulate: clean-input PSNR rose from 27.09 to 29.80 dB.

### 4. Trained on real and synthetic pairs together

Training purely on self-synthesised degradation produced excellent synthetic scores and **near-zero real-world benefit** (+0.26 dB, SSIM *below* bicubic) — the model had learned to invert *our* operator rather than KLA's. Mixing 50% of KLA's actual pairs into training took that to **+4.03 dB**, and synthetic performance improved as well.

---

## Repository

```
evaluate.py              inference — self-contained, runs as-is
train.py                 reproduces the model from scratch
requirements.txt         minimal deps for evaluate.py
requirements-train.txt   full training environment
weights/model.pt         final weights (11 MB, committed)
outputs/                 restored test set (400 images)
src/
  vst.py                 noise estimation + variance-stabilising transform
  model.py               network
  degrade.py             the verified forward operator
  metrics.py             PSNR, SSIM, high-frequency error
docs/
  ablations.md           7 rungs, 6 documented rejections
  figures/               before/after comparisons
```

`evaluate.py` deliberately duplicates the model definition rather than importing from `src/`, so a broken `PYTHONPATH` cannot stop the benchmark from running. `src/` holds the same code in modular, documented form.

---

## Reproducing training

```bash
pip install -r requirements-train.txt
python train.py --data-dir /path/to/train --out weights/model.pt
python train.py --data-dir /path/to/train --out weights/model.pt \
                --hp-loss --epochs 12 --lr 1.5e-4     # final fine-tune
```

`--data-dir` must contain `GT/` (256×256 `.npy`) and `NoisyLR/` (128×128 `.npy`) with matching filenames.

Reference run: A100-SXM4-80GB, 40 epochs, 7200 optimiser steps, **30.2 minutes**.

**fp32 is required.** Mixed precision produces NaN: `sinh` in the inverse VST overflows fp16, and `ms_ssim` underflows in its five-scale product. Measured under autocast: 792 of 2891 batches produced non-finite loss. See `docs/ablations.md` Part 6.

---

## Things we found that are easy to get wrong

**Do not clip the input to [0,1].** The observed input range is `[−0.279, 2.158]` and up to 12% of pixels legitimately exceed 1.0 — speckle pushing values beyond the original signal, exactly as the problem statement describes. A `clamp(0,1)` in preprocessing destroys real signal.

**Do not renormalise the output by min–max.** We tried it; it costs **4.07 dB**. Ground truth has exactly one pixel at 0 and one at 1 because KLA normalised a *clean* image. Applying the same to a prediction rescales by the noise outliers instead of the signal.

**126 ground-truth images (3.9%) are structureless noise**, in contiguous index blocks. Excluding them changes measured gain by only +0.086 dB, so we report on the full set for comparability — but they are unlearnable.

**Some ground truth is itself grainy** (up to σ ≈ 0.038). On those images, perfect denoising *reduces* PSNR because the target retains the grain.

---

## Ablations

Seven rungs, **six components built, measured and removed**: min–max renormalisation, the data-consistency projection, a band-weighted FFT loss, a clean-GT retrain, and a two-model ensemble. Each with the measurement that killed it.

Full detail in **[`docs/ablations.md`](docs/ablations.md)**.

---

## License

MIT
