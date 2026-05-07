# Current SOTA stack — airbench96 baseline (v2)

Source: `KellerJordan/cifar10-airbench/airbench96_faster.py` upstream.
Reported: **96.00 % mean accuracy** (n=200 trials), **27.3 s on 1×GPU 400W**.
On GPU the baseline is faster (~14–18 s per seed; n=10 mean train_s
calibrated by `calibrate_baseline.sh` before swarm launch).

## Architecture (~30M params, scaling_factor 1/9)

CifarNet with 3 wide ConvGroups + a frozen whitening stem.

```
input  →  whiten (frozen Conv2d, no bias)
       →  stem  (Conv 3→128, BN)
       →  stage1 ConvGroup: Conv→MaxPool→BN→Conv→BN×depth  128ch
       →  stage2 ConvGroup: Conv→MaxPool→BN→Conv→BN×depth  384ch
       →  stage3 ConvGroup: Conv→MaxPool→BN→Conv→BN×depth  512ch
       →  AvgPool (2D)  →  flatten  →  Linear(512→10) head
```

Defaults from upstream `hyp` dict:
- proxy widths: `{block1: 32, block2: 64, block3: 64}`, depth=2
- net widths: `{block1: 128, block2: 384, block3: 512}`, depth=3
- scaling_factor = 1/9 (tunes whitening norm)
- TTA level = 2 (mirror + translate at eval)

Two-stage architecture: a smaller "proxy" model trains the whitening
bias for `whiten_bias_epochs=3`, then a separate "trainbias / freezebias"
model takes over.

## Optimizer

**Single SGD with kilostep_scaled lr**:
- lr            = 9.0 (per 1024 examples; rescaled by `kilostep_scale`)
- weight_decay  = 0.012 (per 1024 examples; decoupled from lr)
- momentum      = 0.85, nesterov implicit via PyTorch SGD
- bias_scaler   = 64.0 (BN-bias lr multiplier, no weight decay on biases)
- label_smoothing = 0.2

LR schedule: triangular (linear warmup over 1 epoch, then linear decay
to 0 over remaining `train_epochs - 1` epochs).

## Augmentation

Heavier than airbench94:
- `flip` = True (random horizontal)
- `translate` = 4 px (vs 2 in airbench94)
- `cutout` = 12 px (NEW vs airbench94)
- TTA level 2 at eval

## Training horizon

- `train_epochs` = 45.0 (vs 8 in airbench94)
- `whiten_bias_epochs` = 3
- batch_size = 1024 (with batch_size_masked = 512 for the masked-loss path)

## Why this stack hits 96%

Combination of:
1. **Larger model (~30M params)** — capacity needed past 95%
2. **Stronger aug (cutout=12 + translate=4)** — extra regularization for the bigger model
3. **Longer training (45 epochs)** — proper convergence horizon
4. **Two-model proxy/main split** — whitening freezes after a short proxy phase, letting the main model train with optimal first-layer init
5. **Triangular LR schedule** — gives time for both warmup and cooldown over the longer horizon

## Mutation directions worth trying (initial)

Starting points only — read LESSONS.md for what swarm has tried so far.
**Goal: reduce `train_s` while keeping mean_acc(n=10) ≥ 0.96.**

- **arch**:
  - Reduce widths (e.g. block3 512→384) — capacity vs time tradeoff. Risk:
    drops below 96% gate.
  - Drop ConvGroup depth (3→2) — saves time but may not converge.
  - Replace whitening with a learnable but small init.
- **opt**:
  - Higher lr + shorter schedule (45 epoch → 30 epoch).
  - Switch to Muon (matrix-aware optimizer; airbench94's choice).
  - Lookahead / Sophia / Lion — modern optimizers with reported step-count
    reductions on small CNNs.
  - Asymmetric warmup/decay ratios.
- **aug**:
  - Smaller cutout (12 → 8) might let model converge faster while still
    hitting 96% — try it.
  - Mixup/CutMix as cutout replacement.
  - TTA level reduce (2 → 1) cuts eval time.
- **loss**:
  - Label smoothing tune (0.2 → 0.1 — airbench94's choice).
  - Poly-1 loss (PolyLoss, ICLR 2022) with ε.
- **reg**:
  - EMA decay tuning (default unset; some airbench variants use ~0.99).
  - Weight decay schedule (cosine?).
  - Stochastic depth in stage 2/3.

The biggest "free" wins likely come from **reducing the recipe to the
minimum needed to hit 96%** — e.g. fewer epochs + harder aug.
