# Ablation and sensitivity experiments

Validation split, 389 images, 256×256. All runs on an Apple-silicon MacBook (MPS).

## 1. Loss ablation (U-Net / ResNet-34, seed 42)

Only the loss changes (`configs/ablation/unet_r34_{bce,dice}_only.yaml` vs. `configs/unet_r34.yaml`).

| Loss | Dice | IoU | ISIC thr. Jaccard | IoU < 0.65 | HD95 p95 (px) | best / run epochs |
|---|---:|---:|---:|---:|---:|---:|
| BCE + Dice (baseline) | **0.901** | **0.834** | **0.796** | **32** | 52.4 | 27 / 37 |
| BCE only | 0.897 | 0.828 | 0.778 | 41 | 52.0 | 18 / 28 |
| Dice only | 0.894 | 0.825 | 0.787 | 35 | 60.8 | 10 / 20 |

* The combined loss is best on every overlap metric. Dropping either term costs 0.004 (BCE only) or
  0.007 (Dice only) Dice, which is 2–3× the seed-to-seed std of the baseline (0.002), but the paired
  95% bootstrap intervals over images (−0.012…+0.005 and −0.015…+0.002) still include zero, and each
  variant was trained with one seed only. Treat the ranking as indicative, not significant.
* **BCE only** produces poorly calibrated, "soft" probability maps: its Dice depends strongly on the
  threshold (0.840–0.899 over 0.1–0.9; spread 0.027 over 0.3–0.7) and at 0.5 it already
  under-segments (predicted / true area 0.94), giving 9 more failed images than the baseline.
* **Dice only** gives almost binary outputs (Dice spread 0.002 over 0.3–0.7), but stops early
  (best epoch 10) and has the worst boundary outliers (HD95 p95 60.8 vs 52.4 px): without the
  per-pixel BCE term, isolated wrong regions are penalised less.
* BCE + Dice combines both: well-calibrated probabilities (stable around 0.5) and the best overlap.

Files: `loss_ablation.md/.json`, `loss_ablation_curves.png`, `threshold_sensitivity_loss.*`,
`unet_r34_{bce,dice}_only/`. Reproduce:

```bash
python train.py    --config configs/ablation/unet_r34_bce_only.yaml    # ~36 min on MPS
python evaluate.py --config configs/ablation/unet_r34_bce_only.yaml
python train.py    --config configs/ablation/unet_r34_dice_only.yaml   # ~27 min on MPS
python evaluate.py --config configs/ablation/unet_r34_dice_only.yaml
python scripts/threshold_sensitivity.py --configs configs/unet_r34.yaml configs/ablation/unet_r34_bce_only.yaml \
    configs/ablation/unet_r34_dice_only.yaml --names "BCE + Dice (baseline)" "BCE only" "Dice only" \
    --out results/ablation/threshold_sensitivity_loss
python scripts/summarize_loss_ablation.py
```

## 2. Threshold sensitivity (final models, no retraining)

Checkpoints: U-Net seed 42, DeepLabV3+ seeds 1 and 2 (`threshold_sensitivity.*`;
`python scripts/threshold_sensitivity.py`).

* The fixed threshold 0.5 is practically optimal for both architectures: the best threshold is 0.4
  for all three runs and gains at most +0.0004 Dice, i.e. far below the seed spread. The threshold was
  not tuned on validation, so the reported results are not optimistically biased.
* Over 0.3–0.7 Dice changes by at most 0.006 per run, and the three runs stay within 0.004 Dice of
  each other at every threshold in that range, so the "practically equal" conclusion of the model
  comparison does not depend on the threshold. Only extreme values hurt (0.1 or 0.9: −0.012 to −0.031 Dice, 40–58 failed images
  instead of 31–34).
* Raising the threshold reduces U-Net's extreme boundary errors (HD95 p95 52.4 → 48.6 px at 0.6)
  at almost no cost in Dice: part of U-Net's larger boundary outliers are low-confidence false
  positives.

Self-check: at threshold 0.5 every sweep reproduces the run's `metrics_val.json` exactly
(Dice, IoU, ISIC Jaccard, failures, HD95 median and p95), n = 389 for every run.
