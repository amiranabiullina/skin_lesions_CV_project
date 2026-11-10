# Loss ablation: U-Net / ResNet-34, seed 42, validation (389 images), threshold 0.5

Only the loss changes; data, split, augmentations, optimiser, early stopping and seed are the same.

| Loss | Dice | IoU | ISIC thr. Jaccard | IoU < 0.65 | HD95 median | HD95 p95 | best / run epochs | train min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BCE + Dice (baseline) | 0.9010 | 0.8341 | 0.7959 | 32 | 7.21 | 52.4 | 27 / 37 | 54.9 |
| BCE only | 0.8970 | 0.8284 | 0.7781 | 41 | 7.07 | 52.0 | 18 / 28 | 35.8 |
| Dice only | 0.8943 | 0.8254 | 0.7867 | 35 | 7.62 | 60.8 | 10 / 20 | 26.6 |

Paired difference in per-image Dice vs. the baseline (95% bootstrap CI over images, 10000 resamples):

* BCE only: -0.0040 (-0.0122 to +0.0046); better on 187, worse on 202 images
* Dice only: -0.0067 (-0.0151 to +0.0019); better on 172, worse on 217 images

Threshold sweep 0.1-0.9 on each checkpoint (`threshold_sensitivity_loss.md`):

* BCE + Dice (baseline): best Dice 0.9011 at 0.4; Dice spread over 0.3-0.7 = 0.0037
* BCE only: best Dice 0.8990 at 0.4; Dice spread over 0.3-0.7 = 0.0266
* Dice only: best Dice 0.8957 at 0.9; Dice spread over 0.3-0.7 = 0.0019
