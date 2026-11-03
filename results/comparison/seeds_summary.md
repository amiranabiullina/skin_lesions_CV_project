## Final results: validation split (389 images, 256x256), mean ± std over seeds

| Metric | U-Net | DeepLabV3+ |
|---|---|---|
| Dice | 0.903 ± 0.002 | 0.901 ± 0.002 |
| IoU | 0.837 ± 0.003 | 0.834 ± 0.003 |
| ISIC thresh. Jaccard | 0.797 ± 0.002 | 0.793 ± 0.005 |
| Images with IoU < 0.65 | 32.3 ± 0.6 | 33.3 ± 2.1 |
| HD95 median (px) | 7.14 ± 0.12 | 6.87 ± 0.48 |
| HD95 95th pct. (px) | 47.5 ± 4.2 | 43.1 ± 1.7 |
| Seeds | 1, 2, 42 | 1, 2, 42 |

std = sample standard deviation over seeds (ddof = 1).

## Per seed

| Model | Seed | Dice | IoU | ISIC thresh. Jaccard | Images with IoU < 0.65 | HD95 median (px) | HD95 95th pct. (px) | Best epoch | Epochs | Train min |
|---|---|---|---|---|---|---|---|---|---|---|
| U-Net | 1 | 0.905 | 0.839 | 0.799 | 32 | 7.00 | 45.6 | 17 | 27 | 40.6 |
| U-Net | 2 | 0.904 | 0.838 | 0.796 | 33 | 7.21 | 44.6 | 33 | 43 | 65.3 |
| U-Net | 42 | 0.901 | 0.834 | 0.796 | 32 | 7.21 | 52.4 | 27 | 37 | 54.9 |
| DeepLabV3+ | 1 | 0.903 | 0.837 | 0.794 | 34 | 6.32 | 42.6 | 18 | 28 | 45.2 |
| DeepLabV3+ | 2 | 0.900 | 0.833 | 0.797 | 31 | 7.21 | 44.9 | 18 | 28 | 48.0 |
| DeepLabV3+ | 42 | 0.899 | 0.830 | 0.787 | 35 | 7.07 | 41.6 | 21 | 31 | 49.9 |

## Model difference (DeepLabV3+ − U-Net) vs. spread over seeds

| Metric | Mean difference | Per seed (1, 2, 42) | std U-Net | std DeepLabV3+ |
|---|---|---|---|---|
| Dice | -0.0024 | -0.0023, -0.0035, -0.0015 | 0.0022 | 0.0019 |
| IoU | -0.0034 | -0.0020, -0.0044, -0.0037 | 0.0026 | 0.0033 |
| ISIC thresh. Jaccard | -0.0041 | -0.0044, +0.0013, -0.0091 | 0.0018 | 0.0053 |
| Images with IoU < 0.65 | +1.0000 | +2.0000, -2.0000, +3.0000 | 0.5774 | 2.0817 |
| HD95 median (px) | -0.2718 | -0.6754, +0.0000, -0.1400 | 0.1219 | 0.4766 |
| HD95 95th pct. (px) | -4.4375 | -2.9379, +0.3633, -10.7380 | 4.2382 | 1.6905 |

## Per-image Dice averaged over seeds (paired, DeepLabV3+ − U-Net)

mean -0.0024 (95% bootstrap CI -0.0064 … +0.0014), median -0.0016; DeepLabV3+ better on 172 images, U-Net on 217; Wilcoxon p = 0.0162

## Dice by lesion size (mean ± std over seeds)

| Lesion area | n | U-Net | DeepLabV3+ | Δ |
|---|---|---|---|---|
| < 5% | 90 | 0.883 ± 0.007 | 0.875 ± 0.010 | -0.008 |
| 5-20% | 140 | 0.897 ± 0.004 | 0.898 ± 0.002 | +0.002 |
| 20-50% | 107 | 0.912 ± 0.002 | 0.906 ± 0.004 | -0.006 |
| > 50% | 52 | 0.939 ± 0.005 | 0.941 ± 0.003 | +0.002 |

## Dice by diagnosis (mean ± std over seeds)

| Diagnosis | n | U-Net | DeepLabV3+ | Δ |
|---|---|---|---|---|
| Melanoma (all) | 88 | 0.891 ± 0.005 | 0.883 ± 0.001 | -0.007 |
| Nevus | 263 | 0.915 ± 0.001 | 0.913 ± 0.003 | -0.002 |
| Seborrheic keratosis | 38 | 0.850 ± 0.020 | 0.856 ± 0.007 | +0.006 |
