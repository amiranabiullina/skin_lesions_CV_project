## Summary (validation, 389 images, 256x256, seed 42)

| model | params_M | dice_mean | dice_std | dice_median | iou_mean | iou_std | iou_median | isic_jaccard | n_iou_below_0.65 | hd95_mean | hd95_median | hd95_p95 | train_min | epochs_run | best_epoch | infer_ms | infer_ms_std | device |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| U-Net | 24.44 | 0.901 | 0.1131 | 0.9363 | 0.8341 | 0.1421 | 0.8802 | 0.7959 | 32 | 13.66 | 7.211 | 52.36 | 54.91 | 37 | 27 | 9.002 | 0.4393 | mps |
| DeepLabV3+ | 22.44 | 0.8995 | 0.1067 | 0.9335 | 0.8304 | 0.1375 | 0.8752 | 0.7868 | 35 | 12.27 | 7.071 | 41.62 | 49.85 | 31 | 21 | 14.67 | 0.0574 | mps |


## Paired per-image comparison (DeepLabV3+ vs U-Net)

| metric | mean_diff | median_diff | b_better | a_better | ties | wilcoxon_stat | p_value |
|---|---|---|---|---|---|---|---|
| dice | -0.0015 | -0.003453 | 165 | 224 | 0 | 3.133e+04 | 0.002948 |
| iou | -0.003695 | -0.005917 | 165 | 224 | 0 | 3.117e+04 | 0.002315 |
| hd95 | -1.386 | 0.1716 | 158 | 206 | 25 | 3.128e+04 | 0.3347 |


IoU < 0.65: only U-Net 12, only DeepLabV3+ 15, both 20

## By lesion size

| size | n | dice_unet | dice_dlv3p | delta_dice | hd95_median_unet | hd95_median_dlv3p |
|---|---|---|---|---|---|---|
| < 5% | 90 | 0.8763 | 0.8679 | -0.008373 | 3 | 3.162 |
| 5-20% | 140 | 0.8953 | 0.9 | 0.004708 | 7.071 | 6.366 |
| 20-50% | 107 | 0.9128 | 0.9043 | -0.008445 | 10.44 | 11 |
| > 50% | 52 | 0.9346 | 0.9425 | 0.007972 | 12 | 12.21 |


## By diagnosis

| dx | n | dice_unet | dice_dlv3p | delta_dice | hd95_median_unet | hd95_median_dlv3p |
|---|---|---|---|---|---|---|
| Melanoma (all) | 88 | 0.8859 | 0.884 | -0.001888 | 12.37 | 12.63 |
| Nevus | 263 | 0.9166 | 0.9117 | -0.004927 | 6 | 6 |
| Seborrheic keratosis | 38 | 0.8279 | 0.851 | 0.02311 | 12.22 | 12.57 |

