# Milestone: baseline — skin lesion segmentation (ISIC 2018 Task 1)

Repository: https://github.com/amiranabiullina/skin_lesions_CV_project

**Summary.** A U-Net with an ImageNet-pretrained ResNet-34 encoder, trained at 256×256 on a
leakage-aware 85/15 split of the official ISIC 2018 Task 1 training set, reaches
**Dice 0.901 ± 0.113, IoU 0.834 ± 0.142, ISIC thresholded Jaccard 0.796 and median HD95 7.2 px**
on the 389-image validation split. Training took 54.9 min on a MacBook Air M4 (MPS); inference
takes 9.0 ms per image. The official test set has not been used yet.

## 1. Data

| | |
|---|---|
| Source | ISIC 2018 Challenge, Task 1 (lesion boundary segmentation) |
| Used for this milestone | official **training** set: 2594 dermoscopic RGB images + binary masks |
| Kept aside | official test set (1000 images) — final evaluation only, never used for training, tuning or this report |
| Integrity | the browser unpacked the 11.2 GB zip and removed it, so every extracted file was checked against the size and CRC32 stored in the official zip's central directory (read with HTTP Range requests): 2596/2596 files match (`scripts/verify_extracted.py`) |

**Verification** (`scripts/prepare_data.py`): 2594 images and 2594 masks with identical ids, every file
readable, every mask the same size as its image, every mask strictly binary {0, 255}, no empty
masks, all JPEGs have EXIF orientation 1 (no hidden rotation between image and mask).

**EDA** (`results/eda_train.json`, `results/figures/eda_train.png`)

- Very heterogeneous acquisition: 206 distinct resolutions, from 576×540 to 6748×4499
  (median 3008×2000, mean aspect ratio 1.43). The most common sizes are 4288×2848 and 1024×768
  (~560 images each) and 3008×2000 (~380).
- Lesion area fraction: mean 21%, median 14%, range 0.3%–98.7%. 624 images (24%) have a lesion
  smaller than 5% of the image, 101 images have a lesion larger than 70% (often cut by the border).
- Diagnoses (from ISIC metadata): ~72% nevi, ~8% seborrheic keratoses, ~20% melanomas.

## 2. Preprocessing and data loading

- **Resize cache:** every image is resized once to 256×256 with area interpolation and stored as
  lossless PNG; masks are resized with nearest neighbour and re-binarised. Resizing changes the
  lesion area fraction by at most 0.8 percentage points. Aspect ratio is not preserved (images are
  stretched), which is the usual choice for ISIC baselines.
- **Normalisation:** cv2 BGR → RGB, scaled to [0, 1], ImageNet mean/std (fixed constants, so no
  statistics are computed on any split).
- **Training-only augmentation** (Albumentations 2.0.8): horizontal and vertical flips, random
  90° rotations, affine (scale 0.9–1.1, shift ±6.25%, rotation ±30°, reflect border), colour
  jitter (brightness/contrast/saturation 0.2, hue 0.02). Geometric transforms are applied to image
  and mask together (nearest neighbour for the mask); colour jitter only to the image. Validation
  images are only normalised.
- **Checks** (`scripts/check_augmentations.py`, `results/figures/augmentations.png`): mask
  contours stay aligned with the lesion after augmentation; tensors are (3, 256, 256) float32 in
  [−2.12, 2.64] and (1, 256, 256) float32 with values exactly {0, 1} over 200 random draws.
- **DataLoader:** batch 16, 4 workers (macOS `spawn`), persistent workers, seeded shuffling.
  Loading takes ~4 ms per batch against ~415 ms per training step, so the GPU is the bottleneck.

## 3. Train/validation split and leakage prevention

The same lesion is often photographed several times in ISIC (follow-up visits, different zoom,
crop or colour). If copies end up in both train and validation, validation scores are inflated.
Only part of the images carry identifiers in the ISIC archive metadata
(`scripts/fetch_metadata.py`, cached in `splits/isic_metadata.csv`): `lesion_id` for 1098 and
`patient_id` for 390 of the 2594 images.

**Grouping** (`scripts/make_split.py`) — union-find over three kinds of links:

1. same `patient_id`, 2. same `lesion_id` (55 links in total),
3. near-duplicate images, found by comparing **all 3.36 M image pairs** with SIFT keypoints
   (grayscale + CLAHE, so colour changes don't matter), Lowe's ratio test and a RANSAC similarity
   transform. A pair is a duplicate if it has **≥ 8 geometrically consistent, distinct keypoint
   matches**; fits with an implausible scale (outside 0.5–2) are rejected.

How the duplicate detector was chosen (all decisions checked by looking at the image pairs):

| Method | Result |
|---|---|
| pHash, Hamming ≤ 6 | 182 pairs, but about half of the inspected pairs at the threshold were different lesions with a similar overall look; real duplicates and unrelated images overlap in distance |
| ImageNet ResNet-34 embedding cosine | ~2000 pairs of unrelated lesions have cosine 0.92–0.95, so no usable threshold |
| SIFT + RANSAC, first version | dark dermoscope frames produced degenerate fits (many points mapped onto one point) and chained 850 images into one group → fixed with the scale check and distinct-point counting |
| SIFT + RANSAC, ≥ 20 inliers | clean, but inspection of the rejected 10–19 band showed they were mostly true duplicates, one of them split across train and val |
| **SIFT + RANSAC, ≥ 8 inliers (final)** | at 8–9 inliers pairs are still mostly true duplicates; at ≤ 7 matches are mostly rulers/frames of different lesions |

Figures: `results/figures/duplicate_detection.png`, `phash_pairs.png`, `sift_pairs_borderline.png`.

**Result** (`splits/split_summary.json`): 134 duplicate pairs, 2413 groups (2306 singletons,
90 pairs, the largest group has 34 images — most likely several lesions linked through an
identical ruler; it lies entirely in train, so it is over-cautious rather than leaky). Whole
groups were shuffled with seed 42 and assigned to validation until it reached 15%.

| | images | groups | nevus | melanoma in situ | seb. keratosis | invasive melanoma |
|---|---|---|---|---|---|---|
| train | 2205 | 2052 | 72.6% | 7.2% | 7.7% | 6.1% |
| val | 389 (15.0%) | 361 | 67.6% | 10.0% | 9.8% | 7.2% |

Built-in assertions: no group, image, `patient_id` or `lesion_id` in both parts; the most similar
train–val pair has 7 inliers (below the threshold).

## 4. Model configuration

`configs/unet_r34.yaml`

| | |
|---|---|
| Architecture | U-Net (segmentation-models-pytorch 0.5.0) |
| Encoder | ResNet-34, ImageNet weights |
| Output | 1 channel (logits), sigmoid at inference, threshold 0.5 |
| Parameters | 24.4 M |
| Input | 3×256×256, ImageNet normalisation |

## 5. Training setup

| | |
|---|---|
| Loss | BCE (with logits) + soft Dice, weights 1 : 1 |
| Optimiser | AdamW, lr 1e-4, weight decay 1e-4, constant lr |
| Batch size | 16 (137 steps per epoch, last incomplete batch dropped) |
| Epochs | max 50, early stopping on val Dice with patience 10 |
| Checkpoint | best val Dice |
| Seed | 42 (Python, NumPy, PyTorch, DataLoader generator, per-worker augmentation seeds) |
| Hardware | MacBook Air, Apple M4 (10-core GPU), 24 GB, macOS 26.6, PyTorch 2.14.1 MPS, Python 3.14.3, fp32 |

**Sanity checks before the real run:** the model overfits a single batch to Dice 0.993 in 150
steps; the full pipeline was dry-run on synthetic images built from the real masks.
A benchmark of one training step (batch 16): MPS fp32 0.415 s, MPS fp16 autocast 0.423 s (no gain),
MPS channels_last 1.21 s (3× slower), CPU 3.1 s — so MPS fp32 was used.

## 6. Initial results (validation split, 389 images)

Per-image metrics at 256×256, threshold 0.5, averaged over images
(`results/metrics_val.json`, `results/per_image_val.csv`).

| Metric | Mean ± std | Median | 95th pct |
|---|---|---|---|
| **Dice** | **0.901 ± 0.113** | 0.936 | 0.979 |
| **IoU (Jaccard)** | **0.834 ± 0.142** | 0.880 | 0.958 |
| ISIC thresholded Jaccard (IoU < 0.65 → 0) | 0.796 | — | — |
| **HD95, px @ 256** | 13.7 ± 17.2 | **7.2** | 52.4 |

| Cost | |
|---|---|
| Training time | 54.9 min (37 epochs, early stopped; best epoch 27) |
| Epoch time | 67–117 s (incl. validation) |
| Inference time | 9.0 ± 0.4 ms per image (MPS, batch 1, after warm-up, with `torch.mps.synchronize`) |

Metric conventions: Dice/IoU are 1 when prediction and ground truth are both empty; HD95 is the
95th percentile of symmetric boundary distances in pixels at 256×256, 0 if both are empty and the
image diagonal (362 px) if exactly one is empty. In this run there were **0 empty predictions**;
32 images (8.2%) have IoU < 0.65 and 5 have Dice < 0.5.

Breakdown (from `results/per_image_val.csv`):

| Lesion area | n | Dice | IoU | median HD95 |
|---|---|---|---|---|
| < 5% | 90 | 0.876 | 0.807 | 3.0 px |
| 5–20% | 140 | 0.895 | 0.821 | 7.1 px |
| 20–50% | 107 | 0.913 | 0.849 | 10.4 px |
| > 50% | 52 | 0.935 | 0.885 | 12.0 px |

| Diagnosis | n | Dice | IoU |
|---|---|---|---|
| Nevus | 263 | 0.917 | 0.855 |
| Melanoma (all subtypes) | 88 | 0.886 | 0.814 |
| Seborrheic keratosis | 38 | 0.828 | 0.738 |

Figures: `results/figures/learning_curves.png`, `results/figures/val_best_worst.png`.

## 7. Observations and problems encountered

- **Learning curves:** train loss fell from 0.92 to 0.17, val loss from 0.73 to ~0.22; val Dice
  passed 0.89 after 6 epochs and then plateaued at 0.89–0.90 with ±0.005 epoch-to-epoch noise.
  The gap between train and val loss opens slowly after epoch ~15 — mild overfitting, no collapse.
- **Failure cases** (bottom row of `val_best_worst.png`): low-contrast, light lesions inside
  immersion-fluid/contact-plate artefacts and ink markings; the model segments the bubble or plate
  outline or a darker neighbouring spot instead of the faint lesion. These few images dominate the
  mean HD95 (mean 13.7 px vs median 7.2 px) and the thresholded Jaccard.
- **Harder classes:** seborrheic keratoses (Dice 0.828) and melanomas (0.886) are segmented worse
  than nevi (0.917); they have more irregular, less contrasted borders.
- **Small lesions:** Dice is lower for lesions under 5% of the image (0.876), as expected for an
  overlap metric, although their absolute boundary error is small (median HD95 3 px).
- **Label noise:** some ground-truth masks are coarse or include only part of a lesion
  (e.g. ISIC_0000004), which bounds achievable Dice.
- **Duplicates and leakage:** simple near-duplicate detection (pHash) was not reliable for
  dermoscopy; SIFT + RANSAC on all pairs was needed (Section 3).
- **Augmentation RNG bug (found in code review, fixed before training):** with `spawn`, every
  DataLoader worker got an identical copy of the Albumentations RNG, so all workers produced the
  same augmentation sequence. Fixed with a `worker_init_fn` that reseeds each worker
  (verified: 16/16 identical samples across workers before, 0/16 after).
- **Hardware:** the fanless MacBook Air throttles — epoch time grew from 67 s to ~100 s
  (peak 117 s) over the run; the first part of the run was also on battery.
- **Data download:** the 11.2 GB training zip downloaded at only 1.5–4 MB/s; it was finally
  downloaded in the browser and verified against the official CRC32s.

## 8. Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python scripts/verify_extracted.py ~/Downloads/ISIC2018_Task1-2_Training_Input   # or scripts/download_data.py
python scripts/prepare_data.py
python scripts/make_split.py            # splits/ are committed, so this is optional
python train.py --config configs/unet_r34.yaml
python evaluate.py --config configs/unet_r34.yaml
```
