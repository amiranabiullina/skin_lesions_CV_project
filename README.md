# Skin lesion segmentation — ISIC 2018 Task 1

Binary segmentation of skin lesions in dermoscopic images (ISIC 2018 Challenge, Task 1).
Baseline: U-Net with an ImageNet-pretrained ResNet-34 encoder
([segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)).

Milestone report: [reports/milestone_baseline.md](reports/milestone_baseline.md)

## Repository layout

```
configs/unet_r34.yaml        model / training configuration
scripts/download_data.py     resumable download of the official ISIC 2018 zips
scripts/fetch_metadata.py    lesion_id / patient_id from the ISIC archive API -> splits/isic_metadata.csv
scripts/verify_extracted.py  checks a browser-unzipped folder against the official zip's CRC32s
scripts/prepare_data.py      verification, EDA, 256x256 cache in data/processed
scripts/make_split.py        leakage-aware 85/15 train/val split -> splits/
scripts/check_augmentations.py  augmentation sanity check + figure
src/                         dataset, model, loss, metrics, utils
train.py                     training (logs to results/train_log.csv)
evaluate.py                  validation metrics, timings, figures
splits/                      the exact split used (committed)
results/                     metrics, logs and figures (committed); checkpoints/ and data/ are not
```

## Setup

Tested on a MacBook Air (Apple M4, 24 GB, macOS 26.6) with Python 3.14.3 (Homebrew) and PyTorch MPS.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Only the official **training** set (2594 images) is used for training and validation.
The official 1000-image test set is kept aside for the final evaluation.

```bash
python scripts/download_data.py           # or put ISIC2018_Task1-2_Training_Input.zip and
                                          # ISIC2018_Task1_Training_GroundTruth.zip into data/raw/ (or ~/Downloads)
python scripts/verify_extracted.py ~/Downloads/ISIC2018_Task1-2_Training_Input  # if the browser already unzipped it
python scripts/prepare_data.py            # verify + EDA + resize cache
python scripts/fetch_metadata.py          # optional: splits/isic_metadata.csv is already committed
python scripts/make_split.py              # optional: splits/train.csv, splits/val.csv are committed
```

## Train and evaluate

```bash
python scripts/check_augmentations.py
python train.py --config configs/unet_r34.yaml --overfit-batch   # sanity check
python train.py --config configs/unet_r34.yaml
python evaluate.py --config configs/unet_r34.yaml
```
