"""Build the small example set shown in the live demo (demo/examples/).

    python demo/make_examples.py

For each chosen validation image it writes
  <id>.jpg        the dermoscopic image for display (longest side DISPLAY px, aspect kept)
  <id>_gt.png     the ground-truth mask at display size
  <id>_256.png    the exact model input: original resized to 256x256 with INTER_AREA,
                  i.e. the same pixels as data/processed/images (scripts/prepare_data.py)
  <id>_gt256.png  the ground truth at 256x256 (INTER_NEAREST), used for the live metrics
and examples.json with the case description and the Dice of every trained run from results/.

All images come from our validation split, so the models never saw them during training.
The raw ISIC images are looked up in data/raw/<folder> or ~/Downloads/<folder>.
"""
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "demo" / "examples"
TEST_DIR = ROOT / "demo" / "test_images"  # copies to drag into the page; the app recognises them
IMG_NAME = "ISIC2018_Task1-2_Training_Input"
MASK_NAME = "ISIC2018_Task1_Training_GroundTruth"
SIZE = 256
DISPLAY = 640

# (id, group, short note). Groups: typical, unet, deeplab, failure.
# Chosen from the per-image results of all six runs (results/*/per_image_val.csv): the 'U-Net better' and
# 'DeepLabV3+ better' cases win in every seed, so they are not seed noise; failures are the four cases of
# results/failures/failure_analysis.md, which fail in all six runs.
CASES = [
    ("ISIC_0000145", "typical", "Melanoma with a clear border: both models are near-perfect in every seed."),
    ("ISIC_0000232", "typical", "Typical nevus, medium size."),
    ("ISIC_0014652", "typical", "Large seborrheic keratosis."),
    ("ISIC_0015255", "typical", "Lesion covering most of the image."),
    ("ISIC_0013458", "typical", "Small seborrheic keratosis (4% of the image), still segmented well."),
    ("ISIC_0013333", "unet", "Small seborrheic keratosis (3%): U-Net wins in all three seeds (Dice 0.91-0.93 vs 0.83-0.84)."),
    ("ISIC_0013740", "unet", "Melanoma: U-Net is better in every seed; DeepLabV3+ leaves part of the lesion out."),
    ("ISIC_0012661", "deeplab", "Seborrheic keratosis: DeepLabV3+ wins in all three seeds, with a much tighter border (HD95 7-11 vs 14-43 px)."),
    ("ISIC_0014931", "deeplab", "Melanoma with irregular pigment: DeepLabV3+ wins in all three seeds (Dice 0.89-0.95 vs 0.76-0.84)."),
    ("ISIC_0011117", "failure", "Failure (a): ground truth includes a pale halo that is hardly visible."),
    ("ISIC_0013567", "failure", "Failure (b): faint pink lesion; the models pick a darker spot next to it."),
    ("ISIC_0013966", "failure", "Failure (c): the edge of an immersion-fluid bubble is taken as the border."),
    ("ISIC_0012160", "failure", "Failure (d): small light nevus with a diffuse border, both over-segment."),
]

RUNS = {  # results directory of every trained run
    ("U-Net", 42): "results", ("U-Net", 1): "results/unet_r34_seed1", ("U-Net", 2): "results/unet_r34_seed2",
    ("DeepLabV3+", 42): "results/deeplabv3plus_r34", ("DeepLabV3+", 1): "results/deeplabv3plus_r34_seed1",
    ("DeepLabV3+", 2): "results/deeplabv3plus_r34_seed2",
}


def locate(name):
    for base in (ROOT / "data" / "raw", Path.home() / "Downloads"):
        if (base / name).is_dir():
            return base / name
    raise FileNotFoundError(f"{name} not found in data/raw or ~/Downloads (unzip the official ISIC zip there)")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    img_dir, mask_dir = locate(IMG_NAME), locate(MASK_NAME)
    val_ids = set(pd.read_csv(ROOT / "splits" / "val.csv").isic_id)
    meta = pd.read_csv(ROOT / "results" / "comparison" / "per_image_both.csv").set_index("isic_id")
    per_run = {k: pd.read_csv(ROOT / d / "per_image_val.csv").set_index("isic_id") for k, d in RUNS.items()
               if (ROOT / d / "per_image_val.csv").exists()}

    entries = []
    for isic_id, group, note in CASES:
        assert isic_id in val_ids, f"{isic_id} is not in the validation split"
        img = cv2.imread(str(img_dir / f"{isic_id}.jpg"), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_dir / f"{isic_id}_segmentation.png"), cv2.IMREAD_GRAYSCALE)
        h, w = img.shape[:2]
        s = DISPLAY / max(h, w)
        dw, dh = round(w * s), round(h * s)

        cv2.imwrite(str(OUT / f"{isic_id}.jpg"), cv2.resize(img, (dw, dh), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        gt_disp = cv2.resize(mask, (dw, dh), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(OUT / f"{isic_id}_gt.png"), np.where(gt_disp > 127, 255, 0).astype(np.uint8))
        # identical to scripts/prepare_data.py, so live metrics match the evaluation
        cv2.imwrite(str(OUT / f"{isic_id}_256.png"), cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA))
        gt256 = cv2.resize(mask, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(OUT / f"{isic_id}_gt256.png"), np.where(gt256 > 127, 255, 0).astype(np.uint8))

        m = meta.loc[isic_id]
        dice = {}
        for (model, seed), df in per_run.items():
            dice.setdefault(model, {})[str(seed)] = round(float(df.loc[isic_id, "dice"]), 4)
        entries.append({
            "id": isic_id, "group": group, "note": note, "diagnosis": m.dx,
            "area_pct": round(100 * float(m.area_frac), 1), "size_bin": m["size"],
            "original_size": [w, h], "eval_dice": dice,
        })
        print(isic_id, group, dice)

    TEST_DIR.mkdir(exist_ok=True)
    for f in TEST_DIR.glob("*.jpg"):
        f.unlink()
    for n, e in enumerate(entries, 1):
        dx = {"Melanoma (all)": "melanoma", "Nevus": "nevus", "Seborrheic keratosis": "keratosis"}[e["diagnosis"]]
        shutil.copyfile(OUT / f"{e['id']}.jpg", TEST_DIR / f"{n:02d}_{e['group']}_{dx}_{e['id']}.jpg")

    with open(OUT / "examples.json", "w") as f:
        json.dump(entries, f, indent=2)
    print(f"{len(entries)} examples -> {OUT}")


if __name__ == "__main__":
    main()
