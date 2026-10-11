"""Locate, verify and preprocess the ISIC 2018 Task 1 training data.

1. Finds the training images either in data/raw (zip or extracted folder) or in ~/Downloads
   (read only; the zip is extracted into data/raw).
2. Verifies the data: 2594 images, 2594 masks, matching ids, binary masks.
3. Small EDA: image size distribution and lesion area fraction (figure + json in results/).
4. Caches every image resized to 256x256 (area interpolation) and every mask resized with
   nearest neighbour to data/processed/{images,masks}/<isic_id>.png.
"""
import json
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "results" / "figures"
DOWNLOADS = Path.home() / "Downloads"

IMG_NAME = "ISIC2018_Task1-2_Training_Input"
MASK_NAME = "ISIC2018_Task1_Training_GroundTruth"
EXPECTED_N = 2594
SIZE = 256


def locate(name):
    """Return an extracted folder for `name`, extracting a zip from data/raw or ~/Downloads if needed."""
    for base in (RAW_DIR, DOWNLOADS):
        folder = base / name
        if folder.is_dir() and any(folder.iterdir()):
            return folder
    for base in (RAW_DIR, DOWNLOADS):
        zip_path = base / f"{name}.zip"
        if zip_path.exists():
            print(f"found {zip_path}, testing CRC ...")
            with zipfile.ZipFile(zip_path) as zf:
                bad = zf.testzip()
                if bad is not None:
                    raise RuntimeError(f"{zip_path} is corrupted ({bad}); please download it again")
                zf.extractall(RAW_DIR)
            return RAW_DIR / name
    raise FileNotFoundError(f"{name} not found (zip or folder) in {RAW_DIR} or {DOWNLOADS}")


def process_one(args):
    isic_id, img_path, mask_path = args
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        return {"isic_id": isic_id, "ok": False}
    h, w = img.shape[:2]
    values = np.unique(mask)
    row = {
        "isic_id": isic_id, "ok": True, "height": h, "width": w,
        "mask_same_size": mask.shape == (h, w),
        "mask_binary": set(values.tolist()) <= {0, 255},
        "area_frac": float((mask > 127).mean()),
    }
    img_small = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    mask_small = cv2.resize(mask, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    mask_small = np.where(mask_small > 127, 255, 0).astype(np.uint8)
    cv2.imwrite(str(PROC_DIR / "images" / f"{isic_id}.png"), img_small)
    cv2.imwrite(str(PROC_DIR / "masks" / f"{isic_id}.png"), mask_small)
    row["area_frac_256"] = float((mask_small > 0).mean())
    return row


def eda_figure(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sizes = df.groupby(["width", "height"]).size().sort_values(ascending=False)
    top = sizes.head(10)
    axes[0].barh([f"{w}x{h}" for w, h in top.index][::-1], top.values[::-1])
    axes[0].set_title(f"Top-10 image sizes ({len(sizes)} distinct)")
    axes[0].set_xlabel("images")
    axes[1].scatter(df.width, df.height, s=5, alpha=0.4)
    axes[1].set_xlabel("width, px")
    axes[1].set_ylabel("height, px")
    axes[1].set_title("Original resolution")
    axes[2].hist(df.area_frac, bins=40)
    axes[2].set_xlabel("lesion area / image area")
    axes[2].set_ylabel("images")
    axes[2].set_title("Lesion area fraction")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_train.png", dpi=120)
    plt.close(fig)


def main():
    img_dir = locate(IMG_NAME)
    mask_dir = locate(MASK_NAME)
    img_ids = {p.stem for p in img_dir.glob("ISIC_*.jpg")}
    mask_ids = {p.name.removesuffix("_segmentation.png") for p in mask_dir.glob("ISIC_*_segmentation.png")}
    print(f"images: {len(img_ids)}, masks: {len(mask_ids)}")
    assert len(img_ids) == EXPECTED_N, f"expected {EXPECTED_N} images, found {len(img_ids)}"
    assert img_ids == mask_ids, f"id mismatch: {len(img_ids ^ mask_ids)} ids without a pair"

    (PROC_DIR / "images").mkdir(parents=True, exist_ok=True)
    (PROC_DIR / "masks").mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    jobs = [(i, img_dir / f"{i}.jpg", mask_dir / f"{i}_segmentation.png") for i in sorted(img_ids)]
    with ProcessPoolExecutor(max_workers=8) as pool:
        rows = list(tqdm(pool.map(process_one, jobs, chunksize=16), total=len(jobs)))
    df = pd.DataFrame(rows)

    assert df.ok.all(), f"unreadable files: {df.loc[~df.ok, 'isic_id'].tolist()}"
    assert df.mask_same_size.all(), "some masks differ in size from their image"
    assert df.mask_binary.all(), f"non-binary masks: {df.loc[~df.mask_binary, 'isic_id'].tolist()}"
    df.to_csv(ROOT / "results" / "train_image_stats.csv", index=False)
    eda_figure(df)

    summary = {
        "n_images": len(df),
        "distinct_sizes": int(df.groupby(["width", "height"]).ngroups),
        "width": df.width.describe().round(1).to_dict(),
        "height": df.height.describe().round(1).to_dict(),
        "aspect_ratio_mean": round(float((df.width / df.height).mean()), 3),
        "area_frac": df.area_frac.describe().round(4).to_dict(),
        "area_frac_lt_5pct": int((df.area_frac < 0.05).sum()),
        "area_frac_gt_70pct": int((df.area_frac > 0.70).sum()),
        "empty_masks": int((df.area_frac == 0).sum()),
        "max_area_frac_change_after_resize": round(float((df.area_frac - df.area_frac_256).abs().max()), 4),
    }
    with open(ROOT / "results" / "eda_train.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
