"""Fetch lesion_id / patient_id for every ISIC 2018 Task 1 training image from the ISIC archive API.

The result is cached in splits/isic_metadata.csv so the split can be rebuilt without network access.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
MASK_DIR = ROOT / "data" / "raw" / "ISIC2018_Task1_Training_GroundTruth"
OUT_CSV = ROOT / "splits" / "isic_metadata.csv"
API = "https://api.isic-archive.com/api/v2/images/{}/"


def fetch(isic_id):
    for _ in range(3):
        try:
            r = requests.get(API.format(isic_id), timeout=30)
            r.raise_for_status()
            break
        except requests.RequestException:
            continue
    else:
        return {"isic_id": isic_id, "api_ok": False}
    meta = r.json().get("metadata", {})
    clinical = meta.get("clinical", {})
    acq = meta.get("acquisition", {})
    return {
        "isic_id": isic_id,
        "api_ok": True,
        "lesion_id": clinical.get("lesion_id"),
        "patient_id": clinical.get("patient_id"),
        "diagnosis": clinical.get("diagnosis_3") or clinical.get("diagnosis_1"),
        "pixels_x": acq.get("pixels_x"),
        "pixels_y": acq.get("pixels_y"),
    }


def main():
    ids = sorted(p.name.removesuffix("_segmentation.png") for p in MASK_DIR.glob("*_segmentation.png"))
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(tqdm(pool.map(fetch, ids), total=len(ids)))
    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"saved {len(df)} rows -> {OUT_CSV}")
    print(f"API ok: {df.api_ok.sum()}, with lesion_id: {df.lesion_id.notna().sum()}, "
          f"with patient_id: {df.patient_id.notna().sum()}")


if __name__ == "__main__":
    main()
