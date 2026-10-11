"""Download and unpack ISIC 2018 Task 1 (lesion segmentation) into data/raw.

Downloads are resumable: a partial file (<name>.part) is continued with an HTTP Range request,
so the script can simply be restarted after a network drop. Each zip is checked against the
server's Content-Length and with a CRC test before it is extracted.

    python scripts/download_data.py          # training images + masks (enough for the baseline)
    python scripts/download_data.py --test   # also the official test set (final evaluation only)
"""
import argparse
import time
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

BASE_URL = "https://isic-challenge-data.s3.amazonaws.com/2018/"
TRAIN_FILES = ["ISIC2018_Task1-2_Training_Input.zip", "ISIC2018_Task1_Training_GroundTruth.zip"]
TEST_FILES = ["ISIC2018_Task1-2_Test_Input.zip", "ISIC2018_Task1_Test_GroundTruth.zip"]
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
MAX_RETRIES = 20


def remote_size(url):
    r = requests.head(url, timeout=60)
    r.raise_for_status()
    return int(r.headers["content-length"])


def download(url, dest):
    total = remote_size(url)
    if dest.exists() and dest.stat().st_size == total:
        print(f"already downloaded: {dest.name}")
        return
    part = dest.with_suffix(".part")

    for attempt in range(1, MAX_RETRIES + 1):
        done = part.stat().st_size if part.exists() else 0
        if done >= total:
            break
        headers = {"Range": f"bytes={done}-"} if done else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                r.raise_for_status()
                if done and r.status_code != 206:  # server ignored Range -> start over
                    done = 0
                with open(part, "ab" if done else "wb") as f, \
                        tqdm(total=total, initial=done, unit="B", unit_scale=True, desc=dest.name) as bar:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        bar.update(len(chunk))
        except requests.RequestException as e:
            wait = min(2 ** attempt, 60)
            print(f"\nnetwork error ({e.__class__.__name__}), retry {attempt}/{MAX_RETRIES} in {wait}s")
            time.sleep(wait)

    size = part.stat().st_size
    if size != total:
        raise RuntimeError(f"{dest.name}: got {size} bytes, expected {total}; rerun to resume")
    part.rename(dest)


def verify_and_extract(zip_path):
    out_dir = RAW_DIR / zip_path.stem
    if out_dir.exists():
        print(f"already extracted: {out_dir.name}")
        return
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"{zip_path.name}: corrupted member {bad}; delete the zip and rerun")
        print(f"{zip_path.name}: CRC ok, extracting {len(zf.namelist())} files")
        zf.extractall(RAW_DIR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="also download the official test set")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    files = TRAIN_FILES + (TEST_FILES if args.test else [])
    for name in files:
        zip_path = RAW_DIR / name
        download(BASE_URL + name, zip_path)
        verify_and_extract(zip_path)


if __name__ == "__main__":
    main()
