"""Leakage-aware 85/15 train/val split of the 2594 ISIC 2018 Task 1 training images.

Images are grouped with union-find so that a group never crosses the split:
  * same patient_id (ISIC metadata)  -> same group
  * same lesion_id  (ISIC metadata)  -> same group
  * near-duplicate images -> same group. Every pair of images is compared with SIFT keypoints
    (grayscale + CLAHE, so colour changes don't matter) and a RANSAC similarity transform;
    a pair with >= SIFT_MIN_INLIERS geometrically consistent matches is the same lesion photographed
    again (re-crop, zoom, colour change, follow-up visit).
pHash was tried first but is not reliable on dermoscopy (see the report): its distance for real
duplicates overlaps with unrelated images, so it is only kept as a diagnostic figure.
Whole groups are then shuffled with a fixed seed and assigned to val until it holds 15% of images.
The official test set is never used here.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import imagehash
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
META_CSV = ROOT / "splits" / "isic_metadata.csv"
IMG_DIR = ROOT / "data" / "processed" / "images"
SEED = 42
VAL_FRAC = 0.15
SIFT_MIN_INLIERS = 8
PHASH_THRESHOLD = 6  # diagnostic only
MAX_PLAUSIBLE_GROUP = 20
FIG_DIR = ROOT / "results" / "figures"


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def sift_features(path):
    gray = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    kp, desc = cv2.SIFT_create(nfeatures=500).detectAndCompute(gray, None)
    return np.float32([k.pt for k in kp]), desc


_FEATS = None


def _init_worker(feats):
    global _FEATS
    _FEATS = feats


def count_inliers(a, b, feats):
    (pa, da), (pb, db) = feats[a], feats[b]
    if da is None or db is None or len(pa) < 8 or len(pb) < 8:
        return 0
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [m[0] for m in knn if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]  # Lowe ratio test
    if len(good) < 8:
        return 0
    src = pa[[m.queryIdx for m in good]]
    dst = pb[[m.trainIdx for m in good]]
    M, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0)
    if M is None:
        return 0
    # reject degenerate fits (e.g. everything matched onto one point of a black border):
    # a re-photographed lesion is related by a plausible zoom, not a collapse
    scale = np.hypot(M[0, 0], M[1, 0])
    if not 0.5 <= scale <= 2.0:
        return 0
    inl = mask.ravel().astype(bool)
    # count distinct keypoints on both sides, so one point matched many times counts once
    return min(len(np.unique(src[inl].round(), axis=0)), len(np.unique(dst[inl].round(), axis=0)))


def _match_row(a):
    n = len(_FEATS)
    return [(a, b, k) for b in range(a + 1, n) if (k := count_inliers(a, b, _FEATS)) >= 5]


def sift_pairs(ids, img_dir):
    """All image pairs with >= 5 RANSAC inliers (exhaustive, ~3.4M pairs, parallel)."""
    feats = [sift_features(img_dir / f"{i}.png") for i in tqdm(ids, desc="sift")]
    with ProcessPoolExecutor(max_workers=8, initializer=_init_worker, initargs=(feats,)) as pool:
        rows = list(tqdm(pool.map(_match_row, range(len(ids)), chunksize=8), total=len(ids), desc="match"))
    return [p for row in rows for p in row]


def phash_bits(ids, img_dir):
    bits = []
    for i in tqdm(ids, desc="phash"):
        h = imagehash.phash(Image.open(img_dir / f"{i}.png"))
        bits.append(h.hash.flatten())
    return np.array(bits, dtype=bool)


def hamming_matrix(a, b):
    a = a.astype(np.int16)
    b = b.astype(np.int16)
    # number of differing bits = |a| + |b| - 2 * |a & b|
    return a.sum(1)[:, None] + b.sum(1)[None, :] - 2 * (a @ b.T)


def plot_pairs(pairs, ids, img_dir, label, out):
    if not pairs:
        return
    fig, axes = plt.subplots(len(pairs), 2, figsize=(5, 2.5 * len(pairs)), squeeze=False)
    for row, (a, b, v) in zip(axes, pairs):
        for ax, k in zip(row, (a, b)):
            ax.imshow(Image.open(img_dir / f"{ids[k]}.png").convert("RGB"))
            ax.set_title(f"{ids[k]}  {label}={v}", fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)


def plot_diagnostics(pair_d, phash_pairs, inliers, ids, img_dir):
    """pHash distance histogram, SIFT inlier histogram and borderline pairs for both methods."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].hist(pair_d, bins=np.arange(0, 65), log=True)
    axes[0].axvline(PHASH_THRESHOLD + 0.5, c="red", ls="--", label=f"<= {PHASH_THRESHOLD}")
    axes[0].set_xlabel("pHash Hamming distance (64 bits), all pairs")
    axes[0].set_ylabel("image pairs (log)")
    axes[0].legend()
    axes[1].hist([min(k, 300) for _, _, k in inliers], bins=np.arange(5, 305, 5), log=True)
    axes[1].axvline(SIFT_MIN_INLIERS, c="red", ls="--", label=f">= {SIFT_MIN_INLIERS}")
    axes[1].set_xlabel("SIFT + RANSAC inliers (pairs with >= 5; capped at 300)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "duplicate_detection.png", dpi=120)
    plt.close(fig)

    plot_pairs(sorted(phash_pairs, key=lambda t: -t[2])[:8], ids, img_dir, "d",
               FIG_DIR / "phash_pairs.png")
    accepted = sorted([p for p in inliers if p[2] >= SIFT_MIN_INLIERS], key=lambda t: t[2])
    plot_pairs(accepted[:8], ids, img_dir, "inliers", FIG_DIR / "sift_pairs_borderline.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, default=IMG_DIR)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "splits")
    args = parser.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(META_CSV).sort_values("isic_id").reset_index(drop=True)
    ids = meta.isic_id.tolist()
    n = len(ids)
    uf = UnionFind(n)

    n_meta_links = 0
    for col in ("patient_id", "lesion_id"):
        for _, idx in meta.dropna(subset=[col]).groupby(col).groups.items():
            idx = list(idx)
            for j in idx[1:]:
                uf.union(idx[0], j)
                n_meta_links += 1

    bits = phash_bits(ids, args.image_dir)
    dist = hamming_matrix(bits, bits)
    iu = np.triu_indices(n, k=1)
    pair_d = dist[iu]
    near = pair_d <= PHASH_THRESHOLD
    phash_pairs = [(int(a), int(b), int(d)) for a, b, d in zip(iu[0][near], iu[1][near], pair_d[near])]

    inliers = sift_pairs(ids, args.image_dir)
    dup_pairs = [p for p in inliers if p[2] >= SIFT_MIN_INLIERS]
    for a, b, _ in dup_pairs:
        uf.union(a, b)
    plot_diagnostics(pair_d, phash_pairs, inliers, ids, args.image_dir)

    roots = [uf.find(i) for i in range(n)]
    meta["group"] = pd.factorize(pd.Series(roots))[0]
    group_sizes = meta.group.value_counts()
    if group_sizes.max() > MAX_PLAUSIBLE_GROUP:
        print(f"WARNING: largest group has {group_sizes.max()} images -> threshold is probably too loose")

    rng = np.random.default_rng(SEED)
    groups = np.sort(group_sizes.index.to_numpy())  # sorted copy, so the shuffle only depends on the seed
    rng.shuffle(groups)
    target = round(VAL_FRAC * n)
    val_groups, count = set(), 0
    for g in groups:
        if count >= target:
            break
        val_groups.add(g)
        count += group_sizes[g]

    meta["split"] = np.where(meta.group.isin(val_groups), "val", "train")
    train, val = meta[meta.split == "train"], meta[meta.split == "val"]

    # checks
    assert len(train) + len(val) == n
    assert not set(train.group) & set(val.group), "group leakage"
    for col in ("patient_id", "lesion_id"):
        assert not set(train[col].dropna()) & set(val[col].dropna()), f"{col} leakage"
    split_of = meta.split.to_numpy()
    assert all(split_of[a] == split_of[b] for a, b, _ in dup_pairs), "near-duplicate leakage"
    max_cross = max([k for a, b, k in inliers if split_of[a] != split_of[b]], default=0)

    cols = ["isic_id", "group", "lesion_id", "patient_id", "diagnosis"]
    train[cols].to_csv(out / "train.csv", index=False)
    val[cols].to_csv(out / "val.csv", index=False)
    pd.DataFrame([(ids[a], ids[b], k) for a, b, k in inliers],
                 columns=["isic_id_a", "isic_id_b", "sift_inliers"]).to_csv(
        out / "sift_pairs_ge5.csv", index=False)  # lets the threshold be changed without rematching
    pd.DataFrame([(ids[a], ids[b], d) for a, b, d in dup_pairs],
                 columns=["isic_id_a", "isic_id_b", "sift_inliers"]).to_csv(
        out / "near_duplicates_train.csv", index=False)

    summary = {
        "seed": SEED,
        "sift_min_inliers": SIFT_MIN_INLIERS,
        "n_images": n,
        "with_patient_id": int(meta.patient_id.notna().sum()),
        "with_lesion_id": int(meta.lesion_id.notna().sum()),
        "metadata_links": n_meta_links,
        "near_duplicate_pairs": len(dup_pairs),
        "pairs_with_5_to_7_inliers": sum(5 <= k < SIFT_MIN_INLIERS for _, _, k in inliers),
        "phash_pairs_le_threshold_diagnostic": len(phash_pairs),
        "n_groups": int(meta.group.nunique()),
        "n_groups_size_gt1": int((group_sizes > 1).sum()),
        "largest_group": int(group_sizes.max()),
        "train": {"images": len(train), "groups": int(train.group.nunique())},
        "val": {"images": len(val), "groups": int(val.group.nunique())},
        "val_fraction": round(len(val) / n, 4),
        "max_train_val_sift_inliers": max_cross,
        "diagnosis_train": train.diagnosis.value_counts(normalize=True).round(3).head(6).to_dict(),
        "diagnosis_val": val.diagnosis.value_counts(normalize=True).round(3).head(6).to_dict(),
    }
    with open(out / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
