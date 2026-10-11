"""Leakage-aware 85/15 train/val split of the 2594 ISIC 2018 Task 1 training images.

Images are grouped with union-find so that a group never crosses the split:
  * same patient_id (ISIC metadata)  -> same group
  * same lesion_id  (ISIC metadata)  -> same group
  * near-duplicate images (perceptual hash, Hamming distance <= PHASH_THRESHOLD) -> same group
Whole groups are then shuffled with a fixed seed and assigned to val until it holds 15% of images.
The official test set is never used here.
"""
import argparse
import json
from pathlib import Path

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
PHASH_THRESHOLD = 6  # out of 64 bits
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


def plot_phash(pair_d, dup_pairs, ids, img_dir):
    """Histogram of pairwise distances and a few matched pairs, to sanity-check the threshold."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(pair_d, bins=np.arange(0, 65), log=True)
    ax.axvline(PHASH_THRESHOLD + 0.5, c="red", ls="--", label=f"threshold <= {PHASH_THRESHOLD}")
    ax.set_xlabel("pHash Hamming distance (64 bits)")
    ax.set_ylabel("image pairs (log)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phash_distances.png", dpi=120)
    plt.close(fig)

    show = sorted(dup_pairs, key=lambda t: -t[2])[:8]  # the least similar accepted pairs
    if not show:
        return
    fig, axes = plt.subplots(len(show), 2, figsize=(5, 2.5 * len(show)), squeeze=False)
    for row, (a, b, d) in zip(axes, show):
        for ax, k in zip(row, (a, b)):
            ax.imshow(Image.open(img_dir / f"{ids[k]}.png").convert("RGB"))
            ax.set_title(f"{ids[k]}  d={d}", fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phash_pairs.png", dpi=90)
    plt.close(fig)


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
    dup_pairs = [(int(a), int(b), int(d)) for a, b, d in zip(iu[0][near], iu[1][near], pair_d[near])]
    for a, b, _ in dup_pairs:
        uf.union(a, b)

    plot_phash(pair_d, dup_pairs, ids, args.image_dir)

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
    assert dist[np.ix_(train.index, val.index)].min() > PHASH_THRESHOLD, "near-duplicate leakage"

    cols = ["isic_id", "group", "lesion_id", "patient_id", "diagnosis"]
    train[cols].to_csv(out / "train.csv", index=False)
    val[cols].to_csv(out / "val.csv", index=False)
    pd.DataFrame([(ids[a], ids[b], d) for a, b, d in dup_pairs],
                 columns=["isic_id_a", "isic_id_b", "hamming"]).to_csv(
        out / "near_duplicates_train.csv", index=False)

    summary = {
        "seed": SEED,
        "phash_threshold": PHASH_THRESHOLD,
        "n_images": n,
        "with_patient_id": int(meta.patient_id.notna().sum()),
        "with_lesion_id": int(meta.lesion_id.notna().sum()),
        "metadata_links": n_meta_links,
        "near_duplicate_pairs": len(dup_pairs),
        "hamming_distance_percentiles": {p: int(np.percentile(pair_d, p)) for p in (0.01, 0.1, 1, 50)},
        "n_groups": int(meta.group.nunique()),
        "n_groups_size_gt1": int((group_sizes > 1).sum()),
        "largest_group": int(group_sizes.max()),
        "train": {"images": len(train), "groups": int(train.group.nunique())},
        "val": {"images": len(val), "groups": int(val.group.nunique())},
        "val_fraction": round(len(val) / n, 4),
        "min_train_val_hamming": int(dist[np.ix_(train.index, val.index)].min()),
        "diagnosis_train": train.diagnosis.value_counts(normalize=True).round(3).head(6).to_dict(),
        "diagnosis_val": val.diagnosis.value_counts(normalize=True).round(3).head(6).to_dict(),
    }
    with open(out / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
