"""Visual + numeric check of the training pipeline: augmented image/mask pairs stay aligned and binary.

    python scripts/check_augmentations.py
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dataset import IMAGENET_MEAN, IMAGENET_STD, LesionDataset, get_transforms  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main(image_dir=ROOT / "data/processed/images", mask_dir=ROOT / "data/processed/masks",
         train_csv=ROOT / "splits/train.csv", out=ROOT / "results/figures/augmentations.png"):
    ids = pd.read_csv(train_csv).isic_id.tolist()[:4]
    plain = LesionDataset(ids, image_dir, mask_dir, get_transforms(train=False))
    aug = LesionDataset(ids, image_dir, mask_dir, get_transforms(train=True))

    def show(ax, x, y, title):
        img = np.clip(x.numpy().transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN, 0, 1)
        ax.imshow(img)
        ax.contour(y[0].numpy(), levels=[0.5], colors="lime", linewidths=1.2)
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    torch.manual_seed(0)
    np.random.seed(0)
    fig, axes = plt.subplots(len(ids), 4, figsize=(12, 3 * len(ids)))
    for r, isic_id in enumerate(ids):
        x, y, _ = plain[r]
        show(axes[r, 0], x, y, f"{isic_id} (no aug)")
        for c in range(1, 4):
            xa, ya, _ = aug[r]
            show(axes[r, c], xa, ya, f"augmented #{c}")
    fig.suptitle("Training augmentations; green contour = mask after the same transform")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100)
    plt.close(fig)

    # numeric checks over many random draws
    full = LesionDataset(pd.read_csv(train_csv).isic_id, image_dir, mask_dir, get_transforms(train=True))
    xs, values = [], set()
    for i in np.random.default_rng(0).choice(len(full), 200, replace=False):
        x, y, _ = full[i]
        assert x.shape == (3, 256, 256) and x.dtype == torch.float32
        assert y.shape == (1, 256, 256) and y.dtype == torch.float32
        values |= set(torch.unique(y).tolist())
        xs.append(x)
    xs = torch.stack(xs)
    assert values <= {0.0, 1.0}, f"mask not binary after augmentation: {values}"
    print(f"image tensor {tuple(x.shape)} {x.dtype}, range [{xs.min():.2f}, {xs.max():.2f}], "
          f"per-channel mean {xs.mean((0, 2, 3)).numpy().round(3)}, std {xs.std((0, 2, 3)).numpy().round(3)}")
    print(f"mask tensor {tuple(y.shape)} {y.dtype}, unique values {sorted(values)}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
