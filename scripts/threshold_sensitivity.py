"""Sensitivity of the validation metrics to the binarisation threshold (no retraining).

For each config, the best checkpoint predicts probabilities on the validation split once;
the masks are then binarised at every threshold and scored with the same per-image metrics
as evaluate.py. At threshold 0.5 the numbers must equal the run's metrics_val.json.

    python scripts/threshold_sensitivity.py                      # default runs below
    python scripts/threshold_sensitivity.py --configs configs/unet_r34.yaml --name unet

Outputs: results/ablation/threshold_sensitivity.csv (one row per run x threshold),
         results/ablation/threshold_sensitivity.md, results/ablation/threshold_sensitivity.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import LesionDataset, get_transforms  # noqa: E402
from src.metrics import all_metrics  # noqa: E402
from src.model import build_model  # noqa: E402
from src.utils import get_device, load_config  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DEFAULT_RUNS = [  # runs whose checkpoints are available locally
    ("U-Net, seed 42", "configs/unet_r34.yaml"),
    ("DeepLabV3+, seed 1", "configs/deeplabv3plus_r34_seed1.yaml"),
    ("DeepLabV3+, seed 2", "configs/deeplabv3plus_r34_seed2.yaml"),
]


@torch.no_grad()
def predict_probs(cfg):
    device = get_device(cfg["device"])
    ckpt = torch.load(cfg["output"]["checkpoint"], map_location="cpu", weights_only=False)
    model = build_model({**cfg, "model": {**cfg["model"], "encoder_weights": None}})
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    d = cfg["data"]
    ids = pd.read_csv(d["val_csv"]).isic_id
    ds = LesionDataset(ids, d["image_dir"], d["mask_dir"], get_transforms(train=False))
    loader = torch.utils.data.DataLoader(ds, batch_size=16, num_workers=4)
    probs, gts, names = [], [], []
    for x, y, batch_ids in loader:
        probs.append(torch.sigmoid(model(x.to(device))).cpu().numpy()[:, 0])
        gts.append(y.numpy()[:, 0] > 0.5)
        names += list(batch_ids)
    return np.concatenate(probs), np.concatenate(gts), names


def sweep(probs, gts):
    rows = []
    for t in THRESHOLDS:
        m = pd.DataFrame([all_metrics(p > t, g) for p, g in zip(probs, gts)])
        rows.append({
            "threshold": t,
            "dice": m.dice.mean(),
            "iou": m.iou.mean(),
            "isic_thresholded_jaccard": np.where(m.iou < 0.65, 0.0, m.iou).mean(),
            "n_iou_below_0.65": int((m.iou < 0.65).sum()),
            "hd95_median": m.hd95.median(),
            "hd95_p95": np.percentile(m.hd95, 95),
            "pred_area_ratio": float(sum((p > t).sum() for p in probs) / gts.sum()),
            "n_images": len(m),
        })
    return pd.DataFrame(rows)


def to_markdown(df):
    lines = []
    for run, g in df.groupby("run", sort=False):
        best = g.loc[g.dice.idxmax()]
        lines += [f"### {run}", "",
                  "| threshold | Dice | IoU | ISIC thr. Jaccard | IoU < 0.65 | HD95 median | HD95 p95 | pred / GT area |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in g.rename(columns={"n_iou_below_0.65": "n_bad"}).itertuples():
            mark = " **(used)**" if r.threshold == 0.5 else ""
            lines.append(f"| {r.threshold:.1f}{mark} | {r.dice:.4f} | {r.iou:.4f} | {r.isic_thresholded_jaccard:.4f} | "
                         f"{r.n_bad} | {r.hd95_median:.2f} | {r.hd95_p95:.1f} | {r.pred_area_ratio:.3f} |")
        span = g[g.threshold.between(0.3, 0.7)]
        lines += ["", f"Best Dice {best.dice:.4f} at threshold {best.threshold:.1f}; "
                  f"Dice range over 0.3-0.7: {span.dice.min():.4f}-{span.dice.max():.4f} "
                  f"(spread {span.dice.max() - span.dice.min():.4f}).", ""]
    return "\n".join(lines)


def plot(df, out):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for run, g in df.groupby("run", sort=False):
        axes[0].plot(g.threshold, g.dice, marker="o", label=run)
        axes[1].plot(g.threshold, g["n_iou_below_0.65"], marker="o", label=run)
        axes[2].plot(g.threshold, g.hd95_p95, marker="o", label=run)
    for ax, title in zip(axes, ["Mean Dice", "Images with IoU < 0.65", "HD95, 95th percentile (px)"]):
        ax.axvline(0.5, ls="--", c="gray", lw=1)
        ax.set_xlabel("threshold")
        ax.set_title(title)
    axes[0].legend(fontsize=8)
    fig.suptitle("Threshold sensitivity on the validation split (389 images); dashed line = threshold used")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", help="configs to sweep (default: runs with local checkpoints)")
    parser.add_argument("--names", nargs="+", help="display names, one per config")
    parser.add_argument("--out", default="results/ablation/threshold_sensitivity")
    args = parser.parse_args()
    runs = list(zip(args.names or args.configs, args.configs)) if args.configs else DEFAULT_RUNS

    frames = []
    for name, path in runs:
        cfg = load_config(path)
        probs, gts, _ = predict_probs(cfg)
        df = sweep(probs, gts)
        df.insert(0, "run", name)
        df.insert(1, "config", path)
        frames.append(df)
        print(df.drop(columns="config").to_string(index=False, float_format="%.4f"), flush=True)
    df = pd.concat(frames, ignore_index=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out.with_suffix(".csv"), index=False, float_format="%.6f")
    out.with_suffix(".md").write_text("# Threshold sensitivity (validation, 389 images)\n\n" + to_markdown(df))
    plot(df, out.with_suffix(".png"))


if __name__ == "__main__":
    main()
