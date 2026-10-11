"""Evaluate the best checkpoint on the validation split.

    python evaluate.py --config configs/unet_r34.yaml

Outputs: results/metrics_val.json, results/per_image_val.csv,
         results/figures/learning_curves.png, results/figures/val_best_worst.png
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

from src.dataset import IMAGENET_MEAN, IMAGENET_STD, LesionDataset, get_transforms
from src.metrics import all_metrics
from src.model import build_model
from src.utils import get_device, load_config, sync

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@torch.no_grad()
def predict_all(model, ds, device, threshold):
    rows, preds = [], {}
    loader = torch.utils.data.DataLoader(ds, batch_size=16, num_workers=4)
    for x, y, ids in loader:
        prob = torch.sigmoid(model(x.to(device))).cpu().numpy()[:, 0]
        gt = y.numpy()[:, 0] > 0.5
        for i, p, g in zip(ids, prob, gt):
            pred = p > threshold
            rows.append({"isic_id": i, **all_metrics(pred, g)})
            preds[i] = pred
    return pd.DataFrame(rows), preds


@torch.no_grad()
def time_inference(model, ds, device, n_warmup=10, n_images=100):
    """Wall-clock forward pass per image (batch size 1, includes host->device copy)."""
    xs = [ds[i][0][None] for i in range(min(n_images, len(ds)))]
    for x in xs[:n_warmup]:
        model(x.to(device))
    sync(device)
    times = []
    for x in xs:
        t0 = time.perf_counter()
        torch.sigmoid(model(x.to(device)))
        sync(device)
        times.append(time.perf_counter() - t0)
    return 1000 * np.mean(times), 1000 * np.std(times)


def plot_learning_curves(log_csv, out):
    log = pd.read_csv(log_csv)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(log.epoch, log.train_loss, label="train")
    axes[0].plot(log.epoch, log.val_loss, label="val")
    axes[0].set_title("BCE + Dice loss")
    axes[0].legend()
    axes[1].plot(log.epoch, log.val_dice, label="val Dice")
    axes[1].plot(log.epoch, log.val_iou, label="val IoU")
    best = log.val_dice.idxmax()
    axes[1].axvline(log.epoch[best], ls="--", c="gray", lw=1)
    axes[1].set_title(f"Validation (best Dice {log.val_dice[best]:.4f} @ epoch {log.epoch[best]})")
    axes[1].legend()
    axes[2].plot(log.epoch, log.epoch_time_s)
    axes[2].set_title("Epoch time, s")
    for ax in axes:
        ax.set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def denorm(x):
    img = x.numpy().transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)


def plot_best_worst(ds, df, preds, out, k=4):
    order = df.sort_values("dice")
    picks = list(order.tail(k).itertuples()) + list(order.head(k).itertuples())
    index = {i: n for n, i in enumerate(ds.ids)}
    fig, axes = plt.subplots(2, k, figsize=(4 * k, 8.5))
    for ax, r in zip(axes.flat, picks):
        x, y, _ = ds[index[r.isic_id]]
        ax.imshow(denorm(x))
        ax.contour(y[0].numpy(), levels=[0.5], colors="lime", linewidths=1.5)
        ax.contour(preds[r.isic_id].astype(float), levels=[0.5], colors="red", linewidths=1.5)
        ax.set_title(f"{r.isic_id}\nDice {r.dice:.3f}  HD95 {r.hd95:.1f}px", fontsize=9)
        ax.axis("off")
    fig.suptitle("Top row: best 4, bottom row: worst 4 (green = ground truth, red = prediction)")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def summarise(values):
    return {"mean": float(np.mean(values)), "std": float(np.std(values)),
            "median": float(np.median(values))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unet_r34.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = get_device(cfg["device"])
    threshold = cfg["train"]["threshold"]

    ckpt = torch.load(cfg["output"]["checkpoint"], map_location="cpu", weights_only=False)
    model = build_model({**cfg, "model": {**cfg["model"], "encoder_weights": None}})
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    d = cfg["data"]
    ids = pd.read_csv(d["val_csv"]).isic_id
    ds = LesionDataset(ids, d["image_dir"], d["mask_dir"], get_transforms(train=False))

    df, preds = predict_all(model, ds, device, threshold)
    ms_mean, ms_std = time_inference(model, ds, device)

    results_dir = Path("results")
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_dir / "per_image_val.csv", index=False)

    summary_txt = Path(cfg["output"]["log_csv"]).with_name("train_summary.txt")
    train_info = dict(line.strip().split("=") for line in open(summary_txt)) if summary_txt.exists() else {}
    metrics = {
        "split": "val (15% of ISIC 2018 Task 1 train, leakage-aware)",
        "n_images": len(df),
        "resolution": f"{d['image_size']}x{d['image_size']}",
        "threshold": threshold,
        "best_epoch": ckpt["epoch"],
        "dice": summarise(df.dice),
        "iou": summarise(df.iou),
        "hd95_px": summarise(df.hd95),
        "n_empty_predictions": int(sum(not p.any() for p in preds.values())),
        "n_iou_below_0.65": int((df.iou < 0.65).sum()),
        "inference_ms_per_image": {"mean": ms_mean, "std": ms_std, "batch_size": 1,
                                   "device": str(device)},
        "train_time_s": float(train_info.get("total_train_time_s", "nan")),
        "epochs_run": int(train_info.get("epochs_run", 0)),
    }
    with open(results_dir / "metrics_val.json", "w") as f:
        json.dump(metrics, f, indent=2)

    if Path(cfg["output"]["log_csv"]).exists():
        plot_learning_curves(cfg["output"]["log_csv"], fig_dir / "learning_curves.png")
    plot_best_worst(ds, df, preds, fig_dir / "val_best_worst.png")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
