"""Loss ablation summary: U-Net seed 42 trained with BCE + Dice (baseline), BCE only and soft Dice only.

Run after
    python train.py    --config configs/ablation/unet_r34_{bce,dice}_only.yaml
    python evaluate.py --config configs/ablation/unet_r34_{bce,dice}_only.yaml

    python scripts/summarize_loss_ablation.py

Outputs: results/ablation/loss_ablation.md, results/ablation/loss_ablation.json,
         results/ablation/loss_ablation_curves.png
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUNS = [  # name, results dir, train log
    ("BCE + Dice (baseline)", "results", "results/train_log.csv"),
    ("BCE only", "results/ablation/unet_r34_bce_only", "results/ablation/unet_r34_bce_only/train_log.csv"),
    ("Dice only", "results/ablation/unet_r34_dice_only", "results/ablation/unet_r34_dice_only/train_log.csv"),
]
OUT = Path("results/ablation")
N_BOOT = 10_000


def paired_bootstrap(diff, rng):
    """95% CI of the mean of per-image differences (resampling images)."""
    idx = rng.integers(0, len(diff), size=(N_BOOT, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    rng = np.random.default_rng(0)
    base = pd.read_csv(Path(RUNS[0][1]) / "per_image_val.csv").set_index("isic_id")
    rows = []
    for name, rdir, _ in RUNS:
        m = json.load(open(Path(rdir) / "metrics_val.json"))
        per = pd.read_csv(Path(rdir) / "per_image_val.csv").set_index("isic_id").loc[base.index]
        diff = (per.dice - base.dice).to_numpy()
        lo, hi = paired_bootstrap(diff, rng)
        rows.append({
            "run": name, "n_images": m["n_images"], "dice": m["dice"]["mean"], "iou": m["iou"]["mean"],
            "isic_thresholded_jaccard": m["isic_thresholded_jaccard"], "n_iou_below_0.65": m["n_iou_below_0.65"],
            "hd95_median": m["hd95_px"]["median"], "hd95_p95": m["hd95_px"]["p95"],
            "n_empty_predictions": m["n_empty_predictions"],
            "best_epoch": m["best_epoch"], "epochs_run": m["epochs_run"], "train_time_min": m["train_time_s"] / 60,
            "delta_dice_vs_baseline": float(diff.mean()), "delta_dice_ci95": [lo, hi],
            "n_images_dice_better": int((diff > 0).sum()), "n_images_dice_worse": int((diff < 0).sum()),
        })

    # best threshold of each run, if the threshold sweep was run on these checkpoints
    sweep_csv = OUT / "threshold_sensitivity_loss.csv"
    if sweep_csv.exists():
        sw = pd.read_csv(sweep_csv)
        for r in rows:
            g = sw[sw.run == r["run"]]
            if len(g):
                b = g.loc[g.dice.idxmax()]
                span = g[g.threshold.between(0.3, 0.7)]
                r.update(best_threshold=float(b.threshold), dice_at_best_threshold=float(b.dice),
                         dice_spread_0p3_0p7=float(span.dice.max() - span.dice.min()))

    with open(OUT / "loss_ablation.json", "w") as f:
        json.dump(rows, f, indent=2)

    lines = ["# Loss ablation: U-Net / ResNet-34, seed 42, validation (389 images), threshold 0.5", "",
             "Only the loss changes; data, split, augmentations, optimiser, early stopping and seed are the same.", "",
             "| Loss | Dice | IoU | ISIC thr. Jaccard | IoU < 0.65 | HD95 median | HD95 p95 | best / run epochs | train min |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['run']} | {r['dice']:.4f} | {r['iou']:.4f} | {r['isic_thresholded_jaccard']:.4f} | "
                     f"{r['n_iou_below_0.65']} | {r['hd95_median']:.2f} | {r['hd95_p95']:.1f} | "
                     f"{r['best_epoch']} / {r['epochs_run']} | {r['train_time_min']:.1f} |")
    lines += ["", "Paired difference in per-image Dice vs. the baseline (95% bootstrap CI over images, "
              f"{N_BOOT} resamples):", ""]
    for r in rows[1:]:
        lo, hi = r["delta_dice_ci95"]
        lines.append(f"* {r['run']}: {r['delta_dice_vs_baseline']:+.4f} ({lo:+.4f} to {hi:+.4f}); "
                     f"better on {r['n_images_dice_better']}, worse on {r['n_images_dice_worse']} images")
    if "best_threshold" in rows[0]:
        lines += ["", "Threshold sweep 0.1-0.9 on each checkpoint (`threshold_sensitivity_loss.md`):", ""]
        for r in rows:
            lines.append(f"* {r['run']}: best Dice {r['dice_at_best_threshold']:.4f} at {r['best_threshold']:.1f}; "
                         f"Dice spread over 0.3-0.7 = {r['dice_spread_0p3_0p7']:.4f}")
    (OUT / "loss_ablation.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, _, log_csv in RUNS:
        log = pd.read_csv(log_csv)
        axes[0].plot(log.epoch, log.val_dice, label=name)
        axes[1].plot(log.epoch, log.val_iou, label=name)
    for ax, t in zip(axes, ["Validation Dice (threshold 0.5)", "Validation IoU"]):
        ax.set_xlabel("epoch")
        ax.set_title(t)
        ax.legend()
    fig.suptitle("Loss ablation, U-Net / ResNet-34, seed 42")
    fig.tight_layout()
    fig.savefig(OUT / "loss_ablation_curves.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
