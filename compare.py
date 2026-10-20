"""Compare U-Net and DeepLabV3+ on the same validation images.

    python compare.py

Reads the outputs of evaluate.py for both models and writes to results/comparison/:
    comparison.json        summary metrics, per-group breakdown, paired test
    comparison.md          the same as Markdown tables
    learning_curves.png    validation Dice and losses of both models side by side
    largest_differences.png  images where the two models disagree most
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from scipy.stats import wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODELS = {
    "U-Net": {"dir": Path("results"), "arch": "Unet"},
    "DeepLabV3+": {"dir": Path("results/deeplabv3plus_r34"), "arch": "DeepLabV3Plus"},
}
CONFIGS = {"U-Net": "configs/unet_r34.yaml", "DeepLabV3+": "configs/deeplabv3plus_r34.yaml"}
OUT = Path("results/comparison")
SIZE_BINS = [0, 0.05, 0.2, 0.5, 1.0001]
SIZE_LABELS = ["< 5%", "5-20%", "20-50%", "> 50%"]
NOISE = 0.01  # single seed: Dice differences below this are treated as noise


def load(name, info):
    m = json.load(open(info["dir"] / "metrics_val.json"))
    if "n_params" not in m:  # the U-Net run predates the n_params field
        model = getattr(smp, info["arch"])(encoder_name="resnet34", encoder_weights=None, classes=1)
        m["n_params"] = sum(p.numel() for p in model.parameters())
    df = pd.read_csv(info["dir"] / "per_image_val.csv")
    log = pd.read_csv(info["dir"] / "train_log.csv")
    return m, df, log


def summary_rows(metrics):
    rows = []
    for name, m in metrics.items():
        rows.append({
            "model": name,
            "params_M": m["n_params"] / 1e6,
            "dice_mean": m["dice"]["mean"], "dice_std": m["dice"]["std"], "dice_median": m["dice"]["median"],
            "iou_mean": m["iou"]["mean"], "iou_std": m["iou"]["std"], "iou_median": m["iou"]["median"],
            "isic_jaccard": m["isic_thresholded_jaccard"],
            "n_iou_below_0.65": m["n_iou_below_0.65"],
            "hd95_mean": m["hd95_px"]["mean"], "hd95_median": m["hd95_px"]["median"],
            "hd95_p95": m["hd95_px"]["p95"],
            "train_min": m["train_time_s"] / 60, "epochs_run": m["epochs_run"], "best_epoch": m["best_epoch"],
            "infer_ms": m["inference_ms_per_image"]["mean"], "infer_ms_std": m["inference_ms_per_image"]["std"],
            "device": m["inference_ms_per_image"]["device"],
        })
    return pd.DataFrame(rows)


def paired_test(a, b, col):
    """Wilcoxon signed-rank test, b vs a, on per-image values (zero differences dropped)."""
    diff = b[col] - a[col]
    stat, p = wilcoxon(b[col], a[col], zero_method="wilcox")
    return {"metric": col, "mean_diff": float(diff.mean()), "median_diff": float(diff.median()),
            "b_better": int((diff > 0).sum()), "a_better": int((diff < 0).sum()),
            "ties": int((diff == 0).sum()), "wilcoxon_stat": float(stat), "p_value": float(p)}


def breakdown(merged, by):
    g = merged.groupby(by, observed=True)
    return pd.DataFrame({
        "n": g.size(),
        "dice_unet": g.dice_unet.mean(), "dice_dlv3p": g.dice_dlv3p.mean(),
        "delta_dice": g.dice_dlv3p.mean() - g.dice_unet.mean(),
        "hd95_median_unet": g.hd95_unet.median(), "hd95_median_dlv3p": g.hd95_dlv3p.median(),
    }).reset_index()


def plot_curves(logs, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for (name, log), c in zip(logs.items(), ["tab:blue", "tab:orange"]):
        axes[0].plot(log.epoch, log.val_dice, c=c, label=name)
        best = log.val_dice.idxmax()
        axes[0].scatter(log.epoch[best], log.val_dice[best], c=c, zorder=3)
        axes[1].plot(log.epoch, log.train_loss, c=c, ls="--", label=f"{name} train")
        axes[1].plot(log.epoch, log.val_loss, c=c, label=f"{name} val")
    axes[0].set_title("Validation Dice (dot = best epoch)")
    axes[0].set_ylim(0.8, 0.93)
    axes[1].set_title("BCE + Dice loss")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def md_table(df):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: f"{v:.4g}")
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "".join("| " + " | ".join(map(str, r)) + " |\n" for r in df.itertuples(index=False))


@torch.no_grad()
def plot_largest_differences(merged, out, k=3):
    """Images where one model is much better than the other (needs data/ and both checkpoints)."""
    from evaluate import denorm
    from src.dataset import LesionDataset, get_transforms
    from src.model import build_model
    from src.utils import get_device, load_config

    preds, cfg = {}, None
    picks = list(merged.nlargest(k, "delta_dice").isic_id) + list(merged.nsmallest(k, "delta_dice").isic_id)
    for name, path in CONFIGS.items():
        cfg = load_config(path)
        ckpt = Path(cfg["output"]["checkpoint"])
        if not ckpt.exists():
            print(f"skip largest_differences.png: {ckpt} not found")
            return
        model = build_model({**cfg, "model": {**cfg["model"], "encoder_weights": None}})
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
        device = get_device(cfg["device"])
        model.to(device).eval()
        d = cfg["data"]
        ds = LesionDataset(pd.Series(picks), d["image_dir"], d["mask_dir"], get_transforms(train=False))
        preds[name] = [(torch.sigmoid(model(ds[i][0][None].to(device))) > 0.5).cpu().numpy()[0, 0]
                       for i in range(len(ds))]
    rows = merged.set_index("isic_id")
    fig, axes = plt.subplots(2, k, figsize=(4 * k, 8.8))
    for n, (ax, isic_id) in enumerate(zip(axes.flat, picks)):
        x, y, _ = ds[n]
        ax.imshow(denorm(x))
        ax.contour(y[0].numpy(), levels=[0.5], colors="lime", linewidths=1.5)
        ax.contour(preds["U-Net"][n].astype(float), levels=[0.5], colors="tab:blue", linewidths=1.5)
        ax.contour(preds["DeepLabV3+"][n].astype(float), levels=[0.5], colors="tab:orange", linewidths=1.5)
        r = rows.loc[isic_id]
        ax.set_title(f"{isic_id}\nDice U-Net {r.dice_unet:.3f} | DLV3+ {r.dice_dlv3p:.3f}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Top: DeepLabV3+ much better; bottom: U-Net much better\n"
                 "(green = ground truth, blue = U-Net, orange = DeepLabV3+)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metrics, dfs, logs = {}, {}, {}
    for name, info in MODELS.items():
        metrics[name], dfs[name], logs[name] = load(name, info)

    summary = summary_rows(metrics)

    a = dfs["U-Net"].set_index("isic_id")
    b = dfs["DeepLabV3+"].set_index("isic_id").loc[a.index]
    tests = [paired_test(a, b, "dice"), paired_test(a, b, "iou")]
    # for HD95 lower is better: report "b_better" as images where DeepLabV3+ has the smaller HD95
    t = paired_test(a, b, "hd95")
    t["b_better"], t["a_better"] = t["a_better"], t["b_better"]
    tests.append(t)

    merged = a.add_suffix("_unet").join(b.add_suffix("_dlv3p")).reset_index()
    merged = merged.merge(pd.read_csv("splits/val.csv")[["isic_id", "diagnosis"]], on="isic_id")
    merged = merged.merge(pd.read_csv("results/train_image_stats.csv")[["isic_id", "area_frac"]], on="isic_id")
    merged["size"] = pd.cut(merged.area_frac, SIZE_BINS, labels=SIZE_LABELS, right=False)
    merged["dx"] = merged.diagnosis.where(~merged.diagnosis.str.startswith("Melanoma"), "Melanoma (all)")
    by_size = breakdown(merged, "size")
    by_dx = breakdown(merged, "dx")
    # failures: images below the ISIC threshold for each model
    fail = {"unet_only": int(((merged.iou_unet < 0.65) & (merged.iou_dlv3p >= 0.65)).sum()),
            "dlv3p_only": int(((merged.iou_dlv3p < 0.65) & (merged.iou_unet >= 0.65)).sum()),
            "both": int(((merged.iou_unet < 0.65) & (merged.iou_dlv3p < 0.65)).sum())}
    merged["delta_dice"] = merged.dice_dlv3p - merged.dice_unet
    merged.to_csv(OUT / "per_image_both.csv", index=False)

    result = {"summary": summary.to_dict("records"), "paired_tests": tests,
              "by_size": by_size.to_dict("records"), "by_diagnosis": by_dx.to_dict("records"),
              "iou_below_0.65": fail, "noise_threshold_dice": NOISE}
    json.dump(result, open(OUT / "comparison.json", "w"), indent=2, default=float)

    with open(OUT / "comparison.md", "w") as f:
        f.write("## Summary (validation, 389 images, 256x256, seed 42)\n\n")
        f.write(md_table(summary) + "\n\n")
        f.write("## Paired per-image comparison (DeepLabV3+ vs U-Net)\n\n")
        f.write(md_table(pd.DataFrame(tests)) + "\n\n")
        f.write(f"IoU < 0.65: only U-Net {fail['unet_only']}, only DeepLabV3+ {fail['dlv3p_only']}, "
                f"both {fail['both']}\n\n")
        f.write("## By lesion size\n\n" + md_table(by_size) + "\n\n")
        f.write("## By diagnosis\n\n" + md_table(by_dx) + "\n")

    plot_curves(logs, OUT / "learning_curves.png")
    plot_largest_differences(merged, OUT / "largest_differences.png")
    print(open(OUT / "comparison.md").read())


if __name__ == "__main__":
    main()
