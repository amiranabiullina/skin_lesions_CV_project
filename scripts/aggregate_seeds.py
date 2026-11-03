"""Aggregate the 3 seeds of U-Net and DeepLabV3+ into a final results table and figure.

    python scripts/aggregate_seeds.py

Reads metrics_val.json and per_image_val.csv of the six runs (no data or checkpoints needed) and
writes to results/comparison/:
    seeds_summary.md     per-seed metrics and mean +- std over seeds, model difference vs. seed spread,
                         Dice by lesion size and diagnosis averaged over seeds
    seeds_summary.json   the same numbers
    seeds_dice.png       (a) validation Dice per seed with mean +- std, (b) Dice by lesion size
std is the sample standard deviation over the 3 seeds (ddof=1). A run whose folder is missing is
listed as "in progress" and left out of the means.
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "U-Net": {42: "results", 1: "results/unet_r34_seed1", 2: "results/unet_r34_seed2"},
    "DeepLabV3+": {42: "results/deeplabv3plus_r34", 1: "results/deeplabv3plus_r34_seed1",
                   2: "results/deeplabv3plus_r34_seed2"},
}
COLORS = {"U-Net": "tab:blue", "DeepLabV3+": "tab:orange"}
OUT = ROOT / "results" / "comparison"
N_VAL = 389
SIZE_BINS = [0, 0.05, 0.2, 0.5, 1.0001]
SIZE_LABELS = ["< 5%", "5-20%", "20-50%", "> 50%"]
# (key, label, function of metrics_val.json, decimals)
METRICS = [
    ("dice", "Dice", lambda m: m["dice"]["mean"], 3),
    ("iou", "IoU", lambda m: m["iou"]["mean"], 3),
    ("isic", "ISIC thresh. Jaccard", lambda m: m["isic_thresholded_jaccard"], 3),
    ("below", "Images with IoU < 0.65", lambda m: m["n_iou_below_0.65"], 1),
    ("hd95_med", "HD95 median (px)", lambda m: m["hd95_px"]["median"], 2),
    ("hd95_p95", "HD95 95th pct. (px)", lambda m: m["hd95_px"]["p95"], 1),
]


def load_runs():
    rows, per_image, missing = [], {}, []
    for model, seeds in RUNS.items():
        for seed, d in seeds.items():
            path = ROOT / d / "metrics_val.json"
            if not path.exists():
                missing.append(f"{model} seed {seed} in progress")
                continue
            m = json.load(open(path))
            assert m["n_images"] == N_VAL, f"{d}: n_images = {m['n_images']}, expected {N_VAL}"
            row = {"model": model, "seed": seed, "best_epoch": m["best_epoch"],
                   "epochs_run": m["epochs_run"], "train_min": m["train_time_s"] / 60}
            row.update({k: f(m) for k, _, f, _ in METRICS})
            rows.append(row)
            per_image[(model, seed)] = pd.read_csv(ROOT / d / "per_image_val.csv").set_index("isic_id")
    return pd.DataFrame(rows), per_image, missing


def mean_std(values, decimals):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return f"{values.mean():.{decimals}f}"
    return f"{values.mean():.{decimals}f} ± {values.std(ddof=1):.{decimals}f}"


def md_table(header, rows):
    out = "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n"
    return out + "".join("| " + " | ".join(map(str, r)) + " |\n" for r in rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs, per_image, missing = load_runs()
    models = list(RUNS)

    # all runs must be evaluated on the same 389 images
    ids = {tuple(sorted(df.index)) for df in per_image.values()}
    assert len(ids) == 1, "runs were evaluated on different images"

    # 1) final table: mean +- std over seeds
    final = [[label] + [mean_std(runs[runs.model == mo][k], dec) for mo in models]
             for k, label, _, dec in METRICS]
    final.append(["Seeds"] + [", ".join(map(str, sorted(runs[runs.model == mo].seed))) for mo in models])

    # 2) per-seed table
    per_seed = [[r.model, r.seed] + [str(int(getattr(r, k))) if k == "below" else f"{getattr(r, k):.{dec}f}"
                                     for k, _, _, dec in METRICS]
                + [r.best_epoch, r.epochs_run, f"{r.train_min:.1f}"]
                for r in runs.sort_values(["model", "seed"], ascending=[False, True]).itertuples()]

    # 3) model difference vs. seed spread
    common = sorted(set(runs[runs.model == models[0]].seed) & set(runs[runs.model == models[1]].seed))
    diff = {}
    for k, label, _, dec in METRICS:
        a = runs[runs.model == models[0]].set_index("seed")[k]
        b = runs[runs.model == models[1]].set_index("seed")[k]
        diff[k] = {"label": label, "mean_diff": float(b.mean() - a.mean()),
                   "per_seed_diff": {int(s): float(b[s] - a[s]) for s in common},
                   "std_unet": float(a.std(ddof=1)), "std_dlv3p": float(b.std(ddof=1))}

    # 4) per-image Dice averaged over the seeds of each model -> paired test without seed noise
    seed_avg = {mo: pd.concat([per_image[(mo, s)].dice for s in runs[runs.model == mo].seed], axis=1).mean(axis=1)
                for mo in models}
    d_img = seed_avg[models[1]] - seed_avg[models[0]]
    stat, p = wilcoxon(seed_avg[models[1]], seed_avg[models[0]])
    rng = np.random.default_rng(0)
    boot = [rng.choice(d_img.values, len(d_img)).mean() for _ in range(5000)]
    paired = {"mean_diff": float(d_img.mean()), "median_diff": float(d_img.median()),
              "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
              "dlv3p_better": int((d_img > 0).sum()), "unet_better": int((d_img < 0).sum()),
              "wilcoxon_p": float(p)}

    # 5) Dice by lesion size / diagnosis: per seed, then mean +- std over seeds
    meta = pd.read_csv(ROOT / "splits/val.csv").set_index("isic_id")[["diagnosis"]]
    meta = meta.join(pd.read_csv(ROOT / "results/train_image_stats.csv").set_index("isic_id")[["area_frac"]])
    meta["size"] = pd.cut(meta.area_frac, SIZE_BINS, labels=SIZE_LABELS, right=False)
    meta["dx"] = meta.diagnosis.where(~meta.diagnosis.str.startswith("Melanoma"), "Melanoma (all)")
    groups = {}
    for by in ["size", "dx"]:
        g = {}
        for (mo, s), df in per_image.items():
            g[(mo, s)] = df.dice.groupby(meta.loc[df.index, by], observed=True).mean()
        groups[by] = g

    def group_rows(by):
        cats = list(groups[by][(models[0], 42)].index)
        n = meta[by].value_counts()
        rows = []
        for c in cats:
            vals = {mo: [groups[by][(mo, s)][c] for s in runs[runs.model == mo].seed] for mo in models}
            rows.append([c, int(n[c])] + [mean_std(vals[mo], 3) for mo in models]
                        + [f"{np.mean(vals[models[1]]) - np.mean(vals[models[0]]):+.3f}"])
        return rows

    # write markdown
    with open(OUT / "seeds_summary.md", "w") as f:
        f.write(f"## Final results: validation split ({N_VAL} images, 256x256), mean ± std over seeds\n\n")
        if missing:
            f.write("**" + "; ".join(missing) + "**\n\n")
        f.write(md_table(["Metric"] + models, final) + "\n")
        f.write("std = sample standard deviation over seeds (ddof = 1).\n\n")
        f.write("## Per seed\n\n")
        f.write(md_table(["Model", "Seed"] + [l for _, l, _, _ in METRICS] + ["Best epoch", "Epochs", "Train min"],
                         per_seed) + "\n")
        f.write("## Model difference (DeepLabV3+ − U-Net) vs. spread over seeds\n\n")
        f.write(md_table(["Metric", "Mean difference", "Per seed (" + ", ".join(map(str, common)) + ")",
                          "std U-Net", "std DeepLabV3+"],
                         [[v["label"], f"{v['mean_diff']:+.4f}",
                           ", ".join(f"{x:+.4f}" for x in v["per_seed_diff"].values()),
                           f"{v['std_unet']:.4f}", f"{v['std_dlv3p']:.4f}"] for v in diff.values()]) + "\n")
        f.write("## Per-image Dice averaged over seeds (paired, DeepLabV3+ − U-Net)\n\n")
        f.write(f"mean {paired['mean_diff']:+.4f} (95% bootstrap CI {paired['ci95'][0]:+.4f} … "
                f"{paired['ci95'][1]:+.4f}), median {paired['median_diff']:+.4f}; DeepLabV3+ better on "
                f"{paired['dlv3p_better']} images, U-Net on {paired['unet_better']}; Wilcoxon p = "
                f"{paired['wilcoxon_p']:.3g}\n\n")
        f.write("## Dice by lesion size (mean ± std over seeds)\n\n")
        f.write(md_table(["Lesion area", "n"] + models + ["Δ"], group_rows("size")) + "\n")
        f.write("## Dice by diagnosis (mean ± std over seeds)\n\n")
        f.write(md_table(["Diagnosis", "n"] + models + ["Δ"], group_rows("dx")))

    json.dump({"runs": runs.to_dict("records"), "difference": diff, "paired_seed_avg": paired,
               "missing": missing}, open(OUT / "seeds_summary.json", "w"), indent=2, default=float)

    # figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 1.6]})
    ax = axes[0]
    for i, mo in enumerate(models):
        r = runs[runs.model == mo]
        ax.scatter(np.full(len(r), i) + np.linspace(-0.08, 0.08, len(r)), r.dice,
                   color=COLORS[mo], alpha=0.6, zorder=3)
        for s, x, y in zip(r.seed, np.full(len(r), i) + np.linspace(-0.08, 0.08, len(r)), r.dice):
            ax.annotate(f"seed {s}", (x, y), xytext=(8, -3), textcoords="offset points", fontsize=8, color="gray")
        ax.errorbar(i + 0.3, r.dice.mean(), yerr=r.dice.std(ddof=1), fmt="D", color=COLORS[mo],
                    capsize=6, markersize=7, zorder=4)
    ax.set_xticks(range(len(models)), models)
    ax.set_xlim(-0.5, len(models) - 0.3)
    ax.set_ylabel("Validation Dice (mean over 389 images)")
    ax.set_title("(a) Dice per seed; diamond = mean ± std")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    x = np.arange(len(SIZE_LABELS))
    w = 0.3
    n = meta["size"].value_counts()
    for j, mo in enumerate(models):
        vals = np.array([[groups["size"][(mo, s)][c] for c in SIZE_LABELS] for s in runs[runs.model == mo].seed])
        xs = x + (j - 0.5) * w
        ax.scatter(np.repeat(xs, len(vals)), vals.T.ravel(), color=COLORS[mo], alpha=0.35, s=18)
        ax.errorbar(xs, vals.mean(0), yerr=vals.std(0, ddof=1) if len(vals) > 1 else None, fmt="D",
                    color=COLORS[mo], capsize=5, markersize=7, label=mo)
    ax.set_xticks(x, [f"{c}\n(n = {n[c]})" for c in SIZE_LABELS])
    ax.set_ylim(0.86, 0.95)
    ax.set_xlabel("Lesion area (share of the image)")
    ax.set_ylabel("Dice")
    ax.set_title("(b) Dice by lesion size; dots = seeds, diamond = mean ± std")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "seeds_dice.png", dpi=150)
    plt.close(fig)

    print(open(OUT / "seeds_summary.md").read())


if __name__ == "__main__":
    main()
