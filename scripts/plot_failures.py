"""Failure cases of U-Net and DeepLabV3+: results/failures/failure_cases.png

    python scripts/plot_failures.py
    python scripts/plot_failures.py --dlv3p-config configs/deeplabv3plus_r34.yaml   # seed 42 instead

Needs data/processed and the checkpoints named in the two configs (by default U-Net seed 42,
checkpoints/unet_r34_best.pt, and DeepLabV3+ seed 1, checkpoints/deeplabv3plus_r34_seed1_best.pt).
The four cases were chosen to have different causes and to fail in all three seeds of both models, so they are not one-run accidents (see the Dice table in
results/failures/failure_analysis.md).
GT = green, U-Net = blue, DeepLabV3+ = orange; titles give the Dice of the shown runs and the range over all three seeds.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from evaluate import denorm  # noqa: E402
from src.dataset import LesionDataset, get_transforms  # noqa: E402
from src.model import build_model  # noqa: E402
from src.utils import get_device, load_config  # noqa: E402

OUT = ROOT / "results/failures"
IDS = ["ISIC_0011117", "ISIC_0013567", "ISIC_0013966", "ISIC_0012160"]
SEED_RUNS = {"U-Net": ["results", "results/unet_r34_seed1", "results/unet_r34_seed2"],
             "DeepLabV3+": ["results/deeplabv3plus_r34", "results/deeplabv3plus_r34_seed1",
                            "results/deeplabv3plus_r34_seed2"]}


def seed_dice():
    """Dice of every case in each of the three seeds of both models."""
    cols = {}
    for model, runs in SEED_RUNS.items():
        for run, seed in zip(runs, [42, 1, 2]):
            cols[f"{model} s{seed}"] = pd.read_csv(ROOT / run / "per_image_val.csv").set_index("isic_id").dice
    return pd.DataFrame(cols).loc[IDS]


@torch.no_grad()
def predict(cfg_path):
    cfg = load_config(ROOT / cfg_path)
    model = build_model({**cfg, "model": {**cfg["model"], "encoder_weights": None}})
    ckpt = torch.load(ROOT / cfg["output"]["checkpoint"], map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    device = get_device(cfg["device"])
    model.to(device).eval()
    d = cfg["data"]
    ds = LesionDataset(IDS, ROOT / d["image_dir"], ROOT / d["mask_dir"], get_transforms(train=False))
    preds = [(torch.sigmoid(model(ds[i][0][None].to(device))) > 0.5).cpu().numpy()[0, 0] for i in range(len(ds))]
    return ds, preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet-config", default="configs/unet_r34.yaml")
    parser.add_argument("--dlv3p-config", default="configs/deeplabv3plus_r34_seed1.yaml")
    args = parser.parse_args()
    configs = {"U-Net": args.unet_config, "DeepLabV3+": args.dlv3p_config}
    shown = {m: load_config(ROOT / c)["seed"] for m, c in configs.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(ROOT / "results/comparison/per_image_both.csv").set_index("isic_id")
    dice = seed_dice()
    preds = {}
    for name, cfg_path in configs.items():
        ds, preds[name] = predict(cfg_path)

    fig, axes = plt.subplots(2, 2, figsize=(11, 13))
    for n, (ax, isic_id) in enumerate(zip(axes.flat, IDS)):
        x, y, _ = ds[n]
        ax.imshow(denorm(x))
        ax.contour(y[0].numpy(), levels=[0.5], colors="lime", linewidths=1.8)
        ax.contour(preds["U-Net"][n].astype(float), levels=[0.5], colors="tab:blue", linewidths=1.5)
        ax.contour(preds["DeepLabV3+"][n].astype(float), levels=[0.5], colors="tab:orange", linewidths=1.5)
        r, dr = meta.loc[isic_id], dice.loc[isic_id]
        u = [dr[f"U-Net s{s}"] for s in (42, 1, 2)]
        d = [dr[f"DeepLabV3+ s{s}"] for s in (42, 1, 2)]
        su, sd = shown["U-Net"], shown["DeepLabV3+"]
        ax.set_title(f"({chr(97 + n)}) {isic_id}, {r['dx']}, area {100 * r['area_frac']:.1f}%\n"
                     f"Dice shown: U-Net (s{su}) {dr[f'U-Net s{su}']:.2f} | DLV3+ (s{sd}) {dr[f'DeepLabV3+ s{sd}']:.2f}\n"
                     f"3 seeds: U-Net {min(u):.2f}-{max(u):.2f} | DLV3+ {min(d):.2f}-{max(d):.2f}", fontsize=10)
        ax.axis("off")
    fig.suptitle(f"Failure cases: ground truth = green, U-Net seed {shown['U-Net']} = blue, "
                 f"DeepLabV3+ seed {shown['DeepLabV3+']} = orange")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.01, top=0.88, hspace=0.25, wspace=0.05)
    fig.savefig(OUT / "failure_cases.png", dpi=130)
    plt.close(fig)
    print(dice.round(3).to_string())
    print(f"saved {OUT / 'failure_cases.png'}")


if __name__ == "__main__":
    main()
