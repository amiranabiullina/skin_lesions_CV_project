"""Create failure_cases.png per TZ (Andre, step 58, point 4-6).

Shows 4 failure cases with GT (green), U-Net (blue), DLV3+ (orange) contours,
Dice of both models in title. Uses per_image_both.csv candidates.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch

from evaluate import denorm
from src.dataset import LesionDataset, get_transforms
from src.model import build_model
from src.utils import get_device, load_config

OUT = Path("results/failures")
OUT.mkdir(parents=True, exist_ok=True)

# Candidates from per_image_both.csv (TZ table). Picked 4 with different failure causes:
#   ISIC_0013567 — both fail on melanoma (tiny, 4%)
#   ISIC_0011117 — both fail on nevus (46%)
#   ISIC_0012127 — DLV3+ loses tiny nevus (1.8%) while U-Net keeps it
#   ISIC_0015102 — U-Net breaks on large nevus (65%), DLV3+ nails it
IDS = ["ISIC_0013567", "ISIC_0011117", "ISIC_0012127", "ISIC_0015102"]

merged = pd.read_csv("results/comparison/per_image_both.csv")
merged = merged.set_index("isic_id")
for tid in IDS:
    assert tid in merged.index, f"{tid} not in per_image_both.csv"

# Load both models once
models = {}
for name, cfg_path in {"U-Net": "configs/unet_r34.yaml",
                        "DeepLabV3+": "configs/deeplabv3plus_r34.yaml"}.items():
    cfg = load_config(cfg_path)
    ckpt = Path(cfg["output"]["checkpoint"])
    model = build_model({**cfg, "model": {**cfg["model"], "encoder_weights": None}})
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
    device = get_device(cfg["device"])
    model.to(device).eval()
    d = cfg["data"]
    ds = LesionDataset(pd.Series(IDS), d["image_dir"], d["mask_dir"], get_transforms(train=False))
    preds = []
    for i in range(len(ds)):
        with torch.no_grad():
            p = torch.sigmoid(model(ds[i][0][None].to(device))) > 0.5
        preds.append(p.cpu().numpy()[0, 0])
    models[name] = (ds, preds, device)

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, tid in zip(axes.flat, IDS):
    row = merged.loc[tid]
    x, y, _ = models["U-Net"][0][IDS.index(tid)]
    ax.imshow(denorm(x))
    ax.contour(y[0].numpy(), levels=[0.5], colors="lime", linewidths=1.5)
    ax.contour(models["U-Net"][1][IDS.index(tid)], levels=[0.5], colors="tab:blue", linewidths=1.5)
    ax.contour(models["DeepLabV3+"][1][IDS.index(tid)], levels=[0.5], colors="tab:orange", linewidths=1.5)
    ax.set_title(f"{tid}\nDice U-Net {row.dice_unet:.3f} | DLV3+ {row.dice_dlv3p:.3f} | size {row.size}", fontsize=9)
    ax.axis("off")

fig.suptitle("Failure cases: GT=green, U-Net=blue, DeepLabV3+=orange", fontsize=11)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig(OUT / "failure_cases.png", dpi=130)
plt.close(fig)
print(f"Saved {OUT / 'failure_cases.png'}")