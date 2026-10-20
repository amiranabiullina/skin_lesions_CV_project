"""Time U-Net and DeepLabV3+ on the same machine, so the speed comparison is fair.

Speed does not depend on the trained weights, so randomly initialised models are used and no
checkpoints are needed. Run it on an otherwise idle machine (not while training):

    python scripts/benchmark_speed.py

Measures, for each model with the settings of its config:
  * training: seconds per epoch (2,205 train images, batch 16, same loader and augmentations),
    averaged over --epochs epochs after one warm-up epoch is excluded;
  * inference: ms per image at batch size 1 on 100 validation images (as in evaluate.py).
Writes results/comparison/speed_benchmark.json.
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import time_inference  # noqa: E402
from src.dataset import LesionDataset, get_transforms  # noqa: E402
from src.losses import BCEDiceLoss  # noqa: E402
from src.model import build_model  # noqa: E402
from src.utils import get_device, load_config, set_seed, sync  # noqa: E402
from train import make_loader  # noqa: E402

import pandas as pd  # noqa: E402

CONFIGS = {"U-Net": "configs/unet_r34.yaml", "DeepLabV3+": "configs/deeplabv3plus_r34.yaml"}


def chip_name():
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return platform.processor()


def time_epoch(model, loader, loss_fn, opt, device):
    model.train()
    t0 = time.perf_counter()
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()
    sync(device)
    return time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2, help="timed epochs per model (after 1 warm-up)")
    args = parser.parse_args()

    out = {"chip": chip_name(), "torch": torch.__version__, "timed_epochs": args.epochs, "models": {}}
    for name, path in CONFIGS.items():
        cfg = load_config(path)
        set_seed(cfg["seed"])
        device = get_device(cfg["device"])
        model = build_model({**cfg, "model": {**cfg["model"], "encoder_weights": None}}).to(device)
        loss_fn = BCEDiceLoss(cfg["loss"]["bce_weight"], cfg["loss"]["dice_weight"])
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                                weight_decay=cfg["train"]["weight_decay"])
        loader = make_loader(cfg, cfg["data"]["train_csv"], train=True)

        print(f"{name}: warm-up epoch ...", flush=True)
        time_epoch(model, loader, loss_fn, opt, device)
        times = []
        for e in range(args.epochs):
            times.append(time_epoch(model, loader, loss_fn, opt, device))
            print(f"{name}: epoch {e + 1}/{args.epochs} {times[-1]:.1f} s", flush=True)

        model.eval()
        d = cfg["data"]
        ds = LesionDataset(pd.read_csv(d["val_csv"]).isic_id, d["image_dir"], d["mask_dir"],
                           get_transforms(train=False))
        ms_mean, ms_std = time_inference(model, ds, device)
        out["models"][name] = {"device": str(device), "epoch_s": times,
                               "epoch_s_mean": sum(times) / len(times),
                               "inference_ms_mean": ms_mean, "inference_ms_std": ms_std}
        print(f"{name}: {out['models'][name]['epoch_s_mean']:.1f} s/epoch, "
              f"{ms_mean:.1f} +- {ms_std:.1f} ms/image", flush=True)
        del model, opt, loader

    path = Path("results/comparison/speed_benchmark.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(path, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
