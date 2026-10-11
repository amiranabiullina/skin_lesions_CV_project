"""Train the U-Net ResNet-34 baseline.

    python train.py --config configs/unet_r34.yaml
    python train.py --config configs/unet_r34.yaml --overfit-batch   # sanity check
"""
import argparse
import csv
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset import LesionDataset, get_transforms, seed_worker
from src.losses import BCEDiceLoss
from src.model import build_model
from src.utils import get_device, load_config, set_seed, sync


def make_loader(cfg, csv_path, train):
    d = cfg["data"]
    ids = pd.read_csv(csv_path).isic_id
    ds = LesionDataset(ids, d["image_dir"], d["mask_dir"], get_transforms(train))
    g = torch.Generator().manual_seed(cfg["seed"])
    return DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=train, drop_last=train,
                      num_workers=d["num_workers"], persistent_workers=d["num_workers"] > 0,
                      generator=g, worker_init_fn=seed_worker)


def batch_dice_iou(logits, target, threshold):
    """Per-image hard Dice and IoU for a batch (empty-vs-empty counts as 1)."""
    pred = (torch.sigmoid(logits) > threshold).float()
    dims = (1, 2, 3)
    inter = (pred * target).sum(dims)
    total = pred.sum(dims) + target.sum(dims)
    union = total - inter
    dice = torch.where(total > 0, 2 * inter / total.clamp(min=1), torch.ones_like(total))
    iou = torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(union))
    return dice, iou


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, threshold):
    model.eval()
    losses, dices, ious = [], [], []
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        losses.append(loss_fn(logits, y).item() * len(x))
        d, i = batch_dice_iou(logits, y, threshold)
        dices.append(d.cpu())
        ious.append(i.cpu())
    n = sum(len(d) for d in dices)
    return sum(losses) / n, torch.cat(dices).mean().item(), torch.cat(ious).mean().item()


def overfit_one_batch(cfg, model, loss_fn, device, steps=150):
    """Sanity check: the model must be able to memorise a single batch."""
    x, y, _ = next(iter(make_loader(cfg, cfg["data"]["train_csv"], train=False)))
    x, y = x.to(device), y.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for step in range(1, steps + 1):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        if step % 25 == 0:
            model.eval()
            with torch.no_grad():
                d, _ = batch_dice_iou(model(x), y, cfg["train"]["threshold"])
            model.train()
            print(f"step {step:3d}  loss {loss.item():.4f}  dice {d.mean().item():.4f}")
    return d.mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unet_r34.yaml")
    parser.add_argument("--overfit-batch", action="store_true")
    parser.add_argument("--epochs", type=int, help="override number of epochs (for quick tests)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    tcfg = cfg["train"]
    epochs = args.epochs or tcfg["epochs"]
    print(f"device: {device}")

    model = build_model(cfg).to(device)
    loss_fn = BCEDiceLoss(cfg["loss"]["bce_weight"], cfg["loss"]["dice_weight"])

    if args.overfit_batch:
        final = overfit_one_batch(cfg, model, loss_fn, device)
        print(f"overfit single batch: final dice {final:.4f}")
        return

    train_loader = make_loader(cfg, cfg["data"]["train_csv"], train=True)
    val_loader = make_loader(cfg, cfg["data"]["val_csv"], train=False)
    print(f"train images: {len(train_loader.dataset)}, val images: {len(val_loader.dataset)}")

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])

    ckpt_path = Path(cfg["output"]["checkpoint"])
    log_path = Path(cfg["output"]["log_csv"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["epoch", "train_loss", "val_loss", "val_dice", "val_iou", "epoch_time_s", "lr"]
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(fields)

    best_dice, best_epoch, bad_epochs = -1.0, 0, 0
    t_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        running, seen = 0.0, 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * len(x)
            seen += len(x)
        val_loss, val_dice, val_iou = evaluate(model, val_loader, loss_fn, device, tcfg["threshold"])
        sync(device)
        epoch_time = time.perf_counter() - t0

        row = [epoch, running / seen, val_loss, val_dice, val_iou, epoch_time, opt.param_groups[0]["lr"]]
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([f"{v:.6g}" if isinstance(v, float) else v for v in row])

        improved = val_dice > best_dice
        if improved:
            best_dice, best_epoch, bad_epochs = val_dice, epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_dice": val_dice, "config": cfg},
                       ckpt_path)
        else:
            bad_epochs += 1
        print(f"epoch {epoch:2d} | train {running / seen:.4f} | val loss {val_loss:.4f} "
              f"dice {val_dice:.4f} iou {val_iou:.4f} | {epoch_time:.0f}s{' *' if improved else ''}",
              flush=True)
        if bad_epochs >= tcfg["early_stopping_patience"]:
            print(f"early stopping: no val Dice improvement for {bad_epochs} epochs")
            break

    total = time.perf_counter() - t_start
    print(f"best val dice {best_dice:.4f} at epoch {best_epoch}; total training time {total / 60:.1f} min")
    with open(log_path.with_name("train_summary.txt"), "w") as f:
        f.write(f"best_epoch={best_epoch}\nbest_val_dice={best_dice:.6f}\n"
                f"epochs_run={epoch}\ntotal_train_time_s={total:.1f}\n")


if __name__ == "__main__":
    main()
