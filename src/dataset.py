"""Dataset and augmentations for ISIC 2018 Task 1 (images pre-resized to 256x256)."""
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transforms(train):
    """Augmentations are applied only to the training split; val/test are only normalized."""
    norm = A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    if not train:
        return A.Compose([norm])
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.0625, 0.0625), rotate=(-30, 30),
                 border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.5),
        norm,
    ])


class LesionDataset(Dataset):
    """Reads cached 256x256 images (.jpg -> .png) and binary masks for a list of ISIC ids."""

    def __init__(self, ids, image_dir, mask_dir, transform=None):
        self.ids = list(ids)
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        isic_id = self.ids[idx]
        image = cv2.imread(str(self.image_dir / f"{isic_id}.png"), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.mask_dir / f"{isic_id}.png"), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]

        image = torch.from_numpy(image.transpose(2, 0, 1).copy()).float()
        mask = torch.from_numpy(mask[None].copy()).float()
        return image, mask, isic_id
