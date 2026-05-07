import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def list_pairs(images_dir: str, masks_dir: str) -> List[Tuple[Path, Path]]:
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    image_files = sorted([p for p in images_dir.iterdir() if p.is_file()])
    mask_files = [p for p in masks_dir.iterdir() if p.is_file()]
    mask_by_stem = {p.stem: p for p in mask_files}
    pairs = []
    for image_path in image_files:
        mask_path = mask_by_stem.get(image_path.stem)
        if mask_path is not None:
            pairs.append((image_path, mask_path))
    if not pairs:
        raise ValueError(f"No image-mask pairs found in {images_dir} and {masks_dir}")
    return pairs


def split_train_val(pairs, val_ratio: float = 0.15, seed: int = 42):
    train_pairs, val_pairs = train_test_split(
        pairs, test_size=val_ratio, random_state=seed, shuffle=True
    )
    return train_pairs, val_pairs


def make_splits(images_dir: str, masks_dir: str, seed: int = 42):
    # Backward-compatible helper for 70/15/15 from a single source.
    pairs = list_pairs(images_dir, masks_dir)
    train_pairs, temp_pairs = train_test_split(pairs, test_size=0.30, random_state=seed, shuffle=True)
    val_pairs, test_pairs = train_test_split(temp_pairs, test_size=0.50, random_state=seed, shuffle=True)
    return train_pairs, val_pairs, test_pairs


class SODDataset(Dataset):
    def __init__(self, pairs, image_size: int = 128, augment: bool = False):
        self.pairs = pairs
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def _read(self, image_path: Path, mask_path: Path):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise ValueError(f"Failed to read: {image_path} or {mask_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        image = image.astype(np.float32) / 255.0
        mask = (mask.astype(np.float32) / 255.0)
        mask = np.expand_dims(mask, axis=2)
        return image, mask

    def _augment(self, image: np.ndarray, mask: np.ndarray):
        if random.random() < 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()

        if random.random() < 0.5:
            factor = random.uniform(0.75, 1.25)
            image = np.clip(image * factor, 0.0, 1.0)

        if random.random() < 0.5:
            crop = random.randint(0, self.image_size // 8)
            if crop > 0:
                h, w = image.shape[:2]
                image = image[crop : h - crop, crop : w - crop]
                mask = mask[crop : h - crop, crop : w - crop]
                image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
                mask = np.expand_dims(mask, axis=2) if mask.ndim == 2 else mask
        return image, mask

    def __getitem__(self, idx):
        image_path, mask_path = self.pairs[idx]
        image, mask = self._read(image_path, mask_path)
        if self.augment:
            image, mask = self._augment(image, mask)

        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).permute(2, 0, 1).float()
        return image, mask, image_path.name
