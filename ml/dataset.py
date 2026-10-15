from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import Config

class GestureDataset(Dataset):
    def __init__(self, config: Config, transform: transforms.Compose) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

        self.config = config
        self.transform = transform

        self.classes: tuple[str, ...] = config.GESTURE_CLASSES
        self.class_to_idx: dict[str, int] = {c: i for i, c in enumerate(self.classes)}

        self.samples: list[tuple[Path, int]] = []
        data_dir = Path(config.GESTURE_DATA_DIR)

        for _class in self.classes:
            class_dir = data_dir / _class
            for path in sorted(class_dir.glob("*.jpg")): self.samples.append((path, self.class_to_idx[_class]))

        if not self.samples:
            raise RuntimeError(f"The dataset is empty: {data_dir}. Firstly - collect the data")

        self.logger.info("%d images were found in %d categories", len(self.samples), len(self.classes))

    def __len__(self) -> int: return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label

def build_train_transform(config: Config) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((config.GESTURE_IMG_SIZE, config.GESTURE_IMG_SIZE)),
            transforms.RandomAffine(degrees=20, translate=(0.15, 0.15), scale=(0.85, 1.15), shear=10,),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.3, hue=0.05,),

            transforms.RandomGrayscale(p=0.2),
            transforms.RandomAutocontrast(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(config.GESTURE_MEAN, config.GESTURE_STD),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3), value='random',),
        ]
    )

def build_eval_transform(config: Config) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((config.GESTURE_IMG_SIZE, config.GESTURE_IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(config.GESTURE_MEAN, config.GESTURE_STD),
        ]
    )

def save_meta(config: Config, path: str) -> None:
    meta = {
        "classes": list(config.GESTURE_CLASSES),
        "img_size": config.GESTURE_IMG_SIZE,
        "mean": list(config.GESTURE_MEAN),
        "std": list(config.GESTURE_STD),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f: json.dump(meta, f, ensure_ascii=False, indent=2)

def load_meta(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f: return json.load(f)