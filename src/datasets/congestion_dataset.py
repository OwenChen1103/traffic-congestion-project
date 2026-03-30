"""
congestion_dataset.py
---------------------
PyTorch Dataset class for the traffic congestion classification task.
Reads a split CSV and serves (image_tensor, label_id) pairs.

Reads: data/labels/samples_split_v1.csv (or per-split train/val/test.csv)
Required columns: image_path, label
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Label string → integer mapping (consistent across all modules)
LABEL_MAP = {"low": 0, "medium": 1, "high": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}


def get_train_transforms(image_size: Tuple[int, int]) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_eval_transforms(image_size: Tuple[int, int]) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class _Rotate:
    """Deterministic rotation by a fixed angle (degrees)."""
    def __init__(self, angle):
        self.angle = angle
    def __call__(self, img):
        return transforms.functional.rotate(img, self.angle)


def get_tta_transforms(image_size: Tuple[int, int]) -> List[transforms.Compose]:
    """5 TTA variants: original, h-flip, v-flip, +5° rotation, -5° rotation.
    Drone footage is top-down so all 4 orientations are plausible.
    """
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    resize = transforms.Resize(image_size)
    to_t  = transforms.ToTensor()
    return [
        transforms.Compose([resize, to_t, norm]),
        transforms.Compose([resize, transforms.RandomHorizontalFlip(p=1.0), to_t, norm]),
        transforms.Compose([resize, transforms.RandomVerticalFlip(p=1.0),   to_t, norm]),
        transforms.Compose([resize, _Rotate(+5),  to_t, norm]),
        transforms.Compose([resize, _Rotate(-5),  to_t, norm]),
    ]


class CongestionDataset(Dataset):
    """
    Args:
        csv_path:     Path to a split CSV (train.csv / val.csv / test.csv).
        project_root: Root directory; image_path in CSV is relative to this.
        transform:    torchvision transform to apply to each image.
    """

    def __init__(
        self,
        csv_path: Union[str, Path],
        project_root: Union[str, Path],
        transform: Optional[transforms.Compose] = None,
    ):
        self.project_root = Path(project_root)
        self.df = pd.read_csv(csv_path)
        self.transform = transform

        # Validate required columns
        for col in ["image_path", "label"]:
            assert col in self.df.columns, "Missing column '{}' in {}".format(col, csv_path)

        # Drop rows with invalid label
        self.df = self.df[self.df["label"].isin(LABEL_MAP)].reset_index(drop=True)

    def __len__(self):
        # type: () -> int
        return len(self.df)

    def __getitem__(self, idx):
        # type: (int) -> Tuple
        row = self.df.iloc[idx]
        img_path = self.project_root / row["image_path"]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label_id = LABEL_MAP[row["label"]]
        return image, label_id

    def get_class_weights(self):
        # type: () -> List[float]
        """
        Returns per-class weights inversely proportional to class frequency.
        Pass result as weight= to torch.nn.CrossEntropyLoss when classes are imbalanced.
        """
        counts = self.df["label"].value_counts()
        total = len(self.df)
        weights = []
        for lbl in ["low", "medium", "high"]:
            cnt = counts.get(lbl, 1)
            weights.append(total / (3.0 * cnt))
        return weights
