"""
train.py
--------
Training script. Supports all three model architectures via --model flag.

Usage:
    python src/training/train.py --model baseline_cnn
    python src/training/train.py --model mobilenet_v2
    python src/training/train.py --model resnet50
    python src/training/train.py --model efficientnet_b0

Outputs (per model):
    outputs/checkpoints/<model_name>_best.pt
    outputs/logs/<model_name>_train_log.csv
"""

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config.settings import CFG, PROJECT_ROOT
from src.datasets.congestion_dataset import (
    CongestionDataset,
    get_eval_transforms,
    get_train_transforms,
)
from src.models.baseline_cnn import BaselineCNN
from src.models.transfer_models import build_model


def get_model(name: str, num_classes: int) -> nn.Module:
    if name == "baseline_cnn":
        return BaselineCNN(num_classes=num_classes)
    return build_model(name, num_classes=num_classes, pretrained=True)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(
    model,           # type: nn.Module
    loader,          # type: DataLoader
    criterion,       # type: nn.Module
    optimizer,       # type: Optional[torch.optim.Optimizer]
    device,          # type: torch.device
    is_train,        # type: bool
):
    # type: (...) -> Tuple[float, float]
    """Returns (avg_loss, accuracy)."""
    model.train(is_train)
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def main(model_name: str):
    print("=" * 60)
    print("train.py — training model: {}".format(model_name))
    print("=" * 60)

    cfg_train = CFG["training"]
    cfg_model = CFG["models"]
    splits_dir = PROJECT_ROOT / CFG["split"]["output_dir"]
    image_size = tuple(CFG["frame_extraction"]["image_size"])
    num_classes = cfg_model["num_classes"]
    seed = CFG["project"]["seed"]

    torch.manual_seed(seed)
    device = get_device()
    # pin_memory only helps CUDA; causes issues on MPS/CPU
    pin_memory = device.type == "cuda"
    print("[Device] {}".format(device))

    # Dataset size warning
    train_csv = splits_dir / "train.csv"
    n_train = len(pd.read_csv(train_csv)) if train_csv.exists() else 0
    if n_train < 200:
        print("[WARN] Small training set ({} samples). "
              "Overfitting is likely — watch val_acc closely.".format(n_train))

    # Datasets
    train_ds = CongestionDataset(
        splits_dir / "train.csv", PROJECT_ROOT, get_train_transforms(image_size)
    )
    val_ds = CongestionDataset(
        splits_dir / "val.csv", PROJECT_ROOT, get_eval_transforms(image_size)
    )

    # Class weights for imbalance handling
    class_weights = torch.tensor(train_ds.get_class_weights(), dtype=torch.float).to(device)
    print("[Class weights] {}".format(class_weights.tolist()))

    train_loader = DataLoader(
        train_ds, batch_size=cfg_train["batch_size"],
        shuffle=True, num_workers=cfg_train["num_workers"], pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg_train["batch_size"],
        shuffle=False, num_workers=cfg_train["num_workers"], pin_memory=pin_memory
    )

    # Model
    model = get_model(model_name, num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("[Model] {}  |  Trainable params: {:,}".format(model_name, total_params))

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg_train["learning_rate"],
        weight_decay=cfg_train["weight_decay"],
    )

    if cfg_train["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg_train["epochs"]
        )
    elif cfg_train["scheduler"] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        scheduler = None

    # Output dirs
    ckpt_dir = PROJECT_ROOT / CFG["paths"]["checkpoints"]
    log_dir = PROJECT_ROOT / CFG["paths"]["logs"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    best_ckpt_path = ckpt_dir / "{}_best.pt".format(model_name)

    log_path = log_dir / "{}_train_log.csv".format(model_name)
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    print("\n{:>6} {:>10} {:>9} {:>9} {:>8} {:>10}".format(
        "Epoch", "TrainLoss", "TrainAcc", "ValLoss", "ValAcc", "LR"))
    print("-" * 60)

    for epoch in range(1, cfg_train["epochs"] + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device, is_train=False)

        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler:
            scheduler.step()

        elapsed = time.time() - t0
        print("{:>6}  {:>10.4f}  {:>8.4f}  {:>9.4f}  {:>8.4f}  {:>10.6f}  ({:.1f}s)".format(
            epoch, train_loss, train_acc, val_loss, val_acc, current_lr, elapsed))

        log_writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, current_lr])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_name": model_name,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "class_names": cfg_model["class_names"],
            }, best_ckpt_path)

    log_file.close()
    print("\n[OK] Best val accuracy: {:.4f}".format(best_val_acc))
    print("     Checkpoint saved: {}".format(best_ckpt_path))
    print("     Training log:     {}".format(log_path))
    print("\nNEXT STEPS:")
    print("  Run: python src/evaluation/evaluate.py --model {}".format(model_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="baseline_cnn",
        choices=["baseline_cnn", "mobilenet_v2", "resnet50", "efficientnet_b0"],
        help="Model architecture to train",
    )
    args = parser.parse_args()
    main(args.model)
