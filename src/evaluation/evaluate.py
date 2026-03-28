"""
evaluate.py
-----------
Evaluates a trained model on the test set.
Saves confusion matrix plot and classification report.

Usage:
    python src/evaluation/evaluate.py --model baseline_cnn
    python src/evaluation/evaluate.py --model mobilenet_v2
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config.settings import CFG, PROJECT_ROOT
from src.datasets.congestion_dataset import CongestionDataset, get_eval_transforms
from src.models.baseline_cnn import BaselineCNN
from src.models.transfer_models import build_model


def load_checkpoint(model_name: str) -> dict:
    ckpt_dir = PROJECT_ROOT / CFG["paths"]["checkpoints"]
    ckpt_path = ckpt_dir / f"{model_name}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
    return torch.load(ckpt_path, map_location="cpu")


def get_model(name: str, num_classes: int) -> nn.Module:
    if name == "baseline_cnn":
        return BaselineCNN(num_classes=num_classes)
    return build_model(name, num_classes=num_classes, pretrained=False)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(model_name: str):
    print("=" * 60)
    print("evaluate.py — model: {}".format(model_name))
    print("=" * 60)

    cfg_model = CFG["models"]
    splits_dir = PROJECT_ROOT / CFG["split"]["output_dir"]
    image_size = tuple(CFG["frame_extraction"]["image_size"])
    class_names = cfg_model["class_names"]
    num_classes = cfg_model["num_classes"]

    device = get_device()

    # Load model
    ckpt = load_checkpoint(model_name)
    model = get_model(model_name, num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print("[Checkpoint] Loaded epoch {}  val_acc={:.4f}".format(ckpt["epoch"], ckpt["val_acc"]))

    # Test dataset
    test_ds = CongestionDataset(
        splits_dir / "test.csv", PROJECT_ROOT, get_eval_transforms(image_size)
    )
    test_loader = DataLoader(
        test_ds, batch_size=CFG["training"]["batch_size"],
        shuffle=False, num_workers=0,  # 0 = safe on macOS Python 3.9
    )

    # Inference
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Metrics
    report = classification_report(
        all_labels, all_preds, target_names=class_names, digits=4
    )
    print("\nClassification Report:")
    print(report)

    # Save report
    reports_dir = PROJECT_ROOT / CFG["paths"]["reports"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "{}_classification_report.txt".format(model_name)
    report_path.write_text("Model: {}\n\n{}".format(model_name, report))
    print("[OK] Classification report saved: {}".format(report_path))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Confusion Matrix — {}".format(model_name))
    plt.tight_layout()

    figs_dir = PROJECT_ROOT / CFG["paths"]["figures"]
    figs_dir.mkdir(parents=True, exist_ok=True)
    cm_path = figs_dir / "{}_confusion_matrix.png".format(model_name)
    fig.savefig(cm_path, dpi=150)
    plt.close()
    print("[OK] Confusion matrix saved: {}".format(cm_path))

    # Overall accuracy
    accuracy = (all_preds == all_labels).mean()
    print("\nTest Accuracy: {:.4f}".format(accuracy))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="baseline_cnn",
        choices=["baseline_cnn", "mobilenet_v2", "resnet50", "efficientnet_b0"],
    )
    args = parser.parse_args()
    main(args.model)
