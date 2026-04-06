"""
evaluate.py
-----------
Evaluates trained models on the test set.
Supports single-model, ensemble, TTA, and ensemble+TTA modes.

Usage:
    python src/evaluation/evaluate.py --model mobilenet_v2
    python src/evaluation/evaluate.py --model mobilenet_v2 --tta
    python src/evaluation/evaluate.py --ensemble
    python src/evaluation/evaluate.py --ensemble --tta
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config.settings import CFG, PROJECT_ROOT
from src.datasets.congestion_dataset import (
    CongestionDataset,
    get_eval_transforms,
    get_tta_transforms,
)
from src.models.baseline_cnn import BaselineCNN
from src.models.transfer_models import build_model

ALL_MODELS = ["baseline_cnn", "mobilenet_v2", "resnet50", "efficientnet_b0"]


def get_device():
    # type: () -> torch.device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_name, num_classes, device):
    # type: (str, int, torch.device) -> nn.Module
    ckpt_path = PROJECT_ROOT / CFG["paths"]["checkpoints"] / "{}_best.pt".format(model_name)
    if not ckpt_path.exists():
        raise FileNotFoundError("No checkpoint: {}".format(ckpt_path))
    if model_name == "baseline_cnn":
        model = BaselineCNN(num_classes=num_classes)
    else:
        model = build_model(model_name, num_classes=num_classes, pretrained=False)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print("[Loaded] {}  (epoch {}, val_acc={:.4f})".format(
        model_name, ckpt["epoch"], ckpt["val_acc"]))
    return model


def predict_probs(model, loader, device):
    # type: (nn.Module, DataLoader, torch.device) -> np.ndarray
    """Returns softmax probabilities, shape (N, num_classes)."""
    all_probs = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            probs = torch.softmax(model(images), dim=1).cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def predict_probs_tta(model, csv_path, image_size, device):
    # type: (nn.Module, Path, tuple, torch.device) -> np.ndarray
    """Runs TTA: average probabilities over 5 augmented versions."""
    tta_tfms = get_tta_transforms(image_size)
    batch_size = CFG["training"]["batch_size"]
    all_probs = []
    for tfm in tta_tfms:
        ds = CongestionDataset(csv_path, PROJECT_ROOT, tfm)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        all_probs.append(predict_probs(model, loader, device))
    return np.mean(all_probs, axis=0)  # average across 5 transforms


def get_labels(csv_path, image_size):
    # type: (Path, tuple) -> np.ndarray
    ds = CongestionDataset(csv_path, PROJECT_ROOT, get_eval_transforms(image_size))
    return np.array([ds[i][1] for i in range(len(ds))])


def save_results(label, all_preds, all_labels, class_names):
    # type: (str, np.ndarray, np.ndarray, list) -> None
    reports_dir = PROJECT_ROOT / CFG["paths"]["reports"]
    figs_dir    = PROJECT_ROOT / CFG["paths"]["figures"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print("\nClassification Report:")
    print(report)

    report_path = reports_dir / "{}_classification_report.txt".format(label)
    report_path.write_text("Mode: {}\n\n{}".format(label, report))
    print("[OK] Report saved: {}".format(report_path))

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(
        ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Confusion Matrix — {}".format(label))
    plt.tight_layout()
    cm_path = figs_dir / "{}_confusion_matrix.png".format(label)
    fig.savefig(cm_path, dpi=150)
    plt.close()
    print("[OK] Confusion matrix saved: {}".format(cm_path))

    accuracy = (all_preds == all_labels).mean()
    print("\nTest Accuracy: {:.4f}".format(accuracy))


def main(model_name, use_ensemble, use_tta, splits_dir=None):
    # type: (str, bool, bool, Path) -> None
    print("=" * 60)
    mode_str = []
    if use_ensemble: mode_str.append("ensemble")
    elif model_name:  mode_str.append(model_name)
    if use_tta:      mode_str.append("tta")
    print("evaluate.py — mode: {}".format(" + ".join(mode_str) if mode_str else model_name))
    print("=" * 60)

    cfg_model   = CFG["models"]
    if splits_dir is None:
        splits_dir = PROJECT_ROOT / CFG["split"]["output_dir"]
    splits_dir  = Path(splits_dir)
    image_size  = tuple(CFG["frame_extraction"]["image_size"])
    class_names = cfg_model["class_names"]
    num_classes = cfg_model["num_classes"]
    device      = get_device()
    test_csv    = splits_dir / "test.csv"

    all_labels = get_labels(test_csv, image_size)

    if use_ensemble:
        # Load all available models
        models_to_use = []
        for name in ALL_MODELS:
            try:
                models_to_use.append((name, load_model(name, num_classes, device)))
            except FileNotFoundError:
                print("[SKIP] {} — no checkpoint".format(name))

        if not models_to_use:
            raise RuntimeError("No checkpoints found.")

        # Gather probabilities from each model
        model_probs = []
        for name, model in models_to_use:
            if use_tta:
                print("[TTA] Running {} × 5 transforms...".format(name))
                probs = predict_probs_tta(model, test_csv, image_size, device)
            else:
                ds = CongestionDataset(test_csv, PROJECT_ROOT, get_eval_transforms(image_size))
                loader = DataLoader(ds, batch_size=CFG["training"]["batch_size"],
                                    shuffle=False, num_workers=0)
                probs = predict_probs(model, loader, device)
            model_probs.append(probs)

        avg_probs = np.mean(model_probs, axis=0)
        all_preds = avg_probs.argmax(axis=1)
        label = "ensemble_tta" if use_tta else "ensemble"

    else:
        # Single model
        model = load_model(model_name, num_classes, device)
        if use_tta:
            print("[TTA] Running {} × 5 transforms...".format(model_name))
            avg_probs = predict_probs_tta(model, test_csv, image_size, device)
        else:
            ds = CongestionDataset(test_csv, PROJECT_ROOT, get_eval_transforms(image_size))
            loader = DataLoader(ds, batch_size=CFG["training"]["batch_size"],
                                shuffle=False, num_workers=0)
            avg_probs = predict_probs(model, loader, device)

        all_preds = avg_probs.argmax(axis=1)
        label = "{}_tta".format(model_name) if use_tta else model_name

    save_results(label, all_preds, all_labels, class_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        choices=ALL_MODELS,
                        help="Single model to evaluate (ignored if --ensemble)")
    parser.add_argument("--ensemble", action="store_true",
                        help="Average predictions from all 4 models")
    parser.add_argument("--tta", action="store_true",
                        help="Apply test-time augmentation (5 variants, averaged)")
    parser.add_argument("--split-dir", type=str, default=None,
                        help="Path to splits directory (default: from config)")
    args = parser.parse_args()

    if not args.ensemble and args.model is None:
        parser.error("Provide --model <name> or --ensemble")

    main(args.model, args.ensemble, args.tta, splits_dir=args.split_dir)
