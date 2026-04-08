"""
preview_predictions.py
----------------------
Shows model predictions on test set images.

Layout:
    Row 0 = images predicted LOW
    Row 1 = images predicted MEDIUM
    Row 2 = images predicted HIGH
    3 images per row (randomly sampled from that predicted class)

Each image shows:
    - YOLO bounding boxes (green = counted, red = ROI-excluded)
    - Title: PRED / GT / confidence / vehicle count

Controls:
    n / Enter    next page (new random sample)
    p            previous page
    q            quit

Usage:
    python src/live_pipeline/preview_predictions.py
    python src/live_pipeline/preview_predictions.py --model mobilenet_v2
    python src/live_pipeline/preview_predictions.py --split-dir data/live/splits
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import CFG
from src.datasets.congestion_dataset import get_eval_transforms
from src.models.transfer_models import build_model
from src.models.baseline_cnn import BaselineCNN

# ── Config ────────────────────────────────────────────────────────────────────

SAMPLES_PER_CLASS = 3
CLASS_NAMES = CFG["models"]["class_names"]   # ["high", "low", "medium"] sorted
LABELS      = ["low", "medium", "high"]      # display order

LABEL_COLORS = {"low": "#4CAF50", "medium": "#FF9800", "high": "#F44336"}
GT_MATCH_COLOR    = "#FFFFFF"   # white  — prediction correct
GT_MISMATCH_COLOR = "#FF4444"   # red    — prediction wrong

VEHICLE_CLASS_IDS = {2, 3, 5, 7}

CAMERA_ROI = {}
CAMERA_EXCLUDE = {
    "5_ways_miranda": [(0.0, 0.0, 0.28, 0.55)],
}


# ── Model helpers ─────────────────────────────────────────────────────────────

def load_model(model_name, device):
    ckpt_path = PROJECT_ROOT / CFG["paths"]["checkpoints"] / "{}_best.pt".format(model_name)
    if not ckpt_path.exists():
        raise FileNotFoundError("No checkpoint: {}".format(ckpt_path))
    if model_name == "baseline_cnn":
        model = BaselineCNN(num_classes=len(CLASS_NAMES))
    else:
        model = build_model(model_name, num_classes=len(CLASS_NAMES), pretrained=False)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print("[model] Loaded {}  (epoch {}, val_acc={:.4f})".format(
        model_name, ckpt["epoch"], ckpt["val_acc"]))
    return model


def predict_all(model, df, device, image_size):
    """Run model on all rows; return (pred_labels, confidences) lists."""
    tfm = get_eval_transforms(image_size)
    from PIL import Image as PILImage

    pred_labels  = []
    confidences  = []

    model.eval()
    with torch.no_grad():
        for _, row in df.iterrows():
            img_path = PROJECT_ROOT / row["image_path"]
            try:
                img = PILImage.open(img_path).convert("RGB")
                tensor = tfm(img).unsqueeze(0).to(device)
                logits = model(tensor)
                probs  = F.softmax(logits, dim=1).cpu().numpy()[0]
                pred_idx = int(probs.argmax())
                pred_label = CLASS_NAMES[pred_idx]
                confidence = float(probs[pred_idx])
            except Exception:
                pred_label = "low"
                confidence = 0.0
            pred_labels.append(pred_label)
            confidences.append(confidence)

    return pred_labels, confidences


# ── YOLO helpers ──────────────────────────────────────────────────────────────

def load_yolo():
    from ultralytics import YOLO
    print("[yolo] Loading YOLOv8n...")
    model = YOLO("yolov8n.pt")
    model.conf = 0.3
    print("[yolo] Ready.")
    return model


def get_boxes(yolo, img_path, camera_id=""):
    from PIL import Image as PILImage
    img = PILImage.open(img_path).convert("RGB")
    W, H = img.size

    roi = CAMERA_ROI.get(camera_id)
    rx1, ry1, rx2, ry2 = (roi[0]*W, roi[1]*H, roi[2]*W, roi[3]*H) if roi else (0, 0, W, H)

    exclude_zones = [
        (ex1*W, ey1*H, ex2*W, ey2*H)
        for ex1, ey1, ex2, ey2 in CAMERA_EXCLUDE.get(camera_id, [])
    ]

    results = yolo(img, verbose=False)[0]
    included, excluded = [], []
    for box in results.boxes:
        if int(box.cls[0].item()) not in VEHICLE_CLASS_IDS:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        in_roi     = rx1 <= cx <= rx2 and ry1 <= cy <= ry2
        in_exclude = any(ex1 <= cx <= ex2 and ey1 <= cy <= ey2
                         for ex1, ey1, ex2, ey2 in exclude_zones)
        if in_roi and not in_exclude:
            included.append((x1, y1, x2, y2))
        else:
            excluded.append((x1, y1, x2, y2))
    return included, excluded, (W, H)


# ── Viewer ────────────────────────────────────────────────────────────────────

class PredictionViewer:
    def __init__(self, df, yolo, page_size=SAMPLES_PER_CLASS):
        self.df        = df          # has: image_path, label, camera_id, pred_label, confidence
        self.yolo      = yolo
        self.page_size = page_size
        self.page      = 0
        self.box_cache = {}

        self.fig = plt.figure(figsize=(18, 10))
        self.fig.patch.set_facecolor("#1e1e1e")
        self.fig.canvas.manager.set_window_title("Model Prediction Preview")
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.render()
        plt.show()

    def render(self):
        self.fig.clear()
        rng = np.random.RandomState(self.page * 999)

        gs = gridspec.GridSpec(3, self.page_size, figure=self.fig,
                               hspace=0.45, wspace=0.05)

        for row, pred_label in enumerate(LABELS):
            subset = self.df[self.df["pred_label"] == pred_label]
            samples = subset.sample(
                min(self.page_size, len(subset)), random_state=rng
            ).reset_index(drop=True) if len(subset) > 0 else pd.DataFrame()

            color = LABEL_COLORS[pred_label]

            for col in range(self.page_size):
                ax = self.fig.add_subplot(gs[row, col])
                ax.axis("off")

                if col >= len(samples):
                    ax.set_facecolor("#111")
                    ax.set_title("(no {})".format(pred_label), color="#555", fontsize=8)
                    continue

                r = samples.iloc[col]
                img_path = PROJECT_ROOT / r["image_path"]

                try:
                    img = mpimg.imread(str(img_path))
                    ax.imshow(img)
                except Exception:
                    ax.set_facecolor("#333")
                    img = None

                # YOLO boxes
                cam_id = r.get("camera_id", "")
                ip = str(r["image_path"])
                if ip not in self.box_cache:
                    self.box_cache[ip] = get_boxes(self.yolo, img_path, cam_id)
                included, excluded, (orig_w, orig_h) = self.box_cache[ip]

                try:
                    disp_h, disp_w = img.shape[:2]
                except Exception:
                    disp_h, disp_w = orig_h, orig_w
                sx, sy = disp_w / orig_w, disp_h / orig_h

                for x1, y1, x2, y2 in included:
                    ax.add_patch(patches.Rectangle(
                        (x1*sx, y1*sy), (x2-x1)*sx, (y2-y1)*sy,
                        linewidth=1.5, edgecolor="#00FF00", facecolor="none"
                    ))
                for x1, y1, x2, y2 in excluded:
                    ax.add_patch(patches.Rectangle(
                        (x1*sx, y1*sy), (x2-x1)*sx, (y2-y1)*sy,
                        linewidth=1.0, edgecolor="#FF4444", facecolor="none",
                        linestyle="dashed"
                    ))

                # Title
                gt_label   = r["label"]
                confidence = r["confidence"]
                v_count    = len(included)
                correct    = (pred_label == gt_label)
                gt_color   = GT_MATCH_COLOR if correct else GT_MISMATCH_COLOR
                gt_mark    = "✓" if correct else "✗"

                title = "PRED: {}  conf={:.2f}  v={}\nGT: {} {}".format(
                    pred_label.upper(), confidence, v_count,
                    gt_label.upper(), gt_mark
                )
                ax.set_title(title, fontsize=8, color=color,
                             fontweight="bold", pad=3)

                # Row label
                if col == 0:
                    ax.set_ylabel("PRED\n{}".format(pred_label.upper()),
                                  color=color, fontsize=10,
                                  fontweight="bold", rotation=90, labelpad=6)

        self.fig.suptitle(
            "Page {}   |   n / Enter = new sample    p = prev    q = quit   "
            "|   GT ✓ correct  ✗ wrong".format(self.page + 1),
            color="white", fontsize=10, y=0.98
        )
        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key in ("n", "enter"):
            self.page += 1
            self.render()
        elif event.key == "p" and self.page > 0:
            self.page -= 1
            self.render()
        elif event.key == "q":
            plt.close("all")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mobilenet_v2",
                        choices=["baseline_cnn", "mobilenet_v2", "resnet50", "efficientnet_b0"])
    parser.add_argument("--split-dir", default=str(PROJECT_ROOT / "data/live/splits"))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    print("[device] {}".format(device))

    image_size = tuple(CFG["frame_extraction"]["image_size"])

    # Load data
    csv_path = Path(args.split_dir) / "{}.csv".format(args.split)
    df = pd.read_csv(csv_path)
    print("[data] {} frames from {}".format(len(df), csv_path))

    # Run model inference
    model = load_model(args.model, device)
    print("[inference] Running on {} frames...".format(len(df)))
    pred_labels, confidences = predict_all(model, df, device, image_size)
    df["pred_label"]  = pred_labels
    df["confidence"]  = confidences

    acc = (df["pred_label"] == df["label"]).mean()
    print("[inference] Done.  Accuracy: {:.4f}".format(acc))
    for lbl in LABELS:
        n = (df["label"] == lbl).sum()
        c = ((df["pred_label"] == lbl) & (df["label"] == lbl)).sum()
        print("  {:6s}  recall={:.2f}  ({}/{})".format(lbl, c/n if n else 0, c, n))

    # Load YOLO
    yolo = load_yolo()

    PredictionViewer(df, yolo)


if __name__ == "__main__":
    main()
