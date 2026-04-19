"""
app.py
------
Traffic Congestion Classification — Enhanced Demo GUI

Tabs:
  1. Classify       — Single-model inference, entropy badge, animated traffic light,
                      GradCAM vs YOLO side-by-side
  2. Compare        — 4-model comparison, ensemble summary, GradCAM 2×2 grid
  3. Methodology    — 6-step pipeline explanation
  4. Dataset        — Live dataset stats and results table
  5. Live Feed      — Real-time TfNSW camera fetch + inference
  6. Robustness     — Brightness/contrast/blur stress test
  7. Sequence       — Multi-frame congestion timeline

Run:
  python src/gui/app.py
"""

import base64
import io
import math
import os
import random
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import requests
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not installed — manually parse .env from project root
    _env_path = Path(__file__).resolve().parents[2] / ".env"
    if _env_path.exists():
        for _line in _env_path.read_text().splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config.settings import CFG, PROJECT_ROOT
from src.datasets.congestion_dataset import get_eval_transforms, get_tta_transforms
from src.models.baseline_cnn import BaselineCNN
from src.models.transfer_models import build_model

import gradio as gr


# ── Constants ──────────────────────────────────────────────────────────────────

CLASS_NAMES  = ["low", "medium", "high"]
CLASS_COLORS = {"low": "#3fb950", "medium": "#d29922", "high": "#f85149"}
CLASS_EMOJIS = {"low": "🟢", "medium": "🟡", "high": "🔴"}

MODEL_DISPLAY = {
    "baseline_cnn":     "Baseline CNN",
    "mobilenet_v2":     "MobileNetV2",
    "resnet50":         "ResNet-50",
    "efficientnet_b0":  "EfficientNet-B0 ★",
    "ensemble_tta":     "Ensemble + TTA",
    "efficientnet_tta": "EfficientNet-B0 + TTA ⚡",
}
MODEL_PARAMS = {
    "baseline_cnn":     "619K",
    "mobilenet_v2":     "2.2M",
    "resnet50":         "23.5M",
    "efficientnet_b0":  "4.0M",
    "ensemble_tta":     "4 models",
    "efficientnet_tta": "4.0M",
}
TEST_RESULTS = {
    "baseline_cnn":    {"test_acc": 0.7201, "macro_f1": 0.7246, "low_f1": 0.822, "med_f1": 0.636, "high_f1": 0.716},
    "mobilenet_v2":    {"test_acc": 0.8143, "macro_f1": 0.8204, "low_f1": 0.880, "med_f1": 0.764, "high_f1": 0.818},
    "resnet50":        {"test_acc": 0.7686, "macro_f1": 0.7778, "low_f1": 0.845, "med_f1": 0.715, "high_f1": 0.773},
    "efficientnet_b0": {"test_acc": 0.8290, "macro_f1": 0.8340, "low_f1": 0.899, "med_f1": 0.768, "high_f1": 0.835},
    "ensemble_tta":    {"test_acc": 0.8249, "macro_f1": 0.8308, "low_f1": 0.891, "med_f1": 0.776, "high_f1": 0.825},
    "efficientnet_tta":{"test_acc": 0.8331, "macro_f1": 0.8375, "low_f1": 0.898, "med_f1": 0.774, "high_f1": 0.841},
}

SIGNAL_RULES = {
    "low":    {"action": "Maintain current cycle",      "delta": 0,   "detail": "Traffic is flowing freely. No signal adjustment required."},
    "medium": {"action": "Extend green phase by ~10 s", "delta": +10, "detail": "Moderate congestion detected. Extend current green phase to clear queued vehicles."},
    "high":   {"action": "Extend green phase by ~20 s", "delta": +20, "detail": "Heavy congestion detected. Significant green extension recommended. Consider overflow protocol if queues persist."},
}

# TfNSW Live Feed
TFNSW_API_KEY = os.environ.get("TFNSW_API_KEY", "")
TFNSW_API_URL = "https://api.transport.nsw.gov.au/v1/live/cameras"
TFNSW_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    "Referer": "https://www.livetraffic.com/",
}
CAMERA_IDS = {
    "parramatta_road_camperdown":   "Parramatta Rd at Camperdown (Sydney)",
    "hume_highway_bankstown":       "Hume Hwy at Bankstown (Sydney)",
    "anzac_parade_moore_park":      "Anzac Parade at Moore Park (Sydney)",
    "james_ruse_drive_rosehill":    "James Ruse Dr at Rosehill (Sydney)",
    "princes_highway_st_peters_n":  "Princes Hwy at St Peters N (Sydney)",
    "city_road_newtown":            "City Rd at Newtown (Sydney)",
    "king_georges_road_hurstville": "King Georges Rd at Hurstville (Sydney)",
    "5_ways_miranda":               "5 Ways Miranda (Sydney)",
    "memorial_drive_towradgi":      "Memorial Dr at Towradgi (Wollongong)",
    "shellharbour_road_warilla":    "Shellharbour Rd at Warilla (Wollongong)",
}
CAMERA_DISPLAY_TO_ID = {v: k for k, v in CAMERA_IDS.items()}

CAMERA_EXCLUDE = {
    "5_ways_miranda": [(0.0, 0.0, 0.28, 0.55)],
}

EXAMPLE_IMAGES = [
    str(PROJECT_ROOT / "data/live/raw/memorial_drive_towradgi/0000/frame_00.jpg"),
    str(PROJECT_ROOT / "data/live/raw/memorial_drive_towradgi/0002/frame_03.jpg"),
    str(PROJECT_ROOT / "data/live/raw/memorial_drive_towradgi/0030/frame_00.jpg"),
]


# ── CSS ────────────────────────────────────────────────────────────────────────

CSS = """
body, .gradio-container {
    background: #0d1117 !important;
    color: #e6edf3 !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; }
footer { display: none !important; }

.tab-nav button {
    background: transparent !important;
    color: #8b949e !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 10px 20px !important;
    font-size: 0.9em !important;
    font-weight: 500 !important;
    letter-spacing: 0.03em !important;
    transition: all 0.2s !important;
}
.tab-nav button.selected {
    color: #00d4aa !important;
    border-bottom: 2px solid #00d4aa !important;
    background: transparent !important;
}

.gr-button-primary, button.primary {
    background: linear-gradient(135deg, #00d4aa, #0099cc) !important;
    border: none !important;
    color: #0d1117 !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    transition: opacity 0.2s !important;
}
.gr-button-primary:hover, button.primary:hover { opacity: 0.85 !important; }

.gr-button-secondary, button.secondary {
    background: transparent !important;
    border: 1px solid #30363d !important;
    color: #8b949e !important;
    font-weight: 500 !important;
    font-size: 0.82em !important;
    letter-spacing: 0.05em !important;
    border-radius: 6px !important;
    padding: 6px 12px !important;
    transition: all 0.2s !important;
}
.gr-button-secondary:hover, button.secondary:hover {
    border-color: #00d4aa !important;
    color: #00d4aa !important;
    background: #00d4aa11 !important;
}

select, .gr-dropdown {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 8px !important;
}
.gr-image { border: 1px solid #30363d !important; border-radius: 12px !important; background: #161b22 !important; }
.gr-label { background: #161b22 !important; border: 1px solid #30363d !important; border-radius: 12px !important; }

/* Traffic light animations */
@keyframes pulse-slow {
    0%,100% { opacity:1.0; box-shadow:0 0 18px #3fb950, 0 0 40px #3fb95044; }
    50%      { opacity:0.6; box-shadow:0 0 8px  #3fb950, 0 0 16px #3fb95022; }
}
@keyframes pulse-medium {
    0%,100% { opacity:1.0; box-shadow:0 0 18px #d29922, 0 0 40px #d2992244; }
    50%      { opacity:0.5; box-shadow:0 0 6px  #d29922, 0 0 12px #d2992222; }
}
@keyframes pulse-fast {
    0%,100% { opacity:1.0; box-shadow:0 0 22px #f85149, 0 0 48px #f8514966; }
    50%      { opacity:0.4; box-shadow:0 0 6px  #f85149, 0 0 12px #f8514922; }
}
.bulb-pulse-slow   { animation: pulse-slow   2.4s ease-in-out infinite; }
.bulb-pulse-medium { animation: pulse-medium 1.2s ease-in-out infinite; }
.bulb-pulse-fast   { animation: pulse-fast   0.5s ease-in-out infinite; }

@keyframes blink { 0%,100% { opacity:1; } 50% { opacity:0.3; } }
.example-gallery img { cursor: pointer; transition: opacity 0.15s, transform 0.15s; }
.example-gallery img:hover { opacity: 0.82; transform: scale(1.03); }
.example-gallery .thumbnail-item { border-radius: 8px; overflow: hidden; }

.live-dot {
    display:inline-block; width:8px; height:8px; border-radius:50%;
    background:#3fb950; animation:blink 1.5s ease-in-out infinite; margin-right:6px;
}
"""


# ── Model loading ──────────────────────────────────────────────────────────────

def get_device():
    return torch.device("cpu")


def load_model(model_name):
    # type: (str) -> Optional[nn.Module]
    ckpt_path = PROJECT_ROOT / CFG["paths"]["checkpoints"] / "{}_best.pt".format(model_name)
    if not ckpt_path.exists():
        return None
    num_classes = CFG["models"]["num_classes"]
    if model_name == "baseline_cnn":
        model = BaselineCNN(num_classes=num_classes)
    else:
        model = build_model(model_name, num_classes=num_classes, pretrained=False)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(get_device())
    model.eval()
    return model


def load_all_models():
    # type: () -> Tuple[Dict, Dict]
    models, gradcams = {}, {}
    for name in ["baseline_cnn", "mobilenet_v2", "resnet50", "efficientnet_b0"]:
        m = load_model(name)
        if m is not None:
            models[name] = m
            gradcams[name] = GradCAM(m, name)
            print("[GUI] Loaded {}".format(name))
        else:
            print("[GUI] Skipped {} — no checkpoint".format(name))
    return models, gradcams


def get_transform():
    image_size = tuple(CFG["frame_extraction"]["image_size"])
    return get_eval_transforms(image_size)


# ── GradCAM ────────────────────────────────────────────────────────────────────

class GradCAM:
    def __init__(self, model, model_name):
        self.model      = model
        self.model_name = model_name
        self.gradients  = None
        self.activations = None
        self._handles   = []
        self._register()

    def _get_target_layer(self):
        name = self.model_name
        try:
            if name == "baseline_cnn":
                return self.model.features[-1]
            elif name == "mobilenet_v2":
                return self.model.features[13]
            elif name == "resnet50":
                return self.model.layer3[-1]
            elif name == "efficientnet_b0":
                return self.model.features[5]
        except Exception:
            pass
        for m in reversed(list(self.model.modules())):
            if isinstance(m, nn.Conv2d):
                return m
        return None

    def _register(self):
        layer = self._get_target_layer()
        if layer is None:
            return
        def fwd_hook(_, __, output):
            self.activations = output.detach()
        def bwd_hook(_, __, grad_output):
            self.gradients = grad_output[0].detach()
        self._handles.append(layer.register_forward_hook(fwd_hook))
        self._handles.append(layer.register_backward_hook(bwd_hook))

    def generate(self, image_tensor, target_class):
        # type: (torch.Tensor, int) -> Optional[np.ndarray]
        self.model.zero_grad()
        output = self.model(image_tensor)
        score  = output[0, target_class]
        score.backward()
        if self.gradients is None or self.activations is None:
            return None
        w   = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (w * self.activations).sum(dim=1).squeeze(0)
        cam = torch.clamp(cam, min=0).numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam

    def overlay(self, cam, image_pil):
        # type: (np.ndarray, Image.Image) -> Image.Image
        img_np = np.array(image_pil.convert("RGB"))
        h, w   = img_np.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        blended = (0.45 * heatmap + 0.55 * img_np).astype(np.uint8)
        return Image.fromarray(blended)


# ── YOLO (lazy, thread-safe) ───────────────────────────────────────────────────

_yolo_model = None
_yolo_lock  = threading.Lock()

def get_yolo():
    global _yolo_model
    with _yolo_lock:
        if _yolo_model is not None:
            return _yolo_model if _yolo_model is not False else None
        try:
            from ultralytics import YOLO
            _yolo_model = YOLO("yolov8m.pt")
            _yolo_model.conf = 0.15
            print("[GUI] YOLO loaded")
        except Exception as e:
            print("[GUI] YOLO unavailable: {}".format(e))
            _yolo_model = False
    return _yolo_model if _yolo_model is not False else None


VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck

def yolo_detection_image(image_pil, camera_name=None):
    # type: (Image.Image, Optional[str]) -> Tuple[Optional[Image.Image], int]
    yolo = get_yolo()
    if yolo is None:
        return None, 0

    img_np = np.array(image_pil.convert("RGB"))
    h, w   = img_np.shape[:2]

    results = yolo(img_np, conf=0.15, imgsz=1280, verbose=False)
    exclude_zones = CAMERA_EXCLUDE.get(camera_name, []) if camera_name else []

    annotated = img_np.copy()
    count = 0

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        if cls_id not in VEHICLE_CLASSES:
            continue
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        cx_n = ((x1 + x2) / 2) / w
        cy_n = ((y1 + y2) / 2) / h
        excluded = any(
            ex[0] <= cx_n <= ex[2] and ex[1] <= cy_n <= ex[3]
            for ex in exclude_zones
        )
        if excluded:
            # Red dashed box — draw as alternating segments
            for side in ["top", "bottom", "left", "right"]:
                if side == "top":
                    pts = [(x1 + i, y1) for i in range(0, x2 - x1, 8)]
                elif side == "bottom":
                    pts = [(x1 + i, y2) for i in range(0, x2 - x1, 8)]
                elif side == "left":
                    pts = [(x1, y1 + i) for i in range(0, y2 - y1, 8)]
                else:
                    pts = [(x2, y1 + i) for i in range(0, y2 - y1, 8)]
                for j in range(0, len(pts) - 1, 2):
                    cv2.line(annotated, pts[j], pts[min(j + 1, len(pts) - 1)], (248, 81, 73), 2)
        else:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (63, 185, 80), 2)
            count += 1

    cv2.putText(annotated, "Vehicles: {}".format(count),
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (63, 185, 80), 2)
    return Image.fromarray(annotated), count


# ── Inference ──────────────────────────────────────────────────────────────────

def run_inference(image_pil, model_name, models, gradcams, transform):
    # type: (Image.Image, str, Dict, Dict, transforms.Compose) -> Tuple[Optional[str], Optional[np.ndarray], Optional[Image.Image]]
    if image_pil is None:
        return None, None, None

    image_size = tuple(CFG["frame_extraction"]["image_size"])
    img_rgb    = image_pil.convert("RGB")

    if model_name == "ensemble_tta":
        probs = run_ensemble_tta(img_rgb, models, image_size)
        cam_img = None
    elif model_name == "efficientnet_tta":
        if "efficientnet_b0" not in models:
            return None, None, None
        probs   = run_single_tta(img_rgb, models["efficientnet_b0"], image_size)
        cam_img = None
    else:
        if model_name not in models:
            return None, None, None
        tensor = transform(img_rgb).unsqueeze(0).to(get_device())
        with torch.no_grad():
            logits = models[model_name](tensor)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
        idx     = int(probs.argmax())
        cam_arr = gradcams[model_name].generate(tensor, idx)
        cam_img = gradcams[model_name].overlay(cam_arr, img_rgb) if cam_arr is not None else None

    idx   = int(probs.argmax())
    label = CLASS_NAMES[idx]
    return label, probs, cam_img


def run_ensemble_tta(image_pil, models, image_size):
    # type: (Image.Image, Dict, tuple) -> np.ndarray
    tta_tfms = get_tta_transforms(image_size)
    all_probs = []
    for name, model in models.items():
        for tfm in tta_tfms:
            tensor = tfm(image_pil).unsqueeze(0).to(get_device())
            with torch.no_grad():
                p = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]
            all_probs.append(p)
    return np.mean(all_probs, axis=0)


def run_single_tta(image_pil, model, image_size):
    # type: (Image.Image, nn.Module, tuple) -> np.ndarray
    tta_tfms = get_tta_transforms(image_size)
    all_probs = []
    for tfm in tta_tfms:
        tensor = tfm(image_pil).unsqueeze(0).to(get_device())
        with torch.no_grad():
            p = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]
        all_probs.append(p)
    return np.mean(all_probs, axis=0)


# ── Image utilities ────────────────────────────────────────────────────────────

def pil_to_base64(img):
    # type: (Image.Image) -> str
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def two_image_html(left_img, left_caption, right_img, right_caption):
    # type: (Optional[Image.Image], str, Optional[Image.Image], str) -> str
    def slot(img, caption):
        if img is None:
            return (
                "<div style='flex:1;display:flex;align-items:center;justify-content:center;"
                "background:#161b22;border:1px solid #30363d;border-radius:8px;"
                "color:#6e7681;font-size:0.82em;padding:30px;text-align:center;'>"
                "{}</div>".format(caption + "<br><small style='color:#484f58;'>unavailable</small>")
            )
        b64 = pil_to_base64(img)
        return (
            "<div style='flex:1;text-align:center;'>"
            "<img src='data:image/png;base64,{}' style='width:100%;border-radius:8px;"
            "border:1px solid #30363d;'/>"
            "<div style='font-size:0.76em;color:#8b949e;margin-top:6px;'>{}</div>"
            "</div>"
        ).format(b64, caption)

    return (
        "<div style='display:flex;gap:14px;margin-top:12px;align-items:flex-start;'>"
        "{}{}</div>"
    ).format(slot(left_img, left_caption), slot(right_img, right_caption))


# ── HTML builders ──────────────────────────────────────────────────────────────

def compute_entropy_badge(probs):
    # type: (np.ndarray) -> str
    # H_norm: 0 = perfectly certain, 1 = uniform across all 3 classes
    # < 0.70 → top class ≥ ~75% → "High Confidence"
    # > 0.90 → very uniform (e.g. 50/30/20 split) → "Uncertain"
    H      = -np.sum(probs * np.log2(probs + 1e-9))
    H_norm = H / math.log2(3)
    if H_norm < 0.70:
        return (
            "<span style='display:inline-block;background:#3fb95022;border:1px solid #3fb950;"
            "color:#3fb950;border-radius:20px;padding:3px 12px;font-size:0.78em;font-weight:600;"
            "margin-left:8px;'>High Confidence</span>"
        )
    elif H_norm > 0.90:
        return (
            "<span style='display:inline-block;background:#f8514922;border:1px solid #f85149;"
            "color:#f85149;border-radius:20px;padding:3px 12px;font-size:0.78em;font-weight:600;"
            "margin-left:8px;'>⚠ Uncertain</span>"
        )
    return ""


def prediction_html(label, probs):
    # type: (str, np.ndarray) -> str
    color = CLASS_COLORS[label]
    emoji = CLASS_EMOJIS[label]
    badge = compute_entropy_badge(probs)

    bars = ""
    for n, p in zip(CLASS_NAMES, probs):
        c = CLASS_COLORS[n]
        bars += (
            "<div style='margin:6px 0;'>"
            "<div style='display:flex;justify-content:space-between;margin-bottom:3px;'>"
            "<span style='color:{c};font-weight:600;font-size:0.85em;'>{emoji} {lbl}</span>"
            "<span style='color:#8b949e;font-size:0.85em;'>{pct:.1f}%</span></div>"
            "<div style='background:#21262d;border-radius:4px;height:8px;overflow:hidden;'>"
            "<div style='width:{pct:.1f}%;background:{c};height:100%;border-radius:4px;"
            "transition:width 0.4s ease;'></div></div></div>"
        ).format(c=c, emoji=CLASS_EMOJIS[n], lbl=n.capitalize(), pct=p * 100)

    return (
        "<div style='background:#161b22;border:1px solid #30363d;border-radius:12px;padding:20px;'>"
        "<div style='font-size:0.7em;color:#8b949e;letter-spacing:0.1em;text-transform:uppercase;"
        "margin-bottom:8px;'>Prediction</div>"
        "<div style='display:flex;align-items:center;flex-wrap:wrap;margin-bottom:12px;'>"
        "<div style='font-size:2em;font-weight:800;color:{color};'>{emoji} {label}</div>"
        "{badge}</div>"
        "<div style='font-size:0.75em;color:#8b949e;letter-spacing:0.08em;"
        "text-transform:uppercase;margin-bottom:10px;'>Confidence</div>"
        "{bars}</div>"
    ).format(color=color, emoji=emoji, label=label.upper(), badge=badge, bars=bars)


def traffic_light_html(label):
    # type: (str) -> str
    rule  = SIGNAL_RULES[label]
    color = CLASS_COLORS[label]

    pulse = {"low": "bulb-pulse-slow", "medium": "bulb-pulse-medium", "high": "bulb-pulse-fast"}[label]
    active_colors = {"low": (None, None, "#3fb950"), "medium": (None, "#d29922", None), "high": ("#f85149", None, None)}
    r_col, y_col, g_col = active_colors[label]

    def bulb(col, anim_class=""):
        if col:
            return (
                "<div class='{anim}' style='width:52px;height:52px;border-radius:50%;"
                "background:{col};'></div>"
            ).format(col=col, anim=anim_class)
        return "<div style='width:52px;height:52px;border-radius:50%;background:#21262d;border:1px solid #30363d;'></div>"

    delta_str = "No change" if rule["delta"] == 0 else "+{} seconds".format(rule["delta"])

    return (
        "<div style='background:#161b22;border:1px solid #30363d;border-radius:12px;"
        "padding:20px;margin-top:12px;'>"
        "<div style='font-size:0.7em;color:#8b949e;letter-spacing:0.1em;"
        "text-transform:uppercase;margin-bottom:12px;'>Signal Recommendation</div>"
        "<div style='display:flex;gap:24px;align-items:center;flex-wrap:wrap;'>"
        "<div style='display:inline-flex;flex-direction:column;align-items:center;gap:10px;"
        "background:#0d1117;border:2px solid #30363d;border-radius:16px;padding:18px 22px;'>"
        "{r}{y}{g}"
        "</div>"
        "<div>"
        "<div style='font-size:1em;font-weight:700;color:{color};margin-bottom:6px;'>{action}</div>"
        "<div style='font-size:0.82em;color:#8b949e;margin-bottom:4px;'>Green phase delta: "
        "<span style='color:{color};font-weight:600;'>{delta}</span></div>"
        "<div style='font-size:0.8em;color:#6e7681;max-width:320px;'>{detail}</div>"
        "</div></div>"
        "<div style='font-size:0.7em;color:#484f58;margin-top:12px;border-top:1px solid #21262d;"
        "padding-top:8px;'>Rule-based prototype — not a production traffic control system</div>"
        "</div>"
    ).format(
        r=bulb(r_col, pulse if r_col else ""),
        y=bulb(y_col, pulse if y_col else ""),
        g=bulb(g_col, pulse if g_col else ""),
        color=color, action=rule["action"], delta=delta_str, detail=rule["detail"],
    )


def compare_html(results):
    # type: (List[Tuple]) -> str
    cards = ""
    for name, label, probs in results:
        if label is None:
            continue
        color = CLASS_COLORS[label]
        emoji = CLASS_EMOJIS[label]
        res   = TEST_RESULTS.get(name, {})
        bars  = ""
        for n, p in zip(CLASS_NAMES, probs):
            c = CLASS_COLORS[n]
            bars += (
                "<div style='margin:4px 0;'>"
                "<div style='display:flex;justify-content:space-between;margin-bottom:2px;'>"
                "<span style='color:{c};font-size:0.8em;'>{n}</span>"
                "<span style='color:#8b949e;font-size:0.8em;'>{p:.1f}%</span></div>"
                "<div style='background:#21262d;border-radius:3px;height:6px;'>"
                "<div style='width:{p:.1f}%;background:{c};height:100%;border-radius:3px;'></div>"
                "</div></div>"
            ).format(c=c, n=n.capitalize(), p=p * 100)

        cards += (
            "<div style='background:#161b22;border:1px solid #30363d;border-radius:10px;"
            "padding:16px;border-top:3px solid {color};'>"
            "<div style='font-size:0.75em;color:#8b949e;margin-bottom:6px;'>{display}</div>"
            "<div style='font-size:1.4em;font-weight:800;color:{color};margin-bottom:10px;'>"
            "{emoji} {label}</div>"
            "{bars}"
            "<div style='margin-top:10px;font-size:0.76em;color:#6e7681;border-top:1px solid #21262d;"
            "padding-top:8px;'>"
            "Test Acc {acc:.1f}%  ·  Macro F1 {f1:.4f}  ·  {params}"
            "</div></div>"
        ).format(
            color=color, display=MODEL_DISPLAY.get(name, name),
            emoji=emoji, label=label.upper(), bars=bars,
            acc=res.get("test_acc", 0) * 100, f1=res.get("macro_f1", 0),
            params=MODEL_PARAMS.get(name, "—"),
        )

    return (
        "<div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;'>{}</div>"
    ).format(cards)


def compare_ensemble_html(results, ensemble_probs):
    # type: (List[Tuple], np.ndarray) -> str
    labels = [r[1] for r in results if r[1] is not None]
    if not labels:
        return ""

    vote_counts = Counter(labels)
    majority    = vote_counts.most_common(1)[0][0]
    n_agree     = vote_counts[majority]
    n_models    = len(labels)
    disagree    = 1.0 - (n_agree / n_models)
    dis_color   = "#f85149" if disagree > 0.4 else "#d29922" if disagree > 0 else "#3fb950"

    ens_idx   = int(np.argmax(ensemble_probs))
    ens_label = CLASS_NAMES[ens_idx]

    ens_bars = ""
    for n, p in zip(CLASS_NAMES, ensemble_probs):
        c = CLASS_COLORS[n]
        ens_bars += (
            "<div style='margin:4px 0;'>"
            "<div style='display:flex;justify-content:space-between;margin-bottom:2px;'>"
            "<span style='color:{c};font-size:0.82em;'>{n}</span>"
            "<span style='color:#8b949e;font-size:0.82em;'>{p:.1f}%</span></div>"
            "<div style='background:#21262d;border-radius:3px;height:7px;'>"
            "<div style='width:{p:.1f}%;background:{c};height:100%;border-radius:3px;'></div>"
            "</div></div>"
        ).format(c=c, n=n.capitalize(), p=p * 100)

    model_badges = ""
    for name, lbl, _ in results:
        if lbl is None:
            continue
        agrees = (lbl == majority)
        bc = "#3fb950" if agrees else "#f85149"
        model_badges += (
            "<span style='display:inline-block;margin:3px;padding:3px 10px;border-radius:12px;"
            "border:1px solid {bc};color:{bc};font-size:0.76em;'>"
            "{display}: {'Agrees' if agrees else 'Disagrees'}</span>"
        ).format(bc=bc, display=MODEL_DISPLAY.get(name, name).replace(" ★", ""),
                 **{"'Agrees' if agrees else 'Disagrees'": "Agrees" if agrees else "Disagrees"})

    return (
        "<div style='background:#161b22;border:1px solid #30363d;border-radius:12px;"
        "padding:20px;margin-top:14px;border-top:3px solid #00d4aa;'>"
        "<div style='font-size:0.7em;color:#8b949e;letter-spacing:0.1em;"
        "text-transform:uppercase;margin-bottom:12px;'>Ensemble Summary</div>"
        "<div style='display:flex;gap:24px;flex-wrap:wrap;align-items:flex-start;'>"
        "<div style='flex:1;min-width:160px;'>"
        "<div style='font-size:0.78em;color:#8b949e;margin-bottom:6px;'>Softmax Average</div>"
        "<div style='font-size:1.3em;font-weight:800;color:{ens_color};margin-bottom:8px;'>"
        "{ens_emoji} {ens_label}</div>{ens_bars}</div>"
        "<div style='flex:1;min-width:160px;'>"
        "<div style='font-size:0.78em;color:#8b949e;margin-bottom:4px;'>Majority Vote</div>"
        "<div style='font-size:1.3em;font-weight:800;color:{maj_color};margin-bottom:10px;'>"
        "{maj_emoji} {majority}</div>"
        "<div style='font-size:0.78em;color:#8b949e;margin-bottom:4px;'>Disagreement Index</div>"
        "<div style='font-size:1.4em;font-weight:700;color:{dis_color};margin-bottom:10px;'>"
        "{dis:.0f}%</div>"
        "<div>{model_badges}</div></div></div></div>"
    ).format(
        ens_color=CLASS_COLORS[ens_label], ens_emoji=CLASS_EMOJIS[ens_label],
        ens_label=ens_label.upper(), ens_bars=ens_bars,
        maj_color=CLASS_COLORS[majority], maj_emoji=CLASS_EMOJIS[majority],
        majority=majority.upper(), dis_color=dis_color,
        dis=disagree * 100, model_badges=model_badges,
    )


def compare_gradcam_grid(results, gradcam_imgs):
    # type: (List[Tuple], Dict) -> str
    cells = ""
    for name, label, _ in results:
        if label is None:
            continue
        img = gradcam_imgs.get(name)
        if img is None:
            content = "<div style='color:#6e7681;font-size:0.8em;padding:20px;text-align:center;'>Unavailable</div>"
        else:
            b64 = pil_to_base64(img)
            content = "<img src='data:image/png;base64,{}' style='width:100%;border-radius:6px;'/>".format(b64)
        cells += (
            "<div style='background:#161b22;border:1px solid #30363d;border-radius:10px;overflow:hidden;'>"
            "{content}"
            "<div style='padding:6px 10px;font-size:0.76em;color:#8b949e;'>"
            "{display} — GradCAM</div></div>"
        ).format(content=content, display=MODEL_DISPLAY.get(name, name).replace(" ★", ""))

    return (
        "<div style='margin-top:16px;'>"
        "<div style='font-size:0.7em;color:#8b949e;letter-spacing:0.1em;"
        "text-transform:uppercase;margin-bottom:10px;'>GradCAM Activations</div>"
        "<div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;'>{}</div></div>"
    ).format(cells)


def methodology_html():
    steps = [
        ("#00d4aa", "1. Raw Data Collection",
         "TfNSW Open Data API — 10 fixed intersection cameras across Sydney and Wollongong. "
         "Frames collected every 15 seconds across 3 sessions (weekend + weekday morning/midday/afternoon peak). "
         "Filtered by brightness (mean > 80) and time-of-day (06:00–18:00)."),
        ("#58a6ff", "2. Vehicle Detection",
         "YOLOv8n detects vehicles in each frame (classes: car, motorcycle, bus, truck). "
         "Per-camera ROI and exclusion zones remove off-road vehicles (e.g. car dealership parking lots)."),
        ("#d29922", "3. Per-Frame Labeling",
         "Each frame labeled individually based on YOLO vehicle count against per-camera calibrated thresholds. "
         "Thresholds set by visual inspection. Labels: Low / Medium / High congestion."),
        ("#bc8cff", "4. Window-Stratified Split",
         "4-frame windows kept together to prevent temporal leakage. "
         "Train/Val: Sydney cameras (8 cameras). Test: Wollongong cameras (2 cameras) — fully held-out region."),
        ("#f85149", "5. Model Training",
         "4 CNN architectures trained: Baseline CNN (619K params), MobileNetV2 (2.2M), "
         "ResNet-50 (23.5M), EfficientNet-B0 (4.0M). ImageNet pretrained, fine-tuned with "
         "class-weighted CrossEntropyLoss, cosine LR schedule, 50 epochs."),
        ("#3fb950", "6. Ensemble + TTA",
         "Best single model: EfficientNet-B0 + TTA (83.31% test accuracy). "
         "TTA averages 5 augmented variants (original, H-flip, V-flip, ±5° rotation). "
         "Signal timing recommendation derived from predicted congestion class."),
    ]
    cards = ""
    for color, title, text in steps:
        cards += (
            "<div style='background:#161b22;border:1px solid #30363d;border-left:3px solid {color};"
            "border-radius:8px;padding:16px;'>"
            "<div style='font-weight:700;color:{color};margin-bottom:6px;'>{title}</div>"
            "<div style='font-size:0.85em;color:#8b949e;line-height:1.6;'>{text}</div>"
            "</div>"
        ).format(color=color, title=title, text=text)
    return (
        "<div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px;'>"
        "{}</div>"
    ).format(cards)


def dataset_html():
    stats = [
        ("10",       "Live Cameras",          "#00d4aa", "TfNSW traffic intersections"),
        ("15,881",   "Total Frames",          "#58a6ff", "Across 10 cameras, 3 sessions"),
        ("3",        "Capture Sessions",      "#d29922",  "Weekend + Weekday peak (morn/mid/arvo)"),
        ("224×224",  "Image Size",            "#bc8cff", "Pixels, RGB, JPEG"),
        ("8 / 2",    "Train / Test Cameras",  "#f85149", "Sydney (train), Wollongong (test)"),
        ("70/15/15", "Split Ratio",           "#3fb950", "Train / Val / Test"),
    ]
    stat_cards = ""
    for val, label, color, sub in stats:
        stat_cards += (
            "<div style='background:#161b22;border:1px solid #30363d;border-radius:10px;"
            "padding:20px 16px;text-align:center;'>"
            "<div style='font-size:2em;font-weight:800;color:{color};letter-spacing:-0.02em;'>{val}</div>"
            "<div style='font-weight:600;color:#e6edf3;font-size:0.9em;margin:6px 0 4px;'>{label}</div>"
            "<div style='font-size:0.78em;color:#6e7681;'>{sub}</div>"
            "</div>"
        ).format(color=color, val=val, label=label, sub=sub)

    header = (
        "<tr style='background:#0d1117;border-bottom:2px solid #30363d;'>"
        "<th style='padding:14px 18px;color:#6e7681;text-align:left;font-size:0.8em;"
        "letter-spacing:0.08em;text-transform:uppercase;font-weight:600;'>Model</th>"
        "<th style='padding:14px 18px;color:#6e7681;text-align:left;font-size:0.8em;"
        "letter-spacing:0.08em;text-transform:uppercase;font-weight:600;'>Test Accuracy</th>"
        "<th style='padding:14px 18px;color:#6e7681;text-align:right;font-size:0.8em;"
        "letter-spacing:0.08em;text-transform:uppercase;font-weight:600;'>Macro F1</th>"
        "<th style='padding:14px 18px;color:#3fb950;text-align:right;font-size:0.8em;"
        "letter-spacing:0.08em;text-transform:uppercase;font-weight:600;'>Low</th>"
        "<th style='padding:14px 18px;color:#d29922;text-align:right;font-size:0.8em;"
        "letter-spacing:0.08em;text-transform:uppercase;font-weight:600;'>Med</th>"
        "<th style='padding:14px 18px;color:#f85149;text-align:right;font-size:0.8em;"
        "letter-spacing:0.08em;text-transform:uppercase;font-weight:600;'>High</th>"
        "<th style='padding:14px 18px;color:#6e7681;text-align:right;font-size:0.8em;"
        "letter-spacing:0.08em;text-transform:uppercase;font-weight:600;'>Params</th>"
        "</tr>"
    )

    order = ["baseline_cnn", "mobilenet_v2", "resnet50", "efficientnet_b0", "ensemble_tta", "efficientnet_tta"]
    best  = "efficientnet_tta"
    rows  = ""
    for name in order:
        r        = TEST_RESULTS.get(name, {})
        is_best  = (name == best)
        acc_val  = r.get("test_acc", 0) * 100
        row_bg   = "background:linear-gradient(90deg,#0d2818 0%,#0d1117 100%);" if is_best else ""
        fw       = "700" if is_best else "400"
        name_col = MODEL_DISPLAY.get(name, name)

        # Inline accuracy bar (width scaled to 60–100% range for visual impact)
        bar_w    = max(0, min(100, (acc_val - 60) / 40 * 100))
        bar_color = "#3fb950" if is_best else "#58a6ff"
        acc_bar  = (
            "<div style='display:flex;align-items:center;gap:10px;'>"
            "<div style='flex:1;background:#21262d;border-radius:4px;height:6px;overflow:hidden;'>"
            "<div style='height:6px;border-radius:4px;background:{bc};width:{bw:.0f}%;'></div></div>"
            "<span style='min-width:46px;text-align:right;font-weight:{fw};color:{tc};font-size:1.0em;'>"
            "{acc:.1f}%</span></div>"
        ).format(bc=bar_color, bw=bar_w, fw=fw,
                 tc="#3fb950" if is_best else "#e6edf3", acc=acc_val)

        rows += (
            "<tr style='border-bottom:1px solid #21262d;{bg}'>"
            "<td style='padding:14px 18px;font-weight:{fw};font-size:0.95em;'>"
            "<span style='color:#e6edf3;'>{display}</span></td>"
            "<td style='padding:14px 18px;min-width:180px;'>{acc_bar}</td>"
            "<td style='padding:14px 18px;color:#8b949e;text-align:right;font-size:0.95em;'>{f1:.4f}</td>"
            "<td style='padding:14px 18px;color:#3fb950;text-align:right;font-size:0.95em;'>{low:.3f}</td>"
            "<td style='padding:14px 18px;color:#d29922;text-align:right;font-size:0.95em;'>{med:.3f}</td>"
            "<td style='padding:14px 18px;color:#f85149;text-align:right;font-size:0.95em;'>{high:.3f}</td>"
            "<td style='padding:14px 18px;color:#6e7681;text-align:right;font-size:0.9em;'>{params}</td>"
            "</tr>"
        ).format(
            bg=row_bg, fw=fw, display=name_col, acc_bar=acc_bar,
            f1=r.get("macro_f1", 0),
            low=r.get("low_f1", 0), med=r.get("med_f1", 0), high=r.get("high_f1", 0),
            params=MODEL_PARAMS.get(name, "—"),
        )

    return (
        "<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:24px;'>"
        "{stat_cards}</div>"
        "<div style='font-size:0.72em;color:#6e7681;letter-spacing:0.1em;text-transform:uppercase;"
        "margin-bottom:10px;font-weight:600;'>Results — Test Set (Wollongong cameras, held-out region)</div>"
        "<div style='background:#161b22;border:1px solid #30363d;border-radius:14px;overflow:hidden;'>"
        "<table style='width:100%;border-collapse:collapse;'>"
        "<thead>{header}</thead><tbody>{rows}</tbody>"
        "</table></div>"
    ).format(stat_cards=stat_cards, header=header, rows=rows)


def sequence_timeline_html(labels, confidences):
    # type: (List[str], List[float]) -> str
    level_map  = {"low": 0, "medium": 1, "high": 2}
    y_vals     = [level_map.get(l, 0) for l in labels]
    x_vals     = list(range(len(labels)))
    seg_colors = [CLASS_COLORS[l] for l in labels]

    fig, ax = plt.subplots(figsize=(10, 3.5), facecolor="#0d1117")
    ax.set_facecolor("#161b22")

    for i in range(len(x_vals) - 1):
        ax.plot([x_vals[i], x_vals[i + 1]], [y_vals[i], y_vals[i + 1]],
                color=seg_colors[i], linewidth=2.5, solid_capstyle="round")
    for xi, yi, col in zip(x_vals, y_vals, seg_colors):
        ax.scatter(xi, yi, color=col, s=60, zorder=5)

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Low", "Medium", "High"], color="#8b949e", fontsize=9)
    ax.set_xlabel("Frame Index", color="#8b949e", fontsize=9)
    ax.tick_params(colors="#8b949e")
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#30363d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(-0.5, max(len(x_vals) - 0.5, 1))
    ax.set_ylim(-0.4, 2.4)
    ax.grid(axis="y", color="#30363d", linestyle="--", alpha=0.5)
    patches = [mpatches.Patch(color=v, label=k.capitalize()) for k, v in CLASS_COLORS.items()]
    ax.legend(handles=patches, facecolor="#161b22", edgecolor="#30363d",
              labelcolor="#e6edf3", fontsize=8, loc="upper right")
    plt.tight_layout(pad=0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return (
        "<img src='data:image/png;base64,{}' style='width:100%;border-radius:8px;"
        "border:1px solid #30363d;'/>"
    ).format(b64)


# ── TfNSW API ──────────────────────────────────────────────────────────────────

def fetch_camera_image(camera_id):
    # type: (str) -> Tuple[Optional[Image.Image], str]
    if not TFNSW_API_KEY:
        return None, "TFNSW_API_KEY not set. Add it to your .env file to use Live Feed."

    api_headers = {
        "Authorization": "apikey {}".format(TFNSW_API_KEY),
        "Accept": "application/json",
    }
    try:
        resp = requests.get(TFNSW_API_URL, headers=api_headers, timeout=15)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        return None, "Network error: could not reach TfNSW API."
    except requests.exceptions.Timeout:
        return None, "Request timed out. Try again."
    except requests.exceptions.HTTPError as e:
        if resp.status_code == 401:
            return None, "API key rejected (401). Check TFNSW_API_KEY in .env."
        return None, "HTTP error {}.".format(resp.status_code)
    except Exception as e:
        return None, "Unexpected error: {}".format(str(e))

    # Derive camera_id from href basename (same as collect.py)
    hrefs = {}
    for feature in resp.json().get("features", []):
        href = feature.get("properties", {}).get("href", "")
        if not href:
            continue
        cid = os.path.splitext(os.path.basename(urlparse(href).path))[0]
        hrefs[cid] = href

    if camera_id not in hrefs:
        return None, "Camera '{}' not found in API response (may be offline).".format(camera_id)

    try:
        img_resp = requests.get(hrefs[camera_id], headers=TFNSW_BROWSER_HEADERS, timeout=15, allow_redirects=True)
        img_resp.raise_for_status()
        img = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
        return img, ""
    except Exception as e:
        return None, "Failed to fetch camera image: {}".format(str(e))


def error_card_html(msg):
    # type: (str) -> str
    return (
        "<div style='background:#161b22;border:1px solid #f85149;border-radius:10px;"
        "padding:20px;color:#f85149;font-size:0.88em;line-height:1.6;'>"
        "<b>Error</b><br>{}</div>"
    ).format(msg)


# ── Dataset index for example shuffle ─────────────────────────────────────────

def build_dataset_index():
    # type: () -> Dict[str, List[str]]
    csv = PROJECT_ROOT / "data/live/splits/test.csv"
    if not csv.exists():
        return {}
    try:
        df = pd.read_csv(csv)
        return {lbl: df[df["label"] == lbl]["image_path"].tolist()
                for lbl in ["low", "medium", "high"]}
    except Exception:
        return {}

DATASET_INDEX = build_dataset_index()


def shuffle_examples():
    # type: () -> tuple
    """Return (gallery_list, state_list) — one image per class (low, medium, high)."""
    from PIL import Image as PILImage
    imgs = []
    for lbl in ["low", "medium", "high"]:
        paths = DATASET_INDEX.get(lbl, [])
        if paths:
            try:
                imgs.append(PILImage.open(PROJECT_ROOT / random.choice(paths)).convert("RGB"))
            except Exception:
                imgs.append(None)
        else:
            imgs.append(None)
    gallery = [img for img in imgs if img is not None]
    return gallery, imgs   # gallery → gr.Gallery, imgs → gr.State


def shuffle_sequence():
    # type: () -> tuple
    """Pick 3 LOW + 3 MEDIUM + 3 HIGH from the SAME camera — shows real congestion progression."""
    from PIL import Image as PILImage
    from pathlib import Path as _Path

    # Group paths by camera_id then label
    # Expected path format: data/live/raw/{camera_id}/{window}/frame_xx.jpg
    cam_index = {}  # type: dict
    for lbl, paths in DATASET_INDEX.items():
        for p in paths:
            parts = _Path(p).parts
            try:
                cam_id = parts[3]   # data / live / raw / {camera_id} / ...
            except IndexError:
                continue
            if cam_id not in cam_index:
                cam_index[cam_id] = {"low": [], "medium": [], "high": []}
            cam_index[cam_id][lbl].append(p)

    # Keep only cameras that have all 3 classes
    valid = [cam for cam, lbls in cam_index.items()
             if all(len(lbls[l]) >= 1 for l in ["low", "medium", "high"])]

    if not valid:
        # Fallback: cross-camera if no single camera has all 3
        imgs = []
        for lbl in ["low", "medium", "high"]:
            paths = DATASET_INDEX.get(lbl, [])
            for p in random.sample(paths, min(3, len(paths))):
                try:
                    imgs.append(PILImage.open(PROJECT_ROOT / p).convert("RGB"))
                except Exception:
                    pass
        return imgs, imgs

    cam = random.choice(valid)
    imgs = []
    for lbl in ["low", "medium", "high"]:
        picks = random.sample(cam_index[cam][lbl], min(3, len(cam_index[cam][lbl])))
        for p in picks:
            try:
                imgs.append(PILImage.open(PROJECT_ROOT / p).convert("RGB"))
            except Exception:
                pass
    return imgs, imgs   # gallery → gr.Gallery, state → gr.State


# ── Build App ──────────────────────────────────────────────────────────────────

def build_app():
    models, gradcams = load_all_models()
    transform        = get_transform()
    image_size       = tuple(CFG["frame_extraction"]["image_size"])

    available    = list(models.keys())
    default_model = "efficientnet_b0" if "efficientnet_b0" in available else (available[0] if available else "")
    dropdown_choices = [(MODEL_DISPLAY[n], n) for n in available] + [
        ("Ensemble + TTA", "ensemble_tta"),
        ("EfficientNet-B0 + TTA ⚡", "efficientnet_tta"),
    ]

    # ── Classify closures ──────────────────────────────────────────────────────

    def classify(image, model_name):
        if image is None:
            empty = "<p style='color:#6e7681;padding:20px;'>Upload an image to begin.</p>"
            return empty, empty, empty
        label, probs, cam = run_inference(image, model_name, models, gradcams, transform)
        if label is None:
            err = error_card_html("Model '{}' is not loaded.".format(model_name))
            return err, err, err
        yolo_img, vcount = yolo_detection_image(image)
        visuals = two_image_html(
            cam,      "Model attention (GradCAM)",
            yolo_img, "Vehicle detection — counted: {}".format(vcount),
        )
        return prediction_html(label, probs), traffic_light_html(label), visuals

    def use_example(img, model_name):
        if img is None:
            return None, "", "", ""
        pred, signal, visuals = classify(img, model_name)
        return img, pred, signal, visuals

    # ── Compare closures ───────────────────────────────────────────────────────

    def compare_all(image):
        if image is None:
            return "<p style='color:#6e7681;padding:20px;'>Upload an image to compare all models.</p>"
        results      = []
        gradcam_imgs = {}
        img_rgb = image.convert("RGB")
        for name in ["baseline_cnn", "mobilenet_v2", "resnet50", "efficientnet_b0"]:
            if name not in models:
                continue
            label, probs, _ = run_inference(img_rgb, name, models, gradcams, transform)
            if label is None:
                continue
            results.append((name, label, probs))
            # GradCAM per model
            try:
                idx    = CLASS_NAMES.index(label)
                tensor = transform(img_rgb).unsqueeze(0)
                cam_arr = gradcams[name].generate(tensor, idx)
                gradcam_imgs[name] = gradcams[name].overlay(cam_arr, img_rgb) if cam_arr is not None else None
            except Exception:
                gradcam_imgs[name] = None

        if not results:
            return error_card_html("No models loaded.")

        all_probs  = np.array([r[2] for r in results])
        ens_probs  = np.mean(all_probs, axis=0)
        return (
            compare_html(results)
            + compare_ensemble_html(results, ens_probs)
            + compare_gradcam_grid(results, gradcam_imgs)
        )

    # ── Live Feed closures ─────────────────────────────────────────────────────

    def fetch_and_classify(camera_display):
        camera_id = CAMERA_DISPLAY_TO_ID.get(camera_display)
        if not camera_id:
            return None, error_card_html("Unknown camera selection."), "", ""

        img, err = fetch_camera_image(camera_id)
        if err:
            return None, error_card_html(err), "", ""

        # EfficientNet-B0 + TTA (best single-model)
        if "efficientnet_b0" in models:
            probs = run_single_tta(img.convert("RGB"), models["efficientnet_b0"], image_size)
        else:
            probs = run_ensemble_tta(img.convert("RGB"), models, image_size)

        idx   = int(probs.argmax())
        label = CLASS_NAMES[idx]

        yolo_img, vcount = yolo_detection_image(img, camera_id)
        display_img = yolo_img if yolo_img is not None else img

        ts = time.strftime("%H:%M:%S")
        status_html = (
            "<div style='font-size:0.8em;color:#8b949e;padding:6px 0;'>"
            "<span class='live-dot'></span>"
            "Live — <b style='color:#e6edf3;'>{cam}</b> — fetched at {ts}"
            " · {v} vehicles counted</div>"
        ).format(cam=camera_display, ts=ts, v=vcount)

        return display_img, status_html, prediction_html(label, probs), traffic_light_html(label)

    # ── Robustness closures ────────────────────────────────────────────────────

    def apply_preview(image, brightness, contrast, blur_sigma):
        """Fast image-only preview — no inference. Called on slider release."""
        if image is None:
            return None
        from torchvision.transforms import functional as TF
        import PIL.ImageFilter
        mod = image.convert("RGB").copy()
        if abs(brightness - 1.0) > 0.01:
            mod = TF.adjust_brightness(mod, brightness)
        if abs(contrast - 1.0) > 0.01:
            mod = TF.adjust_contrast(mod, contrast)
        if blur_sigma > 0.05:
            mod = mod.filter(PIL.ImageFilter.GaussianBlur(radius=blur_sigma))
        return mod

    def robustness_evaluate(image, brightness, contrast, blur_sigma):
        if image is None:
            return None, None, "<p style='color:#6e7681;padding:20px;'>Upload an image first.</p>"

        from torchvision.transforms import functional as TF
        import PIL.ImageFilter

        orig = image.convert("RGB")
        mod  = orig.copy()
        if abs(brightness - 1.0) > 0.01:
            mod = TF.adjust_brightness(mod, brightness)
        if abs(contrast - 1.0) > 0.01:
            mod = TF.adjust_contrast(mod, contrast)
        if blur_sigma > 0.05:
            mod = mod.filter(PIL.ImageFilter.GaussianBlur(radius=blur_sigma))

        best = "efficientnet_b0" if "efficientnet_b0" in models else (available[0] if available else None)
        if best is None:
            return orig, mod, error_card_html("No models loaded.")

        orig_label, orig_probs, _ = run_inference(orig, best, models, gradcams, transform)
        mod_label,  mod_probs,  _ = run_inference(mod,  best, models, gradcams, transform)

        if orig_label is None or mod_label is None:
            return orig, mod, error_card_html("Inference failed.")

        same        = (orig_label == mod_label)
        result_color = "#3fb950" if same else "#f85149"
        result_text  = "Prediction stable" if same else "Prediction changed!"

        orig_conf = orig_probs[CLASS_NAMES.index(orig_label)] * 100
        mod_conf  = mod_probs[CLASS_NAMES.index(mod_label)]   * 100

        result_html = (
            "<div style='background:#161b22;border:1px solid {rc};border-radius:12px;"
            "padding:20px;margin-top:12px;'>"
            "<div style='font-size:1.05em;font-weight:700;color:{rc};margin-bottom:14px;'>"
            "{rt}</div>"
            "<div style='display:flex;gap:24px;flex-wrap:wrap;align-items:center;'>"
            "<div>"
            "<div style='font-size:0.75em;color:#8b949e;margin-bottom:4px;'>Original</div>"
            "<div style='font-weight:700;color:{oc};font-size:1.1em;'>{oe} {ol} ({oc_conf:.1f}%)</div>"
            "</div>"
            "<div style='color:#30363d;font-size:1.8em;'>→</div>"
            "<div>"
            "<div style='font-size:0.75em;color:#8b949e;margin-bottom:4px;'>Modified</div>"
            "<div style='font-weight:700;color:{mc};font-size:1.1em;'>{me} {ml} ({mc_conf:.1f}%)</div>"
            "</div></div>"
            "<div style='font-size:0.76em;color:#6e7681;margin-top:12px;"
            "border-top:1px solid #21262d;padding-top:8px;'>"
            "Model: {model} · Brightness {br:.2f}× · Contrast {ct:.2f}× · Blur σ={bl:.1f}"
            "</div></div>"
        ).format(
            rc=result_color, rt=result_text,
            oc=CLASS_COLORS[orig_label], oe=CLASS_EMOJIS[orig_label],
            ol=orig_label.upper(), oc_conf=orig_conf,
            mc=CLASS_COLORS[mod_label], me=CLASS_EMOJIS[mod_label],
            ml=mod_label.upper(), mc_conf=mod_conf,
            model=MODEL_DISPLAY.get(best, best).replace(" ★", ""),
            br=brightness, ct=contrast, bl=blur_sigma,
        )
        return orig, mod, result_html

    # ── Sequence closures ──────────────────────────────────────────────────────

    def analyse_pil_sequence(imgs):
        """Same as analyse_sequence but accepts a list of PIL images (from pre-built shuffle)."""
        if not imgs:
            return "<p style='color:#6e7681;padding:20px;'>No sequence loaded — click Shuffle first.</p>"
        best = "efficientnet_b0" if "efficientnet_b0" in models else (available[0] if available else None)
        if best is None:
            return error_card_html("No models loaded.")
        labels, confs = [], []
        for img in imgs[:15]:
            if img is None:
                continue
            lbl, probs, _ = run_inference(img, best, models, gradcams, transform)
            if lbl is None:
                lbl = "low"; probs = np.array([1.0, 0.0, 0.0])
            labels.append(lbl)
            confs.append(float(np.max(probs)))
        if not labels:
            return error_card_html("Could not load any images.")
        chart_html = sequence_timeline_html(labels, confs)
        cnt = Counter(labels)
        dominant = cnt.most_common(1)[0][0]
        dom_color = CLASS_COLORS[dominant]
        transitions = sum(1 for a, b in zip(labels, labels[1:]) if a != b)
        summary_html = (
            "<div style='background:#161b22;border:1px solid #30363d;border-radius:12px;"
            "padding:16px;margin-top:12px;display:flex;gap:28px;flex-wrap:wrap;'>"
            "<div><div style='font-size:0.74em;color:#8b949e;margin-bottom:4px;'>Dominant Class</div>"
            "<div style='font-weight:700;color:{dc};'>{de} {du}</div></div>"
            "<div><div style='font-size:0.74em;color:#8b949e;margin-bottom:4px;'>Transitions</div>"
            "<div style='font-weight:700;color:#e6edf3;'>{tr}</div></div>"
            "<div><div style='font-size:0.74em;color:#8b949e;margin-bottom:4px;'>Frames Analysed</div>"
            "<div style='font-weight:700;color:#e6edf3;'>{nf}</div></div>"
            "</div>"
        ).format(dc=dom_color, de=CLASS_EMOJIS[dominant], du=dominant.upper(),
                 tr=transitions, nf=len(labels))
        rows = ""
        for i, (lbl, conf) in enumerate(zip(labels, confs)):
            c = CLASS_COLORS[lbl]
            rows += (
                "<tr style='border-bottom:1px solid #21262d;'>"
                "<td style='padding:8px 12px;color:#8b949e;text-align:center;'>{i}</td>"
                "<td style='padding:8px 12px;color:{c};font-weight:600;'>{emoji} {lbl}</td>"
                "<td style='padding:8px 12px;color:#e6edf3;text-align:center;'>{conf:.1f}%</td>"
                "</tr>"
            ).format(i=i + 1, c=c, emoji=CLASS_EMOJIS[lbl], lbl=lbl.upper(), conf=conf * 100)
        table_html = (
            "<div style='background:#161b22;border:1px solid #30363d;border-radius:12px;"
            "overflow:hidden;margin-top:12px;'>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.84em;'>"
            "<thead><tr style='background:#0d1117;border-bottom:2px solid #30363d;'>"
            "<th style='padding:8px 12px;color:#8b949e;text-align:center;'>Frame</th>"
            "<th style='padding:8px 12px;color:#8b949e;text-align:left;'>Label</th>"
            "<th style='padding:8px 12px;color:#8b949e;text-align:center;'>Confidence</th>"
            "</tr></thead><tbody>{}</tbody></table></div>"
        ).format(rows)
        return chart_html + summary_html + table_html

    def analyse_sequence(files):
        if not files:
            return "<p style='color:#6e7681;padding:20px;'>Upload images first.</p>"

        best = "efficientnet_b0" if "efficientnet_b0" in models else (available[0] if available else None)
        if best is None:
            return error_card_html("No models loaded.")

        file_list = files[:15]
        labels, confs = [], []
        for f in file_list:
            path = f.name if hasattr(f, "name") else str(f)
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                labels.append("low"); confs.append(0.0)
                continue
            lbl, probs, _ = run_inference(img, best, models, gradcams, transform)
            if lbl is None:
                lbl = "low"; probs = np.array([1.0, 0.0, 0.0])
            labels.append(lbl)
            confs.append(float(np.max(probs)))

        chart_html = sequence_timeline_html(labels, confs)

        cnt       = Counter(labels)
        dominant  = cnt.most_common(1)[0][0]
        dom_color = CLASS_COLORS[dominant]
        transitions = sum(1 for a, b in zip(labels, labels[1:]) if a != b)

        summary_html = (
            "<div style='background:#161b22;border:1px solid #30363d;border-radius:12px;"
            "padding:16px;margin-top:12px;display:flex;gap:28px;flex-wrap:wrap;'>"
            "<div><div style='font-size:0.74em;color:#8b949e;margin-bottom:4px;'>Dominant Class</div>"
            "<div style='font-weight:700;color:{dc};'>{de} {du}</div></div>"
            "<div><div style='font-size:0.74em;color:#8b949e;margin-bottom:4px;'>Transitions</div>"
            "<div style='font-weight:700;color:#e6edf3;'>{tr}</div></div>"
            "<div><div style='font-size:0.74em;color:#8b949e;margin-bottom:4px;'>Frames Analysed</div>"
            "<div style='font-weight:700;color:#e6edf3;'>{nf}</div></div>"
            "</div>"
        ).format(dc=dom_color, de=CLASS_EMOJIS[dominant], du=dominant.upper(),
                 tr=transitions, nf=len(labels))

        rows = ""
        for i, (lbl, conf) in enumerate(zip(labels, confs)):
            c = CLASS_COLORS[lbl]
            rows += (
                "<tr style='border-bottom:1px solid #21262d;'>"
                "<td style='padding:8px 12px;color:#8b949e;text-align:center;'>{i}</td>"
                "<td style='padding:8px 12px;color:{c};font-weight:600;'>{emoji} {lbl}</td>"
                "<td style='padding:8px 12px;color:#e6edf3;text-align:center;'>{conf:.1f}%</td>"
                "</tr>"
            ).format(i=i, c=c, emoji=CLASS_EMOJIS[lbl], lbl=lbl.upper(), conf=conf * 100)

        table_html = (
            "<div style='background:#161b22;border:1px solid #30363d;border-radius:12px;"
            "overflow:hidden;margin-top:12px;'>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.84em;'>"
            "<thead><tr style='background:#0d1117;border-bottom:2px solid #30363d;'>"
            "<th style='padding:8px 12px;color:#8b949e;text-align:center;'>Frame</th>"
            "<th style='padding:8px 12px;color:#8b949e;text-align:left;'>Label</th>"
            "<th style='padding:8px 12px;color:#8b949e;text-align:center;'>Confidence</th>"
            "</tr></thead><tbody>{}</tbody></table></div>"
        ).format(rows)

        return chart_html + summary_html + table_html

    # ── Layout ─────────────────────────────────────────────────────────────────

    with gr.Blocks(css=CSS, title="Traffic Congestion Classifier") as demo:
        gr.HTML(
            "<div style='padding:20px 0 8px;'>"
            "<div style='font-size:1.5em;font-weight:800;color:#e6edf3;'>Traffic Congestion Classifier</div>"
            "<div style='font-size:0.85em;color:#6e7681;margin-top:4px;'>"
            "Visual congestion classification from intersection footage — UTS 42028 Deep Learning · Road Rangers"
            "</div></div>"
        )

        with gr.Tabs():

            # ── Tab 1: Live Feed ───────────────────────────────────────────────
            with gr.Tab("  Live Feed  "):
                gr.HTML("<div style='padding:4px 0 12px;font-size:0.84em;color:#6e7681;'>"
                        "Fetch real-time frames directly from TfNSW traffic cameras and classify congestion live. "
                        "Requires <code>TFNSW_API_KEY</code> in <code>.env</code>.</div>")
                with gr.Row():
                    with gr.Column(scale=1):
                        cam_drop  = gr.Dropdown(
                            choices=list(CAMERA_IDS.values()),
                            value=list(CAMERA_IDS.values())[0],
                            label="Camera Location",
                        )
                        fetch_btn = gr.Button("Fetch & Classify", variant="primary")
                        live_status = gr.HTML()
                        live_pred   = gr.HTML()
                        live_signal = gr.HTML()
                    with gr.Column(scale=1):
                        live_img = gr.Image(type="pil", label="Camera Frame (with YOLO detection)",
                                            interactive=False)

                fetch_btn.click(
                    fetch_and_classify,
                    inputs=[cam_drop],
                    outputs=[live_img, live_status, live_pred, live_signal],
                )

                # Auto-refresh (Gradio 4.x only)
                try:
                    auto_toggle = gr.Checkbox(label="Auto-refresh every 60 s", value=False)
                    timer = gr.Timer(60)
                    def _auto_fetch(cam, do_refresh):
                        if not do_refresh:
                            return gr.update(), gr.update(), gr.update(), gr.update()
                        return fetch_and_classify(cam)
                    timer.tick(
                        _auto_fetch,
                        inputs=[cam_drop, auto_toggle],
                        outputs=[live_img, live_status, live_pred, live_signal],
                    )
                except AttributeError:
                    gr.HTML("<div style='font-size:0.76em;color:#484f58;margin-top:8px;'>"
                            "Auto-refresh requires Gradio 4.x — manually click Fetch to update.</div>")

            # ── Tab 2: Classify ────────────────────────────────────────────────
            with gr.Tab("  Classify  "):
                with gr.Row():
                    with gr.Column(scale=1):
                        img_input   = gr.Image(type="pil", label="Input Frame")
                        model_drop  = gr.Dropdown(
                            choices=dropdown_choices,
                            value=default_model,
                            label="Model",
                        )
                        classify_btn = gr.Button("Classify", variant="primary")
                        gr.HTML("<div style='font-size:0.76em;color:#8b949e;margin-top:10px;margin-bottom:3px;'>"
                                "Example frames — click any image to use it</div>")
                        cls_gallery = gr.Gallery(
                            rows=1, columns=3, height=130,
                            show_label=False, elem_classes=["example-gallery"],
                        )
                        cls_state = gr.State([None, None, None])
                        cls_shuf  = gr.Button("🔀 Shuffle", variant="secondary")
                    with gr.Column(scale=1):
                        pred_out    = gr.HTML(value="<p style='color:#6e7681;padding:20px;'>Upload an image to begin.</p>")
                        signal_out  = gr.HTML()
                        visuals_out = gr.HTML()

                classify_btn.click(
                    classify,
                    inputs=[img_input, model_drop],
                    outputs=[pred_out, signal_out, visuals_out],
                )
                img_input.change(
                    classify,
                    inputs=[img_input, model_drop],
                    outputs=[pred_out, signal_out, visuals_out],
                )

            # ── Tab 3: Compare ────────────────────────────────────────────────
            with gr.Tab("  Compare  "):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML("<div style='font-size:0.83em;color:#6e7681;margin-bottom:10px;'>"
                                "Compare all 4 models simultaneously — ensemble analysis and GradCAM.</div>")
                        cmp_input = gr.Image(type="pil", label="Input Frame")
                        cmp_btn   = gr.Button("Compare All Models", variant="primary")
                        gr.HTML("<div style='font-size:0.76em;color:#8b949e;margin-top:10px;margin-bottom:3px;'>"
                                "Example frames — click any image to use it</div>")
                        cmp_gallery = gr.Gallery(
                            rows=1, columns=3, height=130,
                            show_label=False, elem_classes=["example-gallery"],
                        )
                        cmp_state = gr.State([None, None, None])
                        cmp_shuf  = gr.Button("🔀 Shuffle", variant="secondary")
                    with gr.Column(scale=2):
                        cmp_out = gr.HTML(
                            value="<div style='color:#6e7681;padding:40px;text-align:center;"
                                  "font-size:0.88em;'>Upload an image and click Compare to see results.</div>"
                        )
                cmp_btn.click(compare_all, inputs=[cmp_input], outputs=[cmp_out])

            # ── Tab 4: Robustness ─────────────────────────────────────────────
            with gr.Tab("  Robustness  "):
                gr.HTML("<div style='padding:4px 0 12px;font-size:0.84em;color:#6e7681;'>"
                        "Test model stability under image degradation. Adjust parameters then click Evaluate. "
                        "Uses EfficientNet-B0 (best single model).</div>")
                with gr.Row():
                    with gr.Column(scale=1):
                        rob_input  = gr.Image(type="pil", label="Source Image")
                        brightness = gr.Slider(0.3, 2.0, value=1.0, step=0.05, label="Brightness")
                        contrast   = gr.Slider(0.3, 2.0, value=1.0, step=0.05, label="Contrast")
                        blur_sigma = gr.Slider(0.0, 3.0, value=0.0, step=0.1,  label="Blur Sigma")
                        eval_btn   = gr.Button("Evaluate", variant="primary")
                        gr.HTML("<div style='font-size:0.76em;color:#8b949e;margin-top:10px;margin-bottom:3px;'>"
                                "Example frames — click any image to use it</div>")
                        rob_gallery = gr.Gallery(
                            rows=1, columns=3, height=130,
                            show_label=False, elem_classes=["example-gallery"],
                        )
                        rob_state = gr.State([None, None, None])
                        rob_shuf  = gr.Button("🔀 Shuffle", variant="secondary")
                    with gr.Column(scale=1):
                        with gr.Row():
                            orig_out = gr.Image(type="pil", label="Original",  interactive=False)
                            mod_out  = gr.Image(type="pil", label="Modified",  interactive=False)
                        rob_result = gr.HTML()

                eval_btn.click(
                    robustness_evaluate,
                    inputs=[rob_input, brightness, contrast, blur_sigma],
                    outputs=[orig_out, mod_out, rob_result],
                )
                # Live preview: update Modified image instantly on slider release
                _preview_inputs  = [rob_input, brightness, contrast, blur_sigma]
                _preview_outputs = [mod_out]
                brightness.release(apply_preview, inputs=_preview_inputs, outputs=_preview_outputs)
                contrast.release(  apply_preview, inputs=_preview_inputs, outputs=_preview_outputs)
                blur_sigma.release(apply_preview, inputs=_preview_inputs, outputs=_preview_outputs)
                rob_input.change(  apply_preview, inputs=_preview_inputs, outputs=_preview_outputs)

            # ── Tab 5: Sequence ───────────────────────────────────────────────
            with gr.Tab("  Sequence  "):
                gr.HTML("<div style='padding:4px 0 12px;font-size:0.84em;color:#6e7681;'>"
                        "Analyse congestion trends across a sequence of frames. "
                        "Uses EfficientNet-B0 for per-frame inference.</div>")
                with gr.Row():
                    with gr.Column(scale=1):
                        # Pre-built sequence section
                        gr.HTML("<div style='font-size:0.82em;font-weight:600;color:#e6edf3;"
                                "margin-bottom:6px;'>Pre-built Sequence</div>"
                                "<div style='font-size:0.76em;color:#8b949e;margin-bottom:8px;'>"
                                "3 × Low → 3 × Medium → 3 × High from the <b style='color:#e6edf3;'>same camera</b> "
                                "— shows real congestion progression at one location. Shuffle for a new set.</div>")
                        seq_preset_gallery = gr.Gallery(
                            rows=3, columns=3, height=320,
                            show_label=False, elem_classes=["example-gallery"],
                        )
                        seq_preset_state = gr.State([])
                        with gr.Row():
                            seq_preset_shuf = gr.Button("🔀 Shuffle", variant="secondary")
                            seq_preset_btn  = gr.Button("▶ Analyse Pre-built", variant="primary")

                        # Divider
                        gr.HTML("<div style='border-top:1px solid #21262d;margin:16px 0 12px;'></div>"
                                "<div style='font-size:0.82em;font-weight:600;color:#e6edf3;"
                                "margin-bottom:6px;'>Custom Upload</div>")
                        seq_files = gr.Files(
                            label="Upload frames (up to 15 images)",
                            file_types=["image"],
                            file_count="multiple",
                        )
                        seq_btn = gr.Button("Analyse Uploaded Sequence", variant="secondary")

                    with gr.Column(scale=2):
                        seq_out = gr.HTML(
                            value="<div style='color:#6e7681;padding:40px;text-align:center;"
                                  "font-size:0.88em;'>Shuffle a pre-built sequence or upload frames, "
                                  "then click Analyse.</div>"
                        )

                seq_preset_shuf.click(shuffle_sequence, outputs=[seq_preset_gallery, seq_preset_state])
                seq_preset_btn.click(analyse_pil_sequence, inputs=[seq_preset_state], outputs=[seq_out])
                seq_btn.click(analyse_sequence, inputs=[seq_files], outputs=[seq_out])
                demo.load(shuffle_sequence, outputs=[seq_preset_gallery, seq_preset_state])

            # ── Tab 6: Methodology ────────────────────────────────────────────
            with gr.Tab("  Methodology  "):
                gr.HTML(
                    "<div style='padding:4px 0 12px;font-size:0.84em;color:#6e7681;'>"
                    "End-to-end pipeline: from TfNSW camera collection to trained classifier.</div>"
                    + methodology_html()
                )

            # ── Tab 7: Dataset & Results ──────────────────────────────────────
            with gr.Tab("  Dataset & Results  "):
                gr.HTML(
                    "<div style='padding:4px 0 12px;font-size:0.84em;color:#6e7681;'>"
                    "Live NSW camera dataset — 10 cameras, 3 collection sessions, "
                    "test set is fully held-out Wollongong region (unseen during training).</div>"
                    + dataset_html()
                )

        # ── Shuffle gallery wiring ─────────────────────────────────────────
        def _use_selected(state, evt: gr.SelectData):
            try:
                return state[evt.index]
            except Exception:
                return None

        cls_shuf.click(shuffle_examples, outputs=[cls_gallery, cls_state])
        cls_gallery.select(_use_selected, inputs=[cls_state], outputs=[img_input])

        cmp_shuf.click(shuffle_examples, outputs=[cmp_gallery, cmp_state])
        cmp_gallery.select(_use_selected, inputs=[cmp_state], outputs=[cmp_input])

        rob_shuf.click(shuffle_examples, outputs=[rob_gallery, rob_state])
        rob_gallery.select(_use_selected, inputs=[rob_state], outputs=[rob_input])

        demo.load(shuffle_examples, outputs=[cls_gallery, cls_state])
        demo.load(shuffle_examples, outputs=[cmp_gallery, cmp_state])
        demo.load(shuffle_examples, outputs=[rob_gallery, rob_state])

    return demo


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",   type=int, default=7860)
    parser.add_argument("--share",  action="store_true")
    parser.add_argument("--debug",  action="store_true")
    args = parser.parse_args()

    demo = build_app()
    demo.launch(server_port=args.port, share=args.share, debug=args.debug)


if __name__ == "__main__":
    main()
