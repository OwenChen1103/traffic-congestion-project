"""
app.py
------
Traffic Congestion Classification — Professional Demo GUI

Features:
  - Single-model inference with GradCAM visualisation
  - In-UI model switching (dropdown)
  - Four-model simultaneous comparison
  - Labeling methodology explanation
  - Dataset statistics and results table
  - Traffic light signal timing visualisation

Run:
  python src/gui/app.py
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config.settings import CFG, PROJECT_ROOT
from src.models.baseline_cnn import BaselineCNN
from src.models.transfer_models import build_model

import gradio as gr


# ── Constants ──────────────────────────────────────────────────────────────────

CLASS_NAMES  = ["low", "medium", "high"]
CLASS_LABELS = ["Low", "Medium", "High"]
CLASS_COLORS = {"low": "#3fb950", "medium": "#d29922", "high": "#f85149"}
CLASS_EMOJIS = {"low": "🟢", "medium": "🟡", "high": "🔴"}

MODEL_DISPLAY = {
    "baseline_cnn":    "Baseline CNN",
    "mobilenet_v2":    "MobileNetV2 ★",
    "resnet50":        "ResNet-50",
    "efficientnet_b0": "EfficientNet-B0",
    "ensemble_tta":    "Ensemble + TTA ⚡",
}

MODEL_PARAMS = {
    "baseline_cnn":    "619K",
    "mobilenet_v2":    "2.2M",
    "resnet50":        "23.5M",
    "efficientnet_b0": "4.0M",
    "ensemble_tta":    "4 models",
}

TEST_RESULTS = {
    "baseline_cnn":    {"test_acc": 0.7241, "macro_f1": 0.7210, "low_f1": 0.618, "med_f1": 0.707, "high_f1": 0.837},
    "mobilenet_v2":    {"test_acc": 0.7874, "macro_f1": 0.7659, "low_f1": 0.650, "med_f1": 0.800, "high_f1": 0.847},
    "resnet50":        {"test_acc": 0.7615, "macro_f1": 0.7380, "low_f1": 0.598, "med_f1": 0.774, "high_f1": 0.842},
    "efficientnet_b0": {"test_acc": 0.7759, "macro_f1": 0.7503, "low_f1": 0.580, "med_f1": 0.793, "high_f1": 0.878},
    "ensemble_tta":    {"test_acc": 0.8218, "macro_f1": 0.7992, "low_f1": 0.667, "med_f1": 0.839, "high_f1": 0.893},
}

SIGNAL_RULES = {
    "low":    {"action": "Maintain current cycle",        "delta": 0,   "detail": "Traffic is flowing freely. No signal adjustment required."},
    "medium": {"action": "Extend green phase by ~10 s",   "delta": +10, "detail": "Moderate congestion detected. Extend current green phase to clear queued vehicles."},
    "high":   {"action": "Extend green phase by ~20 s",   "delta": +20, "detail": "Heavy congestion detected. Significant green extension recommended. Consider overflow protocol if queues persist."},
}

CSS = """
/* ── Base ── */
body, .gradio-container {
    background: #0d1117 !important;
    color: #e6edf3 !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; }
footer { display: none !important; }

/* ── Tabs ── */
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

/* ── Inputs ── */
.gr-button-primary {
    background: linear-gradient(135deg, #00d4aa, #0099cc) !important;
    border: none !important;
    color: #0d1117 !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    transition: opacity 0.2s !important;
}
.gr-button-primary:hover { opacity: 0.85 !important; }

select, .gr-dropdown {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 8px !important;
}

/* ── Upload area ── */
.gr-image { border: 1px solid #30363d !important; border-radius: 12px !important; background: #161b22 !important; }

/* ── Label component ── */
.gr-label { background: #161b22 !important; border: 1px solid #30363d !important; border-radius: 12px !important; }

/* ── Secondary buttons (Use) ── */
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
"""


# ── Model loading ──────────────────────────────────────────────────────────────

def get_device():
    # type: () -> torch.device
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
            models[name]  = m
            gradcams[name] = GradCAM(m, name)
            print("[GUI] Loaded {}".format(name))
        else:
            print("[GUI] Skipped {} (no checkpoint)".format(name))
    return models, gradcams


# ── Transforms ────────────────────────────────────────────────────────────────

def get_transform():
    # type: () -> transforms.Compose
    size = tuple(CFG["frame_extraction"]["image_size"])
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ── GradCAM ───────────────────────────────────────────────────────────────────

class GradCAM:
    def __init__(self, model, model_name):
        # type: (nn.Module, str) -> None
        self.model       = model
        self.model_name  = model_name
        self.gradients   = None
        self.activations = None
        self._handles    = []
        self._register()

    def _target_layer(self):
        # type: () -> Optional[nn.Module]
        n = self.model_name
        # Use higher-resolution intermediate layers (14×14) for better spatial precision
        # instead of the final 7×7 feature maps
        if n == "baseline_cnn":    return self.model.features[-1]   # 28×28
        if n == "mobilenet_v2":    return self.model.features[13]   # 14×14
        if n == "resnet50":        return self.model.layer3[-1]     # 14×14
        if n == "efficientnet_b0": return self.model.features[5]    # 14×14
        last = None
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d): last = m
        return last

    def _register(self):
        layer = self._target_layer()
        if layer is None: return
        def fwd(mod, inp, out): self.activations = out.detach()
        def bwd(mod, gi, go):   self.gradients   = go[0].detach()
        self._handles.append(layer.register_forward_hook(fwd))
        self._handles.append(layer.register_backward_hook(bwd))

    def generate(self, tensor, class_idx):
        # type: (torch.Tensor, int) -> np.ndarray
        self.model.zero_grad()
        out = self.model(tensor)
        out[0, class_idx].backward()
        if self.gradients is None or self.activations is None:
            return np.zeros((tensor.shape[2], tensor.shape[3]))
        w   = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.clamp((w * self.activations).sum(dim=1).squeeze(0), min=0)
        cam = cam.numpy()
        if cam.max() > 0: cam = cam / cam.max()
        return cam


def overlay_cam(pil_img, cam):
    # type: (Image.Image, np.ndarray) -> Image.Image
    import cv2
    orig = np.array(pil_img.convert("RGB"))
    h, w = orig.shape[:2]
    cam_r = cv2.resize(cam.astype(np.float32), (w, h))
    heat  = cv2.applyColorMap((cam_r * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat  = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    blend = (0.45 * heat + 0.55 * orig).astype(np.uint8)
    return Image.fromarray(blend)


# ── Inference ──────────────────────────────────────────────────────────────────

def run_ensemble_tta(image_pil, models, image_size):
    # type: (Image.Image, Dict, tuple) -> np.ndarray
    """Average softmax probs across all models × 5 TTA transforms."""
    from src.datasets.congestion_dataset import get_tta_transforms
    tta_tfms = get_tta_transforms(image_size)
    all_probs = []
    for model in models.values():
        for tfm in tta_tfms:
            tensor = tfm(image_pil).unsqueeze(0)
            with torch.no_grad():
                p = torch.softmax(model(tensor), dim=1).squeeze(0).numpy()
            all_probs.append(p)
    return np.mean(all_probs, axis=0)


def run_inference(image_pil, model_name, models, gradcams, transform):
    # type: (Image.Image, str, Dict, Dict, transforms.Compose) -> Tuple
    if image_pil is None:
        return None, None, None
    image_pil = image_pil.convert("RGB")

    # Ensemble + TTA mode
    if model_name == "ensemble_tta":
        size = tuple(CFG["frame_extraction"]["image_size"])
        probs = run_ensemble_tta(image_pil, models, size)
        idx   = int(np.argmax(probs))
        label = CLASS_NAMES[idx]
        # GradCAM on best single model (mobilenet_v2) for visualisation
        cam_img = None
        best = "mobilenet_v2" if "mobilenet_v2" in models else list(models.keys())[0]
        try:
            t2  = transform(image_pil).unsqueeze(0)
            cam = gradcams[best].generate(t2, idx)
            cam_img = overlay_cam(image_pil, cam)
        except Exception:
            pass
        return label, probs, cam_img

    if model_name not in models:
        return None, None, None

    tensor = transform(image_pil).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(models[model_name](tensor), dim=1).squeeze(0).numpy()

    idx   = int(np.argmax(probs))
    label = CLASS_NAMES[idx]

    cam_img = None
    try:
        t2  = transform(image_pil).unsqueeze(0)
        cam = gradcams[model_name].generate(t2, idx)
        cam_img = overlay_cam(image_pil, cam)
    except Exception:
        pass

    return label, probs, cam_img


# ── HTML builders ──────────────────────────────────────────────────────────────

def prediction_html(label, probs):
    # type: (str, np.ndarray) -> str
    color = CLASS_COLORS[label]
    emoji = CLASS_EMOJIS[label]
    bars  = ""
    for i, (n, p) in enumerate(zip(CLASS_NAMES, probs)):
        c = CLASS_COLORS[n]
        bars += """
        <div style="margin:6px 0;">
          <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
            <span style="color:{c};font-weight:600;font-size:0.85em;">{emoji} {label}</span>
            <span style="color:#8b949e;font-size:0.85em;">{pct:.1f}%</span>
          </div>
          <div style="background:#21262d;border-radius:4px;height:8px;overflow:hidden;">
            <div style="width:{pct:.1f}%;background:{c};height:100%;border-radius:4px;
                        transition:width 0.4s ease;"></div>
          </div>
        </div>""".format(c=c, emoji=CLASS_EMOJIS[n], label=n.capitalize(), pct=p * 100)

    return """
<div style="background:#161b22;border:1px solid #30363d;border-radius:12px;padding:20px;">
  <div style="font-size:0.7em;color:#8b949e;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px;">
    Prediction
  </div>
  <div style="font-size:2em;font-weight:800;color:{color};margin-bottom:16px;letter-spacing:0.02em;">
    {emoji} {label_upper}
  </div>
  <div style="font-size:0.75em;color:#8b949e;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:10px;">
    Confidence
  </div>
  {bars}
</div>""".format(color=color, emoji=emoji, label_upper=label.upper(), bars=bars)


def traffic_light_html(label):
    # type: (str) -> str
    rule   = SIGNAL_RULES[label]
    color  = CLASS_COLORS[label]
    active = {"low": (False, False, True), "medium": (False, True, False), "high": (True, False, False)}
    r, y, g = active[label]

    def bulb(on, col):
        if on:
            return """<div style="width:52px;height:52px;border-radius:50%;background:{col};
                      box-shadow:0 0 18px {col},0 0 40px {col}44;transition:all 0.3s;"></div>""".format(col=col)
        return """<div style="width:52px;height:52px;border-radius:50%;background:#21262d;
                  border:1px solid #30363d;"></div>"""

    light_html = """
<div style="display:inline-flex;flex-direction:column;align-items:center;gap:10px;
            background:#0d1117;border:2px solid #30363d;border-radius:16px;
            padding:18px 22px;box-shadow:0 4px 24px #0004;">
  {r}{y}{g}
</div>""".format(r=bulb(r, "#f85149"), y=bulb(y, "#d29922"), g=bulb(g, "#3fb950"))

    delta_str = "No change" if rule["delta"] == 0 else "+{} seconds".format(rule["delta"])

    return """
<div style="background:#161b22;border:1px solid #30363d;border-radius:12px;padding:20px;">
  <div style="font-size:0.7em;color:#8b949e;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:14px;">
    Signal Timing Recommendation
  </div>
  <div style="display:flex;align-items:center;gap:24px;flex-wrap:wrap;">
    {light}
    <div style="flex:1;min-width:180px;">
      <div style="font-size:1.1em;font-weight:700;color:{color};margin-bottom:6px;">{action}</div>
      <div style="font-size:0.85em;color:#8b949e;margin-bottom:10px;">
        Green phase Δ: <span style="color:{color};font-weight:600;">{delta}</span>
      </div>
      <div style="font-size:0.82em;color:#6e7681;line-height:1.6;">{detail}</div>
      <div style="font-size:0.72em;color:#444;margin-top:10px;">
        ⚠ Rule-based prototype — not a production traffic control system
      </div>
    </div>
  </div>
</div>""".format(light=light_html, color=color, action=rule["action"],
                 delta=delta_str, detail=rule["detail"])


def compare_html(results):
    # type: (List[Tuple]) -> str
    """results = list of (model_name, label, probs) tuples"""
    cards = ""
    for model_name, label, probs in results:
        if label is None:
            continue
        color = CLASS_COLORS[label]
        emoji = CLASS_EMOJIS[label]
        r     = TEST_RESULTS[model_name]
        bars  = ""
        for n, p in zip(CLASS_NAMES, probs):
            c = CLASS_COLORS[n]
            bars += """
            <div style="margin:4px 0;">
              <div style="display:flex;justify-content:space-between;margin-bottom:2px;">
                <span style="color:{c};font-size:0.78em;">{n}</span>
                <span style="color:#8b949e;font-size:0.78em;">{p:.0f}%</span>
              </div>
              <div style="background:#21262d;border-radius:3px;height:6px;">
                <div style="width:{p:.0f}%;background:{c};height:100%;border-radius:3px;"></div>
              </div>
            </div>""".format(c=c, n=n.capitalize(), p=p * 100)

        cards += """
        <div style="background:#161b22;border:1px solid #30363d;border-radius:12px;padding:18px;
                    border-top:3px solid {color};">
          <div style="font-size:0.72em;color:#8b949e;text-transform:uppercase;letter-spacing:0.08em;
                      margin-bottom:4px;">{display}</div>
          <div style="font-size:1.5em;font-weight:800;color:{color};margin-bottom:12px;">
            {emoji} {label_upper}
          </div>
          {bars}
          <div style="margin-top:12px;padding-top:10px;border-top:1px solid #21262d;
                      display:flex;justify-content:space-between;font-size:0.76em;color:#6e7681;">
            <span>Test Acc <b style="color:#e6edf3;">{acc:.1f}%</b></span>
            <span>Macro F1 <b style="color:#e6edf3;">{f1:.3f}</b></span>
            <span>Params <b style="color:#e6edf3;">{params}</b></span>
          </div>
        </div>""".format(
            color=color, display=MODEL_DISPLAY[model_name],
            emoji=emoji, label_upper=label.upper(), bars=bars,
            acc=r["test_acc"] * 100, f1=r["macro_f1"],
            params=MODEL_PARAMS[model_name]
        )

    return """
<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:4px;">
  {cards}
</div>""".format(cards=cards)


def methodology_html():
    # type: () -> str
    steps = [
        ("01", "#00d4aa", "Raw Data",
         "14 intersection recording pairs from the Waterloo Multi-Agent Traffic Dataset. "
         "Each pair contains drone footage (~5 min, 29.97 FPS) and a SQLite database with "
         "per-frame trajectory annotations (position, speed, vehicle type)."),
        ("02", "#58a6ff", "Congestion Labeling",
         "Labels are derived programmatically — not manually annotated. For each 5-second window: "
         "<br><br>"
         "<code style='background:#21262d;padding:2px 6px;border-radius:4px;font-size:0.88em;'>"
         "score = 0.4 × norm_count + 0.4 × (1 − norm_speed) + 0.2 × stop_proxy"
         "</code>"
         "<br><br>Percentile-normalised (5th–95th) within each pair. "
         "Thresholds: Low ≤ 0.33, High ≥ 0.67."),
        ("03", "#d29922", "Overlay Removal",
         "Source footage contains annotation overlays (bounding boxes) embedded by the dataset provider. "
         "These were removed using HSV colour masking and TELEA inpainting before training, "
         "ensuring models learn genuine visual traffic patterns rather than annotation density."),
        ("04", "#f85149", "Frame Extraction",
         "3 frames extracted per window at evenly-spaced interior positions (25%, 50%, 75%). "
         "Frames resized to 224×224. Total: 2,296 samples across 769 windows from 14 pairs."),
        ("05", "#bc8cff", "Window-Stratified Split",
         "Split at the window level (not frame level) to prevent near-duplicate frames from "
         "appearing in both train and test. Composite key (pair_id, window_id) stratified by label. "
         "Ratio: 70% train / 15% val / 15% test."),
        ("06", "#00d4aa", "CNN Classification + Ensemble",
         "Four architectures trained: Baseline CNN, MobileNetV2, ResNet-50, EfficientNet-B0. "
         "Class-weighted CrossEntropyLoss, cosine LR annealing, 50 epochs. "
         "Best single model: MobileNetV2 (78.74%). "
         "Ensemble + TTA (5 augmented variants × 4 models, averaged): <b>82.18% test accuracy</b>."),
    ]

    cards = ""
    for num, color, title, desc in steps:
        cards += """
        <div style="background:#161b22;border:1px solid #30363d;border-radius:12px;padding:20px;
                    border-left:4px solid {color};">
          <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
            <span style="font-size:0.72em;font-weight:700;color:{color};letter-spacing:0.1em;
                         background:{color}22;padding:3px 8px;border-radius:4px;">STEP {num}</span>
            <span style="font-weight:700;color:#e6edf3;font-size:0.95em;">{title}</span>
          </div>
          <div style="font-size:0.84em;color:#8b949e;line-height:1.7;">{desc}</div>
        </div>""".format(color=color, num=num, title=title, desc=desc)

    return """
<div style="padding:4px 0;">
  <div style="font-size:0.72em;color:#8b949e;letter-spacing:0.1em;text-transform:uppercase;
              margin-bottom:16px;">Pipeline Overview</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;">
    {cards}
  </div>
</div>""".format(cards=cards)


def dataset_html():
    # type: () -> str
    stats = [
        ("14", "Recording Pairs",    "#00d4aa", "Intersections 769–785"),
        ("777", "Total Windows",     "#58a6ff", "5-second temporal windows"),
        ("2,296", "Training Samples","#d29922", "3 frames per window"),
        ("29.97", "FPS",             "#bc8cff", "Drone footage frame rate"),
        ("224×224", "Image Size",    "#f85149", "Pixels, RGB, JPEG"),
        ("70/15/15", "Split Ratio",  "#3fb950", "Train / Val / Test"),
    ]

    stat_cards = ""
    for val, label, color, sub in stats:
        stat_cards += """
        <div style="background:#161b22;border:1px solid #30363d;border-radius:12px;padding:18px;
                    text-align:center;">
          <div style="font-size:1.8em;font-weight:800;color:{color};margin-bottom:4px;">{val}</div>
          <div style="font-size:0.82em;font-weight:600;color:#e6edf3;margin-bottom:4px;">{label}</div>
          <div style="font-size:0.74em;color:#6e7681;">{sub}</div>
        </div>""".format(color=color, val=val, label=label, sub=sub)

    rows = ""
    best = "ensemble_tta"
    for name in ["baseline_cnn", "mobilenet_v2", "resnet50", "efficientnet_b0", "ensemble_tta"]:
        r = TEST_RESULTS[name]
        highlight = "background:#1c2128;" if name == best else ""
        star      = " ★" if name == best else ""
        rows += """
        <tr style="{hl}border-bottom:1px solid #21262d;">
          <td style="padding:12px 16px;font-weight:600;color:#e6edf3;">{display}{star}</td>
          <td style="padding:12px 16px;text-align:center;color:#e6edf3;font-weight:700;">{acc:.1f}%</td>
          <td style="padding:12px 16px;text-align:center;color:#e6edf3;">{f1:.4f}</td>
          <td style="padding:12px 16px;text-align:center;color:#3fb950;">{low:.3f}</td>
          <td style="padding:12px 16px;text-align:center;color:#d29922;">{med:.3f}</td>
          <td style="padding:12px 16px;text-align:center;color:#f85149;">{high:.3f}</td>
          <td style="padding:12px 16px;text-align:center;color:#6e7681;">{params}</td>
        </tr>""".format(
            hl=highlight, display=MODEL_DISPLAY[name].replace(" ★", ""), star=star,
            acc=r["test_acc"] * 100, f1=r["macro_f1"],
            low=r["low_f1"], med=r["med_f1"], high=r["high_f1"],
            params=MODEL_PARAMS[name]
        )

    return """
<div>
  <div style="font-size:0.72em;color:#8b949e;letter-spacing:0.1em;text-transform:uppercase;
              margin-bottom:16px;">Dataset Statistics</div>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:28px;">
    {stat_cards}
  </div>

  <div style="font-size:0.72em;color:#8b949e;letter-spacing:0.1em;text-transform:uppercase;
              margin-bottom:12px;">Model Comparison — Test Set (348 samples)</div>
  <div style="background:#161b22;border:1px solid #30363d;border-radius:12px;overflow:hidden;">
    <table style="width:100%;border-collapse:collapse;font-size:0.85em;">
      <thead>
        <tr style="background:#0d1117;border-bottom:2px solid #30363d;">
          <th style="padding:12px 16px;text-align:left;color:#8b949e;font-weight:600;">Model</th>
          <th style="padding:12px 16px;text-align:center;color:#8b949e;font-weight:600;">Test Acc</th>
          <th style="padding:12px 16px;text-align:center;color:#8b949e;font-weight:600;">Macro F1</th>
          <th style="padding:12px 16px;text-align:center;color:#3fb950;font-weight:600;">Low F1</th>
          <th style="padding:12px 16px;text-align:center;color:#d29922;font-weight:600;">Med F1</th>
          <th style="padding:12px 16px;text-align:center;color:#f85149;font-weight:600;">High F1</th>
          <th style="padding:12px 16px;text-align:center;color:#8b949e;font-weight:600;">Params</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
  </div>

  <div style="margin-top:16px;padding:14px;background:#161b22;border:1px solid #30363d;
              border-radius:10px;font-size:0.8em;color:#6e7681;line-height:1.6;">
    <b style="color:#8b949e;">Label derivation:</b> Congestion labels were derived from SQLite trajectory
    annotations using a composite score weighted across vehicle count (0.4), normalised inverse speed (0.4),
    and stop-proxy fraction (0.2). Annotation overlays were removed from all frames via HSV inpainting
    prior to training. Split strategy: window-level stratified across all 14 pairs to prevent
    near-duplicate frame leakage.
  </div>
</div>""".format(stat_cards=stat_cards, rows=rows)


# ── Dataset index ─────────────────────────────────────────────────────────────

def build_dataset_index():
    # type: () -> pd.DataFrame
    """Load the full samples metadata for browsing."""
    try:
        import pandas as pd
        path = PROJECT_ROOT / CFG["labeling"]["samples_metadata_all_csv"]
        if path.exists():
            df = pd.read_csv(path)
            df = df[df["image_path"].apply(lambda p: (PROJECT_ROOT / p).exists())]
            return df.reset_index(drop=True)
    except Exception:
        pass
    return None


# ── Example images ─────────────────────────────────────────────────────────────

def get_examples():
    # type: () -> List[str]
    # Curated frames: MobileNetV2 correctly classifies each with >70% confidence
    curated = [
        "data/processed/frames/769/w00038_f00.jpg",  # low
        "data/processed/frames/769/w00000_f00.jpg",  # medium
        "data/processed/frames/769/w00046_f00.jpg",  # high
    ]
    out = []
    for rel in curated:
        p = PROJECT_ROOT / rel
        if p.exists():
            out.append(str(p))
    return out


# ── Build app ──────────────────────────────────────────────────────────────────

def build_app():
    # type: () -> gr.Blocks
    print("[GUI] Loading all models...")
    models, gradcams = load_all_models()
    transform = get_transform()
    available = list(models.keys())

    if not available:
        raise RuntimeError("No model checkpoints found. Run training first.")

    default_model = "mobilenet_v2" if "mobilenet_v2" in available else available[0]
    # Ensemble+TTA always available if at least one model loaded
    dropdown_choices = [(MODEL_DISPLAY[n], n) for n in available] + \
                       [("Ensemble + TTA ⚡", "ensemble_tta")]

    # ── Dataset index for browsing ──
    import pandas as pd
    dataset_df = build_dataset_index()
    pair_ids   = sorted(dataset_df["pair_id"].astype(str).unique().tolist()) if dataset_df is not None else []
    browse_idx = [0]  # mutable pointer

    def get_filtered(pair_filter, class_filter):
        # type: (str, str) -> pd.DataFrame
        df = dataset_df.copy() if dataset_df is not None else pd.DataFrame()
        if pair_filter != "All":
            df = df[df["pair_id"].astype(str) == pair_filter]
        if class_filter != "All":
            df = df[df["label"] == class_filter.lower()]
        return df.reset_index(drop=True)

    def browse_load(class_filter, direction):
        # type: (str, str) -> tuple
        import random
        df = get_filtered("All", class_filter)
        if df.empty:
            return None, "No frames found"
        n = len(df)
        if direction == "random":
            browse_idx[0] = random.randint(0, n - 1)
        row = df.iloc[browse_idx[0]]
        img = Image.open(PROJECT_ROOT / row["image_path"]).convert("RGB")
        info = "Pair {pair} · Window {wid} · Label: {lbl}".format(
            pair=row["pair_id"], wid=int(row["window_id"]),
            lbl=row["label"].capitalize()
        )
        return img, info

    # ── Inference functions ──
    def classify(image, model_name):
        if image is None:
            empty = "<p style='color:#6e7681;font-family:sans-serif;padding:20px;'>Upload an image to begin.</p>"
            return empty, empty, None

        label, probs, cam = run_inference(image, model_name, models, gradcams, transform)
        if label is None:
            return "", "", None

        return prediction_html(label, probs), traffic_light_html(label), cam

    def compare_all(image):
        if image is None:
            return "<p style='color:#6e7681;font-family:sans-serif;padding:20px;'>Upload an image to begin.</p>"
        results = []
        for name in ["baseline_cnn", "mobilenet_v2", "resnet50", "efficientnet_b0"]:
            if name in models:
                label, probs, _ = run_inference(image, name, models, gradcams, transform)
                results.append((name, label, probs))
        return compare_html(results)

    # ── Layout ──
    with gr.Blocks(title="Traffic Congestion Classifier", css=CSS) as demo:

        gr.HTML("""
<div style="padding:28px 0 20px;border-bottom:1px solid #21262d;margin-bottom:24px;">
  <div style="font-size:0.72em;color:#00d4aa;letter-spacing:0.15em;text-transform:uppercase;
              font-weight:600;margin-bottom:8px;">Road Rangers · Traffic Intelligence</div>
  <div style="font-size:1.9em;font-weight:800;color:#e6edf3;letter-spacing:-0.02em;margin-bottom:8px;">
    Traffic Congestion Classification
  </div>
  <div style="font-size:0.88em;color:#6e7681;max-width:600px;line-height:1.6;">
    CNN-based congestion classification from intersection footage with rule-based
    signal timing recommendations. Trained on 14 intersection pairs · 2,296 samples.
  </div>
</div>""")

        with gr.Tabs():

            # ── Tab 1: Classify ──
            with gr.Tab("  Classify  "):
                with gr.Row():
                    with gr.Column(scale=1):
                        img_input  = gr.Image(type="pil", label="Intersection Frame")
                        model_drop = gr.Dropdown(
                            choices=dropdown_choices,
                            value=default_model,
                            label="Model",
                        )
                        classify_btn = gr.Button("Classify", variant="primary")

                        # ── Example frames ──
                        gr.HTML("""<div style="font-size:0.72em;color:#8b949e;letter-spacing:0.1em;
                                   text-transform:uppercase;margin:16px 0 8px;">
                                   Example Frames</div>""")
                        with gr.Row():
                            ex0 = gr.Image(type="pil", interactive=False, show_label=False, height=110)
                            ex1 = gr.Image(type="pil", interactive=False, show_label=False, height=110)
                            ex2 = gr.Image(type="pil", interactive=False, show_label=False, height=110)
                        with gr.Row():
                            use0 = gr.Button("Use", scale=1, variant="secondary")
                            use1 = gr.Button("Use", scale=1, variant="secondary")
                            use2 = gr.Button("Use", scale=1, variant="secondary")
                        with gr.Row():
                            class_filter = gr.Radio(
                                choices=["Any", "Low", "Medium", "High"],
                                value="Any", label="", container=False, scale=2
                            )
                            shuffle_btn = gr.Button("🔀 Shuffle", scale=1, variant="primary")

                    with gr.Column(scale=1):
                        pred_out   = gr.HTML(
                            value="<p style='color:#6e7681;font-family:sans-serif;padding:20px;'>Upload an image or pick an example to begin.</p>"
                        )
                        signal_out = gr.HTML()
                        cam_out    = gr.Image(type="pil", label="GradCAM — influential regions",
                                             interactive=False)

                classify_btn.click(classify, [img_input, model_drop],
                                   [pred_out, signal_out, cam_out])
                img_input.change(classify, [img_input, model_drop],
                                 [pred_out, signal_out, cam_out])
                model_drop.change(classify, [img_input, model_drop],
                                  [pred_out, signal_out, cam_out])

                # Shuffle — pick 3 new random frames
                def shuffle_examples(class_f):
                    if class_f == "Any":
                        # One of each class: Low, Medium, High
                        imgs = [browse_load(c, "random")[0] for c in ["Low", "Medium", "High"]]
                    else:
                        imgs = [browse_load(class_f, "random")[0] for _ in range(3)]
                    return imgs[0], imgs[1], imgs[2]

                shuffle_btn.click(shuffle_examples, [class_filter], [ex0, ex1, ex2])
                class_filter.change(shuffle_examples, [class_filter], [ex0, ex1, ex2])

                # Use — load selected thumbnail into main input and classify
                def use_example(img, model_name):
                    if img is None:
                        return None, "", "", None
                    p_html, s_html, cam = classify(img, model_name)
                    return img, p_html, s_html, cam

                use_outputs = [img_input, pred_out, signal_out, cam_out]
                use0.click(use_example, [ex0, model_drop], use_outputs)
                use1.click(use_example, [ex1, model_drop], use_outputs)
                use2.click(use_example, [ex2, model_drop], use_outputs)

                # Load initial 3 examples on startup
                demo.load(shuffle_examples, [class_filter], [ex0, ex1, ex2])

            # ── Tab 2: Compare ──
            with gr.Tab("  Compare Models  "):
                gr.HTML("""
<div style="padding:4px 0 16px;font-size:0.84em;color:#6e7681;">
  Run all four models simultaneously on the same frame and compare predictions side by side.
</div>""")
                with gr.Row():
                    with gr.Column(scale=1):
                        cmp_input = gr.Image(type="pil", label="Intersection Frame")
                        cmp_btn   = gr.Button("Compare All Models", variant="primary")
                        gr.Examples(examples=get_examples(), inputs=cmp_input,
                                    label="Example frames")
                    with gr.Column(scale=2):
                        cmp_out = gr.HTML(
                            value="<p style='color:#6e7681;font-family:sans-serif;padding:20px;'>Upload an image to compare all models.</p>"
                        )

                cmp_btn.click(compare_all, cmp_input, cmp_out)
                cmp_input.change(compare_all, cmp_input, cmp_out)

            # ── Tab 3: Methodology ──
            with gr.Tab("  Methodology  "):
                gr.HTML(methodology_html())

            # ── Tab 4: Dataset & Results ──
            with gr.Tab("  Dataset & Results  "):
                gr.HTML(dataset_html())

    return demo


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",  type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_app()
    demo.launch(server_port=args.port, share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()
