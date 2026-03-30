"""
app.py
------
Traffic Congestion Classification Demo — Gradio GUI

Loads the best trained model (ResNet-50 by default) and provides:
  - Image upload for single-frame inference
  - Congestion class prediction with per-class confidence scores
  - Rule-based signal timing recommendation
  - GradCAM visualisation showing which regions drove the prediction

Run:
  python src/gui/app.py
  python src/gui/app.py --model resnet50
  python src/gui/app.py --model baseline_cnn --port 7861
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

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


# ── Constants ─────────────────────────────────────────────────────────────────

CLASS_NAMES  = ["low", "medium", "high"]
CLASS_LABELS = ["Low", "Medium", "High"]
CLASS_EMOJIS = ["🟢", "🟡", "🔴"]

# Rule-based signal timing recommendations
SIGNAL_RULES = {
    "low": {
        "action":      "Maintain current cycle",
        "green_delta": 0,
        "description": "Traffic is flowing freely. No adjustment needed — maintain the standard signal cycle to avoid unnecessary delays on cross-streets.",
        "colour":      "#2ecc71",
    },
    "medium": {
        "action":      "Extend green by ~10 seconds",
        "green_delta": +10,
        "description": "Moderate congestion detected. Extend the current green phase by approximately 10 seconds to clear queued vehicles before switching.",
        "colour":      "#f39c12",
    },
    "high": {
        "action":      "Extend green by ~20 seconds",
        "green_delta": +20,
        "description": "Heavy congestion detected. Extend the current green phase by approximately 20 seconds. Consider activating overflow protocol if queues persist across multiple cycles.",
        "colour":      "#e74c3c",
    },
}


# ── Model loading ──────────────────────────────────────────────────────────────

def get_device():
    # type: () -> torch.device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_name):
    # type: (str) -> Tuple[nn.Module, torch.device]
    ckpt_path = PROJECT_ROOT / CFG["paths"]["checkpoints"] / "{}_best.pt".format(model_name)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            "No checkpoint found for '{}' at {}.\n"
            "Run: python src/training/train.py --model {}".format(model_name, ckpt_path, model_name)
        )

    num_classes = CFG["models"]["num_classes"]
    if model_name == "baseline_cnn":
        model = BaselineCNN(num_classes=num_classes)
    else:
        model = build_model(model_name, num_classes=num_classes, pretrained=False)

    ckpt   = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])

    # GradCAM: register hooks before moving to device
    device = get_device()
    # Use CPU for GUI inference — avoids MPS/CUDA stream issues with hooks
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    epoch   = ckpt.get("epoch", "?")
    val_acc = ckpt.get("val_acc", float("nan"))
    print("[GUI] Loaded {} (epoch {}, val_acc={:.4f})  device={}".format(
        model_name, epoch, val_acc, device))
    return model, device


# ── Preprocessing ──────────────────────────────────────────────────────────────

def get_transform(image_size):
    # type: (Tuple[int, int]) -> transforms.Compose
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ── GradCAM ───────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Lightweight GradCAM implementation.
    Works with any model that has a 'features' attribute or last conv layer.
    """

    def __init__(self, model, model_name):
        # type: (nn.Module, str) -> None
        self.model      = model
        self.model_name = model_name
        self.gradients  = None
        self.activations = None
        self._hook_handles = []
        self._register_hooks()

    def _get_target_layer(self):
        # type: () -> nn.Module
        name = self.model_name
        if name == "baseline_cnn":
            # Last conv block before adaptive pool
            return self.model.features[-1]
        elif name == "mobilenet_v2":
            return self.model.features[-1]
        elif name == "resnet50":
            return self.model.layer4[-1]
        elif name == "efficientnet_b0":
            return self.model.features[-1]
        # Fallback: find last Conv2d
        last_conv = None
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        return last_conv

    def _register_hooks(self):
        # type: () -> None
        layer = self._get_target_layer()
        if layer is None:
            return

        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self._hook_handles.append(layer.register_forward_hook(forward_hook))
        self._hook_handles.append(layer.register_backward_hook(backward_hook))

    def generate(self, input_tensor, class_idx):
        # type: (torch.Tensor, int) -> np.ndarray
        """Returns a HxW heatmap in [0,1]."""
        self.model.zero_grad()
        output = self.model(input_tensor)
        score  = output[0, class_idx]
        score.backward()

        if self.gradients is None or self.activations is None:
            # Return blank heatmap if hooks didn't fire
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))

        # Global average pool the gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam     = (weights * self.activations).sum(dim=1).squeeze(0)  # (H, W)
        cam     = torch.clamp(cam, min=0)

        # Normalise
        cam_np  = cam.numpy()
        if cam_np.max() > 0:
            cam_np = cam_np / cam_np.max()
        return cam_np

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()


def overlay_gradcam(original_pil, cam_array, alpha=0.45):
    # type: (Image.Image, np.ndarray, float) -> Image.Image
    """Blend GradCAM heatmap onto original image."""
    import cv2

    orig_np = np.array(original_pil.convert("RGB"))
    h, w    = orig_np.shape[:2]

    cam_resized = cv2.resize(cam_array.astype(np.float32), (w, h))
    heatmap     = cv2.applyColorMap(
        (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    blended = (alpha * heatmap_rgb + (1 - alpha) * orig_np).astype(np.uint8)
    return Image.fromarray(blended)


# ── Inference ─────────────────────────────────────────────────────────────────

def predict(image_pil, model, gradcam, transform, image_size):
    # type: (Image.Image, nn.Module, GradCAM, transforms.Compose, Tuple[int,int]) -> Tuple
    """
    Run inference on a PIL image.
    Returns (label, confidences_dict, signal_html, gradcam_image).
    """
    if image_pil is None:
        return None, {}, "", None

    image_pil = image_pil.convert("RGB")
    tensor    = transform(image_pil).unsqueeze(0)  # (1, 3, H, W)
    tensor.requires_grad_(False)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze(0).numpy()

    pred_idx   = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_idx]

    # GradCAM (needs gradient, re-run with grad enabled)
    gradcam_img = None
    try:
        t2  = transform(image_pil).unsqueeze(0)
        cam = gradcam.generate(t2, pred_idx)
        gradcam_img = overlay_gradcam(image_pil, cam)
    except Exception:
        pass

    # Confidence dict for gr.Label
    confidences = {
        "{} {}".format(CLASS_EMOJIS[i], CLASS_LABELS[i]): float(probs[i])
        for i in range(len(CLASS_NAMES))
    }

    # Signal timing recommendation HTML
    rule   = SIGNAL_RULES[pred_label]
    colour = rule["colour"]
    delta  = rule["green_delta"]
    delta_str = ("no change" if delta == 0
                 else "+{} seconds".format(delta) if delta > 0
                 else "{} seconds".format(delta))

    signal_html = """
<div style="border-left: 6px solid {colour}; padding: 14px 18px; border-radius: 6px;
            background: #1a1a2e; color: #eee; font-family: sans-serif; margin-top: 8px;">
  <div style="font-size:1.3em; font-weight:700; color:{colour}; margin-bottom:6px;">
    {emoji} {label_upper} CONGESTION
  </div>
  <div style="font-size:1.05em; margin-bottom:4px;">
    <b>Recommended action:</b> {action}
  </div>
  <div style="font-size:0.95em; color:#aaa; margin-bottom:6px;">
    Green phase adjustment: <b style="color:{colour}">{delta_str}</b>
  </div>
  <div style="font-size:0.9em; color:#ccc; line-height:1.5;">
    {description}
  </div>
  <div style="font-size:0.75em; color:#666; margin-top:10px;">
    ⚠ Rule-based prototype only. Not a production traffic control system.
  </div>
</div>
""".format(
        colour=colour,
        emoji=CLASS_EMOJIS[pred_idx],
        label_upper=pred_label.upper(),
        action=rule["action"],
        delta_str=delta_str,
        description=rule["description"],
    )

    return pred_label, confidences, signal_html, gradcam_img


# ── Gradio app ────────────────────────────────────────────────────────────────

def build_app(model_name):
    # type: (str) -> gr.Blocks
    image_size = tuple(CFG["frame_extraction"]["image_size"])
    transform  = get_transform(image_size)

    model, device = load_model(model_name)
    gradcam = GradCAM(model, model_name)

    def run_inference(image):
        label, confs, signal_html, cam_img = predict(
            image, model, gradcam, transform, image_size
        )
        if label is None:
            return {}, "<p style='color:#888'>Upload an image to begin.</p>", None
        return confs, signal_html, cam_img

    with gr.Blocks(
        title="Traffic Congestion Classifier",
        theme=gr.themes.Base(),
        css="""
            .gradio-container { max-width: 1100px !important; }
            footer { display: none !important; }
        """,
    ) as demo:

        gr.Markdown("""
# Traffic Congestion Classification
**Road Rangers — 42028 Deep Learning, UTS**

Upload an intersection frame to classify congestion level (Low / Medium / High)
and receive a rule-based signal timing recommendation.

Model: **{}** (best) &nbsp;|&nbsp; Classes: 🟢 Low &nbsp; 🟡 Medium &nbsp; 🔴 High
        """.format(model_name.replace("_", "-").title()))

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Upload Intersection Frame",
                    tool="editor",
                )
                submit_btn = gr.Button("Classify", variant="primary")

                gr.Examples(
                    examples=_get_example_images(),
                    inputs=image_input,
                    label="Example frames from dataset",
                )

            with gr.Column(scale=1):
                label_output = gr.Label(
                    num_top_classes=3,
                    label="Predicted Congestion Level",
                )
                signal_output = gr.HTML(
                    label="Signal Timing Recommendation",
                    value="<p style='color:#888; font-family:sans-serif'>Upload an image to begin.</p>",
                )

        with gr.Row():
            gradcam_output = gr.Image(
                type="pil",
                label="GradCAM — regions influencing prediction",
                interactive=False,
            )
            gr.Markdown("""
### How to read GradCAM
The heatmap highlights image regions that most influenced the model's prediction.
- **Red/warm** = high influence
- **Blue/cool** = low influence

Congested scenes typically activate on vehicle density, queue length, and lane occupancy patterns.
            """)

        submit_btn.click(
            fn=run_inference,
            inputs=image_input,
            outputs=[label_output, signal_output, gradcam_output],
        )
        image_input.change(
            fn=run_inference,
            inputs=image_input,
            outputs=[label_output, signal_output, gradcam_output],
        )

    return demo


def _get_example_images():
    # type: () -> list
    """Return up to 6 example frame paths from the dataset for the Examples widget."""
    splits_dir = PROJECT_ROOT / CFG["split"]["output_dir"]
    test_csv   = splits_dir / "test.csv"
    examples   = []
    if test_csv.exists():
        import pandas as pd
        df = pd.read_csv(test_csv)
        # One example per class
        for lbl in ["low", "medium", "high"]:
            rows = df[df["label"] == lbl]
            if not rows.empty:
                p = PROJECT_ROOT / rows.iloc[0]["image_path"]
                if p.exists():
                    examples.append(str(p))
    return examples[:6]


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="mobilenet_v2",
        choices=["baseline_cnn", "mobilenet_v2", "resnet50", "efficientnet_b0"],
        help="Model to load for inference (default: resnet50)",
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true",
                        help="Generate a public Gradio share link")
    args = parser.parse_args()

    demo = build_app(args.model)
    demo.launch(server_port=args.port, share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()
