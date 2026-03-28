"""
predict.py
----------
Single-image inference: predict congestion level + signal recommendation.
Used by the GUI and for ad-hoc inference.

Usage:
    python src/inference/predict.py --image path/to/frame.jpg --model mobilenet_v2
    python src/inference/predict.py --image path/to/frame.jpg --model baseline_cnn
    python src/inference/predict.py --image path/to/frame.jpg --model resnet50
    python src/inference/predict.py --image path/to/frame.jpg --model efficientnet_b0
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Union

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config.settings import CFG, PROJECT_ROOT
from src.datasets.congestion_dataset import ID_TO_LABEL, get_eval_transforms
from src.models.baseline_cnn import BaselineCNN
from src.models.transfer_models import build_model
from src.inference.signal_recommendation import recommend


def load_model(model_name):
    # type: (str) -> torch.nn.Module
    ckpt_dir  = PROJECT_ROOT / CFG["paths"]["checkpoints"]
    ckpt_path = ckpt_dir / "{}_best.pt".format(model_name)
    if not ckpt_path.exists():
        raise FileNotFoundError("Checkpoint not found: {}".format(ckpt_path))

    num_classes = CFG["models"]["num_classes"]
    if model_name == "baseline_cnn":
        model = BaselineCNN(num_classes=num_classes)
    else:
        model = build_model(model_name, num_classes=num_classes, pretrained=False)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def predict_image(image_path, model_name="mobilenet_v2"):
    # type: (Union[str, Path], str) -> Dict
    """
    Predict congestion level for a single image.

    Returns dict with keys:
      predicted_label      str    'low' | 'medium' | 'high'
      confidence           float  probability of predicted class [0, 1]
      probabilities        dict   {class_name: probability}
      signal_recommendation SignalRecommendation
    """
    image_size = tuple(CFG["frame_extraction"]["image_size"])
    transform  = get_eval_transforms(image_size)

    image  = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)  # (1, C, H, W)

    model = load_model(model_name)
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)[0]

    class_names  = CFG["models"]["class_names"]
    prob_dict    = {cls: float(probs[i]) for i, cls in enumerate(class_names)}
    pred_id      = int(probs.argmax())
    pred_label   = ID_TO_LABEL[pred_id]
    confidence   = float(probs[pred_id])
    recommendation = recommend(pred_label)

    return {
        "predicted_label":      pred_label,
        "confidence":           confidence,
        "probabilities":        prob_dict,
        "signal_recommendation": recommendation,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--model", default="mobilenet_v2",
        choices=["baseline_cnn", "mobilenet_v2", "resnet50", "efficientnet_b0"],
    )
    args = parser.parse_args()

    result = predict_image(args.image, args.model)
    print("\n" + "=" * 40)
    print("Image:      {}".format(args.image))
    print("Model:      {}".format(args.model))
    print("Prediction: {}  (confidence: {:.1%})".format(
        result["predicted_label"].upper(), result["confidence"]))
    print("\nClass probabilities:")
    for cls, prob in result["probabilities"].items():
        print("  {:8s}: {:.1%}".format(cls, prob))
    print("\nSignal Recommendation:")
    print(result["signal_recommendation"])


if __name__ == "__main__":
    main()
