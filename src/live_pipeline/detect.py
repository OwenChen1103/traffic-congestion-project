"""
detect.py
---------
Runs YOLOv8n vehicle detection on all collected frames in data/live/raw/.
Outputs per-frame detection features to data/live/detections/.

Vehicle classes detected (COCO):
    2 = car, 3 = motorcycle, 5 = bus, 7 = truck

Features per frame:
    vehicle_count       - number of detected vehicles
    bbox_area_ratio     - sum of bbox areas / total frame area
    bottom_roi_count    - vehicles with centroid in bottom third of frame

Usage:
    python src/live_pipeline/detect.py
    python src/live_pipeline/detect.py --raw-dir data/live/raw --conf 0.3
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# COCO class IDs to treat as vehicles
VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck

# Bottom ROI = bottom 1/3 of frame height
BOTTOM_ROI_FRACTION = 1 / 3

# Per-camera valid ROI: (x1_frac, y1_frac, x2_frac, y2_frac)
# Only count vehicles whose centroid falls inside this region.
CAMERA_ROI = {
    "5_ways_miranda": (0.0, 0.0, 1.0, 1.0),  # full frame — exclusion handled by CAMERA_EXCLUDE
}

# Per-camera exclusion zones: list of (x1_frac, y1_frac, x2_frac, y2_frac)
# Centroids inside ANY of these zones are excluded.
# 5_ways_miranda: car dealership sits in the top-left corner of the frame.
CAMERA_EXCLUDE = {
    "5_ways_miranda": [(0.0, 0.0, 0.28, 0.55)],  # top-left corner = dealership
}


# ── Detection helpers ─────────────────────────────────────────────────────────

def load_model(conf_threshold):
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")   # downloads ~6MB on first run
    model.conf = conf_threshold
    return model


def detect_frame(model, img_path, camera_id=""):
    # type: (object, Path, str) -> dict
    """
    Run detection on one frame. Returns feature dict:
        vehicle_count, bbox_area_ratio, bottom_roi_count
    Applies per-camera ROI mask from CAMERA_ROI if defined.
    """
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    frame_area = W * H

    results = model(img, verbose=False)[0]

    # Per-camera valid region (fractions of W, H)
    roi = CAMERA_ROI.get(camera_id)
    if roi:
        rx1, ry1, rx2, ry2 = roi[0]*W, roi[1]*H, roi[2]*W, roi[3]*H
    else:
        rx1, ry1, rx2, ry2 = 0, 0, W, H

    # Per-camera exclusion zones
    exclude_zones = [
        (ex1*W, ey1*H, ex2*W, ey2*H)
        for (ex1, ey1, ex2, ey2) in CAMERA_EXCLUDE.get(camera_id, [])
    ]

    vehicle_count    = 0
    total_bbox_area  = 0.0
    bottom_roi_count = 0
    bottom_threshold = H * (1.0 - BOTTOM_ROI_FRACTION)  # y > this = bottom third

    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        if cls_id not in VEHICLE_CLASS_IDS:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx = (x1 + x2) / 2  # centroid x
        cy = (y1 + y2) / 2  # centroid y

        # Skip if centroid outside valid ROI
        if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
            continue

        # Skip if centroid inside any exclusion zone
        if any(ex1 <= cx <= ex2 and ey1 <= cy <= ey2
               for (ex1, ey1, ex2, ey2) in exclude_zones):
            continue

        box_area = (x2 - x1) * (y2 - y1)
        vehicle_count   += 1
        total_bbox_area += box_area
        if cy >= bottom_threshold:
            bottom_roi_count += 1

    bbox_area_ratio = total_bbox_area / frame_area if frame_area > 0 else 0.0

    # Mean brightness (0–255): used to filter nighttime frames before training
    import numpy as np
    mean_brightness = float(np.array(img.convert("L")).mean())

    return {
        "vehicle_count":    vehicle_count,
        "bbox_area_ratio":  round(bbox_area_ratio, 6),
        "bottom_roi_count": bottom_roi_count,
        "mean_brightness":  round(mean_brightness, 2),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run_detection(raw_dir, output_dir, conf_threshold):
    raw_dir    = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest to know camera/window/frame structure
    manifest_path = raw_dir / "manifest.csv"
    if not manifest_path.exists():
        print("[ERROR] manifest.csv not found in {}".format(raw_dir))
        sys.exit(1)

    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest["status"] == "ok"].copy()
    print("[detect] {} frames to process".format(len(manifest)))

    print("[detect] Loading YOLOv8n...")
    model = load_model(conf_threshold)
    print("[detect] Model ready  (conf={})".format(conf_threshold))

    rows = []
    total = len(manifest)

    for i, (_, row) in enumerate(manifest.iterrows()):
        img_path = PROJECT_ROOT / row["file_path"]

        if not img_path.exists():
            print("  [SKIP] {}".format(row["file_path"]))
            continue

        feats = detect_frame(model, img_path, camera_id=row["camera_id"])

        rows.append({
            "camera_id":        row["camera_id"],
            "role":             row["role"],
            "region":           row["region"],
            "window_id":        row["window_id"],
            "frame_idx":        row["frame_idx"],
            "timestamp":        row["timestamp"],
            "file_path":        row["file_path"],
            "vehicle_count":    feats["vehicle_count"],
            "bbox_area_ratio":  feats["bbox_area_ratio"],
            "bottom_roi_count": feats["bottom_roi_count"],
            "mean_brightness":  feats["mean_brightness"],
        })

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print("  [{}/{}]  {} w{:04d} f{}  count={}  area={:.4f}  bot={}".format(
                i + 1, total,
                row["camera_id"][:30],
                row["window_id"], row["frame_idx"],
                feats["vehicle_count"],
                feats["bbox_area_ratio"],
                feats["bottom_roi_count"],
            ))

    df = pd.DataFrame(rows)
    out_path = output_dir / "detections.csv"
    df.to_csv(out_path, index=False)

    print("\n[detect] Done.")
    print("         Frames processed : {}".format(len(df)))
    print("         Output           : {}".format(out_path))
    print("\nNEXT STEP:")
    print("  python src/live_pipeline/label.py")


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 vehicle detection on collected frames")
    parser.add_argument(
        "--raw-dir", type=str,
        default=str(PROJECT_ROOT / "data" / "live" / "raw"),
        help="Directory containing collected frames (default: data/live/raw)"
    )
    parser.add_argument(
        "--output", type=str,
        default=str(PROJECT_ROOT / "data" / "live" / "detections"),
        help="Output directory for detections CSV (default: data/live/detections)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.3,
        help="YOLO confidence threshold (default: 0.3)"
    )
    args = parser.parse_args()

    run_detection(args.raw_dir, args.output, args.conf)


if __name__ == "__main__":
    main()
