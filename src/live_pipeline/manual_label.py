"""
manual_label.py
---------------
Camera review tool: shows one camera at a time.
Top row = LOW samples, middle row = MEDIUM, bottom row = HIGH.
3 images per label = 9 total per camera.

Controls:
    n / Enter    next camera
    p            previous camera
    q            quit

Usage:
    python src/live_pipeline/manual_label.py
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LABEL_COLORS = {"low": "#4CAF50", "medium": "#FF9800", "high": "#F44336"}
SAMPLES_PER_LABEL = 3   # columns
LABELS = ["low", "medium", "high"]


def load_vehicle_counts():
    lbl_path = PROJECT_ROOT / "data/live/labels/frame_labels_live_all.csv"
    lbl = pd.read_csv(lbl_path, usecols=["file_path", "vehicle_count"])
    return dict(zip(lbl["file_path"], lbl["vehicle_count"]))


VEHICLE_CLASS_IDS = {2, 3, 5, 7}

# Must match detect.py
CAMERA_ROI = {}  # no full-frame ROI crops currently

CAMERA_EXCLUDE = {
    "5_ways_miranda": [(0.0, 0.0, 0.28, 0.55)],  # top-left corner = dealership
}


def load_yolo():
    from ultralytics import YOLO
    print("Loading YOLO model...")
    model = YOLO("yolov8n.pt")
    model.conf = 0.3
    print("YOLO ready.")
    return model


def get_boxes(model, img_path, camera_id=""):
    from PIL import Image as PILImage
    img = PILImage.open(img_path).convert("RGB")
    W, H = img.size
    roi = CAMERA_ROI.get(camera_id)
    if roi:
        rx1, ry1, rx2, ry2 = roi[0]*W, roi[1]*H, roi[2]*W, roi[3]*H
    else:
        rx1, ry1, rx2, ry2 = 0, 0, W, H

    exclude_zones = [
        (ex1*W, ey1*H, ex2*W, ey2*H)
        for (ex1, ey1, ex2, ey2) in CAMERA_EXCLUDE.get(camera_id, [])
    ]

    results = model(img, verbose=False)[0]
    included, excluded = [], []
    for box in results.boxes:
        if int(box.cls[0].item()) not in VEHICLE_CLASS_IDS:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        in_roi = rx1 <= cx <= rx2 and ry1 <= cy <= ry2
        in_exclude = any(ex1 <= cx <= ex2 and ey1 <= cy <= ey2
                         for (ex1, ey1, ex2, ey2) in exclude_zones)
        if in_roi and not in_exclude:
            included.append((x1, y1, x2, y2))
        else:
            excluded.append((x1, y1, x2, y2))
    return included, excluded, (W, H)


class CameraReviewer:
    def __init__(self, df, vehicle_counts, yolo_model):
        self.df = df
        self.vehicle_counts = vehicle_counts
        self.yolo = yolo_model
        self.box_cache = {}

        self.cameras = sorted(df["camera_id"].unique())
        self.cam_idx = 0

        self.fig = plt.figure(figsize=(18, 10))
        self.fig.patch.set_facecolor("#1e1e1e")
        self.fig.canvas.manager.set_window_title("Camera Label Review")
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.render()
        plt.show()

    def render(self):
        self.fig.clear()
        cam_id = self.cameras[self.cam_idx]
        cam_df = self.df[self.df["camera_id"] == cam_id]

        gs = gridspec.GridSpec(3, SAMPLES_PER_LABEL, figure=self.fig,
                               hspace=0.35, wspace=0.05)

        for row, label in enumerate(LABELS):
            label_df = cam_df[cam_df["label"] == label]
            samples = label_df.sample(
                min(SAMPLES_PER_LABEL, len(label_df)), random_state=42
            ).reset_index(drop=True) if len(label_df) > 0 else pd.DataFrame()

            for col in range(SAMPLES_PER_LABEL):
                ax = self.fig.add_subplot(gs[row, col])
                ax.axis("off")

                if col >= len(samples):
                    ax.set_facecolor("#111")
                    ax.set_title(f"(no {label})", color="#555", fontsize=8)
                    continue

                r = samples.iloc[col]
                img_path = PROJECT_ROOT / r["file_path"]
                try:
                    img = mpimg.imread(str(img_path))
                    ax.imshow(img)
                except Exception:
                    ax.set_facecolor("#333")

                # Draw YOLO boxes (green = counted, red = excluded by ROI)
                if self.yolo is not None:
                    ip = r["file_path"]
                    if ip not in self.box_cache:
                        self.box_cache[ip] = get_boxes(self.yolo, PROJECT_ROOT / ip, cam_id)
                    included, excluded, (orig_w, orig_h) = self.box_cache[ip]
                    try:
                        disp_h, disp_w = img.shape[:2]
                    except Exception:
                        disp_h, disp_w = orig_h, orig_w
                    sx, sy = disp_w / orig_w, disp_h / orig_h
                    for (x1, y1, x2, y2) in included:
                        ax.add_patch(patches.Rectangle(
                            (x1*sx, y1*sy), (x2-x1)*sx, (y2-y1)*sy,
                            linewidth=1.5, edgecolor="#00FF00", facecolor="none"
                        ))
                    for (x1, y1, x2, y2) in excluded:
                        ax.add_patch(patches.Rectangle(
                            (x1*sx, y1*sy), (x2-x1)*sx, (y2-y1)*sy,
                            linewidth=1.0, edgecolor="#FF4444", facecolor="none",
                            linestyle="dashed"
                        ))

                count = self.vehicle_counts.get(r["file_path"], "?")
                color = LABEL_COLORS[label]
                ax.set_title(f"{label.upper()}  v={count}",
                             fontsize=9, color=color, fontweight="bold", pad=3)

                # Row label on leftmost column
                if col == 0:
                    ax.set_ylabel(label.upper(), color=color,
                                  fontsize=11, fontweight="bold", rotation=90, labelpad=6)

        self.fig.suptitle(
            f"[{self.cam_idx+1}/{len(self.cameras)}]  {cam_id}   |   "
            f"n / Enter = next camera    p = prev    q = quit",
            color="white", fontsize=11, y=0.98
        )
        self.fig.canvas.draw()

    def on_key(self, event):
        k = event.key
        if k in ("n", "enter"):
            self.cam_idx = (self.cam_idx + 1) % len(self.cameras)
            self.render()
        elif k == "p":
            self.cam_idx = (self.cam_idx - 1) % len(self.cameras)
            self.render()
        elif k == "q":
            plt.close("all")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(PROJECT_ROOT / "data/live/labels/frame_labels_live_all.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.source)
    print(f"Loaded {len(df)} images across {df['camera_id'].nunique()} cameras")

    vehicle_counts = load_vehicle_counts()
    yolo_model = load_yolo()
    CameraReviewer(df, vehicle_counts, yolo_model)


if __name__ == "__main__":
    main()
