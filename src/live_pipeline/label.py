"""
label.py
--------
Assigns per-frame congestion labels directly from each frame's vehicle_count.

Input:
    data/live/detections/detections.csv

Output:
    data/live/labels/frame_labels_live_all.csv
        Columns: camera_id, role, region, window_id, frame_idx, timestamp,
                 file_path, vehicle_count, bbox_area_ratio, bottom_roi_count,
                 mean_brightness, label

Label logic (per-camera absolute thresholds on vehicle_count):
    ≤ low_max  → low
    ≥ high_min → high
    else       → medium
    Falls back to per-camera 30/70 percentile for unknown cameras.

Usage:
    python src/live_pipeline/label.py
    python src/live_pipeline/label.py --detections data/live/detections/detections.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Per-camera absolute thresholds on vehicle_count (per-frame)
# (low_max, high_min): ≤ low_max → low | ≥ high_min → high | else → medium
# Set by visual inspection of each camera's road type and capacity.
CAMERA_THRESHOLDS = {
    "james_ruse_drive_rosehill":    ( 8, 20),  # 4+ lane highway
    "hume_highway_bankstown":       ( 6, 15),  # wide arterial with median
    "5_ways_miranda":               ( 7, 15),  # 5-way intersection
    "parramatta_road_camperdown":   ( 7, 17),  # urban arterial intersection
    "king_georges_road_hurstville": ( 3, 10),  # suburban 2-lane
    "city_road_newtown":            ( 5, 15),  # urban tram road
    "anzac_parade_moore_park":      ( 5, 15),  # wider urban road
    "memorial_drive_towradgi":      ( 5, 13),  # regional highway
    "shellharbour_road_warilla":    ( 7, 15),  # regional arterial
    "princes_highway_st_peters_n":  ( 5, 15),  # glare issues, excluded anyway
}


def assign_frame_labels(det_df):
    # type: (pd.DataFrame) -> pd.DataFrame
    """
    Apply per-camera thresholds directly to each frame's vehicle_count.
    Falls back to per-camera 30/70 percentile for unknown cameras.
    """
    result_parts = []

    for cam_id, cam_df in det_df.groupby("camera_id"):
        df = cam_df.copy()

        if cam_id in CAMERA_THRESHOLDS:
            low_max, high_min = CAMERA_THRESHOLDS[cam_id]
            def to_label(v, lm=low_max, hm=high_min):
                if v <= lm:
                    return "low"
                elif v >= hm:
                    return "high"
                return "medium"
            df["label"] = df["vehicle_count"].apply(to_label)
        else:
            # Fallback: per-camera percentile on vehicle_count
            low_max  = df["vehicle_count"].quantile(0.30)
            high_min = df["vehicle_count"].quantile(0.70)
            df["label"] = df["vehicle_count"].apply(
                lambda v: "low" if v <= low_max else ("high" if v >= high_min else "medium")
            )

        total = len(df)
        dist  = df["label"].value_counts()
        print("  {:45s}  n={:4d}  low={:3d}  med={:3d}  high={:3d}".format(
            cam_id, total,
            dist.get("low",    0),
            dist.get("medium", 0),
            dist.get("high",   0),
        ))
        result_parts.append(df)

    return pd.concat(result_parts, ignore_index=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_labeling(detections_path, output_dir):
    # type: (Path, Path) -> None
    detections_path = Path(detections_path)
    output_dir      = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[label] Reading detections from {}".format(detections_path))
    det_df = pd.read_csv(detections_path)
    print("[label] {} frames  ({} cameras)".format(
        len(det_df), det_df["camera_id"].nunique()))

    print("\n[label] Assigning per-frame labels:")
    labeled_df = assign_frame_labels(det_df)

    # Output column order (keep only columns that exist)
    out_cols = [
        "camera_id", "role", "region", "window_id", "frame_idx", "timestamp",
        "file_path", "vehicle_count", "bbox_area_ratio", "bottom_roi_count",
        "mean_brightness", "label",
    ]
    out_cols = [c for c in out_cols if c in labeled_df.columns]
    labeled_df = labeled_df[out_cols].sort_values(["camera_id", "window_id"]).reset_index(drop=True)

    out_path = output_dir / "frame_labels_live_all.csv"
    labeled_df.to_csv(out_path, index=False)

    total = len(labeled_df)
    dist  = labeled_df["label"].value_counts()
    print("\n[label] Overall: {} frames  low={}({:.0f}%)  med={}({:.0f}%)  high={}({:.0f}%)".format(
        total,
        dist.get("low",    0), 100 * dist.get("low",    0) / total,
        dist.get("medium", 0), 100 * dist.get("medium", 0) / total,
        dist.get("high",   0), 100 * dist.get("high",   0) / total,
    ))
    print("[label] Saved → {}".format(out_path))
    print("\nNEXT STEP:")
    print("  python src/live_pipeline/build_dataset.py")


def main():
    parser = argparse.ArgumentParser(description="Per-frame congestion labeling from vehicle_count")
    parser.add_argument(
        "--detections", type=str,
        default=str(PROJECT_ROOT / "data" / "live" / "detections" / "detections.csv"),
        help="Path to detections CSV (default: data/live/detections/detections.csv)"
    )
    parser.add_argument(
        "--output", type=str,
        default=str(PROJECT_ROOT / "data" / "live" / "labels"),
        help="Output directory (default: data/live/labels)"
    )
    args = parser.parse_args()

    run_labeling(Path(args.detections), Path(args.output))


if __name__ == "__main__":
    main()
