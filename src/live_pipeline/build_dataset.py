"""
build_dataset.py
----------------
Converts per-frame labeled data into train/val/test split CSVs
compatible with the existing CongestionDataset and train.py.

Split strategy:
    - test  cameras (role=test)  → test.csv  only
    - val   cameras (role=val)   → val.csv   only
    - train cameras (role=train) → window-level stratified split
                                   (all frames from a window stay together)
                                   majority label per window used for stratify

Output:
    data/live/splits/train.csv
    data/live/splits/val.csv
    data/live/splits/test.csv

    Columns: image_path (relative to PROJECT_ROOT), label, camera_id, window_id
    → directly readable by CongestionDataset with zero changes

Usage:
    python src/live_pipeline/build_dataset.py
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Fraction of train-role windows used for val top-up
TRAIN_VAL_FRACTION = 0.15

# Cameras to exclude from the training split (label quality too low)
EXCLUDE_FROM_TRAIN = {"princes_highway_st_peters_n"}  # severe glare → YOLO blind

# Override roles for specific cameras (overrides what is stored in the CSV)
ROLE_OVERRIDE = {}


# ── Main ──────────────────────────────────────────────────────────────────────

def build(labels_path, output_dir, seed=42, min_frame_brightness=80.0):
    labels_path = Path(labels_path)
    output_dir  = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_df = pd.read_csv(labels_path)

    # Apply role overrides
    if ROLE_OVERRIDE:
        frame_df["role"] = frame_df["camera_id"].map(ROLE_OVERRIDE).fillna(frame_df["role"])
        for cam, new_role in ROLE_OVERRIDE.items():
            print("[build_dataset] Role override: {} → {}".format(cam, new_role))

    # Filter nighttime frames by timestamp hour
    if "timestamp" in frame_df.columns:
        frame_df["_hour"] = pd.to_datetime(frame_df["timestamp"]).dt.hour
        before = len(frame_df)
        frame_df = frame_df[
            (frame_df["_hour"] >= 6) & (frame_df["_hour"] < 18)
        ].copy()
        frame_df = frame_df.drop(columns=["_hour"])
        print("[build_dataset] Time filter (06:00-18:00): kept {}/{} frames  (dropped {})".format(
            len(frame_df), before, before - len(frame_df)))

    # Per-frame brightness filter
    if "mean_brightness" in frame_df.columns:
        before = len(frame_df)
        frame_df = frame_df[frame_df["mean_brightness"] >= min_frame_brightness].copy()
        if before != len(frame_df):
            print("[build_dataset] Brightness filter (>={:.0f}): kept {}/{} frames  (dropped {})".format(
                min_frame_brightness, len(frame_df), before, before - len(frame_df)))

    print("[build_dataset] {} frames remaining after filters".format(len(frame_df)))

    train_samples = []
    val_samples   = []
    test_samples  = []

    for role in ["train", "val", "test"]:
        role_df = frame_df[frame_df["role"] == role]
        if role_df.empty:
            continue

        for cam_id, cam_df in role_df.groupby("camera_id"):
            if role == "test":
                test_samples.append(cam_df)

            elif role == "val":
                val_samples.append(cam_df)

            elif role == "train":
                if cam_id in EXCLUDE_FROM_TRAIN:
                    print("  [SKIP] {} (excluded from train)".format(cam_id))
                    continue

                # Window-level stratified split to avoid leakage.
                # Use majority label per window for stratification.
                window_majority = (
                    cam_df.groupby("window_id")["label"]
                    .agg(lambda x: x.value_counts().index[0])
                    .reset_index()
                )
                windows_list = window_majority["window_id"].tolist()
                labels_list  = window_majority["label"].tolist()

                if len(windows_list) < 6:
                    tr_wins = set(windows_list)
                    va_wins = set()
                else:
                    try:
                        tr_wins_list, va_wins_list = train_test_split(
                            windows_list,
                            test_size=TRAIN_VAL_FRACTION,
                            stratify=labels_list,
                            random_state=seed,
                        )
                    except ValueError:
                        tr_wins_list, va_wins_list = train_test_split(
                            windows_list,
                            test_size=TRAIN_VAL_FRACTION,
                            stratify=None,
                            random_state=seed,
                        )
                    tr_wins = set(tr_wins_list)
                    va_wins = set(va_wins_list)

                tr_samp = cam_df[cam_df["window_id"].isin(tr_wins)]
                va_samp = cam_df[cam_df["window_id"].isin(va_wins)]
                train_samples.append(tr_samp)
                if not va_samp.empty:
                    val_samples.append(va_samp)

    # Concatenate and save
    splits = {
        "train": pd.concat(train_samples, ignore_index=True) if train_samples else pd.DataFrame(),
        "val":   pd.concat(val_samples,   ignore_index=True) if val_samples   else pd.DataFrame(),
        "test":  pd.concat(test_samples,  ignore_index=True) if test_samples  else pd.DataFrame(),
    }

    print("\n[build_dataset] Split summary:")
    for split_name, df in splits.items():
        if df.empty:
            print("  {:5s}  0 samples".format(split_name))
            continue

        out_path = output_dir / "{}.csv".format(split_name)
        # Rename file_path → image_path for CongestionDataset compatibility
        df = df.rename(columns={"file_path": "image_path"})
        df[["image_path", "label", "camera_id", "window_id"]].to_csv(out_path, index=False)

        dist  = df["label"].value_counts()
        total = len(df)
        print("  {:5s}  {:4d} frames  low={:3d}  med={:3d}  high={:3d}  →  {}".format(
            split_name, total,
            dist.get("low",    0),
            dist.get("medium", 0),
            dist.get("high",   0),
            out_path,
        ))

    print("\n[build_dataset] Done.")
    print("\nNEXT STEP:")
    print("  python src/training/train.py --model mobilenet_v2 \\")
    print("      --split-dir data/live/splits")


def main():
    parser = argparse.ArgumentParser(description="Build train/val/test split from per-frame labels")
    parser.add_argument(
        "--labels", type=str,
        default=str(PROJECT_ROOT / "data" / "live" / "labels" / "frame_labels_live_all.csv"),
    )
    parser.add_argument(
        "--output", type=str,
        default=str(PROJECT_ROOT / "data" / "live" / "splits"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-frame-brightness", type=float, default=80.0,
        help="Skip frames with mean_brightness below this (default: 80)"
    )
    args = parser.parse_args()

    build(args.labels, args.output, args.seed, args.min_frame_brightness)


if __name__ == "__main__":
    main()
