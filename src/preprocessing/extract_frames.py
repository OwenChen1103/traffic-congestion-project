"""
extract_frames.py
-----------------
Pipeline Step 3 (per-pair).

Purpose:
  Extracts representative JPEG frames per labeled window from the source video.
  Writes a per-pair metadata CSV mapping each extracted sample to its window label.
  Must be run after generate_labels.py for the same pair.

Input:
  data/labels/per_pair/{pair_id}_window_labels_v1.csv   (from generate_labels.py)
  data/raw/{pair_id}/{pair_id}.avi  (or any .avi/.mp4 in that directory)

Output:
  data/processed/frames/{pair_id}/w<NNNNN>_f<NN>.jpg
  data/labels/per_pair/{pair_id}_samples_metadata_v1.csv
    Columns:
      pair_id      str    Pair identifier (e.g. '771')
      sample_id    int    Sequential index (0-based, pair-local)
      window_id    int    References window_labels_v1.csv
      image_path   str    Relative path from project root
      timestamp    float  Time in seconds of this specific frame
      label        str    'low' | 'medium' | 'high'
      label_id     int    0 | 1 | 2

Frame extraction strategies (config.yaml → frame_extraction.strategy):
  'middle'  Single frame at temporal midpoint of window (deterministic)
  'random'  Single random frame within window (non-deterministic)
  'multi'   N evenly-spaced frames across the window (default)
            frames_per_window controls how many (e.g. 3)

Run (single pair):
  python src/preprocessing/extract_frames.py --pair 771

Run via orchestrator (all pairs):
  python src/preprocessing/process_pairs.py
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config.settings import (
    CFG, PROJECT_ROOT,
    get_pair_raw_dir,
    get_pair_frames_dir,
    get_pair_window_labels_path,
    get_pair_samples_metadata_path,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def get_video_path(pair_id):
    # type: (str) -> Path
    pair_dir = get_pair_raw_dir(pair_id)
    preferred = pair_dir / "{}.avi".format(pair_id)
    if preferred.exists():
        return preferred
    candidates = list(pair_dir.glob("*.avi")) + list(pair_dir.glob("*.mp4"))
    if candidates:
        print("  [WARN] Using video: {}".format(candidates[0].name))
        return candidates[0]
    raise FileNotFoundError(
        "No video file found in {}. "
        "Expected: {}.avi".format(pair_dir, pair_id)
    )


def get_frame_indices(start_frame, end_frame, strategy, n_frames):
    # type: (int, int, str, int) -> List[int]
    """
    Return a list of frame indices to extract from the window [start_frame, end_frame].
    """
    total = end_frame - start_frame + 1
    if strategy == "middle":
        return [(start_frame + end_frame) // 2]
    elif strategy == "random":
        import random
        return [random.randint(start_frame, end_frame)]
    elif strategy == "multi":
        if n_frames <= 1:
            return [(start_frame + end_frame) // 2]
        # Evenly spaced interior points: divide into n_frames+1 segments
        indices = []
        for i in range(1, n_frames + 1):
            pos = start_frame + int(total * i / (n_frames + 1))
            pos = min(pos, end_frame)
            indices.append(pos)
        return indices
    else:
        raise ValueError(
            "Unknown strategy '{}'. Use 'middle', 'random', or 'multi'.".format(strategy)
        )


def write_frame(cap, frame_idx, out_path, img_size):
    # type: (cv2.VideoCapture, int, Path, Tuple[int, int]) -> bool
    # img_size is (H, W); cv2.resize takes (W, H)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        return False
    resized = cv2.resize(frame, (img_size[1], img_size[0]))
    cv2.imwrite(str(out_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return True


# ── per-pair entry point ───────────────────────────────────────────────────────

def run_pair(pair_id):
    # type: (str) -> Optional[pd.DataFrame]
    """
    Process one pair. Returns the samples metadata DataFrame, or None on failure.
    Called directly by process_pairs.py orchestrator.
    """
    print("\n[Pair {}] extract_frames".format(pair_id))

    cfg_fe   = CFG["frame_extraction"]
    strategy = cfg_fe["strategy"]
    n_frames = int(cfg_fe.get("frames_per_window", 1))
    img_size = tuple(cfg_fe["image_size"])  # [H, W] from yaml

    print("  Strategy: {}  |  frames_per_window: {}".format(strategy, n_frames))

    # Load window labels for this pair
    window_labels_path = get_pair_window_labels_path(pair_id)
    if not window_labels_path.exists():
        print("  [ERROR] Not found: {}".format(window_labels_path))
        print("  Run: python src/labeling/generate_labels.py --pair {}".format(pair_id))
        return None

    windows_df = pd.read_csv(window_labels_path)
    print("  {} windows loaded".format(len(windows_df)))

    # Open video
    try:
        video_path = get_video_path(pair_id)
    except FileNotFoundError as e:
        print("  [ERROR] {}".format(e))
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("  [ERROR] Cannot open video: {}".format(video_path))
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97

    # Output directory for this pair's frames
    frames_dir = get_pair_frames_dir(pair_id)
    frames_dir.mkdir(parents=True, exist_ok=True)

    sample_rows = []
    skipped     = 0
    sample_id   = 0

    for _, row in tqdm(windows_df.iterrows(), total=len(windows_df),
                       desc="  Pair {}".format(pair_id), leave=False):
        wid         = int(row["window_id"])
        start_frame = int(row["start_frame"])
        end_frame   = int(row["end_frame"])

        frame_indices = get_frame_indices(start_frame, end_frame, strategy, n_frames)

        for fi, frame_idx in enumerate(frame_indices):
            fname    = "w{:05d}_f{:02d}.jpg".format(wid, fi)
            out_path = frames_dir / fname

            ok = write_frame(cap, frame_idx, out_path, img_size)
            if not ok:
                skipped += 1
                continue

            sample_rows.append({
                "pair_id":    pair_id,
                "sample_id":  sample_id,
                "window_id":  wid,
                "image_path": str(out_path.relative_to(PROJECT_ROOT)),
                "timestamp":  round(frame_idx / fps, 3),
                "label":      row["label"],
                "label_id":   int(row["label_id"]),
            })
            sample_id += 1

    cap.release()

    if not sample_rows:
        print("  [ERROR] No frames extracted. Check video file and window frame ranges.")
        return None

    output_cols = ["pair_id", "sample_id", "window_id", "image_path", "timestamp", "label", "label_id"]
    metadata_df = pd.DataFrame(sample_rows)[output_cols]

    out_path = get_pair_samples_metadata_path(pair_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_df.to_csv(out_path, index=False)

    total = len(metadata_df)
    dist_parts = []
    for lbl in ["low", "medium", "high"]:
        cnt = int((metadata_df["label"] == lbl).sum())
        dist_parts.append("{}={}({:.0f}%)".format(lbl, cnt, 100.0 * cnt / total))
    print("  {} samples  [{}]  (skipped {})".format(total, "  ".join(dist_parts), skipped))
    print("  Saved → {}".format(out_path.name))

    return metadata_df


# ── CLI entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", required=True, help="Pair ID to process, e.g. 771")
    args = parser.parse_args()

    result = run_pair(str(args.pair))
    if result is None:
        sys.exit(1)

    print("\nNEXT STEPS:")
    print("  python src/preprocessing/build_splits.py")
    print("  (or run all pairs: python src/preprocessing/process_pairs.py)")


if __name__ == "__main__":
    main()
