"""
remove_overlays.py
------------------
Removes annotation overlays (red bounding boxes + text) from all extracted frames
using HSV colour masking and inpainting.

Overwrites frames in-place under data/processed/frames/{pair_id}/.

Run:
  python src/preprocessing/remove_overlays.py
  python src/preprocessing/remove_overlays.py --dry-run   # process 1 frame per pair only
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config.settings import CFG, PROJECT_ROOT


def remove_red_overlay(img_bgr):
    # type: (np.ndarray) -> np.ndarray
    hsv   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0,   150, 150]), np.array([8,   255, 255]))
    mask2 = cv2.inRange(hsv, np.array([172, 150, 150]), np.array([180, 255, 255]))
    mask  = cv2.bitwise_or(mask1, mask2)
    mask  = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    return cv2.inpaint(img_bgr, mask, inpaintRadius=4, flags=cv2.INPAINT_TELEA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Process only the first frame per pair (preview mode)")
    args = parser.parse_args()

    frames_root = PROJECT_ROOT / CFG["paths"]["frames_root"]
    pair_dirs   = sorted([d for d in frames_root.iterdir() if d.is_dir()])

    if not pair_dirs:
        print("[ERROR] No frame directories found under {}".format(frames_root))
        sys.exit(1)

    total_frames = sum(len(list(d.glob("*.jpg"))) for d in pair_dirs)
    print("=" * 60)
    print("remove_overlays.py — inpainting red annotation overlays")
    print("=" * 60)
    print("Pairs:  {}".format(len(pair_dirs)))
    print("Frames: {}{}".format(total_frames, "  (dry-run: 1 per pair)" if args.dry_run else ""))
    print()

    processed = 0
    for pair_dir in pair_dirs:
        frames = sorted(pair_dir.glob("*.jpg"))
        if args.dry_run:
            frames = frames[:1]

        for fpath in tqdm(frames, desc="Pair {}".format(pair_dir.name), leave=False):
            img    = cv2.imread(str(fpath))
            result = remove_red_overlay(img)
            cv2.imwrite(str(fpath), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
            processed += 1

    print("[OK] Processed {} frames — overlays removed in-place.".format(processed))
    if not args.dry_run:
        print("\nNEXT STEPS:")
        print("  python src/training/train.py --model baseline_cnn")
        print("  python src/training/train.py --model mobilenet_v2")
        print("  python src/training/train.py --model resnet50")
        print("  python src/training/train.py --model efficientnet_b0")


if __name__ == "__main__":
    main()
