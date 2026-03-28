"""
discover_pairs.py
-----------------
Utility: scan data/raw/ for valid db+video pairs.

A valid pair is a subdirectory of data/raw/ that contains:
  - at least one .db or .sqlite file
  - at least one .avi or .mp4 file

The pair_id is the subdirectory name (e.g. '771').

Can be run standalone for a diagnostic report, or imported by process_pairs.py.

Run:
  python src/preprocessing/discover_pairs.py

Outputs to stdout only (no files written).
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config.settings import CFG, PROJECT_ROOT, get_path


@dataclass
class PairInfo:
    pair_id: str
    db_path: Path
    video_path: Path

    def __str__(self):
        return "Pair {:>4}  db={}  video={}".format(
            self.pair_id, self.db_path.name, self.video_path.name
        )


def discover_pairs(include=None):
    # type: (Optional[object]) -> List[PairInfo]
    """
    Scan data/raw/ for valid pairs.

    Args:
        include: None or 'all'  → return all valid pairs
                 list of str/int → return only those pair IDs (e.g. ['769', '771'])

    Returns:
        Sorted list of PairInfo for valid pairs only.
    """
    raw_root = get_path("raw_data")
    if not raw_root.exists():
        print("[ERROR] data/raw/ not found: {}".format(raw_root))
        return []

    # Normalise include list
    if include is None or include == "all":
        filter_ids = None
    else:
        filter_ids = set(str(p) for p in include)

    valid = []
    missing = []

    for entry in sorted(raw_root.iterdir()):
        if not entry.is_dir():
            continue

        pair_id = entry.name
        if filter_ids is not None and pair_id not in filter_ids:
            continue

        # Find DB file
        db_candidates = list(entry.glob("*.db")) + list(entry.glob("*.sqlite"))
        # Find video file
        vid_candidates = list(entry.glob("*.avi")) + list(entry.glob("*.mp4"))

        if not db_candidates and not vid_candidates:
            continue  # empty or unrelated dir, skip silently

        db_path = db_candidates[0] if db_candidates else None
        vid_path = vid_candidates[0] if vid_candidates else None

        if db_path and vid_path:
            # Prefer file matching expected naming pattern
            preferred_db = entry / "intsc_data_{}.db".format(pair_id)
            preferred_vid = entry / "{}.avi".format(pair_id)
            if preferred_db.exists():
                db_path = preferred_db
            if preferred_vid.exists():
                vid_path = preferred_vid
            valid.append(PairInfo(pair_id=pair_id, db_path=db_path, video_path=vid_path))
        else:
            missing.append((pair_id, "no db" if not db_candidates else "no video"))

    if missing:
        print("[WARN] Incomplete pairs (skipped):")
        for pid, reason in missing:
            print("  {:>4}: {}".format(pid, reason))

    return valid


def main():
    print("=" * 60)
    print("discover_pairs.py — scanning data/raw/")
    print("=" * 60)

    include = CFG["pairs"]["include"]
    pairs = discover_pairs(include)

    if not pairs:
        print("[ERROR] No valid pairs found in data/raw/")
        print("  Expected structure: data/raw/{pair_id}/intsc_data_{pair_id}.db + {pair_id}.avi")
        sys.exit(1)

    print("\nValid pairs found: {}\n".format(len(pairs)))
    for p in pairs:
        print("  " + str(p))

    print("\nTo process all pairs, run:")
    print("  python src/preprocessing/process_pairs.py")
    print("\nTo process a single pair, run:")
    print("  python src/labeling/generate_labels.py --pair 771")
    print("  python src/preprocessing/extract_frames.py --pair 771")


if __name__ == "__main__":
    main()
