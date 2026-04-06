"""
process_pairs.py
----------------
Pipeline Orchestrator — runs the full pipeline for ALL pairs in one command.

Steps per pair:
  1. generate_labels.py  → data/labels/per_pair/{pair_id}_window_labels_v1.csv
  2. extract_frames.py   → data/processed/frames/{pair_id}/*.jpg
                         → data/labels/per_pair/{pair_id}_samples_metadata_v1.csv

After all pairs:
  3. Merges all per-pair window_labels and samples_metadata CSVs into:
       data/labels/window_labels_v1_all.csv
       data/labels/samples_metadata_v1_all.csv
  4. build_splits.py     → data/processed/splits/{train,val,test}.csv
                         → data/labels/samples_split_v1_all.csv

Usage:
  python src/preprocessing/process_pairs.py              # all pairs in config
  python src/preprocessing/process_pairs.py --pairs 771 775 780

Pairs are discovered automatically from data/raw/ via discover_pairs.py.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config.settings import CFG, PROJECT_ROOT
from src.preprocessing.discover_pairs import discover_pairs
from src.labeling.generate_labels import run_pair as label_pair
from src.preprocessing.extract_frames import run_pair as extract_pair


# ── merge helpers ──────────────────────────────────────────────────────────────

def merge_window_labels(pair_ids):
    # type: (List[str]) -> Optional[pd.DataFrame]
    """Concatenate per-pair window label CSVs into one combined DataFrame."""
    from src.config.settings import get_pair_window_labels_path
    dfs = []
    for pid in pair_ids:
        p = get_pair_window_labels_path(pid)
        if p.exists():
            dfs.append(pd.read_csv(p))
        else:
            print("  [WARN] Missing window labels for pair {}: {}".format(pid, p.name))
    if not dfs:
        return None
    combined = pd.concat(dfs, ignore_index=True)
    return combined


def merge_samples_metadata(pair_ids):
    # type: (List[str]) -> Optional[pd.DataFrame]
    """Concatenate per-pair samples metadata CSVs, reassigning global sample_ids."""
    from src.config.settings import get_pair_samples_metadata_path
    dfs = []
    for pid in pair_ids:
        p = get_pair_samples_metadata_path(pid)
        if p.exists():
            dfs.append(pd.read_csv(p))
        else:
            print("  [WARN] Missing samples metadata for pair {}: {}".format(pid, p.name))
    if not dfs:
        return None
    combined = pd.concat(dfs, ignore_index=True)
    # Reassign global sample_id (pair-local IDs may overlap across pairs)
    combined = combined.reset_index(drop=True)
    combined["sample_id"] = combined.index
    return combined


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run the full pipeline for all (or specified) dataset pairs."
    )
    parser.add_argument(
        "--pairs", nargs="+", metavar="PAIR_ID",
        help="Specific pair IDs to process (default: all discovered pairs)"
    )
    parser.add_argument(
        "--skip-labels", action="store_true",
        help="Skip generate_labels step (re-use existing per-pair CSVs)"
    )
    parser.add_argument(
        "--skip-frames", action="store_true",
        help="Skip extract_frames step (re-use existing per-pair CSVs)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("process_pairs.py — multi-pair pipeline orchestrator")
    print("=" * 60)

    # ── Discover pairs ──────────────────────────────────────────────────────
    include = args.pairs if args.pairs else CFG["pairs"]["include"]
    pairs = discover_pairs(include)

    if not pairs:
        print("[ERROR] No valid pairs found in data/raw/")
        print("  Expected layout: data/raw/{pair_id}/intsc_data_{pair_id}.db + {pair_id}.avi")
        sys.exit(1)

    pair_ids = [p.pair_id for p in pairs]
    print("\nPairs to process ({}): {}".format(len(pair_ids), "  ".join(pair_ids)))

    # ── Per-pair processing ─────────────────────────────────────────────────
    label_ok   = []
    extract_ok = []
    failed     = []

    for pair in pairs:
        pid = pair.pair_id

        # Step 1: generate labels
        if args.skip_labels:
            from src.config.settings import get_pair_window_labels_path
            if get_pair_window_labels_path(pid).exists():
                print("\n[Pair {}] Skipping generate_labels (file exists)".format(pid))
                label_ok.append(pid)
            else:
                print("\n[Pair {}] [WARN] --skip-labels set but no label file found — running anyway".format(pid))
                result = label_pair(pid)
                if result is not None:
                    label_ok.append(pid)
                else:
                    failed.append(pid)
                    continue
        else:
            result = label_pair(pid)
            if result is not None:
                label_ok.append(pid)
            else:
                failed.append(pid)
                continue

        # Step 2: extract frames
        if args.skip_frames:
            from src.config.settings import get_pair_samples_metadata_path
            if get_pair_samples_metadata_path(pid).exists():
                print("[Pair {}] Skipping extract_frames (file exists)".format(pid))
                extract_ok.append(pid)
            else:
                print("[Pair {}] [WARN] --skip-frames set but no metadata file found — running anyway".format(pid))
                result = extract_pair(pid)
                if result is not None:
                    extract_ok.append(pid)
                else:
                    failed.append(pid)
        else:
            result = extract_pair(pid)
            if result is not None:
                extract_ok.append(pid)
            else:
                failed.append(pid)

    print("\n" + "=" * 60)
    print("Per-pair processing complete")
    print("  OK:     {} pairs".format(len(extract_ok)))
    print("  Failed: {} pairs{}".format(
        len(failed), "  ({})".format("  ".join(failed)) if failed else ""
    ))

    if not extract_ok:
        print("[ERROR] No pairs succeeded. Cannot build merged dataset.")
        sys.exit(1)

    # ── Merge per-pair outputs ──────────────────────────────────────────────
    print("\n[Merge] Combining per-pair CSVs...")

    labels_dir = PROJECT_ROOT / CFG["paths"]["labels"]
    labels_dir.mkdir(parents=True, exist_ok=True)

    window_labels_all = merge_window_labels(extract_ok)
    if window_labels_all is not None:
        out = PROJECT_ROOT / CFG["labeling"]["window_labels_all_csv"]
        out.parent.mkdir(parents=True, exist_ok=True)
        window_labels_all.to_csv(out, index=False)
        print("  window_labels_all: {} rows → {}".format(len(window_labels_all), out.name))
    else:
        print("  [WARN] No window labels to merge.")

    samples_all = merge_samples_metadata(extract_ok)
    if samples_all is not None:
        out = PROJECT_ROOT / CFG["labeling"]["samples_metadata_all_csv"]
        out.parent.mkdir(parents=True, exist_ok=True)
        samples_all.to_csv(out, index=False)
        print("  samples_metadata_all: {} rows → {}".format(len(samples_all), out.name))
    else:
        print("  [WARN] No samples metadata to merge.")
        sys.exit(1)

    # ── Build splits ────────────────────────────────────────────────────────
    print("\n[Splits] Running build_splits.py on merged dataset...")
    from src.preprocessing.build_splits import run as build_splits_run
    build_splits_run(samples_all)

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("  Pairs processed: {}".format(len(extract_ok)))
    print("\nNEXT STEPS:")
    print("  python src/training/train.py --model baseline_cnn")
    print("  python src/training/train.py --model mobilenet_v2")
    print("  python src/training/train.py --model resnet50")
    print("  python src/training/train.py --model efficientnet_b0")


if __name__ == "__main__":
    main()
