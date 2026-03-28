"""
build_splits.py
---------------
Pipeline Step 4.

Purpose:
  Assigns each sample to train, val, or test split using window-level stratified
  splitting across ALL pairs combined. Writes canonical training-ready CSVs.

Input (standalone mode):
  data/labels/samples_metadata_v1_all.csv   (merged output from process_pairs.py)
    Required columns: pair_id, sample_id, window_id, image_path, label, label_id

Input (programmatic mode):
  DataFrame passed directly from process_pairs.py via run(df)

Output:
  data/labels/samples_split_v1_all.csv      Final training-ready combined CSV
    Columns: sample_id, pair_id, window_id, image_path, label, label_id, split

  data/processed/splits/train.csv
  data/processed/splits/val.csv
  data/processed/splits/test.csv
  data/processed/splits/split_summary.txt

Split strategy:
  Window-level stratified split (always used):
    - A composite key (pair_id, window_id) uniquely identifies each window.
    - Windows are stratified by label, then split 70/15/15.
    - ALL frames from the same window stay in the same split.
    - Prevents near-duplicate frames (3 frames from same 5s window) from leaking
      across train/val/test boundaries.

Run standalone:
  python src/preprocessing/build_splits.py
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config.settings import CFG, PROJECT_ROOT


# ── split strategy ─────────────────────────────────────────────────────────────

def window_stratified_split(df, train_frac, val_frac, seed):
    # type: (pd.DataFrame, float, float, int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """
    Stratify at the window level across all pairs.

    Uses a composite key (pair_id, window_id) to uniquely identify windows,
    then splits by window label. All frames from a given window go to the same split.
    """
    # Build one row per unique (pair_id, window_id), with its label for stratification
    window_keys = (
        df[["pair_id", "window_id", "label"]]
        .drop_duplicates(subset=["pair_id", "window_id"])
        .reset_index(drop=True)
    )
    window_keys["_key"] = window_keys["pair_id"].astype(str) + "_" + window_keys["window_id"].astype(str)

    test_frac = 1.0 - train_frac - val_frac

    # First split: train vs (val + test)
    train_keys, temp_keys = train_test_split(
        window_keys["_key"],
        test_size=(1.0 - train_frac),
        stratify=window_keys.set_index("_key").loc[window_keys["_key"]]["label"].values,
        random_state=seed,
    )

    # Second split: val vs test from the remainder
    temp_windows = window_keys[window_keys["_key"].isin(temp_keys)].reset_index(drop=True)
    relative_val = val_frac / (val_frac + test_frac)

    val_keys, test_keys = train_test_split(
        temp_windows["_key"],
        test_size=(1.0 - relative_val),
        stratify=temp_windows["label"].values,
        random_state=seed,
    )

    df = df.copy()
    df["_key"] = df["pair_id"].astype(str) + "_" + df["window_id"].astype(str)

    train_df = df[df["_key"].isin(train_keys)].drop(columns=["_key"])
    val_df   = df[df["_key"].isin(val_keys)].drop(columns=["_key"])
    test_df  = df[df["_key"].isin(test_keys)].drop(columns=["_key"])

    return train_df, val_df, test_df


# ── reporting ──────────────────────────────────────────────────────────────────

def split_report(name, df):
    # type: (str, pd.DataFrame) -> List[str]
    n_windows = df[["pair_id", "window_id"]].drop_duplicates().shape[0] if "pair_id" in df.columns else df["window_id"].nunique()
    lines = ["\n{} ({} samples, {} windows):".format(name.upper(), len(df), n_windows)]
    for lbl in ["low", "medium", "high"]:
        cnt = int((df["label"] == lbl).sum())
        pct = 100.0 * cnt / len(df) if len(df) > 0 else 0.0
        lines.append("  {:8s}: {:4d}  ({:.1f}%)".format(lbl, cnt, pct))
    return lines


# ── core logic (callable from process_pairs.py) ────────────────────────────────

def run(df=None):
    # type: (Optional[pd.DataFrame]) -> None
    """
    Build and save train/val/test splits.

    Args:
        df: Pre-loaded samples DataFrame. If None, loads from samples_metadata_all_csv.
    """
    print("\n" + "=" * 60)
    print("build_splits.py — train/val/test split generation")
    print("=" * 60)

    cfg_split = CFG["split"]
    cfg_label = CFG["labeling"]
    seed      = CFG["project"]["seed"]

    if df is None:
        metadata_path = PROJECT_ROOT / cfg_label["samples_metadata_all_csv"]
        if not metadata_path.exists():
            print("[ERROR] Not found: {}".format(metadata_path))
            print("  Run: python src/preprocessing/process_pairs.py first")
            sys.exit(1)
        df = pd.read_csv(metadata_path)
        print("[Input] {} samples from {}".format(len(df), metadata_path.name))
    else:
        print("[Input] {} samples (passed from orchestrator)".format(len(df)))

    # Validate required columns
    required = {"pair_id", "sample_id", "window_id", "image_path", "label", "label_id"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        print("[ERROR] Missing columns in input: {}".format(missing_cols))
        sys.exit(1)

    # Drop rows with missing or invalid data
    before = len(df)
    df = df[df["image_path"].notna() & df["label"].isin(["low", "medium", "high"])].copy()
    if len(df) < before:
        print("[WARN] Dropped {} rows with missing image_path or invalid label".format(before - len(df)))

    if len(df) == 0:
        print("[ERROR] No valid samples to split.")
        sys.exit(1)

    # Sort for reproducibility
    df = df.sort_values(["pair_id", "window_id", "sample_id"]).reset_index(drop=True)

    # Summary of input
    n_pairs   = df["pair_id"].nunique()
    n_windows = df[["pair_id", "window_id"]].drop_duplicates().shape[0]
    print("  {} pairs  |  {} windows  |  {} samples".format(n_pairs, n_windows, len(df)))

    train_frac = cfg_split["train"]
    val_frac   = cfg_split["val"]

    print("[Split] Strategy: window-level stratified across all pairs")
    train_df, val_df, test_df = window_stratified_split(df, train_frac, val_frac, seed)

    train_df["split"] = "train"
    val_df["split"]   = "val"
    test_df["split"]  = "test"

    # Combined training-ready CSV
    output_cols = ["sample_id", "pair_id", "window_id", "image_path", "label", "label_id", "split"]
    combined = pd.concat([train_df, val_df, test_df]).sort_values("sample_id").reset_index(drop=True)

    split_csv_path = PROJECT_ROOT / cfg_label["samples_split_all_csv"]
    split_csv_path.parent.mkdir(parents=True, exist_ok=True)
    combined[output_cols].to_csv(split_csv_path, index=False)
    print("[OK] samples_split_v1_all.csv → {}".format(split_csv_path))

    # Per-split convenience files (for DataLoader)
    splits_dir = PROJECT_ROOT / cfg_split["output_dir"]
    splits_dir.mkdir(parents=True, exist_ok=True)

    # DataLoader-ready columns (image_path, label, label_id only)
    dl_cols = ["sample_id", "image_path", "label", "label_id"]
    for name, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
        sdf[dl_cols].to_csv(splits_dir / "{}.csv".format(name), index=False)

    # Summary report
    n_pairs_info = df["pair_id"].nunique() if "pair_id" in df.columns else "?"
    summary_lines = [
        "Split Summary",
        "=" * 40,
        "Strategy:      window-level stratified (all pairs combined)",
        "Total samples: {}".format(len(combined)),
        "Total windows: {}".format(n_windows),
        "Pairs:         {}".format(n_pairs_info),
    ]
    for name, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
        summary_lines.extend(split_report(name, sdf))

    # Class imbalance warning
    train_low_pct = 100.0 * (train_df["label"] == "low").mean()
    if train_low_pct < 15.0:
        summary_lines.append(
            "\n[WARN] Low class underrepresented in train ({:.1f}%).".format(train_low_pct)
        )
        summary_lines.append(
            "  Class weights are applied automatically in train.py to compensate."
        )

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    summary_path = splits_dir / "split_summary.txt"
    summary_path.write_text(summary_text)
    print("\n[OK] Summary → {}".format(summary_path))

    print("\nNEXT STEPS:")
    print("  python src/training/train.py --model baseline_cnn")


# ── CLI entry point ────────────────────────────────────────────────────────────

def main():
    run(df=None)


if __name__ == "__main__":
    main()
