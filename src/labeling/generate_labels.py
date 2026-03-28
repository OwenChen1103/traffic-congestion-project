"""
generate_labels.py
------------------
Pipeline Step 2 (per-pair).

Purpose:
  Generates congestion labels for one data pair from its SQLite metadata.
  Must be run once per pair before extract_frames.py.

Input:
  data/raw/{pair_id}/intsc_data_{pair_id}.db
    Tables: TRACKS, TRAJECTORIES_{pair_id}, L1_ACTIONS (optional)

Output:
  data/labels/per_pair/{pair_id}_window_labels_v1.csv
    Columns:
      pair_id         str    Pair identifier (e.g. '771')
      window_id       int    Sequential window index within this pair (0-based)
      start_frame     int    Approximate start frame (floor(start_time * fps))
      end_frame       int    Approximate end frame
      start_time      float  Window start in seconds
      end_time        float  Window end in seconds
      vehicle_count   int    Unique vehicle tracks active in window
      avg_speed       float  Mean SPEED of vehicle trajectory rows (m/s)
      stop_proxy      float  Fraction of rows where SPEED < 0.5 m/s [0,1]
      congestion_score float Weighted composite [0,1]
      label           str    'low' | 'medium' | 'high'
      label_id        int    0 | 1 | 2

Schema notes:
  - Trajectories table is auto-detected as TRAJECTORIES_{pair_id} or any TRAJECTORIES_* table.
  - L1_ACTIONS is detected but NOT used for stop_proxy in v1 (action codes are opaque).
  - stop_proxy is speed-based only (SPEED < stop_speed_threshold m/s).

Run (single pair):
  python src/labeling/generate_labels.py --pair 771

Run via orchestrator (all pairs):
  python src/preprocessing/process_pairs.py
"""

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config.settings import (
    CFG, PROJECT_ROOT,
    get_pair_raw_dir,
    get_pair_window_labels_path,
    get_per_pair_labels_dir,
)


# ── database helpers ──────────────────────────────────────────────────────────

def get_db_path(pair_id):
    # type: (str) -> Path
    pair_dir = get_pair_raw_dir(pair_id)
    preferred = pair_dir / "intsc_data_{}.db".format(pair_id)
    if preferred.exists():
        return preferred
    candidates = list(pair_dir.glob("*.db")) + list(pair_dir.glob("*.sqlite"))
    if candidates:
        print("[WARN] Using DB: {}".format(candidates[0].name))
        return candidates[0]
    raise FileNotFoundError(
        "No database found in {}. "
        "Expected: intsc_data_{}.db".format(pair_dir, pair_id)
    )


def list_tables(con):
    # type: (sqlite3.Connection) -> List[str]
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    return [r[0] for r in cur.fetchall()]


def table_columns(con, table):
    # type: (sqlite3.Connection, str) -> List[str]
    try:
        cur = con.cursor()
        cur.execute("PRAGMA table_info({});".format(table))
        return [r[1].lower() for r in cur.fetchall()]
    except Exception:
        return []


def detect_schema(con, pair_id):
    # type: (sqlite3.Connection, str) -> Dict
    tables = list_tables(con)
    tables_upper = [t.upper() for t in tables]

    schema = {}  # type: Dict

    # ── TRACKS ──
    tc = table_columns(con, "TRACKS")
    schema["track_id_col"] = next(
        (c for c in ["track_id", "id", "trackid"] if c in tc), "track_id"
    )
    schema["track_type_col"] = next(
        (c for c in ["type", "class", "agent_type", "road_user_type", "object_type"] if c in tc),
        None,
    )

    # ── TRAJECTORIES — prefer TRAJECTORIES_{pair_id}, fall back to any TRAJECTORIES_* ──
    preferred_traj = "TRAJECTORIES_{}".format(pair_id.zfill(4))
    traj_table = None
    for t in tables:
        if t.upper() == preferred_traj.upper():
            traj_table = t
            break
    if traj_table is None:
        for t in tables:
            if t.upper().startswith("TRAJECTORIES_"):
                traj_table = t
                break
    if traj_table is None and "TRAJECTORIES" in tables_upper:
        traj_table = tables[tables_upper.index("TRAJECTORIES")]

    schema["traj_table"] = traj_table
    if traj_table:
        rc = table_columns(con, traj_table)
        schema["traj_track_col"] = next(
            (c for c in ["track_id", "trackid", "id"] if c in rc), "track_id"
        )
        schema["traj_time_col"] = next(
            (c for c in ["time", "timestamp", "t"] if c in rc), None
        )
        schema["traj_speed_col"] = next(
            (c for c in ["speed", "velocity"] if c in rc), None
        )
        schema["traj_xvel_col"] = next((c for c in ["xvelocity", "vx"] if c in rc), None)
        schema["traj_yvel_col"] = next((c for c in ["yvelocity", "vy"] if c in rc), None)
        print("  Trajectories: {}  cols: {}".format(traj_table, rc))
    else:
        raise RuntimeError(
            "No TRAJECTORIES table found for pair {}. "
            "Run inspect_dataset.py to check table names.".format(pair_id)
        )

    # ── L1_ACTIONS (optional) ──
    actions_table = None
    for t in tables:
        if t.upper() in ("L1_ACTIONS", "HIGH_LEVEL_ACTIONS", "ACTIONS"):
            actions_table = t
            break
    schema["actions_table"] = actions_table
    if actions_table:
        print("  Actions table: {}  (detected, not used in v1)".format(actions_table))

    print("  TRACKS: id_col={}, type_col={}".format(
        schema["track_id_col"], schema["track_type_col"]))
    return schema


# ── data loading ──────────────────────────────────────────────────────────────

def load_vehicle_track_ids(con, schema, vehicle_classes):
    # type: (sqlite3.Connection, Dict, List[str]) -> Set
    df = pd.read_sql("SELECT * FROM TRACKS", con)
    df.columns = df.columns.str.lower()

    type_col = schema["track_type_col"]
    id_col   = schema["track_id_col"]

    if type_col and type_col in df.columns:
        df = df[df[type_col].isin(vehicle_classes)].copy()
        print("  TRACKS: {} vehicle tracks (type in {})".format(len(df), vehicle_classes))
    else:
        print("  [WARN] No type column in TRACKS — using all {} tracks".format(len(df)))

    col = id_col if id_col in df.columns else df.columns[0]
    return set(df[col].tolist())


def load_trajectories(con, schema, vehicle_ids):
    # type: (sqlite3.Connection, Dict, Set) -> pd.DataFrame
    traj_table = schema["traj_table"]
    df = pd.read_sql("SELECT * FROM {}".format(traj_table), con)
    df.columns = df.columns.str.lower()

    track_col = schema["traj_track_col"]
    time_col  = schema["traj_time_col"]
    speed_col = schema["traj_speed_col"]
    xvel_col  = schema["traj_xvel_col"]
    yvel_col  = schema["traj_yvel_col"]

    if track_col in df.columns:
        df = df[df[track_col].isin(vehicle_ids)].copy()

    if time_col and time_col in df.columns:
        df = df.rename(columns={track_col: "_track_id", time_col: "_time"})
    else:
        raise RuntimeError(
            "No TIME column found in {}. Cannot assign windows.".format(traj_table)
        )

    if speed_col and speed_col in df.columns:
        df["_speed"] = df[speed_col].abs()
    elif xvel_col and yvel_col and xvel_col in df.columns and yvel_col in df.columns:
        df["_speed"] = np.sqrt(df[xvel_col] ** 2 + df[yvel_col] ** 2)
    else:
        df["_speed"] = np.nan

    df = df[["_track_id", "_time", "_speed"]].copy()
    if "_speed" not in df.columns:
        df["_speed"] = np.nan

    print("  Trajectories: {} rows, {} vehicle tracks".format(
        len(df), df["_track_id"].nunique()))
    return df


# ── window features ───────────────────────────────────────────────────────────

def compute_window_features(traj_df, fps, window_sec, stop_threshold):
    # type: (pd.DataFrame, float, int, float) -> pd.DataFrame
    df = traj_df.copy()
    df["window_id"] = (df["_time"] // window_sec).astype(int)

    rows = []
    for window_id, group in df.groupby("window_id"):
        start_time    = float(window_id * window_sec)
        end_time      = float((window_id + 1) * window_sec)
        vehicle_count = int(group["_track_id"].nunique())
        avg_speed     = float(group["_speed"].mean()) if not group["_speed"].isna().all() else float("nan")

        if not group["_speed"].isna().all():
            stop_proxy = float((group["_speed"] < stop_threshold).mean())
        else:
            stop_proxy = 0.0

        rows.append({
            "window_id":     int(window_id),
            "start_frame":   int(np.floor(start_time * fps)),
            "end_frame":     max(int(np.floor(end_time * fps)) - 1, int(np.floor(start_time * fps))),
            "start_time":    round(start_time, 3),
            "end_time":      round(end_time, 3),
            "vehicle_count": vehicle_count,
            "avg_speed":     round(avg_speed, 4) if not np.isnan(avg_speed) else float("nan"),
            "stop_proxy":    round(stop_proxy, 4),
        })

    return pd.DataFrame(rows).sort_values("window_id").reset_index(drop=True)


def assign_labels(features_df, cfg_label):
    # type: (pd.DataFrame, Dict) -> pd.DataFrame
    df = features_df.copy()

    lo_c, hi_c = df["vehicle_count"].quantile(0.05), df["vehicle_count"].quantile(0.95)
    df["norm_count"] = ((df["vehicle_count"] - lo_c) / (hi_c - lo_c + 1e-6)).clip(0, 1)

    if df["avg_speed"].isna().all():
        df["norm_speed_inv"] = 0.0
    else:
        lo_s, hi_s = df["avg_speed"].quantile(0.05), df["avg_speed"].quantile(0.95)
        df["norm_speed_inv"] = (1.0 - ((df["avg_speed"] - lo_s) / (hi_s - lo_s + 1e-6))).clip(0, 1)

    df["stop_proxy"] = df["stop_proxy"].clip(0, 1)

    w = cfg_label["weights"]
    df["congestion_score"] = (
        w["count"] * df["norm_count"]
        + w["speed"] * df["norm_speed_inv"]
        + w["stop"]  * df["stop_proxy"]
    ).round(4)

    low_max  = cfg_label["score_thresholds"]["low_max"]
    high_min = cfg_label["score_thresholds"]["high_min"]

    def to_label(s):
        if s <= low_max:   return "low"
        elif s >= high_min: return "high"
        return "medium"

    df["label"]    = df["congestion_score"].apply(to_label)
    df["label_id"] = df["label"].map({"low": 0, "medium": 1, "high": 2})
    df = df.drop(columns=["norm_count", "norm_speed_inv"], errors="ignore")
    return df


# ── FPS helper ────────────────────────────────────────────────────────────────

def get_fps(pair_id):
    # type: (str) -> float
    try:
        import cv2
        pair_dir = get_pair_raw_dir(pair_id)
        candidates = (
            [pair_dir / "{}.avi".format(pair_id)]
            + list(pair_dir.glob("*.avi"))
            + list(pair_dir.glob("*.mp4"))
        )
        for p in candidates:
            if p.exists():
                cap = cv2.VideoCapture(str(p))
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                if fps > 0:
                    return float(fps)
    except Exception:
        pass
    print("  [WARN] Could not read FPS — defaulting to 29.97")
    return 29.97


# ── main ──────────────────────────────────────────────────────────────────────

def run_pair(pair_id):
    # type: (str) -> Optional[pd.DataFrame]
    """
    Process one pair. Returns the labeled DataFrame, or None on failure.
    Called directly by process_pairs.py orchestrator.
    """
    print("\n[Pair {}] generate_labels".format(pair_id))

    cfg_label       = CFG["labeling"]
    vehicle_classes = CFG["pairs"]["vehicle_classes"]
    window_sec      = cfg_label["window_seconds"]
    stop_threshold  = cfg_label.get("stop_speed_threshold", 0.5)

    try:
        db_path = get_db_path(pair_id)
    except FileNotFoundError as e:
        print("  [ERROR] {}".format(e))
        return None

    con    = sqlite3.connect(db_path)
    schema = detect_schema(con, pair_id)

    vehicle_ids = load_vehicle_track_ids(con, schema, vehicle_classes)
    if not vehicle_ids:
        print("  [ERROR] No vehicle tracks found. Check vehicle_classes in config.yaml.")
        con.close()
        return None

    traj_df = load_trajectories(con, schema, vehicle_ids)
    con.close()

    if traj_df.empty:
        print("  [ERROR] Empty trajectory data.")
        return None

    fps   = get_fps(pair_id)
    t_min = traj_df["_time"].min()
    t_max = traj_df["_time"].max()
    print("  Time range: {:.1f}s – {:.1f}s  ({:.1f}s)  FPS={:.2f}".format(
        t_min, t_max, t_max - t_min, fps))

    features_df = compute_window_features(traj_df, fps, window_sec, stop_threshold)
    labeled_df  = assign_labels(features_df, cfg_label)

    # Add pair_id as first column
    labeled_df.insert(0, "pair_id", pair_id)

    # Enforce output column order
    output_cols = [
        "pair_id", "window_id", "start_frame", "end_frame", "start_time", "end_time",
        "vehicle_count", "avg_speed", "stop_proxy", "congestion_score", "label", "label_id",
    ]
    labeled_df = labeled_df[output_cols]

    # Distribution
    total = len(labeled_df)
    dist_parts = []
    for lbl in ["low", "medium", "high"]:
        cnt = int((labeled_df["label"] == lbl).sum())
        dist_parts.append("{}={}({:.0f}%)".format(lbl, cnt, 100.0 * cnt / total))
    print("  {} windows  [{}]".format(total, "  ".join(dist_parts)))

    # Save
    out_path = get_pair_window_labels_path(pair_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labeled_df.to_csv(out_path, index=False)
    print("  Saved → {}".format(out_path.name))

    return labeled_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", required=True, help="Pair ID to process, e.g. 771")
    args = parser.parse_args()

    result = run_pair(str(args.pair))
    if result is None:
        sys.exit(1)

    print("\nNEXT STEPS:")
    print("  python src/preprocessing/extract_frames.py --pair {}".format(args.pair))


if __name__ == "__main__":
    main()
