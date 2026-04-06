"""
inspect_dataset.py
------------------
Step 1 of the pipeline. Run this FIRST after placing the dataset in data/raw/.

What it does:
  - Connects to the SQLite database
  - Prints all table names and their schemas
  - Prints row counts and sample rows for key tables
  - Prints video file metadata (duration, FPS, resolution)
  - Writes a summary report to outputs/reports/dataset_inspection.txt

Run:
    python src/preprocessing/inspect_dataset.py
"""

import sqlite3
import sys
import textwrap
from pathlib import Path
from typing import List, Optional
import cv2

# Allow running from project root without installing as package
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config.settings import CFG, PROJECT_ROOT, get_path


def get_db_path() -> Path:
    raw_dir = get_path("raw_data")
    db_name = CFG["dataset"]["db_filename"]
    p = raw_dir / db_name
    if not p.exists():
        # Try to find any .sqlite / .db file in raw/
        candidates = list(raw_dir.glob("*.sqlite")) + list(raw_dir.glob("*.db"))
        if candidates:
            print(f"[WARN] db_filename '{db_name}' not found. Using: {candidates[0].name}")
            return candidates[0]
        raise FileNotFoundError(
            f"No SQLite database found in {raw_dir}. "
            f"Please place the dataset there and update config.yaml."
        )
    return p


def get_video_path() -> Optional[Path]:
    raw_dir = get_path("raw_data")
    vid_name = CFG["dataset"]["video_filename"]
    p = raw_dir / vid_name
    if not p.exists():
        candidates = list(raw_dir.glob("*.mp4")) + list(raw_dir.glob("*.avi"))
        if candidates:
            print(f"[WARN] video_filename '{vid_name}' not found. Using: {candidates[0].name}")
            return candidates[0]
        print(f"[WARN] No video file found in {raw_dir}. Skipping video inspection.")
        return None
    return p


def inspect_database(db_path: Path, out_lines: List[str]) -> None:
    def log(msg: str = ""):
        print(msg)
        out_lines.append(msg)

    log(f"DATABASE: {db_path}")
    log("=" * 60)

    con = sqlite3.connect(db_path)
    cur = con.cursor()

    # All tables
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = [r[0] for r in cur.fetchall()]
    log(f"\nTables found ({len(tables)}): {tables}\n")

    for table in tables:
        log(f"--- TABLE: {table} ---")

        # Schema
        cur.execute(f"PRAGMA table_info({table});")
        cols = cur.fetchall()
        col_summary = ", ".join(f"{c[1]}({c[2]})" for c in cols)
        log(f"  Columns: {col_summary}")

        # Row count
        cur.execute(f"SELECT COUNT(*) FROM {table};")
        count = cur.fetchone()[0]
        log(f"  Row count: {count}")

        # Sample rows (up to 3)
        if count > 0:
            cur.execute(f"SELECT * FROM {table} LIMIT 3;")
            rows = cur.fetchall()
            col_names = [c[1] for c in cols]
            log(f"  Sample rows (columns: {col_names}):")
            for row in rows:
                log(f"    {row}")

        log()

    # Key statistics for labeling
    log("KEY STATISTICS FOR LABELING")
    log("=" * 60)

    # Vehicle type distribution (TRACKS table)
    for possible_col in ["type", "class", "agent_type", "object_type", "road_user_type"]:
        try:
            cur.execute(f"SELECT {possible_col}, COUNT(*) FROM TRACKS GROUP BY {possible_col};")
            rows = cur.fetchall()
            log(f"\nTRACKS.{possible_col} distribution:")
            for r in rows:
                log(f"  {r[0]}: {r[1]}")
            break
        except sqlite3.OperationalError:
            continue

    # Speed range in TRAJECTORIES (look for xVelocity, yVelocity, speed, or similar)
    try:
        cur.execute("PRAGMA table_info(TRAJECTORIES);")
        traj_cols = [c[1].lower() for c in cur.fetchall()]
        speed_col = next(
            (c for c in ["speed", "xvelocity", "xVelocity", "velocity"] if c.lower() in traj_cols),
            None
        )
        if speed_col:
            # Use original case
            cur.execute("PRAGMA table_info(TRAJECTORIES);")
            orig = {c[1].lower(): c[1] for c in cur.fetchall()}
            real_col = orig[speed_col.lower()]
            cur.execute(f"SELECT MIN({real_col}), MAX({real_col}), AVG({real_col}) FROM TRAJECTORIES;")
            mn, mx, av = cur.fetchone()
            log(f"\nTRAJECTORIES.{real_col}: min={mn:.3f}, max={mx:.3f}, avg={av:.3f}")
    except Exception as e:
        log(f"\n[WARN] Could not inspect TRAJECTORIES speed column: {e}")

    # Time range
    for t_table in ["TRAJECTORIES", "TRACKS"]:
        for t_col in ["frame", "frameNum", "frame_id", "time", "timestamp"]:
            try:
                cur.execute(f"SELECT MIN({t_col}), MAX({t_col}) FROM {t_table};")
                mn, mx = cur.fetchone()
                log(f"\n{t_table}.{t_col} range: {mn} → {mx}")
                break
            except sqlite3.OperationalError:
                continue

    con.close()


def inspect_video(video_path: Path, out_lines: List[str]) -> None:
    def log(msg: str = ""):
        print(msg)
        out_lines.append(msg)

    log(f"\nVIDEO: {video_path}")
    log("=" * 60)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log("[ERROR] Could not open video file.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = total_frames / fps if fps > 0 else 0

    log(f"  FPS: {fps}")
    log(f"  Total frames: {total_frames}")
    log(f"  Resolution: {width} x {height}")
    log(f"  Duration: {duration_s:.1f} seconds ({duration_s/60:.1f} minutes)")
    log(f"  Estimated 10s windows: {int(duration_s // 10)}")
    cap.release()


def main():
    report_dir = get_path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "dataset_inspection.txt"

    out_lines: List[str] = []

    try:
        db_path = get_db_path()
        inspect_database(db_path, out_lines)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    video_path = get_video_path()
    if video_path:
        inspect_video(video_path, out_lines)

    # Write report
    report_path.write_text("\n".join(out_lines))
    print(f"\n[OK] Report saved to: {report_path}")
    print("\nNEXT STEPS:")
    print("  1. Review outputs/reports/dataset_inspection.txt")
    print("  2. Update src/config/config.yaml:")
    print("     - dataset.db_filename / video_filename (actual filenames)")
    print("     - dataset.vehicle_classes (actual type strings from TRACKS)")
    print("     - labeling.thresholds (calibrate from actual speed/count ranges)")
    print("  3. Run: python src/labeling/generate_labels.py")


if __name__ == "__main__":
    main()
