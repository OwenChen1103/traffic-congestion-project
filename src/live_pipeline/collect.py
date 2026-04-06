"""
collect.py
----------
Captures frames from selected TfNSW live traffic cameras every 15 seconds.
Organises captures into 1-minute windows (4 frames per window per camera).

Setup:
    Create .env in project root:
        TFNSW_API_KEY=your_key_here

Usage:
    python src/live_pipeline/collect.py --duration 120   # collect for 120 minutes
    python src/live_pipeline/collect.py --duration 60    # collect for 60 minutes

Output layout:
    data/live/raw/
        {camera_id}/
            {window_id:04d}/
                frame_00.jpg   # t = 0s
                frame_01.jpg   # t = 15s
                frame_02.jpg   # t = 30s
                frame_03.jpg   # t = 45s
        manifest.csv           # one row per frame attempt
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ── .env loader (no extra dependency needed) ──────────────────────────────────

def _load_dotenv():
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

_load_dotenv()


# ── Selected cameras ──────────────────────────────────────────────────────────
# (role, camera_id, region)
# role drives the split later: train / val / test

CAMERAS = [
    # Sydney metro — train/val (roles finalised after data review via ROLE_OVERRIDE)
    ("train", "parramatta_road_camperdown",   "SYD_MET"),
    ("train", "hume_highway_bankstown",       "SYD_SOUTH"),
    ("train", "anzac_parade_moore_park",      "SYD_MET"),
    ("train", "james_ruse_drive_rosehill",    "SYD_WEST"),
    ("train", "princes_highway_st_peters_n",  "SYD_MET"),
    ("train", "city_road_newtown",            "SYD_MET"),
    ("train", "king_georges_road_hurstville", "SYD_SOUTH"),
    ("train", "5_ways_miranda",               "SYD_SOUTH"),
    # Wollongong region — test
    ("test",  "memorial_drive_towradgi",      "REG_WOLLONGONG"),
    ("test",  "shellharbour_road_warilla",    "REG_WOLLONGONG"),
]

_CAMERA_IDS  = {cam_id for _, cam_id, _ in CAMERAS}
_ROLE_MAP    = {cam_id: role   for role, cam_id, _      in CAMERAS}
_REGION_MAP  = {cam_id: region for _,    cam_id, region in CAMERAS}

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    "Referer": "https://www.livetraffic.com/",
}

CAMERA_API_URL = "https://api.transport.nsw.gov.au/v1/live/cameras"

FRAMES_PER_WINDOW = 4   # 4 × 15s = 60s per window
FRAME_INTERVAL    = 15  # seconds between frames


# ── Helpers ───────────────────────────────────────────────────────────────────

def fetch_hrefs(api_key):
    # type: (str) -> dict
    """Fetch camera catalog; return {camera_id: href} for our selected cameras."""
    headers = {
        "Authorization": "apikey {}".format(api_key),
        "Accept": "application/json",
    }
    resp = requests.get(CAMERA_API_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    hrefs = {}
    for feature in data.get("features", []):
        props = feature.get("properties", {})
        href = props.get("href")
        if not href:
            continue
        cam_id = os.path.splitext(os.path.basename(urlparse(href).path))[0]
        if cam_id in _CAMERA_IDS:
            hrefs[cam_id] = href

    return hrefs


def fetch_image(href):
    # type: (str) -> object
    """Download one camera snapshot. Returns PIL Image or None on failure."""
    try:
        resp = requests.get(
            href, headers=BROWSER_HEADERS, timeout=15, allow_redirects=True
        )
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "").lower()
        if not ctype.startswith("image/"):
            return None
        return Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


def _next_window_id(output_dir):
    # type: (Path) -> int
    """Scan existing camera subdirs to find the highest window_id used so far."""
    max_id = -1
    for cam_dir in output_dir.iterdir():
        if not cam_dir.is_dir() or cam_dir.name == "manifest.csv":
            continue
        for win_dir in cam_dir.iterdir():
            if win_dir.is_dir() and win_dir.name.isdigit():
                max_id = max(max_id, int(win_dir.name))
    return max_id + 1


# ── Main collection loop ──────────────────────────────────────────────────────

def collect(duration_minutes, output_dir, api_key):
    # type: (int, Path, str) -> None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[collect] Fetching camera catalog...")
    hrefs = fetch_hrefs(api_key)
    print("[collect] Resolved {}/{} camera hrefs".format(len(hrefs), len(_CAMERA_IDS)))
    for cam_id in _CAMERA_IDS:
        if cam_id not in hrefs:
            print("  [WARN] No href resolved for: {}".format(cam_id))

    if not hrefs:
        print("[ERROR] No cameras available. Check API key.")
        return

    # Resume from highest existing window_id (safe to re-run)
    window_id = _next_window_id(output_dir)
    frame_idx  = 0

    # Manifest — append mode so re-runs accumulate
    manifest_path = output_dir / "manifest.csv"
    write_header  = not manifest_path.exists()
    manifest_file = open(str(manifest_path), "a", newline="")
    writer = csv.writer(manifest_file)
    if write_header:
        writer.writerow([
            "camera_id", "role", "region",
            "window_id", "frame_idx",
            "timestamp", "file_path", "status",
        ])

    total_seconds = duration_minutes * 60
    start_time    = time.time()

    print("\n[collect] Starting — duration={}min  cameras={}".format(
        duration_minutes, len(hrefs)))
    print("          Output: {}".format(output_dir))
    print("          Starting window_id: {}".format(window_id))
    print()

    try:
        while time.time() - start_time < total_seconds:
            tick_start = time.time()
            timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            print("[w{:04d} f{}  {}]".format(window_id, frame_idx, timestamp))

            for cam_id, href in hrefs.items():
                img = fetch_image(href)

                if img is not None:
                    win_dir    = output_dir / cam_id / "{:04d}".format(window_id)
                    win_dir.mkdir(parents=True, exist_ok=True)
                    frame_path = win_dir / "frame_{:02d}.jpg".format(frame_idx)
                    img.save(str(frame_path), "JPEG", quality=90)
                    rel_path   = frame_path.relative_to(PROJECT_ROOT)
                    status     = "ok"
                    print("  ok   {}".format(cam_id))
                else:
                    rel_path = ""
                    status   = "fail"
                    print("  FAIL {}".format(cam_id))

                writer.writerow([
                    cam_id,
                    _ROLE_MAP.get(cam_id, "unknown"),
                    _REGION_MAP.get(cam_id, "unknown"),
                    window_id,
                    frame_idx,
                    timestamp,
                    str(rel_path),
                    status,
                ])

            manifest_file.flush()

            # Advance frame/window counters
            frame_idx += 1
            if frame_idx >= FRAMES_PER_WINDOW:
                frame_idx  = 0
                window_id += 1

            # Sleep until next 15s tick
            sleep_for = max(0.0, FRAME_INTERVAL - (time.time() - tick_start))
            if sleep_for > 0:
                time.sleep(sleep_for)

    except KeyboardInterrupt:
        print("\n[collect] Interrupted by user.")

    manifest_file.close()

    completed_windows = window_id if frame_idx == 0 else window_id
    print("\n[collect] Done.")
    print("          Windows collected : {}".format(completed_windows))
    print("          Manifest          : {}".format(manifest_path))
    print("\nNEXT STEP:")
    print("  python src/live_pipeline/detect.py")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Collect TfNSW live camera frames"
    )
    parser.add_argument(
        "--duration", type=int, default=60,
        help="Collection duration in minutes (default: 60)"
    )
    parser.add_argument(
        "--output", type=str,
        default=str(PROJECT_ROOT / "data" / "live" / "raw"),
        help="Output directory (default: data/live/raw)"
    )
    args = parser.parse_args()

    api_key = os.environ.get("TFNSW_API_KEY")
    if not api_key:
        print("[ERROR] TFNSW_API_KEY not set.")
        print("        Add to .env:  TFNSW_API_KEY=your_key_here")
        sys.exit(1)

    collect(args.duration, Path(args.output), api_key)


if __name__ == "__main__":
    main()
