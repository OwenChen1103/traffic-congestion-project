"""
settings.py
-----------
Central config loader and path helpers.
All pipeline scripts import CFG, PROJECT_ROOT, and helpers from here.
"""

import yaml
from pathlib import Path

# Project root = two levels up from src/config/settings.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config(path=CONFIG_PATH):
    # type: (Path) -> dict
    with open(path, "r") as f:
        return yaml.safe_load(f)


# Singleton — loaded once at import time
CFG = load_config()


def get_path(key):
    # type: (str) -> Path
    """Absolute path for a key under cfg['paths']."""
    return PROJECT_ROOT / CFG["paths"][key]


def get_pair_raw_dir(pair_id):
    # type: (str) -> Path
    """Absolute path to the raw data directory for a specific pair.
    Expected layout: data/raw/{pair_id}/intsc_data_{pair_id}.db + {pair_id}.avi
    """
    return get_path("raw_data") / str(pair_id)


def get_pair_frames_dir(pair_id):
    # type: (str) -> Path
    """Absolute path to the extracted frames directory for a specific pair."""
    return get_path("frames_root") / str(pair_id)


def get_per_pair_labels_dir():
    # type: () -> Path
    """Absolute path to the per-pair labels directory."""
    return get_path("per_pair_labels")


def get_pair_window_labels_path(pair_id):
    # type: (str) -> Path
    return get_per_pair_labels_dir() / "{}_window_labels_v1.csv".format(pair_id)


def get_pair_samples_metadata_path(pair_id):
    # type: (str) -> Path
    return get_per_pair_labels_dir() / "{}_samples_metadata_v1.csv".format(pair_id)
