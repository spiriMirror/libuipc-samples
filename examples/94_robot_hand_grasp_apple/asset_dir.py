"""Locate shared assets and this example's output directory."""

import os
from pathlib import Path


CASE_DIR = Path(__file__).resolve().parent
SAMPLES_ROOT = CASE_DIR.parent.parent
ASSET_ROOT = SAMPLES_ROOT / "assets" / "sim_data"


def urdf_path() -> Path:
    return ASSET_ROOT / "urdf" / "robot_hand" / "robot_hand.urdf"


def output_dir() -> Path:
    path = SAMPLES_ROOT / "output" / "examples" / CASE_DIR.name
    os.makedirs(path, exist_ok=True)
    return path
