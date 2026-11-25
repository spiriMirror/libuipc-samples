"""Locate shared sample assets and this case's output directory."""

import os
import pathlib

_CASE_ROOT = pathlib.Path(__file__).parent.resolve()
_SAMPLES_ROOT = _CASE_ROOT.parent.parent
_ASSETS = _SAMPLES_ROOT / 'assets' / 'sim_data'
_OUTPUT = _SAMPLES_ROOT / 'output'


def tetmesh_dir() -> str:
    return str(_ASSETS / 'tetmesh')


def trimesh_dir() -> str:
    return str(_ASSETS / 'trimesh')


def urdf_dir() -> str:
    return str(_ASSETS / 'urdf')


def case_dir() -> pathlib.Path:
    return _CASE_ROOT


def output_dir() -> str:
    """Output folder mirrors the case's path relative to the samples root."""
    target = _OUTPUT / _CASE_ROOT.relative_to(_SAMPLES_ROOT)
    os.makedirs(target, exist_ok=True)
    return str(target)
