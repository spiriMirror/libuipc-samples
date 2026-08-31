"""Shared helpers for reproducible headless sample benchmarks."""

from __future__ import annotations

import json
import os
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from uipc import Timer


def timer_profiling_enabled() -> bool:
    """Return whether synchronized Timer scopes should be collected."""
    return os.environ.get("UIPC_BENCHMARK_TIMERS", os.environ.get("WB_TIMER", "0")) == "1"


def configure_benchmark_timers() -> None:
    """Keep throughput runs timer-free and enable scopes only on request."""
    if timer_profiling_enabled():
        Timer.enable_all()
    else:
        Timer.disable_all()


def snapshot_frame_stats(engine: Any) -> dict[str, Any]:
    """Copy the backend's small per-frame statistics object."""
    return dict(engine.frame_stats())


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def emit_benchmark_result(
    frame_ms: Sequence[float],
    frame_stats: Sequence[Mapping[str, Any]],
    *,
    observables: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Print the stable summary and one machine-readable result line."""
    if not frame_ms:
        raise ValueError("A benchmark result requires at least one frame")
    if len(frame_ms) != len(frame_stats):
        raise ValueError("frame_ms and frame_stats must have the same length")

    payload = {
        "frame_ms": [float(value) for value in frame_ms],
        "frame_stats": [dict(value) for value in frame_stats],
        "observables": dict(observables or {}),
    }
    mean_ms = statistics.mean(frame_ms)
    median_ms = statistics.median(frame_ms)
    p95_ms = _percentile(frame_ms, 0.95)
    print(
        f"TOTAL frames={len(frame_ms)} mean={mean_ms:.1f}ms "
        f"median={median_ms:.1f}ms p95={p95_ms:.1f}ms",
        flush=True,
    )
    print(
        "BENCHMARK_RESULT " + json.dumps(payload, separators=(",", ":")),
        flush=True,
    )
    return payload


def report_timers_if_enabled() -> None:
    """Print the merged synchronized timer tree for diagnostic runs only."""
    if timer_profiling_enabled():
        Timer.report()
