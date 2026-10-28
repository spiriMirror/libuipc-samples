# libuipc-samples

A sample library for Libuipc.

This library is a collection of sample programs that demonstrate how to use the Libuipc library.

## Featured robotics example

[`94_robot_hand_grasp_apple`](examples/94_robot_hand_grasp_apple/) loads the
sample-87 URDF hand and performs a verified four-finger pick, transfer and
release of a free ABD apple. Run its GUI or use `python main.py --headless 500`
for the full numerical validation.

## Directional cloth example

[`95_anisotropic_cloth`](examples/95_anisotropic_cloth/) compares shared stretch,
a-stiff and b-stiff sheets under identical biaxial loading. Run its GUI or
`python main.py --headless 120 --validate` for numerical checks. It requires
pyuipc built with the new directional cloth API; see the example README.

## Headless performance workloads

Examples 6, 88, 89, 93 and 95 support `python main.py --headless <frames>`
without opening a viewer. They emit `BENCHMARK_RESULT` containing per-frame
wall times, backend iteration/completion statistics and final physical
observables. Throughput runs disable synchronized Timer scopes by default;
set `UIPC_BENCHMARK_TIMERS=1` for a separate diagnostic run.

Start with example 95 using `--resolution 8 --headless 30` for a small procedural
cloth workload. The parent repository owns the versioned workloads and
[comparison protocol](https://github.com/spiriMirror/libuipc/blob/main/docs/development/testing_and_benchmarks.md).
Compare versions on the same machine and settings; hardware is measurement
metadata, with no prescribed GPU model or universal frame-time limit.

## Install (uv recommended)

Install base dependencies (without `pyuipc`):

```bash
uv sync
```

### pyuipc from PyPI

```bash
uv sync --group pypi
```

> **Note:** The PyPI version of `pyuipc` only supports CUDA 12.8. If you need a different CUDA version, use the "Build from source" method below.

### pyuipc from source

Build `pyuipc` following the [Libuipc documentation](https://spirimirror.github.io/libuipc-doc/build_install/), then install the wheel:

```bash
uv pip install pyuipc-<version>.whl
```
