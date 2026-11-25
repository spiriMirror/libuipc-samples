"""Shared utilities for the 41_distance_diagnoser examples."""

import numpy as np
import keyboard
from polyscope import imgui

from uipc import Logger, view
from uipc.core import Engine, World, Scene, DistanceDiagnoserFeature
from asset_dir import AssetDir

Logger.set_level(Logger.Level.Warn)


# Engine / World / DDF bootstrap

def create_ddf(caller_file):
    """Spin up a minimal CUDA engine + world and return (engine, world, ddf).

    ``caller_file`` should be ``__file__`` from the calling script so that
    the workspace directory is derived from its path.  The caller must keep
    a reference to ``engine`` to prevent it from being garbage-collected.
    """
    workspace = AssetDir.output_path(caller_file)
    engine = Engine("cuda", workspace)
    world = World(engine)
    scene = Scene(Scene.default_config())
    world.init(scene)
    ddf = world.features().find(DistanceDiagnoserFeature)
    assert ddf is not None, "DistanceDiagnoserFeature not found. Is the CUDA backend built?"
    return engine, world, ddf


# Keyboard

def poll_keyboard(speed):
    """Return (dx, dy, dz) displacement from WASDQE keys."""
    dx = dy = dz = 0.0
    if keyboard.is_pressed("d"): dx += speed
    if keyboard.is_pressed("a"): dx -= speed
    if keyboard.is_pressed("e"): dy += speed
    if keyboard.is_pressed("q"): dy -= speed
    if keyboard.is_pressed("w"): dz -= speed
    if keyboard.is_pressed("s"): dz += speed
    return dx, dy, dz


# d_hat write-back

def set_d_hat(sc, val):
    """Write *val* to every vertex of the ``d_hat`` attribute on *sc*."""
    attr = sc.vertices().find("d_hat")
    if attr is None:
        return
    v = view(attr)
    v[:] = val


# Newton direction

def compute_newton_dir(g_arr, hess_attr, flag_val, n_verts):
    """Compute ``inv(H) * g`` with flag-based diagonal regularisation.

    Returns the Newton direction array or *None* when inputs are missing.
    """
    if g_arr is None or hess_attr is None:
        return None
    n = 3 * n_verts
    H = hess_attr.view().flatten()[:n * n].astype(np.float64).reshape(n, n)
    if flag_val is not None:
        f = flag_val.flatten()
        for i in range(n_verts):
            if int(f[i]) == 0:
                H[3 * i:3 * i + 3, 3 * i:3 * i + 3] = np.eye(3)
    return np.linalg.lstsq(H, g_arr, rcond=None)[0]


# Vector collapsing-header section

def vector_section(name, table_id, arr, labels, cloud):
    """Draw a collapsing header that toggles a Polyscope vector quantity.

    When the header is open the vectors are enabled and a table of
    per-vertex x/y/z values is shown.  Returns the ``show`` bool.
    """
    n_verts = len(labels)
    show = imgui.CollapsingHeader(name)
    if arr is not None:
        cloud.add_vector_quantity(name, arr.reshape(n_verts, 3),
                                 vectortype='ambient', enabled=show)
    if show and arr is not None:
        if imgui.BeginTable(table_id, 4):
            imgui.TableSetupColumn("")
            imgui.TableSetupColumn("x")
            imgui.TableSetupColumn("y")
            imgui.TableSetupColumn("z")
            imgui.TableHeadersRow()
            for i, lb in enumerate(labels):
                imgui.TableNextRow()
                imgui.TableNextColumn(); imgui.Text(lb)
                for k in range(3):
                    imgui.TableNextColumn()
                    imgui.Text(f"{arr[3 * i + k]:.4e}")
            imgui.EndTable()
    return show


# Flag label

def flag_label(flag):
    """Interpret a distance-type flag vector as a human-readable string."""
    n = int(np.sum(flag.flatten()))
    if n == 2: return "PP"
    if n == 3: return "PE"
    if n == 4:
        if len(flag.flatten()) == 4:
            return "PT" if flag.flatten()[0] == 0 else "EE"
    return f"?({n})"


# Voronoi helpers

def build_edge_voronoi(e0, e1, size=2.0):
    """Voronoi separator planes at edge endpoints."""
    d = e1 - e0
    dl = np.linalg.norm(d)
    if dl < 1e-12:
        return None, None
    d_hat = d / dl
    up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(d_hat, up)) > 0.9:
        up = np.array([1.0, 0.0, 0.0])
    u = np.cross(d_hat, up)
    u /= np.linalg.norm(u)
    v = np.cross(d_hat, u)
    hh = size / 2.0
    V, F = [], []
    vi = 0
    for pt in [e0, e1]:
        for c in [pt - hh * u - hh * v, pt + hh * u - hh * v,
                  pt + hh * u + hh * v, pt - hh * u + hh * v]:
            V.append(c)
        F.append([vi, vi + 1, vi + 2])
        F.append([vi, vi + 2, vi + 3])
        vi += 4
    return np.array(V), np.array(F)


def build_pt_voronoi(t0, t1, t2, h=2.0, ext=1.5):
    """Voronoi separator quads for point-triangle closest-point regions.

    Blue  planes = edge walls   (face <-> edge boundary)
    Orange planes = vertex walls (edge <-> vertex boundary)
    Returns (verts, tri_faces, face_colors).
    """
    n = np.cross(t1 - t0, t2 - t0)
    nl = np.linalg.norm(n)
    if nl < 1e-12:
        return None, None, None
    n = n / nl
    centroid = (t0 + t1 + t2) / 3.0
    hh = h / 2.0
    V, F, C = [], [], []
    vi = 0
    for a, b, _opp in [(t0, t1, t2), (t1, t2, t0), (t2, t0, t1)]:
        d = b - a
        dl = np.linalg.norm(d)
        if dl < 1e-12:
            continue
        d_hat = d / dl
        outward = np.cross(n, d_hat)
        if np.dot(outward, a - centroid) < 0:
            outward = -outward
        for v in [a - hh * n, b - hh * n, b + hh * n, a + hh * n]:
            V.append(v)
        F.append([vi, vi + 1, vi + 2]); F.append([vi, vi + 2, vi + 3])
        C.extend([[0.3, 0.6, 1.0]] * 2)
        vi += 4
        for v in [a - hh * n, a + hh * n,
                  a + hh * n + ext * outward, a - hh * n + ext * outward]:
            V.append(v)
        F.append([vi, vi + 1, vi + 2]); F.append([vi, vi + 2, vi + 3])
        C.extend([[1.0, 0.6, 0.3]] * 2)
        vi += 4
        for v in [b - hh * n, b + hh * n,
                  b + hh * n + ext * outward, b - hh * n + ext * outward]:
            V.append(v)
        F.append([vi, vi + 1, vi + 2]); F.append([vi, vi + 2, vi + 3])
        C.extend([[1.0, 0.6, 0.3]] * 2)
        vi += 4
    return np.array(V), np.array(F), np.array(C)


# Rotation matrices

def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
