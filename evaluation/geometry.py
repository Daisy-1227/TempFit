import json, math
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

SUPPORT_POINTS = 256


def load_template(primitive_dir: Path, primitive_id: int, cache: dict) -> np.ndarray:
    primitive_id = int(primitive_id)
    if primitive_id not in cache:
        payload = json.loads(
            (primitive_dir / f"{primitive_id}.json").read_text(encoding="utf-8")
        )
        cache[primitive_id] = np.asarray(payload["coords"], dtype=np.float64).reshape(
            -1, 2, 2
        )
    return cache[primitive_id]


def transform_template(
    lines: np.ndarray,
    scale: np.ndarray,
    rotation_deg: float,
    translation: np.ndarray,
    flip: int,
    scene_bound: float,
) -> np.ndarray:
    """Transform a raw-mm primitive into the dataset's normalized scene space.

    Industry scenes are normalized by ``scene_bound / 2`` (30000 mm maps to
    the width of [-1, 1]).  Translation is already stored in that normalized
    space.  Keep this identical to ``proj_from_ins_to_segment_simple.py``.
    """
    points = np.asarray(lines, dtype=np.float64).reshape(-1, 2).copy()
    if int(flip) == 1:
        points[:, 0] *= -1.0
    points *= np.asarray(scale, dtype=np.float64).reshape(1, 2)
    angle = math.radians(float(rotation_deg))
    c, s = math.cos(angle), math.sin(angle)
    points = points @ np.asarray([[c, -s], [s, c]], dtype=np.float64).T
    points = points / (float(scene_bound) / 2.0)
    points += np.asarray(translation, dtype=np.float64).reshape(1, 2)
    return points.reshape(-1, 2, 2)


def uniform_support(lines: np.ndarray, count: int = SUPPORT_POINTS) -> np.ndarray:
    """Length-uniform samples of the segment union, independent of tessellation."""
    lines = np.asarray(lines, np.float64).reshape(-1, 2, 2)
    if not len(lines):
        return np.zeros((count, 2), np.float32)
    delta = lines[:, 1] - lines[:, 0]
    length = np.linalg.norm(delta, axis=1)
    valid = length > 1e-12
    if not valid.any():
        return np.repeat(lines[:1, 0], count, axis=0).astype(np.float32)
    lines, delta, length = lines[valid], delta[valid], length[valid]
    cumulative = np.cumsum(length)
    positions = (np.arange(count, dtype=np.float64) + 0.5) / count * cumulative[-1]
    index = np.searchsorted(cumulative, positions, side="right")
    previous = np.where(index > 0, cumulative[np.maximum(index - 1, 0)], 0.0)
    t = (positions - previous) / length[index]
    return (lines[index, 0] + t[:, None] * delta[index]).astype(np.float32)


def support_chamfer(a: np.ndarray, b: np.ndarray) -> float:
    extent = np.concatenate([a, b])
    diagonal = max(float(np.linalg.norm(np.ptp(extent, axis=0))), 1e-8)
    return float(
        0.5 * (cKDTree(b).query(a)[0].mean() + cKDTree(a).query(b)[0].mean()) / diagonal
    )


def list_values(x, n):
    return x[0, :n].detach().cpu().tolist()


def tensor_records(result, gt, n, mode):
    d = dict(mode=mode)
    for key in ("semantic", "primitive"):
        d[key] = list_values(result["pred_" + key], n)
    d["flip"] = list_values(result["pred_flip"], n)
    d["translation"] = list_values(result["pred_discrete_cont_translation"], n)
    d["rotation"] = list_values(result["pred_discrete_cont_rotation"], n)
    shape_key = "size" if result.get("pred_discrete_cont_size") is not None else "scale"
    d["shape_key"] = shape_key
    d["shape"] = list_values(result["pred_discrete_cont_" + shape_key], n)
    if mode == "h":
        d["eos_would_stop"] = (
            result["eos_would_stop"][0, : n + 1].detach().cpu().tolist()
        )
    return d


def center_assignment(pred, gt, gate=0.15):
    if not len(pred) or not len(gt):
        return []
    d = np.linalg.norm(pred[:, None, :] - gt[None, :, :], axis=-1)
    score = np.maximum(0.0, 1.0 - d / (gate * (min(d.shape) + 1)))
    ri, ci = linear_sum_assignment(-np.where(d <= gate, score, 0.0))
    return [(int(i), int(j), float(d[i, j])) for i, j in zip(ri, ci) if d[i, j] <= gate]
