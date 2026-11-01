from __future__ import annotations

import numpy as np


RELATIVE_SCALE = "relative"
ABSOLUTE_LOCAL_EXTENT = "absolute_local_extent"
VALID_SCALE_PARAMETERIZATIONS = {RELATIVE_SCALE, ABSOLUTE_LOCAL_EXTENT}


def validate_scale_parameterization(value: str) -> str:
    value = str(value)
    if value not in VALID_SCALE_PARAMETERIZATIONS:
        raise ValueError(
            "scale_parameterization must be relative/absolute_local_extent, "
            f"got {value!r}"
        )
    return value


def _points(coords) -> np.ndarray:
    array = np.asarray(coords, dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot compute a template extent from empty coordinates")
    if array.shape[-1] == 4:
        return array.reshape(-1, 2)
    if array.shape[-2:] == (2, 2):
        return array.reshape(-1, 2)
    if array.shape[-1] == 2:
        return array.reshape(-1, 2)
    raise ValueError(f"unsupported coordinate shape {array.shape}")


def raw_template_extent(coords, eps: float = 1e-8) -> np.ndarray:
    """Return raw per-axis bbox extent without changing the template origin."""
    points = _points(coords)
    extent = points.max(axis=0) - points.min(axis=0)
    if not np.isfinite(extent).all() or (extent <= eps).any():
        raise ValueError(f"invalid primitive template extent {extent.tolist()}")
    return extent


def scene_template_extent(coords, scene_bound: float) -> np.ndarray:
    """Return raw-template bbox extent in normalized scene coordinates."""
    half_bound = float(scene_bound) / 2.0
    if not np.isfinite(half_bound) or half_bound <= 0:
        raise ValueError(f"scene_bound must be finite and positive, got {scene_bound}")
    return raw_template_extent(coords) / half_bound


def normalize_coords_to_local_extent(coords) -> np.ndarray:
    """Divide x/y by raw bbox width/height while preserving input shape/origin."""
    array = np.asarray(coords, dtype=np.float64)
    extent = raw_template_extent(array)
    result = array.copy()
    if result.shape[-1] == 4:
        result[..., [0, 2]] /= extent[0]
        result[..., [1, 3]] /= extent[1]
    else:
        result = result.reshape(-1, 2)
        result /= extent.reshape(1, 2)
        result = result.reshape(array.shape)
    return result
