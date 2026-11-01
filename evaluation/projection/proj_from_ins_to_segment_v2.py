import argparse
import json
import math
import os
import traceback
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm
from evaluation.projection.projection_ownership_v2 import (
    soft_area_assignment,
    consolidate_source_primitives,
)


# ============================================================================ #
#  几何工具函数（与 infer_cad_attr_vecformer_industry_simple_ar.py 中保持一致）
# ============================================================================ #


def transform_lines(lines, flip, scale, rotation_deg, translation):
    """对线段集合应用 flip → scale → rotation → translation（与 infer 脚本一致的顺序）。

    Args:
        lines: (M, 2, 2) 或 (M, 4) 的线段坐标
        flip: 0/1，是否做 x 镜像
        scale: 标量或 shape (2,) 的 (sx, sy)
        rotation_deg: 角度（度）
        translation: shape (2,) 的 (tx, ty)

    Returns:
        (M, 2, 2) 变换后的线段
    """
    arr = np.asarray(lines, dtype=np.float64).copy()
    if arr.ndim == 2 and arr.shape[1] == 4:
        arr = arr.reshape(-1, 2, 2)
    n = arr.shape[0]
    pts = arr.reshape(-1, 2)

    if int(flip) == 1:
        pts[:, 0] = -pts[:, 0]
    if np.isscalar(scale):
        pts = pts * float(scale)
    else:
        pts = pts * np.asarray(scale, dtype=np.float64).reshape(-1)

    th = math.radians(float(rotation_deg))
    R = np.array(
        [[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]],
        dtype=np.float64,
    )
    pts = pts @ R.T
    pts = pts + np.asarray(translation, dtype=np.float64).reshape(-1)
    return pts.reshape(n, 2, 2)


def _seg_intersects_aabb(p0, p1, xmin, ymin, xmax, ymax, eps=1e-9):
    """Liang–Barsky：单条线段与闭轴对齐矩形是否相交（含触碰）。"""
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    dx, dy = x1 - x0, y1 - y0
    p_arr = (-dx, dx, -dy, dy)
    q_arr = (x0 - xmin, xmax - x0, y0 - ymin, ymax - y0)
    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p_arr, q_arr):
        if abs(pi) < eps:
            if qi < -eps:
                return False
            continue
        r = qi / pi
        if pi < 0:
            u1 = max(u1, r)
        else:
            u2 = min(u2, r)
        if u1 > u2 + eps:
            return False
    return True


def _minimum_area_template_obb(template_lines, long_margin=1.0, normal_margin=0.75):
    """Build an anisotropically expanded minimum-area OBB for a template.

    The candidate orientations are the finite template-edge directions modulo
    90 degrees, matching the Floorplan E1a spatial prior.  Coordinates and
    margins are expected in the same pixel-equivalent space used by the
    projector cost.
    """
    lines = np.asarray(template_lines, dtype=np.float64).reshape(-1, 2, 2)
    points = lines.reshape(-1, 2)
    if points.shape[0] == 0:
        raise ValueError("cannot build OBB for an empty template")

    # A minimum-area rectangle needs only convex-hull edge orientations.  The
    # Industry primitive library can contain thousands of internal edges, so
    # enumerating every template edge would turn this into an avoidable O(M^2)
    # operation.  Monotonic-chain hull construction plus a vectorized search
    # preserves the exact finite point-set OBB while keeping this practical.
    unique_points = np.unique(points, axis=0)
    if unique_points.shape[0] <= 2:
        hull = unique_points
    else:
        ordered = unique_points[np.lexsort((unique_points[:, 1], unique_points[:, 0]))]

        def cross(origin, first, second):
            a = first - origin
            b = second - origin
            return float(a[0] * b[1] - a[1] * b[0])

        lower = []
        for point in ordered:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
                lower.pop()
            lower.append(point)
        upper = []
        for point in ordered[::-1]:
            while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
                upper.pop()
            upper.append(point)
        hull = np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)

    if hull.shape[0] <= 1:
        angles = np.asarray([0.0], dtype=np.float64)
    else:
        edge_delta = np.roll(hull, -1, axis=0) - hull
        finite = np.linalg.norm(edge_delta, axis=1) > 1e-9
        edge_delta = edge_delta[finite]
        angles = np.arctan2(edge_delta[:, 1], edge_delta[:, 0]) % (math.pi / 2.0)
        angles = np.unique(np.round(np.concatenate(([0.0], angles)), decimals=12))

    cos_a, sin_a = np.cos(angles), np.sin(angles)
    axis_u = np.stack([cos_a, sin_a], axis=1)
    axis_v = np.stack([-sin_a, cos_a], axis=1)
    proj_u = hull @ axis_u.T
    proj_v = hull @ axis_v.T
    low_u, high_u = proj_u.min(axis=0), proj_u.max(axis=0)
    low_v, high_v = proj_v.min(axis=0), proj_v.max(axis=0)
    areas = np.maximum(high_u - low_u, 1e-9) * np.maximum(high_v - low_v, 1e-9)
    best_index = int(np.argmin(areas))
    axes = np.stack([axis_u[best_index], axis_v[best_index]], axis=1)
    low = np.asarray([low_u[best_index], low_v[best_index]])
    high = np.asarray([high_u[best_index], high_v[best_index]])
    low, high = low.copy(), high.copy()
    sizes = high - low
    long_axis = int(np.argmax(sizes))
    normal_axis = 1 - long_axis
    low[long_axis] -= float(long_margin)
    high[long_axis] += float(long_margin)
    low[normal_axis] -= float(normal_margin)
    high[normal_axis] += float(normal_margin)
    return {"axes": axes, "low": low, "high": high}


def _segment_fraction_inside_obb_batch(lines, obb):
    """Return each finite segment's Liang-Barsky clip fraction inside an OBB."""
    segments = np.asarray(lines, dtype=np.float64).reshape(-1, 2, 2)
    if segments.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)

    axes, low, high = obb["axes"], obb["low"], obb["high"]
    point_a = segments[:, 0] @ axes
    point_b = segments[:, 1] @ axes
    delta = point_b - point_a
    enter = np.zeros(segments.shape[0], dtype=np.float64)
    leave = np.ones(segments.shape[0], dtype=np.float64)
    valid = np.ones(segments.shape[0], dtype=bool)

    for dimension in range(2):
        d = delta[:, dimension]
        parallel = np.abs(d) < 1e-12
        valid &= ~(
            parallel
            & (
                (point_a[:, dimension] < low[dimension])
                | (point_a[:, dimension] > high[dimension])
            )
        )
        nonparallel = ~parallel
        first = np.zeros_like(d)
        second = np.zeros_like(d)
        first[nonparallel] = (low[dimension] - point_a[nonparallel, dimension]) / d[
            nonparallel
        ]
        second[nonparallel] = (high[dimension] - point_a[nonparallel, dimension]) / d[
            nonparallel
        ]
        lo = np.minimum(first, second)
        hi = np.maximum(first, second)
        enter[nonparallel] = np.maximum(enter[nonparallel], lo[nonparallel])
        leave[nonparallel] = np.minimum(leave[nonparallel], hi[nonparallel])
        valid &= leave > enter

    fraction = np.maximum(0.0, np.minimum(1.0, leave) - np.maximum(0.0, enter))
    fraction[~valid] = 0.0
    return fraction


def lines_intersect_axis_aligned_square(lines, lo=-1.0, hi=1.0):
    """检查任一线段是否与正方形 [lo,hi]^2 相交。"""
    if lines is None:
        return False
    arr = np.asarray(lines, dtype=np.float64)
    if arr.size == 0:
        return False
    if arr.ndim == 2 and arr.shape[1] == 4:
        arr = arr.reshape(-1, 2, 2)
    for i in range(arr.shape[0]):
        if _seg_intersects_aabb(arr[i, 0], arr[i, 1], lo, lo, hi, hi):
            return True
    return False


def snap_quarter_turn(rotation):
    """把角度（度）按最近邻量化到 {0, 90, 180, 270}。"""
    r = float(np.asarray(rotation, dtype=np.float64).reshape(-1)[0]) % 360.0
    return float(round(r / 90.0) * 90 % 360)


def derive_scale_from_size(size_xy, rotation_deg, prim_lines_raw, scene_bound=30000.0):
    """size_pred=True 时由预测的 size + 量化后的旋转反推 scale（镜像 infer 脚本 L1344-1372）。"""
    half_bound = scene_bound / 2.0
    pts = np.asarray(prim_lines_raw, dtype=np.float64).reshape(-1, 2)
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    w_prim = (max_xy[0] - min_xy[0]) / half_bound
    h_prim = (max_xy[1] - min_xy[1]) / half_bound

    rot_snapped = round(float(rotation_deg) / 90.0) * 90 % 360
    eps = 1e-6
    sz = np.asarray(size_xy, dtype=np.float64).reshape(-1)
    if rot_snapped == 0 or rot_snapped == 180:
        sx = sz[0] / w_prim if w_prim > eps else 1.0
        sy = sz[1] / h_prim if h_prim > eps else 1.0
    else:
        sx = sz[1] / w_prim if w_prim > eps else 1.0
        sy = sz[0] / h_prim if h_prim > eps else 1.0
    return np.array([sx, sy], dtype=np.float64)


# ============================================================================ #
#  代价函数（向量化版本，对应参考脚本的 compute_robust_cost_single）
# ============================================================================ #


def _compute_min_cost_per_input_vs_template(
    inp_lines,  # (Nr, 2, 2)
    tpl_lines,  # (Mt, 2, 2)
    perp_tolerance,
    w_perp,
    w_overshoot,
    w_angle,
    short_line_min,
    short_line_max,
    cost_caps,
    bidirectional_overshoot=False,
):
    """对每条输入线段计算到模板线段集合的最小鲁棒代价。

    完全对应参考脚本 compute_robust_cost_single + compute_segment_to_template_cost。
    所有代价/容忍度参数都假定与 inp/tpl 的坐标单位匹配。
    """
    Nr = inp_lines.shape[0]
    Mt = tpl_lines.shape[0]
    if Nr == 0 or Mt == 0:
        return np.full(Nr, cost_caps, dtype=np.float64)

    eps = 1e-9

    # 输入线段：方向 / 长度
    inp_vec = inp_lines[:, 1] - inp_lines[:, 0]  # (Nr, 2)
    inp_len = np.linalg.norm(inp_vec, axis=-1)  # (Nr,)
    inp_dir = inp_vec / np.maximum(inp_len, eps)[:, None]  # (Nr, 2)

    # 模板线段：方向 / 长度
    tpl_vec = tpl_lines[:, 1] - tpl_lines[:, 0]  # (Mt, 2)
    tpl_len = np.linalg.norm(tpl_vec, axis=-1)  # (Mt,)
    tpl_dir = tpl_vec / np.maximum(tpl_len, eps)[:, None]  # (Mt, 2)

    # 广播：(Nr, 1, 2) vs (1, Mt, 2)
    inp_a = inp_lines[:, 0][:, None, :]  # (Nr, 1, 2)
    inp_b = inp_lines[:, 1][:, None, :]  # (Nr, 1, 2)
    tpl_a = tpl_lines[:, 0][None, :, :]  # (1, Mt, 2)
    tpl_dir_b = tpl_dir[None, :, :]  # (1, Mt, 2)
    tpl_len_b = tpl_len[None, :]  # (1, Mt)

    # 1) 垂直距离（输入两端点到模板直线）
    da = inp_a - tpl_a  # (Nr, Mt, 2)
    db = inp_b - tpl_a  # (Nr, Mt, 2)
    proj_a = np.sum(da * tpl_dir_b, axis=-1)  # (Nr, Mt)
    proj_b = np.sum(db * tpl_dir_b, axis=-1)  # (Nr, Mt)
    perp_a_vec = da - proj_a[..., None] * tpl_dir_b
    perp_b_vec = db - proj_b[..., None] * tpl_dir_b
    avg_perp = (
        np.linalg.norm(perp_a_vec, axis=-1) + np.linalg.norm(perp_b_vec, axis=-1)
    ) / 2.0
    perp_cost = np.maximum(0.0, avg_perp - perp_tolerance)  # (Nr, Mt)

    # 2) 纵向投影超出（防"墙泄漏"）
    min_proj = np.minimum(proj_a, proj_b)
    max_proj = np.maximum(proj_a, proj_b)
    overshoot_left = np.maximum(0.0, -min_proj)
    overshoot_right = np.maximum(0.0, max_proj - tpl_len_b)
    source_outside = overshoot_left + overshoot_right  # (Nr, Mt)

    if bidirectional_overshoot:
        # Reverse extent: project both template endpoints onto the input line.
        tpl_a_full = tpl_lines[:, 0][None, :, :]
        tpl_b_full = tpl_lines[:, 1][None, :, :]
        inp_a_full = inp_lines[:, 0][:, None, :]
        inp_dir_full = inp_dir[:, None, :]
        rev_a = np.sum((tpl_a_full - inp_a_full) * inp_dir_full, axis=-1)
        rev_b = np.sum((tpl_b_full - inp_a_full) * inp_dir_full, axis=-1)
        rev_lo = np.minimum(rev_a, rev_b)
        rev_hi = np.maximum(rev_a, rev_b)
        template_outside = np.maximum(0.0, -rev_lo) + np.maximum(
            0.0, rev_hi - inp_len[:, None]
        )
        overshoot_len = 0.5 * source_outside + 0.5 * template_outside
    else:
        overshoot_len = source_outside

    # 3) 角度差 → 分段非线性惩罚
    cos_theta = np.abs(np.sum(inp_dir[:, None, :] * tpl_dir_b, axis=-1))
    angle_penalty = np.zeros_like(cos_theta)
    mid_mask = (cos_theta >= 0.5) & (cos_theta < 0.866)
    low_mask = cos_theta < 0.5
    angle_penalty[mid_mask] = (0.866 - cos_theta[mid_mask]) / (0.866 - 0.5) * 5.0
    angle_penalty[low_mask] = 5.0 + (0.5 - cos_theta[low_mask]) * 20.0

    # 4) 短线段降权
    inp_len_b = inp_len[:, None]  # (Nr, 1)
    min_len = np.minimum(inp_len_b, tpl_len_b)  # (Nr, Mt)
    awf = np.where(
        min_len < short_line_min,
        0.1,
        np.where(
            min_len < short_line_max,
            0.1
            + 0.9
            * (min_len - short_line_min)
            / max(short_line_max - short_line_min, eps),
            1.0,
        ),
    )

    final_cost = (
        w_perp * perp_cost + w_overshoot * overshoot_len + w_angle * angle_penalty * awf
    )
    final_cost = np.minimum(final_cost, cost_caps)
    return final_cost.min(axis=1)


def _compute_min_fragment_cost_per_input_vs_template(
    inp_lines,
    tpl_lines,
    perp_tolerance,
    w_perp,
    w_overshoot,
    angle_tol_deg,
    min_overlap_ratio,
):
    """Return a source-containment competitor for fragmented raw geometry.

    A short raw fragment may cover only part of a longer template support. This
    cost therefore omits reverse/template-outside overshoot, while retaining a
    finite-overlap test, a direction gate, perpendicular distance, and source
    overshoot. Ineligible rows remain infinite so callers can combine this with
    the normal cost using an elementwise minimum.
    """
    inp = np.asarray(inp_lines, dtype=np.float64).reshape(-1, 2, 2)
    tpl = np.asarray(tpl_lines, dtype=np.float64).reshape(-1, 2, 2)
    nr, mt = inp.shape[0], tpl.shape[0]
    if nr == 0 or mt == 0:
        return np.full(nr, np.inf, dtype=np.float64)

    eps = 1e-9
    inp_vec = inp[:, 1] - inp[:, 0]
    inp_len = np.linalg.norm(inp_vec, axis=1)
    inp_dir = inp_vec / np.maximum(inp_len, eps)[:, None]
    tpl_vec = tpl[:, 1] - tpl[:, 0]
    tpl_len = np.linalg.norm(tpl_vec, axis=1)
    tpl_dir = tpl_vec / np.maximum(tpl_len, eps)[:, None]

    delta_a = inp[:, 0][:, None, :] - tpl[:, 0][None, :, :]
    delta_b = inp[:, 1][:, None, :] - tpl[:, 0][None, :, :]
    direction = tpl_dir[None, :, :]
    scalar_a = np.sum(delta_a * direction, axis=2)
    scalar_b = np.sum(delta_b * direction, axis=2)
    source_lo = np.minimum(scalar_a, scalar_b)
    source_hi = np.maximum(scalar_a, scalar_b)
    tpl_len_b = tpl_len[None, :]
    overlap = np.maximum(
        0.0,
        np.minimum(tpl_len_b, source_hi) - np.maximum(0.0, source_lo),
    )

    perp_a = np.linalg.norm(
        delta_a - scalar_a[:, :, None] * direction,
        axis=2,
    )
    perp_b = np.linalg.norm(
        delta_b - scalar_b[:, :, None] * direction,
        axis=2,
    )
    perpendicular = np.maximum(perp_a, perp_b)
    perpendicular_cost = (
        w_perp * np.maximum(0.0, perpendicular - perp_tolerance) + perpendicular
    )
    source_outside = np.maximum(0.0, -source_lo) + np.maximum(
        0.0,
        source_hi - tpl_len_b,
    )
    cost = perpendicular_cost + w_overshoot * source_outside

    cosine = np.abs(inp_dir @ tpl_dir.T)
    projected_length = np.maximum(np.abs(inp_vec @ tpl_dir.T), eps)
    eligible = (
        (inp_len[:, None] > eps)
        & (tpl_len_b > eps)
        & (cosine >= math.cos(math.radians(float(angle_tol_deg))))
        & (overlap > eps)
        & (overlap / projected_length >= float(min_overlap_ratio))
    )
    return np.min(np.where(eligible, cost, np.inf), axis=1)


def _line_offset_to_support(lines, support_line):
    """Largest endpoint distance from each line to an infinite support line."""
    lines = np.asarray(lines, dtype=np.float64).reshape(-1, 2, 2)
    support = np.asarray(support_line, dtype=np.float64).reshape(2, 2)
    vector = support[1] - support[0]
    length = float(np.linalg.norm(vector))
    if length <= 1e-9:
        return np.full(lines.shape[0], np.inf, dtype=np.float64)
    direction = vector / length
    delta = lines - support[0]
    distance = np.abs(direction[0] * delta[:, :, 1] - direction[1] * delta[:, :, 0])
    return distance.max(axis=1)


def _build_support_chains(
    lines,
    path_ids,
    join_gap=0.5,
    angle_tol_deg=5.0,
    line_residual=0.5,
):
    """Merge only consecutive, connected, near-collinear edges of one source primitive."""
    segments = np.asarray(lines, dtype=np.float64).reshape(-1, 2, 2)
    path_ids = np.asarray(path_ids, dtype=np.int64)
    if path_ids.shape != (segments.shape[0],):
        raise ValueError("support path ids must align with segments")

    grouped = {}
    for index, path_id in enumerate(path_ids):
        grouped.setdefault(int(path_id), []).append(index)

    chains = []
    cosine_limit = math.cos(math.radians(float(angle_tol_deg)))

    def append_chain(members):
        representative = np.stack(
            [
                segments[members[0], 0],
                segments[members[-1], 1],
            ]
        )
        if float(np.linalg.norm(representative[1] - representative[0])) <= 1e-9:
            lengths = np.linalg.norm(
                segments[members, 1] - segments[members, 0],
                axis=1,
            )
            representative = segments[members[int(np.argmax(lengths))]].copy()
        selected = segments[members]
        chains.append(
            {
                "members": np.asarray(members, dtype=np.int32),
                "line": representative,
                "bbox": np.asarray(
                    [
                        selected[:, :, 0].min(),
                        selected[:, :, 1].min(),
                        selected[:, :, 0].max(),
                        selected[:, :, 1].max(),
                    ]
                ),
            }
        )

    for indices in grouped.values():
        members = []
        for index in indices:
            can_extend = bool(members)
            if can_extend:
                previous = segments[members[-1]]
                candidate = segments[index]
                can_extend &= float(
                    np.linalg.norm(previous[1] - candidate[0])
                ) <= float(join_gap)
                previous_vector = previous[1] - previous[0]
                candidate_vector = candidate[1] - candidate[0]
                previous_length = float(np.linalg.norm(previous_vector))
                candidate_length = float(np.linalg.norm(candidate_vector))
                can_extend &= previous_length > 1e-9 and candidate_length > 1e-9
                if can_extend:
                    cosine = abs(
                        float(
                            np.dot(
                                previous_vector / previous_length,
                                candidate_vector / candidate_length,
                            )
                        )
                    )
                    can_extend &= cosine >= cosine_limit
                if can_extend:
                    chord = np.stack([segments[members[0], 0], candidate[1]])
                    can_extend &= bool(
                        np.max(
                            _line_offset_to_support(
                                segments[members + [index]],
                                chord,
                            )
                        )
                        <= float(line_residual)
                    )
            if members and not can_extend:
                append_chain(members)
                members = []
            members.append(index)
        if members:
            append_chain(members)
    return chains


def _compute_min_support_cost(raw_lines, template_lines, params):
    """Symmetric either-side-containment cost between support-chain chords."""
    raw = np.asarray(raw_lines, dtype=np.float64).reshape(-1, 2, 2)
    tpl = np.asarray(template_lines, dtype=np.float64).reshape(-1, 2, 2)
    nr, nt = raw.shape[0], tpl.shape[0]
    if nr == 0 or nt == 0:
        return np.full(nr, np.inf, dtype=np.float64)
    eps = 1e-9
    raw_vec = raw[:, 1] - raw[:, 0]
    raw_len = np.linalg.norm(raw_vec, axis=1)
    raw_dir = raw_vec / np.maximum(raw_len, eps)[:, None]
    tpl_vec = tpl[:, 1] - tpl[:, 0]
    tpl_len = np.linalg.norm(tpl_vec, axis=1)
    tpl_dir = tpl_vec / np.maximum(tpl_len, eps)[:, None]
    cosine = np.abs(raw_dir @ tpl_dir.T)

    delta_a = raw[:, 0][:, None, :] - tpl[:, 0][None, :, :]
    delta_b = raw[:, 1][:, None, :] - tpl[:, 0][None, :, :]
    scalar_a = np.sum(delta_a * tpl_dir[None, :, :], axis=2)
    scalar_b = np.sum(delta_b * tpl_dir[None, :, :], axis=2)
    raw_lo = np.minimum(scalar_a, scalar_b)
    raw_hi = np.maximum(scalar_a, scalar_b)
    tpl_len_b = tpl_len[None, :]
    overlap = np.maximum(
        0.0,
        np.minimum(tpl_len_b, raw_hi) - np.maximum(0.0, raw_lo),
    )
    raw_projected_length = np.maximum(raw_hi - raw_lo, eps)
    raw_coverage = np.minimum(1.0, overlap / raw_projected_length)
    template_coverage = np.minimum(1.0, overlap / np.maximum(tpl_len_b, eps))
    containment = 1.0 - np.maximum(raw_coverage, template_coverage)

    raw_perp_a = np.linalg.norm(
        delta_a - scalar_a[:, :, None] * tpl_dir[None, :, :],
        axis=2,
    )
    raw_perp_b = np.linalg.norm(
        delta_b - scalar_b[:, :, None] * tpl_dir[None, :, :],
        axis=2,
    )
    tpl_delta_a = tpl[:, 0][None, :, :] - raw[:, 0][:, None, :]
    tpl_delta_b = tpl[:, 1][None, :, :] - raw[:, 0][:, None, :]
    tpl_scalar_a = np.sum(tpl_delta_a * raw_dir[:, None, :], axis=2)
    tpl_scalar_b = np.sum(tpl_delta_b * raw_dir[:, None, :], axis=2)
    tpl_perp_a = np.linalg.norm(
        tpl_delta_a - tpl_scalar_a[:, :, None] * raw_dir[:, None, :],
        axis=2,
    )
    tpl_perp_b = np.linalg.norm(
        tpl_delta_b - tpl_scalar_b[:, :, None] * raw_dir[:, None, :],
        axis=2,
    )
    perpendicular = np.maximum.reduce(
        [
            raw_perp_a,
            raw_perp_b,
            tpl_perp_a,
            tpl_perp_b,
        ]
    )
    perpendicular_cost = (
        params["w_perp"]
        * np.maximum(
            0.0,
            perpendicular - params["perp_tolerance"],
        )
        + perpendicular
    )
    length_ratio = np.maximum(raw_len[:, None], tpl_len_b) / np.maximum(
        np.minimum(raw_len[:, None], tpl_len_b),
        eps,
    )
    cost = (
        perpendicular_cost
        + params["support_containment_weight"] * containment
        + params["support_length_ratio_weight"]
        * np.maximum(
            0.0,
            length_ratio - params["support_length_ratio_free"],
        )
        + params["support_angle_weight"] * (1.0 - cosine)
    )
    eligible = (
        (raw_len[:, None] > eps)
        & (tpl_len_b > eps)
        & (
            cosine
            >= math.cos(math.radians(float(params["support_match_angle_tol_deg"])))
        )
    )
    return np.min(np.where(eligible, cost, np.inf), axis=1)


def _bounded_pairs(function, pair_limit=200000):
    def call(inputs, templates, *args, **kwargs):
        chunk = max(1, pair_limit // max(1, len(templates)))
        if len(inputs) <= chunk:
            return function(inputs, templates, *args, **kwargs)
        return np.concatenate(
            [
                function(inputs[i : i + chunk], templates, *args, **kwargs)
                for i in range(0, len(inputs), chunk)
            ]
        )

    return call


_compute_min_cost_per_input_vs_template = _bounded_pairs(
    _compute_min_cost_per_input_vs_template
)
_compute_min_fragment_cost_per_input_vs_template = _bounded_pairs(
    _compute_min_fragment_cost_per_input_vs_template
)
_compute_min_support_cost = _bounded_pairs(_compute_min_support_cost)


def _bbox_overlap(b1, b2):
    """非严格 bbox 重叠（含触碰）。"""
    return not (b1[2] < b2[0] or b1[0] > b2[2] or b1[3] < b2[1] or b1[1] > b2[3])


# ============================================================================ #
#  可视化
# ============================================================================ #


def _visualize_three_panel(
    scene_lines_norm,  # (Ns, 2, 2) 输入线段，归一化空间
    instance_tpl_norm,  # List[(Mt_i, 2, 2)] 每个 pred instance 变换后的模板线段
    pred_instance_labels,  # (P,) pred 各 kept instance 的 sem
    assign_inst_seg_lvl,  # (Ns,) 每条 segment 的 pred instance 索引（-1 = 未分配）
    seg_gt_inst,  # (Ns,) 每条 segment 的 GT instance 索引（-1 = bg）
    gt_instance_labels,  # (G,) gt 各 kept instance 的 sem
    out_path,
    title_prefix="",
    iou_summary=None,  # 可选：list of per-GT best IoU（同 sem 加权）
    xlim=(-1.05, 1.05),
    ylim=(-1.05, 1.05),
):
    """生成三联可视化：投影前 pred 模板 / 投影后 segment 分配 / GT segment。

    颜色策略：pred-instance 之间用 tab20，gt-instance 之间也用 tab20（不要求颜色跨 panel 一致，
    因为 pred-inst 与 gt-inst 是两个独立的索引空间）。未分配/背景统一灰色。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import matplotlib.cm as cm

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    cmap = cm.get_cmap("tab20")
    grey = (0.78, 0.78, 0.78, 0.45)

    # ---- 颜色：pred instance idx → color；只为 kept instance 分配，其它走 -1 灰 ----
    n_inst_total = len(instance_tpl_norm)
    pred_colors = [cmap(i % 20) for i in range(n_inst_total)]

    # gt 颜色：按 unique gt inst id 排序后映射
    gt_unique = sorted(
        [x for x in set(int(v) for v in seg_gt_inst.tolist()) if x != -1]
    )
    gt_colors = {gid: cmap(i % 20) for i, gid in enumerate(gt_unique)}

    Ns = scene_lines_norm.shape[0]

    # ---- Panel 1: 投影前 — pred 模板覆盖在场景之上 ----
    ax = axes[0]
    if Ns > 0:
        ax.add_collection(
            LineCollection(scene_lines_norm, colors=[grey] * Ns, linewidths=0.6)
        )
    for inst_i, tpl in enumerate(instance_tpl_norm):
        if tpl.shape[0] == 0:
            continue
        col = pred_colors[inst_i]
        ax.add_collection(
            LineCollection(tpl, colors=[col] * tpl.shape[0], linewidths=1.6, alpha=0.9)
        )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_title(
        f"[1] Pre-projection: pred templates over scene\n(scene=gray, color=pred_inst idx)"
    )
    ax.grid(True, alpha=0.3)
    # legend with pred labels
    legend_handles = []
    for inst_i in range(n_inst_total):
        if instance_tpl_norm[inst_i].shape[0] == 0:
            continue
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                color=pred_colors[inst_i],
                lw=2,
                label=f'pred[{inst_i}] sem={int(pred_instance_labels[inst_i]) if inst_i < len(pred_instance_labels) else "?"}',
            )
        )
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            fontsize=7,
            loc="upper right",
            framealpha=0.85,
            ncol=1 if len(legend_handles) <= 8 else 2,
        )

    # ---- Panel 2: 投影后 — segment 颜色 = 分配的 pred instance ----
    ax = axes[1]
    seg_colors = []
    for s in range(Ns):
        ai = int(assign_inst_seg_lvl[s])
        if ai < 0 or ai >= n_inst_total:
            seg_colors.append(grey)
        else:
            seg_colors.append(pred_colors[ai])
    if Ns > 0:
        ax.add_collection(
            LineCollection(scene_lines_norm, colors=seg_colors, linewidths=1.0)
        )
    n_assigned = int((assign_inst_seg_lvl != -1).sum())
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_title(
        f"[2] Post-projection: segment colored by assigned pred_inst\n"
        f"({n_assigned}/{Ns} assigned, {100*n_assigned/max(Ns,1):.1f}%)"
    )
    ax.grid(True, alpha=0.3)

    # ---- Panel 3: GT — segment 颜色 = GT instance ----
    ax = axes[2]
    seg_colors_gt = []
    for s in range(Ns):
        gi = int(seg_gt_inst[s])
        if gi == -1 or gi not in gt_colors:
            seg_colors_gt.append(grey)
        else:
            seg_colors_gt.append(gt_colors[gi])
    if Ns > 0:
        ax.add_collection(
            LineCollection(scene_lines_norm, colors=seg_colors_gt, linewidths=1.0)
        )
    n_fg = int((seg_gt_inst != -1).sum())
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    iou_str = ""
    if iou_summary is not None and len(iou_summary) > 0:
        iou_arr = np.asarray(iou_summary, dtype=np.float64)
        iou_str = f"\nmean per-GT best-IoU={iou_arr.mean():.3f}, TP@0.5={(iou_arr>=0.5).sum()}/{iou_arr.size}"
    ax.set_title(f"[3] GT: segment colored by GT_inst\n({n_fg}/{Ns} fg){iou_str}")
    ax.grid(True, alpha=0.3)

    fig.suptitle(title_prefix, fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _per_gt_best_iou(pred_masks, pred_labels, gt_masks, gt_labels, seg_lengths):
    """按 log1p 长度加权的 per-GT best-IoU（仅同 semantic 之间比）。"""
    if pred_masks.shape[0] == 0 or gt_masks.shape[0] == 0:
        return []
    weights = np.round(np.log1p(seg_lengths.astype(np.float64)), 3)
    out = []
    for gi in range(gt_masks.shape[0]):
        tgt = gt_masks[gi].astype(bool)
        tlab = int(gt_labels[gi])
        best = 0.0
        for pi in range(pred_masks.shape[0]):
            if int(pred_labels[pi]) != tlab:
                continue
            src = pred_masks[pi].astype(bool)
            inter = float(weights[tgt & src].sum())
            union = float(weights[tgt | src].sum())
            iou = inter / (union + 1e-6)
            if iou > best:
                best = iou
        out.append(best)
    return out


# ============================================================================ #
#  主流程：处理单个样本
# ============================================================================ #


DEFAULT_COST_PARAMS = {
    "ownership_mode": "soft_area_group",
    "area_penalty": 1.0,
    "primitive_vote_agreement": 0.5,
    # 像素等效空间（×SCALE_FACTOR=70）的默认阈值。诊断显示 industry 数据上预测的位置/尺寸
    # 偏差通常落在 600~1500 mm（≈3~7 px）—— 参考脚本默认 cost_thresh=15 / cost_caps=50
    # 会把 60-80% 仍可分辨的"接近但偏移"段判为未分配，因此把阈值上调到能救回大多数。
    # 单独调时可通过 CLI 覆盖。
    "perp_tolerance": 2.0,
    "w_perp": 10.0,
    "w_overshoot": 3.0,
    "w_angle": 1.0,
    # Retained Industry best-PQ preset (2026-09-03 ablation, 494 samples).
    "cost_thresh": 70.0,
    "cost_caps": 100.0,  # was 50.0
    "roi_margin": 0.5,
    "short_line_min": 5.0,
    "short_line_max": 15.0,
    "scale_factor": 70.0,  # 归一化坐标 ×70 → 像素等效（与参考 floorplancad 画布尺度一致）
    "attribute_source": "hard",
    "prefer_small": True,  # True: 重叠时优先把线分给模板 bbox 面积更小的 instance
    "fused": False,  # True: per-block 评估 fused 预测时，把 ROI 在 [-1,1] block 框内裁剪
    "assignment_mode": "bidirectional_overshoot",
    "spatial_prior": "obb",
    "obb_long_margin": 1.0,
    "obb_normal_margin": 0.75,
    "obb_min_inside_ratio": 0.5,
    "fragment_mode": "support_chain_competitive",
    "fragment_angle_tol_deg": 30.0,
    "fragment_min_overlap_ratio": 0.05,
    "support_chain_join_gap": 0.5,
    "support_chain_angle_tol_deg": 5.0,
    "support_chain_line_residual": 0.5,
    "support_match_angle_tol_deg": 30.0,
    "support_containment_weight": 15.0,
    "support_length_ratio_free": 3.0,
    "support_length_ratio_weight": 5.0,
    "support_angle_weight": 5.0,
}


def process_one_npz(
    npz_path,
    primitive_lib_folder,
    out_dir,
    params,
    vis_dir=None,
    assignable_segment_mask=None,
):
    """处理单个 npz 文件。返回 filename（成功）或 None（失败）。

    若 vis_dir 不为 None，则额外输出三联可视化到 ${vis_dir}/${filename}_proj.png。
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
        filename = str(data["filename"])
        scene_bound = float(data["scene_bound"])
        size_pred = bool(data["size_pred"])
        scale_parameterization = (
            str(np.asarray(data["scale_parameterization"]).item())
            if "scale_parameterization" in data.files
            else "relative"
        )
        if scale_parameterization not in {"relative", "absolute_local_extent"}:
            raise ValueError(
                f"unknown scale_parameterization: {scale_parameterization!r}"
            )

        # ----- pred 属性 -----
        pred_prim_id = data["pred_primitive_id_i"].astype(np.int64)
        pred_sem_id = data["pred_semantic_id_i"].astype(np.int64)
        pred_flip = data["pred_flip_i"].reshape(-1).astype(np.int64)
        attribute_source = params.get("attribute_source", "hard")
        if attribute_source == "soft":
            required = ("pred_cont_translation_i", "pred_cont_rotation_i")
            missing = [key for key in required if key not in data]
            if missing:
                raise KeyError(f"soft attribute arrays missing: {missing}")
            pred_trans = (
                data["pred_cont_translation_i"].reshape(-1, 2).astype(np.float64)
            )
            pred_rot = data["pred_cont_rotation_i"].reshape(-1).astype(np.float64)
        else:
            pred_trans = (
                data["pred_discrete_cont_translation_i"]
                .reshape(-1, 2)
                .astype(np.float64)
            )
            pred_rot = (
                data["pred_discrete_cont_rotation_i"].reshape(-1).astype(np.float64)
            )
        if size_pred:
            size_key = (
                "pred_cont_size_i"
                if attribute_source == "soft"
                else "pred_discrete_cont_size_i"
            )
            pred_size = data[size_key].reshape(-1, 2).astype(np.float64)
            pred_scale = None
        else:
            pred_size = None
            scale_key = (
                "pred_cont_scale_i"
                if attribute_source == "soft"
                else "pred_discrete_cont_scale_i"
            )
            pred_scale = data[scale_key].reshape(-1, 2).astype(np.float64)

        # A poorly converged AR decoder may repeat the exact same hard-token
        # instance many times before EOS.  Exact duplicates generate identical
        # transformed templates, costs, areas, and semantic labels; WTA always
        # selects their first occurrence, so later copies can never own a
        # segment.  Optionally discard only these provably redundant copies to
        # avoid needlessly expanding the projector cost matrix.
        if params.get("deduplicate_exact", False) and pred_prim_id.size:
            geom_attr = pred_size if size_pred else pred_scale
            duplicate_key = np.column_stack(
                [
                    pred_prim_id.reshape(-1, 1),
                    pred_sem_id.reshape(-1, 1),
                    pred_flip.reshape(-1, 1),
                    pred_trans,
                    pred_rot.reshape(-1, 1),
                    geom_attr,
                ]
            )
            _, first_indices = np.unique(duplicate_key, axis=0, return_index=True)
            first_indices.sort()
            pred_prim_id = pred_prim_id[first_indices]
            pred_sem_id = pred_sem_id[first_indices]
            pred_flip = pred_flip[first_indices]
            pred_trans = pred_trans[first_indices]
            pred_rot = pred_rot[first_indices]
            if size_pred:
                pred_size = pred_size[first_indices]
            else:
                pred_scale = pred_scale[first_indices]

        n_pred = int(pred_prim_id.shape[0])

        # ----- 场景线段 + per-segment GT -----
        scene_lines = data["scene_lines"].astype(np.float64)  # (Ns, 4) 或 (Ns, 2, 2)
        if scene_lines.ndim == 2 and scene_lines.shape[1] == 4:
            scene_lines = scene_lines.reshape(-1, 2, 2)
        Ns = int(scene_lines.shape[0])
        if assignable_segment_mask is None:
            assignable = np.ones(Ns, dtype=bool)
        else:
            assignable = np.asarray(assignable_segment_mask, dtype=bool)
            if assignable.shape != (Ns,):
                raise ValueError(
                    f"assignable_segment_mask must have shape {(Ns,)}, "
                    f"got {assignable.shape}"
                )
        assignment_mode = params.get("assignment_mode", "robust")
        if assignment_mode not in {"robust", "bidirectional_overshoot"}:
            raise ValueError(f"unknown assignment_mode: {assignment_mode}")
        seg_prim_idx = data["scene_segment_primitive_ids"]
        prim_sem = data[
            "scene_primitive_semantic_ids"
        ]  # (Nprim,) segment 级约定（=instance 级 - 1）
        prim_inst = data["scene_primitive_instance_ids"]  # (Nprim,) -1 = bg
        prim_len = data["scene_primitive_lengths"]

        # 数据集 segment 级 sem 比 instance 级 sem 少 1（preprocessing 统一做了 -1 shift）；
        # 这里把 segment 级 sem +1 转回 model / instance 约定 [0, num_semantic-1]，方便后续指标计算。
        # 保留原值 < 0（潜在 bg 哨兵）不动。
        prim_sem_model_conv = prim_sem.astype(np.int32).copy()
        valid_sem = prim_sem_model_conv >= 0
        prim_sem_model_conv[valid_sem] += int(params.get("semantic_offset", 1))

        seg_gt_sem = prim_sem_model_conv[seg_prim_idx].astype(np.int32)
        seg_gt_inst = prim_inst[seg_prim_idx].astype(np.int32)
        seg_lengths = prim_len[seg_prim_idx].astype(np.float32)

        # ----- 生成每个 pred instance 的变换后模板线段（在归一化空间，再 ×scale_factor 进入代价计算空间）-----
        SCALE_FACTOR = float(params["scale_factor"])
        scene_lines_px = scene_lines * SCALE_FACTOR

        instance_tpl_norm = (
            []
        )  # List[(Mt_i, 2, 2)] 每个 pred instance 在 [-1,1] 空间的模板线段
        instance_tpl_px = (
            []
        )  # List[(Mt_i, 2, 2)] 同上 ×SCALE_FACTOR 后的版本（代价计算用）
        instance_tpl_path_ids = (
            []
        )  # 原始 primitive provenance，用于 support-chain consolidation
        instance_keep = np.zeros(n_pred, dtype=bool)

        for inst_i in range(n_pred):
            prim_id_i = int(pred_prim_id[inst_i])
            prim_path = os.path.join(primitive_lib_folder, f"{prim_id_i}.json")
            if not os.path.isfile(prim_path):
                instance_tpl_norm.append(np.zeros((0, 2, 2), dtype=np.float64))
                instance_tpl_px.append(np.zeros((0, 2, 2), dtype=np.float64))
                instance_tpl_path_ids.append(np.zeros(0, dtype=np.int64))
                continue
            with open(prim_path, "r", encoding="utf-8") as f:
                prim_json = json.load(f)
            prim_lines_raw = np.asarray(
                prim_json["coords"], dtype=np.float64
            )  # (M, 4) 原始 mm 坐标
            prim_path_ids = np.asarray(
                prim_json.get("primitive_ids", np.arange(len(prim_lines_raw))),
                dtype=np.int64,
            )
            if prim_path_ids.shape != (len(prim_lines_raw),):
                prim_path_ids = np.arange(len(prim_lines_raw), dtype=np.int64)
            # Relative scale multiplies the historical scene-normalized raw
            # template.  Absolute local extent multiplies an origin-preserving
            # unit-bbox template.  Both reconstruct identical scene geometry.
            if scale_parameterization == "absolute_local_extent":
                primitive_points = prim_lines_raw.reshape(-1, 2)
                primitive_extent = primitive_points.max(axis=0) - primitive_points.min(
                    axis=0
                )
                if (primitive_extent <= 1e-8).any():
                    raise ValueError(
                        f"primitive {prim_id_i} has invalid extent "
                        f"{primitive_extent.tolist()}"
                    )
                prim_lines_norm = prim_lines_raw.copy()
                prim_lines_norm[:, [0, 2]] /= primitive_extent[0]
                prim_lines_norm[:, [1, 3]] /= primitive_extent[1]
            else:
                # 把 primitive 坐标归一化到 [-1, 1]（与 vec_data 一致：除以 scene_bound/2）
                prim_lines_norm = prim_lines_raw / (scene_bound / 2.0)
            # Floorplan attributes use translation as the instance center.
            if bool(params.get("center_primitive", False)):
                prim_lines_norm = prim_lines_norm - 1.0
            if prim_lines_norm.ndim == 2 and prim_lines_norm.shape[1] == 4:
                prim_lines_norm = prim_lines_norm.reshape(-1, 2, 2)

            rot_deg = snap_quarter_turn(pred_rot[inst_i])
            if size_pred:
                sc = derive_scale_from_size(
                    pred_size[inst_i], rot_deg, prim_lines_raw, scene_bound
                )
            else:
                sc = pred_scale[inst_i]

            tpl_norm = transform_lines(
                prim_lines_norm, int(pred_flip[inst_i]), sc, rot_deg, pred_trans[inst_i]
            )
            if not lines_intersect_axis_aligned_square(tpl_norm):
                instance_tpl_norm.append(np.zeros((0, 2, 2), dtype=np.float64))
                instance_tpl_px.append(np.zeros((0, 2, 2), dtype=np.float64))
                instance_tpl_path_ids.append(np.zeros(0, dtype=np.int64))
                continue

            instance_tpl_norm.append(tpl_norm.astype(np.float64))
            instance_tpl_px.append((tpl_norm * SCALE_FACTOR).astype(np.float64))
            instance_tpl_path_ids.append(prim_path_ids)
            instance_keep[inst_i] = True

        raw_support_chains = None
        if params.get("fragment_mode", "none") == "support_chain_competitive":
            raw_support_chains = _build_support_chains(
                scene_lines_px,
                seg_prim_idx,
                join_gap=params["support_chain_join_gap"],
                angle_tol_deg=params["support_chain_angle_tol_deg"],
                line_residual=params["support_chain_line_residual"],
            )

        # ----- 构造 cost_matrix（Ns, n_pred），ROI 早剪枝 -----
        cost_matrix = np.full((Ns, n_pred), np.inf, dtype=np.float64)

        # 每个 instance 的模板 bbox 面积（像素等效空间），用于 size-priority 排序。
        # 没有模板（被 ROI/相交剪掉的）记为 inf —— 永远不会被任何 segment 选中。
        inst_areas = np.full(n_pred, np.inf, dtype=np.float64)

        # 预算每条 segment 的 bbox（in pixel-equivalent space）
        seg_xmin = np.minimum(scene_lines_px[:, 0, 0], scene_lines_px[:, 1, 0])
        seg_xmax = np.maximum(scene_lines_px[:, 0, 0], scene_lines_px[:, 1, 0])
        seg_ymin = np.minimum(scene_lines_px[:, 0, 1], scene_lines_px[:, 1, 1])
        seg_ymax = np.maximum(scene_lines_px[:, 0, 1], scene_lines_px[:, 1, 1])

        # Fused 模式下，把 ROI 在 block 可见范围 [-1,1]（pixel-space ±SCALE_FACTOR）内裁剪：
        # 跨边界 instance 不应该用其在邻 block 的那一截影响本 block 的代价匹配 / size-priority。
        fused_clip = bool(params.get("fused", False))
        block_half = SCALE_FACTOR

        for inst_i in range(n_pred):
            tpl = instance_tpl_px[inst_i]
            if tpl.shape[0] == 0:
                continue
            xs = tpl[:, :, 0].reshape(-1)
            ys = tpl[:, :, 1].reshape(-1)
            roi_x0 = xs.min() - params["roi_margin"]
            roi_x1 = xs.max() + params["roi_margin"]
            roi_y0 = ys.min() - params["roi_margin"]
            roi_y1 = ys.max() + params["roi_margin"]

            if fused_clip:
                roi_x0 = max(roi_x0, -block_half)
                roi_x1 = min(roi_x1, block_half)
                roi_y0 = max(roi_y0, -block_half)
                roi_y1 = min(roi_y1, block_half)
                if roi_x1 <= roi_x0 or roi_y1 <= roi_y0:
                    continue

            inst_areas[inst_i] = max(roi_x1 - roi_x0, 0.0) * max(roi_y1 - roi_y0, 0.0)

            in_roi = (
                (seg_xmax >= roi_x0)
                & (seg_xmin <= roi_x1)
                & (seg_ymax >= roi_y0)
                & (seg_ymin <= roi_y1)
                & assignable
            )
            idx_roi = np.where(in_roi)[0]
            if idx_roi.size == 0:
                continue

            if params.get("spatial_prior", "none") == "obb":
                obb = _minimum_area_template_obb(
                    tpl,
                    long_margin=params["obb_long_margin"],
                    normal_margin=params["obb_normal_margin"],
                )
                inside_fraction = _segment_fraction_inside_obb_batch(
                    scene_lines_px[idx_roi],
                    obb,
                )
                idx_roi = idx_roi[inside_fraction >= params["obb_min_inside_ratio"]]
                if idx_roi.size == 0:
                    continue

            cost = _compute_min_cost_per_input_vs_template(
                scene_lines_px[idx_roi],
                tpl,
                perp_tolerance=params["perp_tolerance"],
                w_perp=params["w_perp"],
                w_overshoot=params["w_overshoot"],
                w_angle=params["w_angle"],
                short_line_min=params["short_line_min"],
                short_line_max=params["short_line_max"],
                cost_caps=params["cost_caps"],
                bidirectional_overshoot=(assignment_mode == "bidirectional_overshoot"),
            )
            if params.get("fragment_mode", "none") == "source_containment_competitive":
                fragment_cost = _compute_min_fragment_cost_per_input_vs_template(
                    scene_lines_px[idx_roi],
                    tpl,
                    perp_tolerance=params["perp_tolerance"],
                    w_perp=params["w_perp"],
                    w_overshoot=params["w_overshoot"],
                    angle_tol_deg=params["fragment_angle_tol_deg"],
                    min_overlap_ratio=params["fragment_min_overlap_ratio"],
                )
                cost = np.minimum(cost, fragment_cost)
            elif params.get("fragment_mode", "none") == "support_chain_competitive":
                template_chains = _build_support_chains(
                    tpl,
                    instance_tpl_path_ids[inst_i],
                    join_gap=params["support_chain_join_gap"],
                    angle_tol_deg=params["support_chain_angle_tol_deg"],
                    line_residual=params["support_chain_line_residual"],
                )
                candidate_chains = [
                    chain
                    for chain in raw_support_chains
                    if _bbox_overlap(
                        chain["bbox"],
                        (
                            roi_x0,
                            roi_y0,
                            roi_x1,
                            roi_y1,
                        ),
                    )
                ]
                if candidate_chains and template_chains:
                    support_values = _compute_min_support_cost(
                        np.stack([chain["line"] for chain in candidate_chains]),
                        np.stack([chain["line"] for chain in template_chains]),
                        params,
                    )
                    support_cost = np.full(Ns, np.inf, dtype=np.float64)
                    allowed_members = np.zeros(Ns, dtype=bool)
                    allowed_members[idx_roi] = True
                    for chain, value in zip(candidate_chains, support_values):
                        members = chain["members"]
                        members = members[allowed_members[members]]
                        if members.size:
                            support_cost[members] = np.minimum(
                                support_cost[members],
                                value,
                            )
                    cost = np.minimum(cost, support_cost[idx_roi])
            cost_matrix[idx_roi, inst_i] = cost

        # ----- 分配 -----
        # 默认 (prefer_small=True): 重叠区域优先把线分给小 instance。
        #   规则: 对每条 segment, 在 cost < cost_thresh 的候选 instance 中, 按 inst_areas
        #         升序排, 取第一个; 若所有候选并列, 按 cost 升序兜底。
        # prefer_small=False: 退回原始 Winner-Takes-All（直接 argmin cost）。
        assign_inst = np.full(Ns, -1, dtype=np.int32)
        assign_sem = np.full(Ns, -1, dtype=np.int32)
        if n_pred > 0:
            cost_thresh = params["cost_thresh"]
            valid_mask = (cost_matrix < cost_thresh) & np.isfinite(
                cost_matrix
            )  # (Ns, n_pred)

            # Ownership selection is intentionally independent from the
            # segment/template extent cost.  This keeps the historical
            # robust+prefer-small default unchanged while allowing a proper
            # 2x2 ablation with bidirectional overshoot.
            if bool(params.get("prefer_small", True)):
                # rank: 主键 area, 副键 cost. 用 (area, cost) 排序后选 rank 最小者。
                # 把不满足 valid 的位置惩罚到 +inf 让它们不会被选。
                area_arr = inst_areas[None, :]  # (1, n_pred)
                # area_score: invalid → +inf
                area_score = np.where(valid_mask, area_arr, np.inf)  # (Ns, n_pred)
                # 取每行最小 area（满足 valid 的）
                best_inst = np.argmin(area_score, axis=1)  # (Ns,)
                # 若该行全 invalid, area_score 整行都是 inf，best_inst 取到 0 但实际未分配
                row_min_area = area_score[np.arange(Ns), best_inst]
                row_has_valid = np.isfinite(row_min_area)

                # 同 area 并列时（极少见，但比如两个完全重合的 inst），用 cost 兜底
                # —— 检测同行内 area 等于 row_min_area 的列, 在这些列里按 cost 取 argmin
                tie = (area_score == row_min_area[:, None]) & row_has_valid[:, None]
                if tie.any():
                    cost_for_tie = np.where(tie, cost_matrix, np.inf)
                    best_inst_tie = np.argmin(cost_for_tie, axis=1)
                    # 仅在确实有并列时覆盖
                    n_ties_per_row = tie.sum(axis=1)
                    mask_tie_rows = n_ties_per_row > 1
                    best_inst[mask_tie_rows] = best_inst_tie[mask_tie_rows]

                mask_assign = row_has_valid
            else:
                # 原 WTA: 直接最小 cost
                min_cost = cost_matrix.min(axis=1)
                best_inst = cost_matrix.argmin(axis=1)
                mask_assign = (min_cost < cost_thresh) & np.isfinite(min_cost)

            assign_inst[mask_assign] = best_inst[mask_assign].astype(np.int32)
            assign_sem[mask_assign] = pred_sem_id[best_inst[mask_assign]].astype(
                np.int32
            )

        # v2: geometry-led ownership, followed by observable CAD provenance consensus.
        ownership_mode = params.get(
            "ownership_mode", DEFAULT_COST_PARAMS["ownership_mode"]
        )
        if ownership_mode not in {
            "legacy",
            "soft_area",
            "group_only",
            "soft_area_group",
        }:
            raise ValueError(f"unknown ownership_mode: {ownership_mode}")
        if ownership_mode in {"soft_area", "soft_area_group"}:
            assign_inst = soft_area_assignment(
                cost_matrix,
                inst_areas,
                params["cost_thresh"],
                params.get("area_penalty", DEFAULT_COST_PARAMS["area_penalty"]),
            )
        if ownership_mode in {"group_only", "soft_area_group"}:
            agreement = float(
                params.get(
                    "primitive_vote_agreement",
                    DEFAULT_COST_PARAMS["primitive_vote_agreement"],
                )
            )
            if not 0.0 < agreement <= 1.0:
                raise ValueError("primitive_vote_agreement must be in (0, 1]")
            assign_inst = consolidate_source_primitives(
                assign_inst, seg_prim_idx, agreement
            )
        # Respect an explicitly supplied external eligibility mask even after consolidation.
        assign_inst[~assignable] = -1
        assign_sem = np.full(Ns, -1, dtype=np.int32)
        owned = assign_inst >= 0
        assign_sem[owned] = pred_sem_id[assign_inst[owned]]

        # ----- 构造 pred / gt instance 字典 -----
        pred_masks_list, pred_labels_list = [], []
        for inst_i in range(n_pred):
            mask = assign_inst == inst_i
            if mask.sum() == 0:
                continue
            pred_masks_list.append(mask.astype(np.uint8))
            pred_labels_list.append(int(pred_sem_id[inst_i]))

        gt_masks_list, gt_labels_list = [], []
        for inst_id in sorted(np.unique(seg_gt_inst).tolist()):
            if inst_id == -1:
                continue
            mask = seg_gt_inst == inst_id
            if mask.sum() == 0:
                continue
            # 取该 instance 内所有 segment 的 sem_id 众数作为 label。
            # seg_gt_sem 已在上面 +1 转到 model 约定，所以这里直接和 pred_instance_labels 同空间。
            sems_in_inst = seg_gt_sem[mask]
            valid = sems_in_inst[sems_in_inst >= 0]
            if valid.size == 0:
                continue
            label = int(np.bincount(valid).argmax())
            gt_masks_list.append(mask.astype(np.uint8))
            gt_labels_list.append(label)

        pred_instance_masks = (
            np.stack(pred_masks_list, axis=0)
            if len(pred_masks_list) > 0
            else np.zeros((0, Ns), dtype=np.uint8)
        )
        pred_instance_labels = np.asarray(pred_labels_list, dtype=np.int32)
        gt_instance_masks = (
            np.stack(gt_masks_list, axis=0)
            if len(gt_masks_list) > 0
            else np.zeros((0, Ns), dtype=np.uint8)
        )
        gt_instance_labels = np.asarray(gt_labels_list, dtype=np.int32)

        out_path = os.path.join(out_dir, f"{filename}.npz")
        np.savez(
            out_path,
            assign_sem=assign_sem,
            assign_inst=assign_inst,
            seg_gt_sem=seg_gt_sem,
            seg_gt_inst=seg_gt_inst,
            seg_lengths=seg_lengths,
            pred_instance_masks=pred_instance_masks,
            pred_instance_labels=pred_instance_labels,
            gt_instance_masks=gt_instance_masks,
            gt_instance_labels=gt_instance_labels,
            filename=np.array(filename),
            cost_matrix=cost_matrix.astype(np.float32),
            ownership_mode=np.array(ownership_mode),
            projection_version=np.array("v2-20261023"),
            area_penalty=np.float64(
                params.get("area_penalty", DEFAULT_COST_PARAMS["area_penalty"])
            ),
            primitive_vote_agreement=np.float64(
                params.get(
                    "primitive_vote_agreement",
                    DEFAULT_COST_PARAMS["primitive_vote_agreement"],
                )
            ),
            scene_bound=np.float32(scene_bound),
        )

        # ---- 可视化（可选）----
        if vis_dir is not None:
            vis_path = os.path.join(vis_dir, f"{filename}_proj.png")
            iou_per_gt = _per_gt_best_iou(
                pred_instance_masks,
                pred_instance_labels,
                gt_instance_masks,
                gt_instance_labels,
                seg_lengths,
            )
            try:
                _visualize_three_panel(
                    scene_lines_norm=scene_lines.astype(np.float64),
                    instance_tpl_norm=instance_tpl_norm,
                    pred_instance_labels=np.asarray(pred_sem_id, dtype=np.int32),
                    assign_inst_seg_lvl=assign_inst,
                    seg_gt_inst=seg_gt_inst,
                    gt_instance_labels=gt_instance_labels,
                    out_path=vis_path,
                    title_prefix=f"{filename}",
                    iou_summary=iou_per_gt,
                )
            except Exception as e:
                print(f"[vis-error] {filename}: {e}")

        return filename
    except Exception as e:
        print(f"[error] processing {npz_path}: {e}")
        traceback.print_exc()
        return None


def _process_one_npz_kw(npz_path, **kw):
    """方便在 multiprocessing.Pool.imap 中按位置参数调用。"""
    return process_one_npz(npz_path, **kw)


# ============================================================================ #
#  入口
# ============================================================================ #


def main():
    parser = argparse.ArgumentParser(
        description="Stage B: project AR instance-level predictions to line-level segment labels."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Same as the inference -o; expects ${output}/results_<split>/*.npz to exist.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Subdir suffix matched to inference: results_<split>/, output to line_level_<split>/.",
    )
    parser.add_argument(
        "--primitive_lib",
        type=str,
        required=True,
        help="Primitive library folder containing <prim_id>.json files.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="If >0, use multiprocessing.Pool; if 0, run sequentially.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="If set, only process the first N npz files (smoke test).",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Resume projection by skipping samples whose line-level npz already exists.",
    )
    parser.add_argument(
        "--deduplicate_exact",
        action="store_true",
        help="Drop exact duplicate hard-token instances before projection; WTA output is unchanged.",
    )
    # 阈值 / 权重（与参考脚本默认一致；可单独调）
    parser.add_argument(
        "--perp_tolerance", type=float, default=DEFAULT_COST_PARAMS["perp_tolerance"]
    )
    parser.add_argument(
        "--cost_thresh", type=float, default=DEFAULT_COST_PARAMS["cost_thresh"]
    )
    parser.add_argument(
        "--cost_caps", type=float, default=DEFAULT_COST_PARAMS["cost_caps"]
    )
    parser.add_argument(
        "--roi_margin", type=float, default=DEFAULT_COST_PARAMS["roi_margin"]
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=DEFAULT_COST_PARAMS["scale_factor"],
        help='Coord rescale to "pixel-equivalent" space; default 70 mirrors floorplancad.',
    )
    parser.add_argument(
        "--attribute_source",
        choices=("hard", "soft"),
        default=DEFAULT_COST_PARAMS["attribute_source"],
        help="Use argmax-bin attributes (official) or differentiable expectations (diagnostic).",
    )
    parser.add_argument(
        "--assignment_mode",
        choices=("robust", "bidirectional_overshoot"),
        default=DEFAULT_COST_PARAMS["assignment_mode"],
        help="Segment/template extent cost: source-only robust or bidirectional overshoot.",
    )
    parser.add_argument(
        "--spatial_prior",
        choices=("none", "obb"),
        default=DEFAULT_COST_PARAMS["spatial_prior"],
        help="Optional predicted-template spatial candidate gate; default none.",
    )
    parser.add_argument(
        "--obb_long_margin",
        type=float,
        default=DEFAULT_COST_PARAMS["obb_long_margin"],
        help="OBB expansion along the longer template axis.",
    )
    parser.add_argument(
        "--obb_normal_margin",
        type=float,
        default=DEFAULT_COST_PARAMS["obb_normal_margin"],
        help="OBB expansion along the shorter template axis.",
    )
    parser.add_argument(
        "--obb_min_inside_ratio",
        type=float,
        default=DEFAULT_COST_PARAMS["obb_min_inside_ratio"],
        help="Minimum finite-segment fraction inside the expanded OBB.",
    )
    parser.add_argument(
        "--fragment_mode",
        choices=(
            "none",
            "source_containment_competitive",
            "support_chain_competitive",
        ),
        default=DEFAULT_COST_PARAMS["fragment_mode"],
        help="Optional all-class fragment or provenance-aware support-chain competitor.",
    )
    parser.add_argument(
        "--fragment_angle_tol_deg",
        type=float,
        default=DEFAULT_COST_PARAMS["fragment_angle_tol_deg"],
    )
    parser.add_argument(
        "--fragment_min_overlap_ratio",
        type=float,
        default=DEFAULT_COST_PARAMS["fragment_min_overlap_ratio"],
    )
    parser.add_argument(
        "--support_chain_join_gap",
        type=float,
        default=DEFAULT_COST_PARAMS["support_chain_join_gap"],
    )
    parser.add_argument(
        "--support_chain_angle_tol_deg",
        type=float,
        default=DEFAULT_COST_PARAMS["support_chain_angle_tol_deg"],
    )
    parser.add_argument(
        "--support_chain_line_residual",
        type=float,
        default=DEFAULT_COST_PARAMS["support_chain_line_residual"],
    )
    parser.add_argument(
        "--support_match_angle_tol_deg",
        type=float,
        default=DEFAULT_COST_PARAMS["support_match_angle_tol_deg"],
    )
    parser.add_argument(
        "--support_containment_weight",
        type=float,
        default=DEFAULT_COST_PARAMS["support_containment_weight"],
    )
    parser.add_argument(
        "--support_length_ratio_free",
        type=float,
        default=DEFAULT_COST_PARAMS["support_length_ratio_free"],
    )
    parser.add_argument(
        "--support_length_ratio_weight",
        type=float,
        default=DEFAULT_COST_PARAMS["support_length_ratio_weight"],
    )
    parser.add_argument(
        "--support_angle_weight",
        type=float,
        default=DEFAULT_COST_PARAMS["support_angle_weight"],
    )
    parser.add_argument(
        "--ownership_mode",
        choices=("legacy", "soft_area", "group_only", "soft_area_group"),
        default=DEFAULT_COST_PARAMS["ownership_mode"],
        help="Ownership policy; v2 default is soft_area_group. Legacy uses prefer_small/no_prefer_small.",
    )
    parser.add_argument(
        "--area_penalty", type=float, default=DEFAULT_COST_PARAMS["area_penalty"]
    )
    parser.add_argument(
        "--primitive_vote_agreement",
        type=float,
        default=DEFAULT_COST_PARAMS["primitive_vote_agreement"],
    )
    parser.add_argument(
        "--semantic_offset",
        type=int,
        default=1,
        help="Offset applied to non-negative scene semantic ids; use 0 for FloorplanCAD.",
    )
    parser.add_argument(
        "--prefer_small",
        dest="prefer_small",
        action="store_true",
        help="Legacy ownership only: in overlap, assign segments to the pred instance with smallest template bbox first.",
    )
    parser.add_argument(
        "--no_prefer_small",
        dest="prefer_small",
        action="store_false",
        help="Legacy ownership only: disable size-priority and use plain Winner-Takes-All by min cost.",
    )
    parser.set_defaults(prefer_small=DEFAULT_COST_PARAMS["prefer_small"])
    parser.add_argument(
        "--fused",
        action="store_true",
        help="Per-block eval of fused (folder-merged) predictions: clip each pred "
        "instance ROI to the block frame [-1,1] (pixel-space ±scale_factor) "
        "so a fused instance only contributes its in-block portion to "
        "inst_areas / segment matching.",
    )
    parser.add_argument(
        "--vis",
        action="store_true",
        help="Write 3-panel projection visualization PNGs to ${output}/line_level_vis_<split>/.",
    )
    args = parser.parse_args()

    in_dir = os.path.join(args.output, f"results_{args.split}")
    out_suffix = "" if args.attribute_source == "hard" else "_soft"
    out_dir = os.path.join(args.output, f"line_level_{args.split}{out_suffix}")
    os.makedirs(out_dir, exist_ok=True)
    vis_dir = None
    if args.vis:
        vis_dir = os.path.join(args.output, f"line_level_vis_{args.split}")
        os.makedirs(vis_dir, exist_ok=True)
        print(f"Vis enabled, output to: {vis_dir}")

    if not os.path.isdir(in_dir):
        raise FileNotFoundError(f"Input dir not found: {in_dir}")

    npz_files = sorted(
        [os.path.join(in_dir, x) for x in os.listdir(in_dir) if x.endswith(".npz")]
    )
    if args.max_samples is not None:
        npz_files = npz_files[: args.max_samples]
    if args.skip_existing:
        before = len(npz_files)
        npz_files = [
            path
            for path in npz_files
            if not os.path.isfile(
                os.path.join(
                    out_dir,
                    os.path.basename(path),
                )
            )
        ]
        print(f"Skip-existing removed {before - len(npz_files)} completed samples")
    print(f"Found {len(npz_files)} npz files in {in_dir}")

    params = dict(DEFAULT_COST_PARAMS)
    params["perp_tolerance"] = args.perp_tolerance
    params["cost_thresh"] = args.cost_thresh
    params["cost_caps"] = args.cost_caps
    params["roi_margin"] = args.roi_margin
    params["scale_factor"] = args.scale_factor
    params["attribute_source"] = args.attribute_source
    params["assignment_mode"] = args.assignment_mode
    params["spatial_prior"] = args.spatial_prior
    params["obb_long_margin"] = args.obb_long_margin
    params["obb_normal_margin"] = args.obb_normal_margin
    params["obb_min_inside_ratio"] = args.obb_min_inside_ratio
    params["fragment_mode"] = args.fragment_mode
    params["fragment_angle_tol_deg"] = args.fragment_angle_tol_deg
    params["fragment_min_overlap_ratio"] = args.fragment_min_overlap_ratio
    params["support_chain_join_gap"] = args.support_chain_join_gap
    params["support_chain_angle_tol_deg"] = args.support_chain_angle_tol_deg
    params["support_chain_line_residual"] = args.support_chain_line_residual
    params["support_match_angle_tol_deg"] = args.support_match_angle_tol_deg
    params["support_containment_weight"] = args.support_containment_weight
    params["support_length_ratio_free"] = args.support_length_ratio_free
    params["support_length_ratio_weight"] = args.support_length_ratio_weight
    params["support_angle_weight"] = args.support_angle_weight
    params["ownership_mode"] = args.ownership_mode
    params["area_penalty"] = args.area_penalty
    params["primitive_vote_agreement"] = args.primitive_vote_agreement
    params["semantic_offset"] = args.semantic_offset
    params["prefer_small"] = args.prefer_small
    params["fused"] = args.fused
    params["deduplicate_exact"] = args.deduplicate_exact

    protocol_path = os.path.join(out_dir, "PROJECTION_PROTOCOL.json")
    protocol = {
        "version": "v2-20261023",
        "params": params,
        "primitive_lib": os.path.abspath(args.primitive_lib),
    }
    if os.path.isfile(protocol_path):
        with open(protocol_path, encoding="utf-8") as handle:
            previous_protocol = json.load(handle)
        if previous_protocol != protocol:
            raise ValueError(
                "Output directory contains a different projection protocol; choose a new output directory."
            )
    elif any(name.endswith(".npz") for name in os.listdir(out_dir)):
        raise ValueError(
            "Output directory contains unversioned projections; choose a new output directory."
        )
    with open(protocol_path, "w", encoding="utf-8") as handle:
        json.dump(protocol, handle, ensure_ascii=False, indent=2)

    print(f"Cost params: {params}")

    fn = partial(
        _process_one_npz_kw,
        primitive_lib_folder=args.primitive_lib,
        out_dir=out_dir,
        params=params,
        vis_dir=vis_dir,
    )

    if args.num_workers > 0 and len(npz_files) > 1:
        nw = min(args.num_workers, cpu_count(), len(npz_files))
        print(f"Using {nw} workers")
        with Pool(nw) as pool:
            results = list(
                tqdm(pool.imap_unordered(fn, npz_files), total=len(npz_files))
            )
    else:
        results = [fn(p) for p in tqdm(npz_files)]

    n_ok = sum(1 for r in results if r is not None)
    print(f"Processed {n_ok}/{len(npz_files)} samples successfully")
    print(f"Outputs at: {out_dir}")
    if n_ok != len(npz_files):
        raise RuntimeError(f"{len(npz_files) - n_ok} projections failed")


if __name__ == "__main__":
    main()
