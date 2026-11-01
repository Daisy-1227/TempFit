import numpy as np


def soft_area_assignment(cost_matrix, instance_areas, threshold=70.0, penalty=1.0):
    cost = np.asarray(cost_matrix, dtype=np.float64)
    area = np.asarray(instance_areas, dtype=np.float64)
    if cost.ndim != 2 or area.shape != (cost.shape[1],):
        raise ValueError("Cost must be [segments, predictions] and areas [predictions]")
    if cost.shape[1] == 0:
        return np.full(cost.shape[0], -1, dtype=np.int32)
    eligible = np.isfinite(cost) & (cost < threshold) & np.isfinite(area)[None, :]
    score = np.where(
        eligible, cost + penalty * np.log(np.maximum(area, 1e-9))[None, :], np.inf
    )
    assignment = score.argmin(axis=1).astype(np.int32)
    assignment[~eligible.any(axis=1)] = -1
    return assignment


def consolidate_source_primitives(assignment, source_primitive_ids, agreement=0.5):
    """Unify a raw CAD primitive when >= agreement of its segments vote for one instance.

    Source IDs are observable SVG/CAD provenance, never GT instance labels.
    Unassigned segments stay in the vote denominator. A winning foreground
    instance can inherit previously ineligible segments of the same primitive.
    This extension beyond per-segment ROI is intentional and must be validated
    for datasets whose CAD primitives are allowed to span multiple objects.
    """
    assignment = np.asarray(assignment, dtype=np.int32)
    ids = np.asarray(source_primitive_ids)
    if ids.shape != assignment.shape:
        raise ValueError("Source primitive IDs must align with segment assignments")
    if not np.any(assignment >= 0):
        return assignment.copy()
    _, inverse = np.unique(ids, return_inverse=True)
    n = int(assignment.max()) + 1
    votes = np.zeros((int(inverse.max()) + 1, n + 1))
    np.add.at(votes, (inverse, assignment + 1), 1.0)
    winner = votes[:, 1:].argmax(1)
    fraction = votes[:, 1:].max(1) / np.maximum(votes.sum(1), 1.0)
    out = assignment.copy()
    change = fraction[inverse] >= agreement
    out[change] = winner[inverse[change]]
    return out
