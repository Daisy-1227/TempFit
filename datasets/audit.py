import numpy as np


def source_metadata(sample):
    """Support zero-GT Val and one-instance/all-positive source without dropping it."""
    svg = sample["svg_data"]
    v = sample["vec_data"]
    ids = base.arr(v.prim_ids).astype(int)
    assert np.array_equal(ids, base.arr(svg.primitive_ids))
    p = len(svg.instance_ids)
    assert p > 0 and np.array_equal(np.unique(ids), np.arange(p))
    coords = base.arr(v.coords)
    count = np.bincount(ids, minlength=p)
    centers = np.stack(
        [np.bincount(ids, weights=coords[:, j], minlength=p) / count for j in (0, 1)], 1
    )
    raw = base.arr(svg.instance_ids).astype(int)
    n = len(sample["semantic_id"])
    inverse = base.arr(sample["original_indices"]).astype(int)
    assert np.array_equal(np.sort(inverse), np.arange(n))
    sorted_ids = np.full(p, -1, int)
    fg = raw >= 0
    sorted_ids[fg] = inverse[raw[fg]]
    targets = sorted_ids[None, :] == np.arange(n)[:, None]
    assert targets.any(1).all() and (targets.sum(0) <= 1).all()
    assert np.array_equal(np.argsort(inverse), base.arr(sample["sort_indices"]))
    counts = np.bincount(sorted_ids[ids][sorted_ids[ids] >= 0], minlength=n)
    assert np.array_equal(counts, base.arr(sample["recon_source_raw_count"]))
    segments = base.arr(svg.coords).reshape(-1, 2, 2)
    for j in range(n):
        actual = segments[targets[j][ids]]
        packed = base.arr(sample["recon_source_segments"][j])[
            base.arr(sample["recon_source_mask"][j]).astype(bool)
        ]
        assert all(
            np.any(np.all(np.isclose(actual, s, atol=2e-7, rtol=0), axis=(1, 2)))
            for s in packed
        )
    return dict(
        ids=ids,
        centers=centers,
        count=count,
        targets=targets,
        sorted_ids=sorted_ids,
        raw_ids=raw,
    )
