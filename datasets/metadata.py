import numpy as np
import torch


def metadata(sample, audit=False):
    svg, vec = sample["svg_data"], sample["vec_data"]
    ids = np.asarray(vec.prim_ids, dtype=np.int64)
    assert np.array_equal(ids, np.asarray(svg.primitive_ids))
    raw = np.asarray(svg.instance_ids, dtype=np.int64)
    p, n = len(raw), len(sample["semantic_id"])
    count = np.bincount(ids, minlength=p)
    assert p > 0 and (count > 0).all()
    co = np.asarray(vec.coords)
    centers = np.stack(
        [np.bincount(ids, weights=co[:, j], minlength=p) / count for j in (0, 1)], 1
    )
    inverse = np.asarray(sample["original_indices"], dtype=np.int64)
    assert np.array_equal(np.sort(inverse), np.arange(n))
    own = np.full(p, -1, dtype=np.int64)
    valid = raw >= 0
    own[valid] = inverse[raw[valid]]
    targets = own[None, :] == np.arange(n)[:, None]
    assert targets.any(1).all()
    assert np.array_equal(
        np.bincount(own[ids][own[ids] >= 0], minlength=n),
        sample["recon_source_raw_count"],
    )
    if audit:
        from datasets.audit import source_metadata

        reference = source_metadata(sample)
        assert np.array_equal(targets, reference["targets"])
        assert np.allclose(centers, reference["centers"])
    return dict(centers=centers.astype(np.float32), targets=targets)


def schedule(length, epoch, seed, world, rank, shuffle=True):
    order = np.arange(length)
    if shuffle:
        np.random.default_rng(seed + epoch).shuffle(order)
    items = [(int(i), True) for i in order]
    while len(items) % world:
        items.append((int(order[0]), False))
    return items[rank::world]


def identity(item):
    return item
