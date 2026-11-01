import copy
import torch
from datasets.metadata import metadata, identity, schedule
from datasets.primitive_dataset import custom_collate


def groups(data):
    return data if isinstance(data, list) else [data]


def batch(data, device):
    from util import io as base

    items = groups(data)
    valid = [d for d in items if d.get("real", True)]
    # A fully dummy rank still participates in SyncBN; its entire objective has weight zero.
    items = valid if valid else items[:1]
    samples = [copy.deepcopy(d["sample"]) for d in items]
    result = base.move(custom_collate(samples), device)
    B = len(items)
    P = max(len(d["metadata"]["centers"]) for d in items)
    N = max(len(d["sample"]["primitive_id"]) for d in items)
    coords = torch.zeros(B, P, 2, device=device)
    token_mask = torch.zeros(B, P, device=device, dtype=torch.bool)
    instance_mask = torch.zeros(B, N, device=device, dtype=torch.bool)
    target = torch.zeros(B, N, P, device=device)
    for i, d in enumerate(items):
        md = d["metadata"]
        p = len(md["centers"])
        n = len(d["sample"]["primitive_id"])
        coords[i, :p] = torch.as_tensor(md["centers"], device=device)
        token_mask[i, :p] = True
        instance_mask[i, :n] = True
        target[i, :n, :p] = torch.as_tensor(
            md["targets"], device=device, dtype=torch.float32
        )
    assert torch.equal(instance_mask, result["primitive_id"] >= 0)
    assert (target.sum(-1)[instance_mask] > 0).all()
    assert (target.sum(1) <= 1).all()
    return (
        result,
        coords,
        dict(
            values=target,
            tokens=token_mask,
            instances=instance_mask,
            real=float(bool(valid)),
        ),
    )
