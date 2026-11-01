import copy, math
import numpy as np
import torch
from datasets.primitive_dataset import PrimitiveDataset


class AuditedDataset(PrimitiveDataset):
    def _handle_invalid_sample(self, idx, retry_count, reason, *, retryable):
        self.retry_events.append(
            dict(
                filename=self.data_filename[idx],
                retry=retry_count,
                reason=reason,
                retryable=retryable,
            )
        )
        return super()._handle_invalid_sample(
            idx, retry_count, reason, retryable=retryable
        )


def owner(svg):
    return np.asarray(svg.instance_ids)[np.asarray(svg.primitive_ids)]


def box(coords):
    p = coords.reshape(-1, 2)
    return np.r_[p.min(0), p.max(0)]


def ious(b, others):
    if not len(others):
        return np.zeros(0)
    inter = np.maximum(
        0, np.minimum(b[2:], others[:, 2:]) - np.maximum(b[:2], others[:, :2])
    ).prod(1)
    union = (
        np.prod(b[2:] - b[:2]) + np.prod(others[:, 2:] - others[:, :2], axis=1) - inter
    )
    return inter / np.maximum(union, 1e-12)


class IndependentInstanceScale:
    def __init__(
        self,
        enabled,
        p=0.5,
        sample_ratio=0.3,
        scale_range=(0.75, 1.25),
        max_bbox_iou=0.02,
        scale_bounds=(0.03, 12.0),
        capture=False,
    ):
        self.enabled = enabled
        self.p = p
        self.sample_ratio = sample_ratio
        self.scale_range = scale_range
        self.max_bbox_iou = max_bbox_iou
        self.scale_bounds = scale_bounds
        self.capture = capture
        self.seed = 0
        self.calls = 0
        self.events = []

    def reset(self, seed):
        self.seed = seed
        self.calls = 0
        self.events = []

    def __call__(self, d):
        rng = np.random.default_rng(self.seed + self.calls * 10000019)
        self.calls += 1
        if self.capture:
            self.before = copy.deepcopy(d)
        svg = d["svg_data"]
        co = np.asarray(svg.coords).reshape(-1, 2, 2)
        own = owner(svg)
        row = dict(
            enabled=self.enabled,
            triggered=False,
            eligible=0,
            requested=0,
            accepted=[],
            rejected=[],
        )
        if self.enabled and rng.random() < self.p:
            row["triggered"] = True
            boxes = {
                i: box(co[own == i]) for i in range(len(d["scale"])) if np.any(own == i)
            }
            eligible = [
                i
                for i, b in boxes.items()
                if np.min(b[:2]) >= -1 and np.max(b[2:]) <= 1
            ]
            row["eligible"] = len(eligible)
            n = (
                min(len(eligible), max(1, math.ceil(len(eligible) * self.sample_ratio)))
                if eligible
                else 0
            )
            row["requested"] = n
            chosen = rng.choice(eligible, size=n, replace=False) if n else []
            co = co.copy()
            d["scale"] = np.asarray(d["scale"]).copy()
            if "size" in d:
                d["size"] = np.asarray(d["size"]).copy()
            for iid in chosen:
                iid = int(iid)
                k = float(np.exp(rng.uniform(*np.log(self.scale_range))))
                tr = np.asarray(d["translation"][iid])
                mask = own == iid
                proposed = (co[mask] - tr) * k + tr
                b = box(proposed)
                sc = d["scale"][iid] * k
                reason = None
                others = np.array(
                    [v for key, v in boxes.items() if key != iid]
                ).reshape(-1, 4)
                original_iou = ious(boxes[iid], others)
                new_iou = ious(b, others)
                if np.min(b[:2]) < -1 or np.max(b[2:]) > 1:
                    reason = "outside_frame"
                elif sc.min() < self.scale_bounds[0] or sc.max() > self.scale_bounds[1]:
                    reason = "scale_head_range"
                elif np.any(
                    new_iou > np.maximum(original_iou, self.max_bbox_iou) + 1e-7
                ):
                    reason = "new_foreground_overlap"
                if reason:
                    row["rejected"].append(dict(instance=iid, factor=k, reason=reason))
                    continue
                co[mask] = proposed
                d["scale"][iid] = sc
                if "size" in d:
                    d["size"][iid] *= k
                boxes[iid] = b
                row["accepted"].append(
                    dict(
                        instance=iid,
                        factor=k,
                        lines=int(mask.sum()),
                        semantic=int(d["semantic_id"][iid]),
                        primitive=int(d["primitive_id"][iid]),
                        translation=tr.tolist(),
                        new_scale=sc.tolist(),
                    )
                )
            svg.coords = co.reshape(np.asarray(svg.coords).shape)
        self.events.append(row)
        if self.capture:
            self.after = copy.deepcopy(d)
        return d
