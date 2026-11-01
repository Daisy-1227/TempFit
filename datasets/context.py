import copy
import math
from collections import Counter
import numpy as np
from datasets.scale import SceneDataset as ScaleDataset, batch, summarize_exposure
from datasets.full_source import FullSourceInRange
from datasets.instance_scale import owner, box, ious


class AnchoredContext:
    def __init__(
        self,
        sample_ratio=0.3,
        max_delta=0.3,
        min_displacement=0.05,
        max_attempts=8,
        max_bbox_iou=0.02,
        translation_bounds=(-2.0, 2.0),
        capture=False,
    ):
        assert 0 <= sample_ratio <= 1 and 0 <= min_displacement < max_delta
        self.sample_ratio = sample_ratio
        self.max_delta = max_delta
        self.min_displacement = min_displacement
        self.max_attempts = max_attempts
        self.max_bbox_iou = max_bbox_iou
        self.translation_bounds = translation_bounds
        self.capture = capture
        self.reset(0, False)

    def reset(self, seed, enabled):
        self.seed = seed
        self.enabled = enabled
        self.calls = 0
        self.events = []

    def __call__(self, d):
        rng = np.random.default_rng(self.seed + self.calls * 10000019)
        self.calls += 1
        if self.capture:
            self.before = copy.deepcopy(d)
        co = np.asarray(d["svg_data"].coords).reshape(-1, 2, 2)
        own = owner(d["svg_data"])
        tr = np.asarray(d["translation"])
        lo, hi = self.translation_bounds
        boxes = {
            j: box(co[own == j])
            for j in range(len(tr))
            if (own == j).any() and np.isfinite(co[own == j]).all()
        }
        valid = [
            j
            for j in boxes
            if np.isfinite(tr[j]).all() and ((tr[j] >= lo) & (tr[j] <= hi)).all()
        ]
        anchor = int(rng.choice(valid)) if valid else None
        others = [j for j in valid if j != anchor]
        n = math.ceil(len(others) * self.sample_ratio) if self.enabled else 0
        chosen = rng.choice(others, size=n, replace=False) if n else []
        event = dict(
            enabled=self.enabled,
            anchor=anchor,
            eligible_others=len(others),
            requested=n,
            moved=[],
            rejected=[],
        )
        original_anchor = co[own == anchor].copy() if anchor is not None else None
        if n:
            co = co.copy()
            d["translation"] = tr.copy()
            for iid in chosen:
                iid = int(iid)
                comparison = np.array(
                    [b for j, b in boxes.items() if j != iid]
                ).reshape(-1, 4)
                old_iou = ious(boxes[iid], comparison)
                reason = "no_legal_proposal"
                for attempt in range(self.max_attempts):
                    delta = rng.uniform(-self.max_delta, self.max_delta, 2).astype(
                        co.dtype
                    )
                    if np.linalg.norm(delta) < self.min_displacement:
                        reason = "small_displacement"
                        continue
                    shifted = tr[iid] + delta
                    if not ((shifted >= lo) & (shifted <= hi)).all():
                        reason = "translation_out_of_range"
                        continue
                    proposed = co[own == iid] + delta
                    if not np.isfinite(proposed).all():
                        reason = "nonfinite_geometry"
                        continue
                    bb = box(proposed)
                    if np.any(
                        ious(bb, comparison)
                        > np.maximum(old_iou, self.max_bbox_iou) + 1e-7
                    ):
                        reason = "new_foreground_overlap"
                        continue
                    co[own == iid] = proposed
                    d["translation"][iid] = shifted
                    boxes[iid] = bb
                    event["moved"].append(
                        dict(
                            instance=iid,
                            delta=delta.tolist(),
                            attempts=attempt + 1,
                            semantic=int(d["semantic_id"][iid]),
                            primitive=int(d["primitive_id"][iid]),
                        )
                    )
                    break
                else:
                    event["rejected"].append(dict(instance=iid, reason=reason))
            d["svg_data"].coords = co.reshape(np.asarray(d["svg_data"].coords).shape)
        if anchor is not None:
            np.testing.assert_array_equal(co[own == anchor], original_anchor)
            np.testing.assert_array_equal(d["translation"][anchor], tr[anchor])
        self.events.append(event)
        if self.capture:
            self.after = copy.deepcopy(d)
        return d


class SceneDataset(ScaleDataset):
    def __init__(self, cfg, split="train", capture=False, deterministic=False):
        super().__init__(cfg, split, capture, deterministic)
        self.context = None
        if split == "train" and not deterministic:
            assert (
                self.repeat == 4
            ), "Two upstream seeds, each with native/context views"
            self.context = AnchoredContext(
                **cfg["context_augmentation"], capture=capture
            )
            chain = self.ds.transform.transforms
            pos = next(
                i for i, t in enumerate(chain) if isinstance(t, FullSourceInRange)
            )
            chain.insert(pos, self.context)

    def __getitem__(self, key):
        if self.context is None:
            return super().__getitem__(key)
        index, real = key if isinstance(key, tuple) else (int(key), True)
        count = len(self.names)
        repeat, local = divmod(index, count)
        enabled = bool(repeat % 2)
        upstream_index = local + (repeat // 2) * count
        seed = int(
            self.cfg["experiment"]["data_seed"] + self.epoch * 1000003 + upstream_index
        )
        self.context.reset(seed + 900000001, enabled)
        result = super().__getitem__((upstream_index, real))
        result.update(index=index, pair_index=upstream_index, context_branch=enabled)
        result["context_events"] = copy.deepcopy(self.context.events)
        result["context_exposure"] = (
            None
            if result["scale_fallback"] or not self.context.events
            else copy.deepcopy(self.context.events[-1])
        )
        return result


class RealSmoke(SceneDataset):
    def __len__(self):
        return 129

    def __getitem__(self, key):
        index, real = key if isinstance(key, tuple) else (int(key), True)
        paired_index = index // 2 + (index % 2) * len(self.names)
        result = super().__getitem__((paired_index, real))
        result["index"] = index
        return result


def summarize_context(rows):
    totals = Counter()
    reasons = Counter()
    for r in rows:
        if not r.get("real", True):
            continue
        totals["views"] += 1
        totals["context_views"] += int(r.get("context_branch", False))
        event = r.get("context_exposure")
        if event is None:
            totals["unobserved_views"] += 1
            continue
        totals["requested"] += event["requested"]
        totals["moved"] += len(event["moved"])
        totals["actually_changed_views"] += int(bool(event["moved"]))
        reasons.update(e["reason"] for e in event["rejected"])
    return dict(
        **{
            k: totals[k]
            for k in (
                "views",
                "context_views",
                "unobserved_views",
                "requested",
                "moved",
                "actually_changed_views",
            )
        },
        rejection_reasons=dict(reasons)
    )
