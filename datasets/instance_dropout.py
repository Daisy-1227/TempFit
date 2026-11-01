import copy, math
from collections import Counter
import numpy as np
from datasets.context import (
    SceneDataset as ContextDataset,
    batch,
    summarize_context,
    summarize_exposure,
)

GT_KEYS = (
    "semantic_id",
    "primitive_id",
    "scale",
    "rotation",
    "flip_mode",
    "translation",
    "flip_invariant_primitive",
    "size",
)
AUDIT_INSTANCE_KEYS = (
    "clip_count_ratio",
    "clip_length_ratio",
    "clip_bbox_ratio",
    "clip_input_segment_counts",
    "clip_input_segment_lengths",
    "full_translation_pre_clip",
    "instance_origin_kind",
    "instance_jittered",
)


class InstanceDropout:
    def __init__(self, p=0.2, drop_ratio=0.2, capture=False):
        assert 0 <= p <= 1 and 0 <= drop_ratio < 1
        self.p = p
        self.drop_ratio = drop_ratio
        self.capture = capture
        self.reset(0, 0, 0)

    def reset(self, seed, epoch, index):
        self.key = (int(seed), int(epoch), int(index))
        self.calls = 0
        self.events = []
        self.triggered = bool(
            np.random.default_rng(np.random.SeedSequence(self.key)).random() < self.p
        )

    def __call__(self, d):
        call = self.calls
        self.calls += 1
        if self.capture:
            self.before = copy.deepcopy(d)
        svg = d["svg_data"]
        n = len(d["translation"])
        prim = np.asarray(svg.primitive_ids, dtype=np.int64)
        inst = np.asarray(svg.instance_ids, dtype=np.int64)
        own = inst[prim]
        assert np.all((inst >= -1) & (inst < n))
        k = (
            min(max(0, n - 1), int(math.floor(n * self.drop_ratio + 0.5)))
            if self.triggered
            else 0
        )
        rng = np.random.default_rng(np.random.SeedSequence(self.key + (call, 1771)))
        anchor = int(rng.integers(n)) if n else None
        removed = (
            np.sort(
                rng.choice([i for i in range(n) if i != anchor], size=k, replace=False)
            )
            if k
            else np.array([], dtype=int)
        )
        retain = np.ones(n, bool)
        retain[removed] = False
        linekeep = np.ones(len(own), bool)
        fg = own >= 0
        linekeep[fg] = retain[own[fg]]
        audit = d.setdefault("_recon_audit", {})
        pre_ids = np.asarray(audit.get("preclip_full_line_instance_ids", []), dtype=int)
        row = dict(
            key=list(self.key),
            attempt=call,
            triggered=self.triggered,
            applied=bool(k),
            anchor=anchor,
            instances_before=n,
            instances_after=n - k,
            removed=removed.tolist(),
            kept=np.flatnonzero(retain).tolist(),
            segments_before=len(own),
            segments_after=int(linekeep.sum()),
            source_foreground_before=len(pre_ids),
            source_foreground_after=int(np.isin(pre_ids, np.flatnonzero(retain)).sum()),
            removed_semantic=np.asarray(d["semantic_id"])[removed].tolist(),
            removed_primitive=np.asarray(d["primitive_id"])[removed].tolist(),
            skip_reason=(
                None
                if k
                else ("probability" if not self.triggered else "too_few_instances")
            ),
        )
        if k:
            assert anchor not in removed and n - k >= 1
            pk = np.ones(len(inst), bool)
            pfg = inst >= 0
            pk[pfg] = retain[inst[pfg]]
            pm = np.full(len(inst), -1, dtype=np.int32)
            pm[pk] = np.arange(pk.sum(), dtype=np.int32)
            im = np.full(n, -1, dtype=np.int32)
            im[retain] = np.arange(retain.sum(), dtype=np.int32)
            for key in ("coords", "colors", "widths", "layer_ids"):
                setattr(svg, key, np.asarray(getattr(svg, key))[linekeep].copy())
            svg.primitive_ids = pm[prim[linekeep]]
            newinst = inst[pk].astype(np.int32, copy=True)
            newfg = newinst >= 0
            newinst[newfg] = im[newinst[newfg]]
            svg.instance_ids = newinst
            for key in ("semantic_ids", "primitive_lengths"):
                setattr(svg, key, np.asarray(getattr(svg, key))[pk].copy())
            for key in GT_KEYS:
                if key in d and d[key] is not None:
                    assert len(d[key]) == n, (key, len(d[key]), n)
                    d[key] = np.asarray(d[key])[retain].copy()
            for key in AUDIT_INSTANCE_KEYS:
                if key in audit:
                    assert len(audit[key]) == n, (key, len(audit[key]), n)
                    audit[key] = np.asarray(audit[key])[retain].copy()
            for idkey, cokey in [
                ("preclip_full_line_instance_ids", "preclip_full_coords"),
                ("pre_grid_line_instance_ids", "pre_grid_instance_coords"),
            ]:
                if idkey in audit:
                    ids = np.asarray(audit[idkey], dtype=np.int64)
                    assert np.all((ids >= 0) & (ids < n))
                    keep = retain[ids]
                    audit[cokey] = np.asarray(audit[cokey])[keep].copy()
                    audit[idkey] = im[ids[keep]]
            for key in ("grid_coord", "_vec_grid_sample_idx"):
                if key in d:
                    assert len(d[key]) == len(linekeep), key
                    d[key] = np.asarray(d[key])[linekeep].copy()
            assert set(
                np.asarray(svg.instance_ids)[np.asarray(svg.instance_ids) >= 0]
            ) == set(range(n - k))
        self.events.append(row)
        if self.capture:
            self.after = copy.deepcopy(d)
        return d


class SceneDataset(ContextDataset):
    def __init__(self, cfg, split="train", capture=False, deterministic=False):
        super().__init__(cfg, split, capture, deterministic)
        self.dropout = None
        spec = cfg.get("instance_dropout", {})
        if split == "train" and not deterministic and spec.get("enabled", False):
            assert spec["count_basis"] == "all_foreground_instances_before_dropout"
            assert (
                spec["count_rounding"] == "nearest_half_up"
                and spec["preserve_anchor"]
                and spec["keep_background"]
            )
            assert (
                spec["minimum_scene_line_retention"] is None
                and not spec["apply_to_validation"]
            )
            self.dropout = InstanceDropout(
                spec["p"], spec["drop_ratio"], capture=capture
            )
            chain = self.ds.transform.transforms
            positions = [
                i for i, t in enumerate(chain) if type(t).__name__ == "VecGridSample"
            ]
            assert len(positions) == 1
            chain.insert(positions[0] + 1, self.dropout)

    def __getitem__(self, key):
        index, real = key if isinstance(key, tuple) else (int(key), True)
        if self.dropout:
            self.dropout.reset(self.cfg["instance_dropout"]["seed"], self.epoch, index)
        result = super().__getitem__(key)
        event = (
            self.dropout.events[-1]
            if self.dropout
            and self.dropout.events
            and not result.get("scale_fallback", False)
            else None
        )
        result["dropout_exposure"] = (
            copy.deepcopy(event)
            if event is not None
            else dict(
                triggered=False,
                applied=False,
                skip_reason=(
                    "deterministic_fallback"
                    if result.get("scale_fallback")
                    else "disabled"
                ),
                instances_before=len(result["sample"]["semantic_id"]),
                instances_after=len(result["sample"]["semantic_id"]),
                removed=[],
                removed_semantic=[],
                removed_primitive=[],
            )
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


def summarize_dropout(rows):
    c = Counter()
    sem = Counter()
    prim = Counter()
    ratios = []
    for r in rows:
        if not r.get("real", True):
            continue
        d = r.get("dropout_exposure") or {}
        c["views"] += 1
        c["triggered"] += int(d.get("triggered", False))
        c["applied"] += int(d.get("applied", False))
        c["instances_before"] += d.get("instances_before", 0)
        c["instances_after"] += d.get("instances_after", 0)
        c["removed"] += len(d.get("removed", []))
        sem.update(d.get("removed_semantic", []))
        prim.update(d.get("removed_primitive", []))
        if d.get("segments_before", 0):
            ratios.append(d["segments_after"] / d["segments_before"])
    return dict(
        c,
        trigger_rate=c["triggered"] / max(c["views"], 1),
        applied_rate=c["applied"] / max(c["views"], 1),
        line_retention_mean=float(np.mean(ratios)) if ratios else None,
        removed_by_semantic=dict(sem),
        removed_by_primitive=dict(prim),
    )
