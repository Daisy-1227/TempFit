import copy
import math
from collections import Counter
import numpy as np
from datasets.instance_scale import IndependentInstanceScale, owner, box, ious
from datasets.full_source import SceneDataset as BaseSceneDataset, batch


class FullOnlyInstanceScale(IndependentInstanceScale):
    def __init__(self, *args, translation_bounds=(-2.0, 2.0), **kwargs):
        super().__init__(*args, **kwargs)
        self.translation_bounds = translation_bounds
        assert 0 <= self.p <= 1 and 0 <= self.sample_ratio <= 1
        assert 0 < self.scale_range[0] <= self.scale_range[1]
        assert 0 < self.scale_bounds[0] <= self.scale_bounds[1]
        assert translation_bounds[0] < translation_bounds[1]

    def __call__(self, d):
        rng = np.random.default_rng(self.seed + self.calls * 10000019)
        self.calls += 1
        if self.capture:
            self.before = copy.deepcopy(d)
        svg = d["svg_data"]
        co = np.asarray(svg.coords).reshape(-1, 2, 2)
        own = owner(svg)
        lo, hi = self.translation_bounds
        tr_all = np.asarray(d["translation"])
        valid = np.isfinite(tr_all).all(1) & ((tr_all >= lo) & (tr_all <= hi)).all(1)
        boxes, eligible, ineligible = {}, [], Counter()
        for i in range(len(d["scale"])):
            points = co[own == i]
            if not len(points):
                reason = "missing_source"
            elif not np.isfinite(points).all():
                reason = "nonfinite_geometry"
            elif not valid[i]:
                reason = "translation_out_of_range"
            elif (
                not np.isfinite(d["scale"][i]).all()
                or (np.asarray(d["scale"][i]) <= 0).any()
            ):
                reason = "invalid_scale"
            else:
                reason = None
            if len(points) and np.isfinite(points).all():
                boxes[i] = box(points)
            if reason:
                ineligible[reason] += 1
            else:
                eligible.append(i)
        row = dict(
            enabled=self.enabled,
            triggered=False,
            instance_exposures=len(tr_all),
            retained_exposures=int(valid.sum()),
            eligible=len(eligible),
            ineligible=dict(ineligible),
            requested=0,
            accepted=[],
            rejected=[],
        )
        if self.enabled and rng.random() < self.p:
            row["triggered"] = True
            n = min(len(eligible), math.ceil(len(eligible) * self.sample_ratio))
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
                sc = d["scale"][iid] * k
                size = np.asarray(d["size"][iid]) * k if "size" in d else None
                reason = None
                if not np.isfinite(proposed).all():
                    reason = "nonfinite_geometry"
                elif (
                    not np.isfinite(sc).all()
                    or sc.min() < self.scale_bounds[0]
                    or sc.max() > self.scale_bounds[1]
                ):
                    reason = "scale_head_range"
                elif size is not None and not np.isfinite(size).all():
                    reason = "nonfinite_size"
                else:
                    b = box(proposed)
                    others = np.array(
                        [v for key, v in boxes.items() if key != iid]
                    ).reshape(-1, 4)
                    if np.any(
                        ious(b, others)
                        > np.maximum(ious(boxes[iid], others), self.max_bbox_iou) + 1e-7
                    ):
                        reason = "new_foreground_overlap"
                info = dict(
                    instance=iid,
                    factor=k,
                    direction="grow" if k > 1 else "shrink",
                    semantic=int(d["semantic_id"][iid]),
                    primitive=int(d["primitive_id"][iid]),
                )
                if reason:
                    row["rejected"].append(dict(**info, reason=reason))
                    continue
                co[mask] = proposed
                d["scale"][iid] = sc
                if size is not None:
                    d["size"][iid] = size
                boxes[iid] = b
                row["accepted"].append(
                    dict(
                        **info,
                        lines=int(mask.sum()),
                        translation=tr.tolist(),
                        new_scale=sc.tolist(),
                    )
                )
            svg.coords = co.reshape(np.asarray(svg.coords).shape)
        self.events.append(row)
        if self.capture:
            self.after = copy.deepcopy(d)
        return d


class SceneDataset(BaseSceneDataset):
    def __init__(self, cfg, split="train", capture=False, deterministic=False):
        # Base constructor accepts only the sealed transform's original arguments.
        base_cfg = copy.deepcopy(cfg)
        options = base_cfg["instance_scale"].pop("translation_bounds", (-2.0, 2.0))
        super().__init__(base_cfg, split, capture, deterministic)
        self.cfg = cfg
        if self.scale is not None:
            previous = self.scale
            self.scale = FullOnlyInstanceScale(
                **base_cfg["instance_scale"],
                translation_bounds=options,
                capture=capture,
            )
            chain = self.ds.transform.transforms
            chain[chain.index(previous)] = self.scale
        assert list(options) == list(cfg["model"]["continuous_range_translation"])
        assert list(cfg["instance_scale"]["scale_bounds"]) == list(
            cfg["model"]["continuous_range_scale"]
        )

    def __getitem__(self, key):
        result = super().__getitem__(key)
        fallback = bool(
            result["sample"]["_recon_audit"].get(
                "full_only_deterministic_fallback", False
            )
        )
        # Retries are attempts, not additional training exposures. Only the delivered
        # final attempt contributes to effective augmentation rates.
        result["scale_exposure"] = (
            None
            if fallback or not result["scale_events"]
            else copy.deepcopy(result["scale_events"][-1])
        )
        result["scale_fallback"] = fallback
        return result


class RealSmoke(SceneDataset):
    """128 real scenes plus a one-scene accumulation tail; no stale cached inputs."""

    def __len__(self):
        return 129


def summarize_exposure(rows):
    totals = Counter()
    reasons = Counter()
    ineligible = Counter()
    strata = {}
    for row in rows:
        if not row.get("real", True):
            continue
        totals["scenes"] += 1
        totals["retry_attempts"] += len(row.get("retries", []))
        totals["fallback_scenes"] += int(row.get("scale_fallback", False))
        event = row.get("scale_exposure")
        if event is None:
            if row.get("scale_fallback", False):
                totals["instance_exposures"] += row["instances"]
                totals["retained_exposures"] += row["instances"]
            continue
        totals["observed_scenes"] += 1
        ineligible.update(event["ineligible"])
        for key in (
            "instance_exposures",
            "retained_exposures",
            "eligible",
            "requested",
        ):
            totals[key] += event[key]
        totals["triggered_scenes"] += int(event["triggered"])
        totals["accepted"] += len(event["accepted"])
        totals["rejected"] += len(event["rejected"])
        for status in ("accepted", "rejected"):
            for item in event[status]:
                if status == "rejected":
                    reasons[item["reason"]] += 1
                for label in (
                    item["direction"],
                    f"semantic_{item['semantic']}",
                    f"primitive_{item['primitive']}",
                ):
                    c = strata.setdefault(label, Counter())
                    c["requested"] += 1
                    c[status] += 1
    result = {
        key: totals[key]
        for key in (
            "scenes",
            "observed_scenes",
            "retry_attempts",
            "fallback_scenes",
            "instance_exposures",
            "retained_exposures",
            "eligible",
            "requested",
            "triggered_scenes",
            "accepted",
            "rejected",
        )
    }
    for key, numerator, denominator in (
        ("eligible_per_exposure", "eligible", "instance_exposures"),
        ("requested_per_eligible", "requested", "eligible"),
        ("accepted_per_requested", "accepted", "requested"),
        ("accepted_per_exposure", "accepted", "instance_exposures"),
        ("accepted_per_retained", "accepted", "retained_exposures"),
    ):
        result[key] = (
            totals[numerator] / totals[denominator] if totals[denominator] else None
        )
    result["rejection_reasons"] = dict(reasons)
    result["ineligibility_reasons"] = dict(ineligible)
    result["strata"] = {k: dict(v) for k, v in strata.items()}
    return result
