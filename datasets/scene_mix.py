import copy
from collections import Counter
import numpy as np
import torch
from datasets.metadata import metadata
from datasets.primitive_dataset import discretize

GT_KEYS = {
    "semantic_id",
    "primitive_id",
    "scale",
    "rotation",
    "flip_mode",
    "translation",
    "flip_invariant_primitive",
    "size",
}
SPECIAL = {
    "filename",
    "svg_data",
    "vec_data",
    "sort_indices",
    "original_indices",
    "grid_coord",
    "io_timing",
}


def shifted(ids, offset):
    result = ids.clone()
    result[result >= 0] += offset
    return result


def merge_pair(left, right, cfg):
    a, b = left["sample"], right["sample"]
    na, nb = len(a["primitive_id"]), len(b["primitive_id"])
    pa = len(a["svg_data"].instance_ids)
    assert a.keys() == b.keys(), (a.keys(), b.keys())
    tr = torch.cat([a["translation"], b["translation"]])
    bins = discretize(
        tr,
        num_discrete=cfg["model"]["num_discrete_translation"],
        continuous_range=cfg["model"]["continuous_range_translation"],
    )
    order = torch.from_numpy(np.lexsort((bins[:, 1].numpy(), bins[:, 0].numpy())))
    # SVG instance IDs retain the raw order; GT arrays already use decoder order.
    raw_order = torch.cat([a["sort_indices"], b["sort_indices"] + na])[order]
    sample = {}
    for key in a:
        if key.startswith("_") or key in SPECIAL:
            continue
        assert key in GT_KEYS or key.startswith("recon_"), (
            "Unclassified sample field",
            key,
        )
        assert torch.is_tensor(a[key]) and len(a[key]) == na and len(b[key]) == nb, key
        sample[key] = torch.cat([a[key], b[key]], dim=0)[order]
    sample["sort_indices"] = raw_order
    sample["original_indices"] = torch.argsort(raw_order)
    sample["filename"] = "scene_mix::" + str(a["filename"]) + "::" + str(b["filename"])
    svg = copy.copy(a["svg_data"])
    for key in (
        "coords",
        "colors",
        "widths",
        "layer_ids",
        "semantic_ids",
        "primitive_lengths",
    ):
        setattr(
            svg,
            key,
            torch.cat([getattr(a["svg_data"], key), getattr(b["svg_data"], key)]),
        )
    svg.primitive_ids = torch.cat(
        [a["svg_data"].primitive_ids, b["svg_data"].primitive_ids + pa]
    )
    svg.instance_ids = torch.cat(
        [a["svg_data"].instance_ids, shifted(b["svg_data"].instance_ids, na)]
    )
    points = svg.coords.reshape(-1, 2)
    svg.viewBox = torch.cat([points.amin(0), points.amax(0) - points.amin(0)])
    vec = copy.copy(a["vec_data"])
    for key in ("coords", "feats", "layer_ids", "sem_ids", "prim_lengths"):
        setattr(
            vec,
            key,
            torch.cat([getattr(a["vec_data"], key), getattr(b["vec_data"], key)]),
        )
    vec.prim_ids = torch.cat([a["vec_data"].prim_ids, b["vec_data"].prim_ids + pa])
    vec.inst_ids = torch.cat(
        [a["vec_data"].inst_ids, shifted(b["vec_data"].inst_ids, na)]
    )
    # Namespaced CAD layer IDs prevent accidental equality between unrelated files.
    layer_offset = int(a["svg_data"].layer_ids.max()) + 1
    svg.layer_ids[len(a["svg_data"].coords) :] = shifted(
        b["svg_data"].layer_ids, layer_offset
    )
    vec.layer_ids[len(a["vec_data"].coords) :] = shifted(
        b["vec_data"].layer_ids, layer_offset
    )
    vec.data_path = sample["filename"]
    sample.update(svg_data=svg, vec_data=vec)
    if "grid_coord" in a:
        grid = torch.floor(vec.coords / cfg["dataset_train"]["grid_size"]).to(
            a["grid_coord"].dtype
        )
        sample["grid_coord"] = grid - grid.amin(0)
    sample["io_timing"] = {
        key: a["io_timing"][key] + b["io_timing"][key] for key in a["io_timing"]
    }
    sample["_scene_mix"] = dict(
        source_indices=[left["index"], right["index"]],
        source_names=[a["filename"], b["filename"]],
        decoder_source_order=order.tolist(),
        source_instances=[na, nb],
    )
    result = dict(
        sample=sample, metadata=metadata(sample), index=left["index"], real=True
    )
    assert len(sample["primitive_id"]) == na + nb
    assert len(vec.coords) == len(a["vec_data"].coords) + len(b["vec_data"].coords)
    return result


def mix_group(group, cfg, epoch, micro_offset):
    spec = cfg["scene_mix"]
    if not spec["enabled"] or spec["pair_probability"] == 0:
        return group, []
    rng = np.random.default_rng(
        np.random.SeedSequence([spec["seed"], epoch, micro_offset])
    )
    candidates = [i for i, d in enumerate(group) if d["real"]]
    pairs = rng.permutation(candidates).tolist()
    replacements, consumed, events = {}, set(), []
    for pos in range(0, len(pairs) - 1, 2):
        i, j = sorted(pairs[pos : pos + 2])
        a, b = group[i], group[j]
        lines = len(a["sample"]["vec_data"].coords) + len(
            b["sample"]["vec_data"].coords
        )
        instances = len(a["sample"]["primitive_id"]) + len(b["sample"]["primitive_id"])
        triggered = bool(rng.random() < spec["pair_probability"])
        reason = None
        if not triggered:
            reason = "probability"
        elif a["sample"]["filename"] == b["sample"]["filename"]:
            reason = "same_source"
        elif lines > spec["max_postgrid_segments"]:
            reason = "line_limit"
        elif instances > spec["max_instances"]:
            reason = "instance_limit"
        row = dict(
            indices=[a["index"], b["index"]],
            source_names=[a["sample"]["filename"], b["sample"]["filename"]],
            triggered=triggered,
            applied=reason is None,
            reason=reason,
            merged_lines=lines,
            merged_instances=instances,
        )
        if reason is None:
            replacements[i] = merge_pair(a, b, cfg)
            consumed.add(j)
        events.append(row)
    result = [replacements.get(i, d) for i, d in enumerate(group) if i not in consumed]
    assert sum(len(d["sample"]["primitive_id"]) for d in result) == sum(
        len(d["sample"]["primitive_id"]) for d in group
    )
    assert sum(len(d["sample"]["vec_data"].coords) for d in result) == sum(
        len(d["sample"]["vec_data"].coords) for d in group
    )
    return result, events


def summarize_mix(events, source_scenes):
    applied = sum(e["applied"] for e in events)
    return dict(
        source_scenes=source_scenes,
        model_scenes=source_scenes - applied,
        pairs=len(events),
        triggered=sum(e["triggered"] for e in events),
        applied=applied,
        mixed_source_fraction=2 * applied / max(source_scenes, 1),
        rejection_reasons=dict(
            Counter(e["reason"] for e in events if e["triggered"] and not e["applied"])
        ),
        max_mixed_lines=max(
            (e["merged_lines"] for e in events if e["applied"]), default=0
        ),
        max_mixed_instances=max(
            (e["merged_instances"] for e in events if e["applied"]), default=0
        ),
    )
