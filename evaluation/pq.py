from pathlib import Path
from util.io import read, save, sha
from util.guard import ResourceGuard


def run(stage_a, templates, manifest, output):
    from evaluation.projection import proj_from_ins_to_segment_v2 as projector
    from evaluation.pq_protocol import _new_evaluator, _load_and_update, _metrics

    src = Path(stage_a)
    dest = Path(output)
    names = read(manifest)
    if isinstance(names, dict):
        names = names.get("filenames", names.get("test"))
    if not names or len(set(names)) != len(names):
        raise ValueError("Explicit unique scene manifest required")
    files = [src / (Path(n).stem + ".npz") for n in names]
    if any(not p.is_file() for p in files):
        raise FileNotFoundError(
            "Prediction files are missing for the requested manifest"
        )
    if dest.exists():
        raise FileExistsError(dest)
    dest.mkdir(parents=True)
    guard = ResourceGuard(dest)
    ev = _new_evaluator(23)
    rows = []
    protocol = dict(
        projector="proj_from_ins_to_segment_v2.py",
        sha256=sha(projector.__file__),
        params=projector.DEFAULT_COST_PARAMS,
        geometry_dtype="float64",
        bounded_pair_temporaries=200000,
        gt_attributes_used_for_projection=False,
        manifest_sha256=sha(manifest),
        population="test",
        semantic_encoding="Stage-A scene_semantic_offset; legacy files default to 1",
    )
    save(dest / "PROTOCOL.json", protocol)
    try:
        for n, p in zip(names, files):
            if guard.boundary():
                raise RuntimeError("Projection stopped at resource/user boundary")
            target = dest / "line_level_val"
            target.mkdir(exist_ok=True)
            result = projector.process_one_npz(
                str(p), str(templates), str(target), dict(projector.DEFAULT_COST_PARAMS)
            )
            if result != p.stem:
                raise RuntimeError("Projection failed: " + p.name)
            single = _new_evaluator(23)
            npred, ngt = _load_and_update(
                str(target / p.name), single, 23, str(src), True
            )
            for k in ("tp_classes", "tp_classes_values", "fp_classes", "fn_classes"):
                getattr(ev, k)[:] += getattr(single, k)
            rows.append(
                dict(
                    filename=n,
                    n_pred=npred,
                    n_gt=ngt,
                    prediction_sha256=sha(p),
                    projection_sha256=sha(target / p.name),
                    **_metrics(single)
                )
            )
            save(dest / "per_scene.json", rows)
        summary = dict(
            scenes=len(rows), n_gt=sum(r["n_gt"] for r in rows), **_metrics(ev)
        )
        save(dest / "SUMMARY.json", summary)
        return summary
    finally:
        guard.close()
