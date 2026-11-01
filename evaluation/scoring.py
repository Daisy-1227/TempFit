from pathlib import Path

import numpy as np

from evaluation.metrics import ARScorer, ar_summary
from util.io import read, save, sha


def fail(message):
    raise ValueError(message)


def evaluate_predictions(cfg, predictions_dir, output_dir, allow_partial=False):
    source = Path(predictions_dir).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        fail(f"Output already exists: {output}")
    manifest = read(cfg["paths"]["manifest"])
    evaluated = read(source / "EVALUATED_MANIFEST.json")
    if evaluated["source_manifest_sha256"] != sha(cfg["paths"]["manifest"]):
        fail("Configuration manifest differs from the inference manifest")
    predictions = read(source / "predictions.json")
    names = [p["filename"] for p in predictions]
    expected = manifest["val"]
    if not names or len(names) != len(set(names)) or not set(names) <= set(expected):
        fail(
            "Predictions must be a nonempty, unique subset of the configured validation manifest"
        )
    if names != evaluated["filenames"]:
        fail("Predictions disagree with EVALUATED_MANIFEST.json")
    complete = set(names) == set(expected)
    if not complete and not allow_partial:
        fail(
            f"Incomplete population: {len(names)}/{len(expected)}; use --allow-partial for an explicit subset"
        )
    original = read(source / "SUMMARY.json")
    if original["data_version"] != cfg["release"]["data_version"]:
        fail("Configuration data version differs from inference")
    if any(Path(name).name != name for name in names):
        fail("Manifest entries must be plain filenames")
    files = [source / "results_val" / (Path(name).stem + ".npz") for name in names]
    if len(set(files)) != len(files) or any(not p.is_file() for p in files):
        fail("Missing or ambiguous Stage-A files")

    scorer = ARScorer(cfg["paths"]["template_root"])
    instances = []
    hashes = {}
    gt_keys = {
        "semantic_id": "gt_semantic_i",
        "primitive_id": "gt_primitive_i",
        "translation": "gt_cont_translation_i",
        "scale": "gt_cont_scale_i",
        "rotation": "gt_cont_rotation_i",
        "flip_mode": "gt_flip_i",
    }
    for i, (record, path) in enumerate(zip(predictions, files), 1):
        with np.load(path, allow_pickle=False) as data:
            if str(data["filename"].item()) != path.stem:
                raise ValueError(f"Scene filename mismatch: {path}")
            if (
                float(data["scene_bound"]) != 30000
                or bool(data["size_pred"])
                or str(data["scale_parameterization"].item()) != "relative"
            ):
                raise ValueError(f"Unsupported geometry convention: {path}")
            gt = {key: data[field].tolist() for key, field in gt_keys.items()}
        rows = scorer.score(record["pred"], gt)
        instances.extend(dict(filename=record["filename"], **row) for row in rows)
        hashes[record["filename"]] = sha(path)
        if i % 50 == 0 or i == len(files):
            print(f"AR evaluation {i}/{len(files)}", flush=True)

    summary = dict(
        scenes=len(names),
        population=cfg["release"].get("evaluation_population", "explicit manifest"),
        complete_population=complete,
        with_ar=True,
        data_version=cfg["release"]["data_version"],
        protocol="AR center gate 0.1; 256 support points; normalized symmetric CD < 0.03; identity requires semantic AND primitive",
        beta=original.get("beta"),
        ar=ar_summary(instances, predictions),
        evaluation_mode="CPU rescore of saved predictions against exported GT; no inference or PQ",
    )
    output.mkdir(parents=True, exist_ok=False)
    save(output / "instances.json", instances)
    save(output / "SUMMARY.json", summary)
    save(output / "EVALUATED_MANIFEST.json", evaluated)
    save(
        output / "PROVENANCE.json",
        dict(
            predictions_sha256=sha(source / "predictions.json"),
            source_manifest_sha256=sha(cfg["paths"]["manifest"]),
            scorer_sha256=sha(Path(__file__).parent / "metrics.py"),
            geometry_sha256=sha(Path(__file__).parent / "geometry.py"),
            template_sha256={
                p.name: sha(p)
                for p in sorted(Path(cfg["paths"]["template_root"]).glob("*.json"))
            },
            stage_a_sha256=hashes,
        ),
    )
    return summary
