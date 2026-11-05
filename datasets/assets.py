from pathlib import Path
import json, pickle
import numpy as np
from util.io import read, save, sha


def scene_name(name):
    if (
        not isinstance(name, str)
        or Path(name).name != name
        or "\\" in name
        or not name.endswith(".json")
    ):
        raise ValueError("Expected a plain .json scene filename")
    return name


def template_id(annotation):
    if "template_id" in annotation:
        return annotation["template_id"]
    return annotation["primitive_id"]  # Historical GT and raw caches.


def validate_scene(raw, gt):
    a = {k: np.asarray(v) for k, v in raw.items()}
    n = len(a["coords"])
    p = len(a["instance_ids"])
    if not n or not gt or a["coords"].shape != (n, 4):
        raise ValueError("Empty or invalid scene")
    for k in ("colors", "widths", "primitive_ids", "layer_ids"):
        if len(a[k]) != n:
            raise ValueError("Segment array length: " + k)
    for k in ("semantic_ids", "primitive_lengths"):
        if len(a[k]) != p:
            raise ValueError("Primitive array length: " + k)
    ids = a["primitive_ids"]
    if not np.equal(ids, ids.astype(np.int64)).all() or ids.min() < 0 or ids.max() >= p:
        raise ValueError("Invalid primitive provenance index")
    for k, v in a.items():
        if not np.isfinite(v).all():
            raise ValueError("Nonfinite data: " + k)
    for g in gt:
        for k in ("scale", "rotate", "translation", "size", "crop_window"):
            if not np.isfinite(g[k]).all():
                raise ValueError("Nonfinite GT: " + k)
    owners = a["instance_ids"].astype(np.int64)
    if owners.min() < 1 or owners.max() > len(gt):
        raise ValueError("Input instance reference outside GT")
    expected_semantic = np.asarray([g["semantic_id"] for g in gt])[owners - 1]
    if not np.array_equal(a["semantic_ids"], expected_semantic):
        raise ValueError("Input semantic IDs must match zero-based GT classes")
    return dict(
        raw_instances=len(gt),
        segments=n,
        primitive_groups=p,
        negative_semantic_or_template=sum(
            g["semantic_id"] < 0 or template_id(g) < 0 for g in gt
        ),
    )


def build_cache(root, train_manifest, test_manifest, output):
    """Write loader caches to a new directory.

    Place the generated cache_pkl directories alongside the matching JSON files
    under paths.data_root.
    """
    root = Path(root)
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    names = {"train": read(train_manifest), "test": read(test_manifest)}
    out.mkdir(parents=True)
    rows = []
    for key, split in [("train", "train"), ("test", "test")]:
        for name in names[key]:
            name = scene_name(name)
            ip = root / split / "svg" / name
            gp = root / split / "gt" / name
            raw = read(ip)
            gt = read(gp)
            stats = validate_scene(raw, gt)
            dst = out / split / "cache_pkl" / Path(name).with_suffix(".pkl")
            dst.parent.mkdir(parents=True, exist_ok=True)
            with dst.open("wb") as f:
                pickle.dump(
                    dict(file_name=Path(name).stem, svg_raw=raw, gt_raw=gt),
                    f,
                    protocol=4,
                )
            with dst.open("rb") as f:
                c = pickle.load(f)
            if c["svg_raw"] != raw or c["gt_raw"] != gt:
                raise ValueError("Cache roundtrip differs")
            rows.append(
                dict(
                    path=str(dst.relative_to(out)),
                    sha256=sha(dst),
                    input_sha256=sha(ip),
                    gt_sha256=sha(gp),
                    **stats
                )
            )
    save(
        out / "CACHE_MANIFEST.json",
        dict(schema="industrycad-loader-raw-v1", files=rows),
    )
    return dict(scenes=len(rows))


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--test-manifest", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    print(build_cache(a.root, a.train_manifest, a.test_manifest, a.output))
