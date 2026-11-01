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
    # Historical annotations contain unknown/background identities. Preserve and
    # count them; the native loader's validity filtering remains authoritative.
    return dict(
        raw_instances=len(gt),
        segments=n,
        primitive_groups=p,
        nonpositive_semantic_or_negative_primitive=sum(
            g["semantic_id"] < 1 or g["primitive_id"] < 0 for g in gt
        ),
    )


def build_cache(root, manifest, output):
    """New cache root, never overwrite dataset files. Raw schema used by the loader.

    This is deliberately not the historical analysis-cache gt_arrays schema.
    Configure dataset.cache_dir via paths.data_root only after placing the generated
    cache_pkl directories alongside the corresponding verified JSON files.
    """
    root = Path(root)
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    names = read(manifest)
    out.mkdir(parents=True)
    rows = []
    for key, split in [("train", "train"), ("val", "test")]:
        for name in names[key]:
            name = scene_name(name)
            ip = root / split / "svg" / name
            gp = root / split / "gt_update_size" / name
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
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    print(build_cache(a.root, a.manifest, a.output))
