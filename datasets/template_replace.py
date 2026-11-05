from pathlib import Path
import math, json, shutil
import numpy as np
from util.io import read, save, sha
from datasets.assets import scene_name, template_id, validate_scene


def transform_lines(lines, flip, scale, rotation_deg, translation):
    arr = np.asarray(lines, dtype=np.float64).copy()
    pts = arr.reshape(-1, 2)
    if int(flip) == 1:
        pts[:, 0] = -pts[:, 0]
    pts = (
        pts * float(scale)
        if np.isscalar(scale)
        else pts * np.asarray(scale, dtype=np.float64).reshape(-1)
    )
    th = math.radians(float(rotation_deg))
    R = np.array(
        [[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]], dtype=np.float64
    )
    return (pts @ R.T + np.asarray(translation, dtype=np.float64).reshape(-1)).reshape(
        -1, 2, 2
    )


def replace_scene(raw, gt, templates, scene_bound=30000.0):
    validate_scene(raw, gt)
    old = np.asarray(raw["coords"], np.float64).reshape(-1, 2, 2)
    ids = np.asarray(raw["instance_ids"], int)[np.asarray(raw["primitive_ids"], int)]
    if ids.min() < 0 or ids.max() > len(gt):
        raise ValueError(
            "Template replacement requires documented 1-based instances, background=0"
        )
    fields = {
        k: []
        for k in (
            "coords",
            "colors",
            "widths",
            "primitive_ids",
            "layer_ids",
            "semantic_ids",
            "instance_ids",
            "primitive_lengths",
        )
    }
    offset = 0
    errors = []

    def append(lines, src, mask, sem, inst):
        nonlocal offset
        pp = np.asarray(src["primitive_ids"], int)[mask]
        unique, inv = np.unique(pp, return_inverse=True)
        n = len(unique)
        fields["coords"].append(lines.reshape(-1, 4))
        for k in ("colors", "widths", "layer_ids"):
            fields[k].append(np.asarray(src[k])[mask])
        fields["primitive_ids"].append(inv + offset)
        fields["semantic_ids"].append(
            np.asarray(sem)[unique] if not np.isscalar(sem) else np.full(n, sem, int)
        )
        fields["instance_ids"].append(
            np.asarray(inst)[unique] if not np.isscalar(inst) else np.full(n, inst, int)
        )
        fields["primitive_lengths"].append(
            np.bincount(
                inv,
                weights=np.linalg.norm(lines[:, 1] - lines[:, 0], axis=1),
                minlength=n,
            )
        )
        offset += n

    bg = ids == 0
    if bg.any():
        append(old[bg], raw, bg, raw["semantic_ids"], raw["instance_ids"])
    for iid, g in enumerate(gt, 1):
        if not (ids == iid).any():
            raise ValueError("GT has no source instance")
        t = templates(int(template_id(g)))
        pos = (
            np.asarray(g["translation"])
            + scene_bound / 2
            + np.asarray(g["crop_window"][:2])
        )
        if not np.allclose(
            pos,
            np.asarray(g["global_translation"]) + g["crop_window"][:2],
            rtol=0,
            atol=1e-7,
        ):
            raise ValueError("Coordinate convention differs")
        lines = transform_lines(t["coords"], g["flip"], g["scale"], g["rotate"], pos)
        err = float(np.abs(np.ptp(lines.reshape(-1, 2), axis=0) - g["size"]).max())
        if err >= 1e-6:
            raise ValueError("Template extent differs from GT size")
        errors.append(err)
        append(lines, t, np.ones(len(lines), bool), int(g["semantic_id"]), iid)
    a = {k: np.concatenate(v) for k, v in fields.items()}
    xy = a["coords"].reshape(-1, 2)
    lo = xy.min(0)
    hi = xy.max(0)
    a["viewBox"] = np.r_[lo, hi - lo]
    own = a["instance_ids"][a["primitive_ids"]]
    if not np.array_equal(a["coords"][own == 0], old[bg].reshape(-1, 4)):
        raise ValueError("Background changed")
    if not np.array_equal(np.unique(own[own > 0]), np.arange(1, len(gt) + 1)):
        raise ValueError("Lost instance")
    result = {k: v.tolist() for k, v in a.items()}
    stats = validate_scene(result, gt)
    stats.update(
        size_max_abs_error=max(errors),
        background_segments=int(bg.sum()),
        downsampling=False,
        clipping=False,
    )
    return result, stats


def generate(root, templates, manifest, output, scene_bound=30000.0):
    root = Path(root)
    templates = Path(templates)
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    names = read(manifest)
    if not names or len(set(names)) != len(names):
        raise ValueError("Unique explicit replacement filename list required")
    names = [scene_name(n) for n in names]
    out.mkdir(parents=True)
    rows = []
    tpl = {}

    def template(pid):
        if pid not in tpl:
            tpl[pid] = read(templates / f"{pid}.json")
        return tpl[pid]

    for i, name in enumerate(names):
        ip = root / "train/svg" / name
        gp = root / "train/gt" / name
        result, stats = replace_scene(read(ip), read(gp), template, scene_bound)
        dst = out / "train/svg" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
        gtout = out / "train/gt" / name
        gtout.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(gp, gtout)
        rows.append(
            dict(
                filename=name,
                input_sha256=sha(ip),
                gt_sha256=sha(gp),
                output_sha256=sha(dst),
                **stats,
            )
        )
        print("Template replacement", i + 1, "/", len(names), flush=True)
    save(
        out / "GENERATION.json",
        dict(
            scope="replacement overlay only; combine with untouched scenes in a NEW dataset root",
            scene_bound=scene_bound,
            template_hashes={str(pid): sha(templates / f"{pid}.json") for pid in tpl},
            scenes=rows,
        ),
    )
    return dict(scenes=len(rows))


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--templates", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--scene-bound", type=float, default=30000.0)
    a = p.parse_args()
    print(generate(a.root, a.templates, a.manifest, a.output, a.scene_bound))
