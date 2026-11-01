from pathlib import Path
import json, shutil
from util.io import read, save, sha


def materialize(plan_path, source_root, output):
    plan = read(plan_path)
    root = Path(source_root).resolve()
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError(out)
    validated = []
    seen = set()
    for row in plan["files"]:
        src = (root / row["source"]).resolve()
        dst = (out / row["path"]).resolve()
        if not src.is_relative_to(root) or not dst.is_relative_to(out):
            raise ValueError("Asset path escapes its root")
        if dst == out or dst == out / "DATA_MANIFEST.json":
            raise ValueError("Reserved destination")
        if dst in seen:
            raise ValueError("Duplicate destination")
        if any(dst in prior.parents or prior in dst.parents for prior in seen):
            raise ValueError("Conflicting file destinations")
        seen.add(dst)
        if sha(src) != row["sha256"]:
            raise ValueError("Source hash mismatch: " + row["source"])
        validated.append((src, dst, row))
    out.mkdir(parents=True)
    for src, dst, row in validated:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        if sha(dst) != row["sha256"]:
            raise ValueError("Copy hash mismatch")
    save(
        out / "DATA_MANIFEST.json",
        dict(
            version=plan["version"],
            files=[
                dict(path=r["path"], sha256=r["sha256"], bytes=d.stat().st_size)
                for _, d, r in validated
            ],
        ),
    )
    return dict(files=len(validated), data_version=plan["version"])
