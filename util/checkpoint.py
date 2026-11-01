from pathlib import Path
import hashlib, json, random, os
import numpy as np
import torch
from util.config import semantic_digest
from util.io import sha, save


def rng_state():
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        cpu=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state(),
    )


def restore_rng(r):
    random.setstate(r["python"])
    np.random.set_state(r["numpy"])
    torch.set_rng_state(r["cpu"])
    torch.cuda.set_rng_state(r["cuda"])


def object_digest(value):
    h = hashlib.sha256()

    def add(v):
        if torch.is_tensor(v):
            a = v.detach().cpu().contiguous()
            h.update(str((a.dtype, tuple(a.shape))).encode())
            if a.numel():
                h.update(
                    a.reshape(-1)
                    .clone(memory_format=torch.contiguous_format)
                    .view(torch.uint8)
                    .numpy()
                    .tobytes()
                )
        elif isinstance(v, np.ndarray):
            h.update(str((v.dtype, v.shape)).encode())
            h.update(v.tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=str):
                h.update(str(k).encode())
                add(v[k])
        elif isinstance(v, (tuple, list)):
            for x in v:
                add(x)
        else:
            h.update(repr(v).encode())

    add(value)
    return h.hexdigest()


def binding(cfg):
    sources = source_fingerprint()
    assets = {}
    for k in ("manifest", "primitive_mapping", "invalid_train"):
        assets[k] = sha(cfg["paths"][k])
    for k in ("template_root", "donor_bank"):
        if k == "donor_bank" and not cfg["paths"].get(k):
            continue
        path = Path(cfg["paths"][k])
        assets[k] = {
            str(p.relative_to(path)): sha(p)
            for p in sorted(path.rglob("*"))
            if p.is_file()
        }
    # Require the prepared data manifest: content, not the data-root location, identifies input versions.
    path = Path(cfg["paths"]["data_root"]) / "DATA_MANIFEST.json"
    if not path.is_file():
        raise ValueError(
            "Prepared data requires DATA_MANIFEST.json; run prepare-data first"
        )
    record = json.loads(path.read_text())
    if record["version"] != cfg["release"]["data_version"]:
        raise ValueError("Configured data version differs from prepared manifest")
    for row in record["files"]:
        p = Path(cfg["paths"]["data_root"]) / row["path"]
        if sha(p) != row["sha256"]:
            raise ValueError("Prepared data content changed: " + row["path"])
    assets["data_manifest"] = sha(path)
    return dict(
        layout="research-v1", config=semantic_digest(cfg), source=sources, assets=assets
    )


def atomic_pt(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def write(
    path, raw, optimizer, scheduler, step, epoch, offset, totals, identity, best=None
):
    payload = dict(
        format="industrycad-training-v2",
        model=raw.model.state_dict(),
        selector=raw.selector.state_dict(),
        optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(),
        step=step,
        epoch=epoch,
        micro_offset=offset,
        epoch_totals=list(totals),
        rng=rng_state(),
        binding=identity,
        best_val=best,
    )
    payload["state_digest"] = object_digest(
        {"model": payload["model"], "selector": payload["selector"]}
    )
    atomic_pt(path, payload)
    save(
        Path(path).with_suffix(".json"),
        dict(
            step=step,
            epoch=epoch,
            micro_offset=offset,
            sha256=sha(path),
            state_digest=payload["state_digest"],
        ),
    )


def load_model(raw, path, cfg=None):
    if Path(path).suffix == ".safetensors":
        from util.weights import load

        return load(raw, path, cfg)
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    raw.model.load_state_dict(payload["model"], strict=True)
    raw.selector.load_state_dict(payload["selector"], strict=True)
    return payload


def source_fingerprint():
    root = Path(__file__).resolve().parent.parent
    files = [
        root / name for name in ("train.py", "inference.py", "evaluate.py")
    ]
    for folder in ("models", "datasets", "evaluation", "util"):
        files.extend((root / folder).rglob("*.py"))
    return {str(path.relative_to(root)): sha(path) for path in sorted(files)}


def validate_resume_package(package, identity):
    if package.get("format") != "industrycad-training-v2":
        raise ValueError(
            "Resume requires a checkpoint from this source layout. Restore older training states with their archived runtime; inference weights remain supported."
        )
    if package["binding"] != identity:
        raise ValueError("Resume requires matching source, configuration and assets")


def restore_training(model, optimizer, scheduler, path, identity):
    from util.engine import TrainingState

    package = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    validate_resume_package(package, identity)
    model.model.load_state_dict(package["model"], strict=True)
    model.selector.load_state_dict(package["selector"], strict=True)
    optimizer.load_state_dict(package["optimizer"])
    scheduler.load_state_dict(package["scheduler"])
    if object_digest(optimizer.state_dict()) != object_digest(package["optimizer"]):
        raise ValueError("Optimizer restore mismatch")
    if (
        object_digest(
            dict(model=model.model.state_dict(), selector=model.selector.state_dict())
        )
        != package["state_digest"]
    ):
        raise ValueError("Model restore mismatch")
    if scheduler.state_dict() != package["scheduler"]:
        raise ValueError("Scheduler restore mismatch")
    state = TrainingState(
        epoch=package["epoch"],
        offset=package["micro_offset"],
        step=package["step"],
        totals=package["epoch_totals"],
        best=package["best_val"],
    )
    return state, package["rng"]


def save_inference(path, model, epoch, step, identity):
    atomic_pt(
        path,
        dict(
            model=model.model.state_dict(),
            selector=model.selector.state_dict(),
            epoch=epoch,
            step=step,
            inference_only=True,
            binding=identity,
        ),
    )
