from pathlib import Path
import torch
from safetensors.torch import save_file, load_file
from util.io import read, save, sha


def model_spec(cfg):
    return {
        k: v
        for k, v in cfg["model"].items()
        if k not in ("primitive_dir", "primitive_mapping_path")
    }


def export(checkpoint, output, metadata):
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    # Only invoke on explicitly trusted local training assets.
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    tensors = {
        prefix + "." + k: v.detach().cpu().contiguous().clone()
        for prefix in ("model", "selector")
        for k, v in ck[prefix].items()
    }
    out.mkdir(parents=True)
    dest = out / "model.safetensors"
    save_file(tensors, str(dest), metadata={"format": "industrycad-inference-v1"})
    restored = load_file(str(dest))
    assert set(restored) == set(tensors)
    for k, v in tensors.items():
        if (
            restored[k].dtype != v.dtype
            or restored[k].shape != v.shape
            or not torch.equal(restored[k], v)
        ):
            raise ValueError("Export tensor mismatch: " + k)
    m = dict(
        metadata,
        format="industrycad-inference-v1",
        source_checkpoint_sha256=sha(checkpoint),
        weights_sha256=sha(dest),
        tensor_count=len(tensors),
        step=ck.get("step"),
        epoch=ck.get("epoch"),
        micro_offset=ck.get("micro_offset"),
        inference_only=True,
        contains_optimizer=False,
        contains_rng=False,
    )
    save(out / "model.json", m)
    save(
        out / "EXPORT_VERIFIED.json",
        dict(
            passed=True,
            tensors=len(tensors),
            all_values_dtypes_shapes_exact=True,
            source_checkpoint_sha256=m["source_checkpoint_sha256"],
            weights_sha256=m["weights_sha256"],
        ),
    )
    return m


def load(raw, path, cfg=None):
    path = Path(path)
    meta = read(path.with_suffix(".json"))
    if (
        meta["format"] != "industrycad-inference-v1"
        or sha(path) != meta["weights_sha256"]
    ):
        raise ValueError("Invalid inference package or weights hash")
    if cfg is not None:
        if model_spec(cfg) != meta["model_spec"]:
            raise ValueError(
                "Model config differs from exported architecture/decoding config"
            )
        for name, expected in meta["template_hashes"].items():
            if sha(Path(cfg["paths"]["template_root"]) / name) != expected:
                raise ValueError("Template content differs: " + name)
        if sha(cfg["paths"]["primitive_mapping"]) != meta["model_mapping_sha256"]:
            raise ValueError("Model mapping differs")
    state = load_file(str(path))
    if len(state) != meta["tensor_count"] or any(
        not k.startswith(("model.", "selector.")) for k in state
    ):
        raise ValueError("Unexpected tensors")
    for prefix, module in [("model", raw.model), ("selector", raw.selector)]:
        module.load_state_dict(
            {
                k[len(prefix) + 1 :]: v
                for k, v in state.items()
                if k.startswith(prefix + ".")
            },
            strict=True,
        )
    return meta
