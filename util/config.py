from pathlib import Path
import copy, hashlib, json, yaml
from util.io import read


def load(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    cfg = copy.deepcopy(cfg)
    paths = cfg["paths"]
    required = [
        "data_root",
        "template_root",
        "primitive_mapping",
        "manifest",
        "invalid_train",
    ]
    if any(
        t["type"] == "SourceInstanceCutMix"
        for t in cfg["dataset_train"].get("instance_transform", []) or []
    ):
        required.append("donor_bank")
    for key in required:
        value = paths.get(key)
        if not value or str(value).startswith("REQUIRED"):
            raise ValueError(f"Configure paths.{key} in {path.name}")
        p = Path(value).expanduser()
        p = p if p.is_absolute() else path.parent / p
        if not p.exists():
            raise FileNotFoundError(f"paths.{key}: {p}")
        paths[key] = str(p.resolve())
    for key in ["dataset_train", "dataset_val"]:
        dc = cfg[key]
        dc.setdefault("filter_translation", False)
        for field in ["cache_dir", "svg_dir", "gt_dir"]:
            dc[field] = paths["data_root"]
        dc["primitive_dir"] = paths["template_root"]
        dc["invalid_scene_file"] = (
            paths["invalid_train"]
            if key == "dataset_train"
            else str(Path(paths["data_root"]) / "invalid_scene.txt")
        )
        for t in dc.get("instance_transform", []) or []:
            if t["type"] == "SourceInstanceCutMix":
                t["donor_bank"] = paths["donor_bank"]
    cfg["model"]["primitive_dir"] = paths["template_root"]
    cfg["model"]["primitive_mapping_path"] = paths["primitive_mapping"]
    cfg["experiment"]["manifest"] = paths["manifest"]
    m = read(paths["manifest"])
    if not all(k in m and len(m[k]) == len(set(m[k])) for k in ["train", "val"]):
        raise ValueError("manifest requires unique train/val filename lists")
    from evaluation.seeds import indices, preprocessing

    indices(m)
    preprocessing(m)
    if cfg["distributed"]["world_size"] != 1:
        raise ValueError("Only single GPU execution is supported")
    if cfg["model"]["shape_condition_model_type"] != "point_transformer_v3" or cfg[
        "model"
    ].get("use_img_conditioner"):
        raise ValueError(
            "This implementation supports the audited PTv3 Industry path only"
        )
    if cfg["model"]["backbone_config"].get("attn_drop", 0) != 0:
        raise ValueError(
            "The installed FlashAttention build requires attn_drop=0; instance dropout is separate"
        )
    if (
        cfg["model"]["continuous_decode_mode"] != "soft_expectation"
        or cfg["model"]["recon_decode_mode"] != "soft_expectation"
    ):
        raise ValueError(
            "Continuous and reconstruction decoding must both be soft_expectation"
        )
    if (
        cfg["train"]["batch_size"] * cfg["train"]["grad_accum_every"]
        != cfg["distributed"]["effective_batch"]
    ):
        raise ValueError("batch_size × accumulation must match source effective batch")
    if str(cfg["release"]["data_version"]).startswith("REQUIRED"):
        raise ValueError("Set release.data_version explicitly")
    return cfg


def semantic_digest(cfg):
    value = copy.deepcopy(cfg)
    # File locations can move; identity is separately enforced through asset hashes.
    value.pop("paths", None)
    for key in ["dataset_train", "dataset_val"]:
        for name in [
            "cache_dir",
            "svg_dir",
            "gt_dir",
            "primitive_dir",
            "invalid_scene_file",
        ]:
            value[key].pop(name, None)
        for t in value[key].get("instance_transform", []) or []:
            t.pop("donor_bank", None)
    for name in ["primitive_dir", "primitive_mapping_path"]:
        value["model"].pop(name, None)
    value["experiment"].pop("manifest", None)
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
