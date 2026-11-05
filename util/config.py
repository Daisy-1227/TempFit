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
        "train_manifest",
        "test_manifest",
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
    for key in ("train_manifest", "test_manifest"):
        names = read(paths[key])
        if not isinstance(names, list) or not names or any(
            not isinstance(n, str) or Path(n).name != n or not n.endswith(".json")
            for n in names
        ) or len(names) != len(set(names)):
            raise ValueError(f"{key} requires a unique filename list")
    world_size = cfg["distributed"]["world_size"]
    if not isinstance(world_size, int) or world_size < 1:
        raise ValueError("distributed.world_size must be a positive integer")
    if cfg["model"]["shape_condition_model_type"] != "point_transformer_v3" or cfg[
        "model"
    ].get("use_img_conditioner"):
        raise ValueError(
            "Use shape_condition_model_type=point_transformer_v3 and use_img_conditioner=false"
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
        world_size * cfg["train"]["batch_size"] * cfg["train"]["grad_accum_every"]
        != cfg["distributed"]["effective_batch"]
    ):
        raise ValueError("world_size × batch_size × grad_accum_every must equal effective_batch")
    if str(cfg["release"]["data_version"]).startswith("REQUIRED"):
        raise ValueError("Set release.data_version explicitly")
    return cfg


def configure_training(cfg, world_size, batch_size=None, grad_accum_every=None):
    cfg = copy.deepcopy(cfg)
    batch = cfg["train"]["batch_size"] if batch_size is None else batch_size
    if world_size < 1 or batch < 1:
        raise ValueError("Process count and batch size must be positive")
    if grad_accum_every is None:
        accumulation, remainder = divmod(cfg["distributed"]["effective_batch"], world_size * batch)
        if remainder or accumulation < 1:
            raise ValueError("Set --grad_accum_every for this process count and batch size")
    else:
        accumulation = grad_accum_every
    if accumulation < 1:
        raise ValueError("Gradient accumulation must be positive")
    cfg["distributed"]["world_size"] = world_size
    cfg["train"]["batch_size"] = batch
    cfg["train"]["grad_accum_every"] = accumulation
    cfg["distributed"]["effective_batch"] = world_size * batch * accumulation
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
