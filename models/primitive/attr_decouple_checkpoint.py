from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def _copy_if_shape_matches(target, source, target_key, source_key=None):
    source_key = source_key or target_key
    if source_key not in source or target_key not in target:
        return False
    if target[target_key].shape != source[source_key].shape:
        return False
    target[target_key] = source[source_key].detach().clone()
    return True


def migrate_semgeom_checkpoint_to_attr_decouple(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Load a sem/geom checkpoint into a six-token attr-decoupled model.

    Shared modules are copied exactly.  The two old input projections are
    sliced into the six new projections according to their original packed
    feature layout.  Incompatible new tensors keep their initialized values.
    Optimizer/scheduler state is deliberately not loaded: this is weight-only
    architectural fine-tuning, not a training resume.
    """
    checkpoint_path = Path(checkpoint_path)
    package = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = package.get("model", package)
    source = {
        key.removeprefix("module."): value
        for key, value in source.items()
        if torch.is_tensor(value)
    }
    target = model.state_dict()

    copied_exact = []
    for key in list(target):
        if _copy_if_shape_matches(target, source, key):
            copied_exact.append(key)

    mapped = []

    def map_slice(target_key, source_key, column_slice=None):
        if target_key not in target or source_key not in source:
            return
        value = source[source_key]
        if column_slice is not None:
            value = value[:, column_slice]
        if value.shape == target[target_key].shape:
            target[target_key] = value.detach().clone()
            mapped.append(f"{target_key} <- {source_key}")

    sem_dim = model.semantic_embed.embedding_dim
    prim_dim = model.primitive_embed.embedding_dim
    trans_dim = model.translation_embed.embedding_dim * 2
    flip_dim = model.flip_embed.embedding_dim
    rot_dim = model.rotation_embed.embedding_dim
    shape_dim = (
        model.size_embed.embedding_dim * 2
        if model.size_pred
        else model.scale_embed.embedding_dim * 2
    )
    # Old semantic token was [semantic, primitive].
    map_slice("sem_proj_in.weight", "sem_proj_in.weight", slice(0, sem_dim))
    map_slice(
        "prim_proj_in.weight",
        "sem_proj_in.weight",
        slice(sem_dim, sem_dim + prim_dim),
    )
    # Old geometry token was [translation, flip, rotation, shape].
    trans_end = trans_dim
    flip_end = trans_end + flip_dim
    rot_end = flip_end + rot_dim
    map_slice("trans_proj_in.weight", "geom_proj_in.weight", slice(0, trans_end))
    map_slice("flip_proj_in.weight", "geom_proj_in.weight", slice(trans_end, flip_end))
    map_slice("rot_proj_in.weight", "geom_proj_in.weight", slice(flip_end, rot_end))
    map_slice(
        "shape_proj_in.weight",
        "geom_proj_in.weight",
        slice(rot_end, rot_end + shape_dim),
    )
    for target_key, source_key in (
        ("sem_proj_in.bias", "sem_proj_in.bias"),
        ("prim_proj_in.bias", "sem_proj_in.bias"),
        ("trans_proj_in.bias", "geom_proj_in.bias"),
        ("flip_proj_in.bias", "geom_proj_in.bias"),
        ("rot_proj_in.bias", "geom_proj_in.bias"),
        ("shape_proj_in.bias", "geom_proj_in.bias"),
    ):
        if _copy_if_shape_matches(target, source, target_key, source_key):
            mapped.append(f"{target_key} <- {source_key}")

    # Token type 0 was semantic, token type 1 was geometry.
    if "attr_type_embed.weight" in source and "attr_type_embed.weight" in target:
        old_type = source["attr_type_embed.weight"]
        new_type = target["attr_type_embed.weight"].clone()
        if (
            old_type.ndim == 2
            and old_type.shape[0] >= 2
            and old_type.shape[1] == new_type.shape[1]
        ):
            new_type[0:2] = old_type[0]
            new_type[2:6] = old_type[1]
            target["attr_type_embed.weight"] = new_type
            mapped.append("attr_type_embed.weight <- expanded 2 types to 6 types")

    # Geometry heads no longer concatenate preceding GT attributes.  Preserve
    # the hidden-state block of the old first layer; size-H additionally keeps
    # the final width-embedding block used by its factorized conditioning.
    for head in ("to_flip_logits", "to_rotation_logits", "to_size_w_logits"):
        map_slice(f"{head}.0.weight", f"{head}.0.weight", slice(0, model.dim))
    old_size_h = source.get("to_size_h_logits.0.weight")
    new_size_h_key = "to_size_h_logits.0.weight"
    if old_size_h is not None and new_size_h_key in target:
        width_dim = model.size_embed.embedding_dim
        candidate = torch.cat(
            [old_size_h[:, : model.dim], old_size_h[:, -width_dim:]], dim=1
        )
        if candidate.shape == target[new_size_h_key].shape:
            target[new_size_h_key] = candidate.detach().clone()
            mapped.append(f"{new_size_h_key} <- decoder state + width condition")

    load_result = model.load_state_dict(target, strict=True)
    initialized = sorted(
        key
        for key in target
        if key not in copied_exact
        and not any(entry.startswith(f"{key} <-") for entry in mapped)
    )
    return {
        "checkpoint": str(checkpoint_path),
        "source_epoch": package.get("epoch"),
        "copied_exact_count": len(copied_exact),
        "mapped": mapped,
        "kept_initialized": initialized,
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
    }


def load_compatible_attr_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    reset_prefixes: tuple[str, ...] = (
        "translation_embed.",
        "scale_embed.",
        "trans_proj_in.",
        "shape_proj_in.",
        "to_translation_logits.",
        "to_trans_x_logits.",
        "to_trans_y_logits.",
        "to_scale_logits.",
    ),
) -> dict[str, Any]:
    """Load shared attr-decouple weights after changing transform semantics.

    Translation bins/range and the shape token change in the partial-recon
    experiment, so their embeddings, history projections, and output heads
    must remain freshly initialized.  All other shape-compatible tensors are
    copied.  Optimizer/scheduler state is deliberately not restored.
    """
    checkpoint_path = Path(checkpoint_path)
    package = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = package.get("model", package)
    source = {
        key.removeprefix("module."): value
        for key, value in source.items()
        if torch.is_tensor(value)
    }
    target = model.state_dict()
    copied = []
    shape_mismatch = []
    reset = []
    missing = []
    for key in target:
        if key.startswith(reset_prefixes):
            reset.append(key)
        elif key not in source:
            missing.append(key)
        elif source[key].shape != target[key].shape:
            shape_mismatch.append(key)
        else:
            target[key] = source[key].detach().clone()
            copied.append(key)
    load_result = model.load_state_dict(target, strict=True)
    return {
        "checkpoint": str(checkpoint_path),
        "source_epoch": package.get("epoch"),
        "copied_count": len(copied),
        "reset_prefixes": list(reset_prefixes),
        "reset": sorted(reset),
        "shape_mismatch": sorted(shape_mismatch),
        "missing_in_source": sorted(missing),
        "unexpected_in_source": sorted(set(source) - set(target)),
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
    }
