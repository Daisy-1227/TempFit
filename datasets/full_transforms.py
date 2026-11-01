import numpy as np
from datasets.transform import TRANSFORMS


class EmptyInputAfterTranslationFilter(RuntimeError):
    """Never silently replace a scene if filtering removes its entire input."""


@TRANSFORMS.register_module()
class DropOutOfRangeTranslation:
    """Drop whole input instances and GT, after augmentation and BEFORE clipping.

    Bounds refer to normalized GT template origins, not geometry endpoints.
    Background is kept; valid objects crossing the bounds are kept whole.
    Must precede creation of preclip recon snapshots and GridSample.
    """

    def __init__(self, continuous_range=(-2.0, 2.0)):
        self.lower, self.upper = map(float, continuous_range)
        if not np.isfinite([self.lower, self.upper]).all() or self.lower >= self.upper:
            raise ValueError("Invalid translation bounds")

    def __call__(self, d):
        translation = np.asarray(d["translation"]).reshape(-1, 2)
        if not np.isfinite(translation).all():
            raise ValueError("Nonfinite translation; not an ordinary range violation")
        audit = d.setdefault("_recon_audit", {})
        if "preclip_full_coords" in audit or "clip_count_ratio" in audit:
            raise RuntimeError(
                "Translation filtering must run before clip/recon snapshot"
            )
        n = len(translation)
        keep = ((translation >= self.lower) & (translation <= self.upper)).all(axis=1)
        removed = np.flatnonzero(~keep)
        audit["translation_filter_removed_instance_ids"] = removed.copy()
        audit["translation_filter_removed_instances"] = int(len(removed))
        audit["translation_filter_removed_segments"] = 0
        audit["translation_filter_input_instances"] = n
        if not len(removed):
            return d

        svg = d["svg_data"]
        prim_ids = np.asarray(svg.primitive_ids, dtype=np.int64)
        inst_ids = np.asarray(svg.instance_ids, dtype=np.int64)
        if np.any(inst_ids >= n) or np.any(inst_ids < -1):
            raise ValueError("Invalid source-to-GT instance membership")
        keep_prim = np.ones(len(inst_ids), dtype=bool)
        fg = inst_ids >= 0
        keep_prim[fg] = keep[inst_ids[fg]]
        keep_line = keep_prim[prim_ids]
        if not keep_line.any():
            raise EmptyInputAfterTranslationFilter(
                f"Translation filter removed entire input: {d.get('filename', '<unknown>')}"
            )
        # Remove point/line rows, then compact primitive and instance indices.
        for key in ("coords", "colors", "widths", "layer_ids"):
            setattr(svg, key, np.asarray(getattr(svg, key))[keep_line])
        prim_map = np.full(len(inst_ids), -1, dtype=np.int32)
        prim_map[keep_prim] = np.arange(keep_prim.sum(), dtype=np.int32)
        svg.primitive_ids = prim_map[prim_ids[keep_line]]
        inst_map = np.full(n, -1, dtype=np.int32)
        inst_map[keep] = np.arange(keep.sum(), dtype=np.int32)
        new_ids = inst_ids[keep_prim].astype(np.int32, copy=True)
        new_fg = new_ids >= 0
        new_ids[new_fg] = inst_map[new_ids[new_fg]]
        svg.instance_ids = new_ids
        for key in ("semantic_ids", "primitive_lengths"):
            setattr(svg, key, np.asarray(getattr(svg, key))[keep_prim])
        for key in (
            "semantic_id",
            "primitive_id",
            "scale",
            "rotation",
            "flip_mode",
            "translation",
            "flip_invariant_primitive",
            "size",
        ):
            if key in d and d[key] is not None:
                if len(d[key]) != n:
                    raise ValueError(f"GT length mismatch: {key}")
                d[key] = np.asarray(d[key])[keep]
        for key in ("instance_origin_kind", "instance_jittered"):
            if key in audit:
                if len(audit[key]) != n:
                    raise ValueError(f"Audit length mismatch: {key}")
                audit[key] = np.asarray(audit[key])[keep]
        audit["translation_filter_removed_segments"] = int((~keep_line).sum())
        return d
