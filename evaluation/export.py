from pathlib import Path


def export_input(sample, g, pred, path):
    import numpy as np

    svg = sample["svg_data"]
    np.savez_compressed(
        path,
        filename=np.array(Path(sample["filename"]).stem),
        scene_bound=np.float32(30000.0),
        size_pred=np.bool_(False),
        scale_parameterization=np.array("relative"),
        scene_lines=np.asarray(svg.coords),
        scene_segment_primitive_ids=np.asarray(svg.primitive_ids),
        scene_primitive_semantic_ids=np.asarray(svg.semantic_ids),
        scene_semantic_offset=np.int32(0),
        scene_primitive_instance_ids=np.asarray(svg.instance_ids),
        scene_primitive_lengths=np.asarray(svg.primitive_lengths),
        gt_semantic_i=np.asarray(g["semantic_id"], np.int64),
        gt_primitive_i=np.asarray(g["primitive_id"], np.int64),
        gt_flip_i=np.asarray(g["flip_mode"], np.int64),
        gt_cont_translation_i=np.asarray(g["translation"], np.float32).reshape(-1, 2),
        gt_cont_rotation_i=np.asarray(g["rotation"], np.float32).reshape(-1),
        gt_cont_scale_i=np.asarray(g["scale"], np.float32).reshape(-1, 2),
        pred_semantic_id_i=np.asarray(pred["semantic"], np.int64),
        pred_primitive_id_i=np.asarray(pred["primitive"], np.int64),
        pred_flip_i=np.asarray(pred["flip"], np.int64),
        pred_discrete_cont_translation_i=np.asarray(
            pred["translation"], np.float32
        ).reshape(-1, 2),
        pred_discrete_cont_rotation_i=np.asarray(pred["rotation"], np.float32).reshape(
            -1, 1
        ),
        pred_discrete_cont_scale_i=np.asarray(pred["shape"], np.float32).reshape(-1, 2),
    )
