import os
import numpy as np
from evaluation.instance_eval_hungarian import InstanceEval_Hungarian


def _metrics(evaluator):
    tp = evaluator.tp_classes.astype(np.float64)
    tp_iou = evaluator.tp_classes_values.astype(np.float64)
    fp = evaluator.fp_classes.astype(np.float64)
    fn = evaluator.fn_classes.astype(np.float64)
    rq = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-6)
    sq = tp_iou / (tp + 1e-6)
    pq = rq * sq
    total_tp = float(tp.sum())
    total_fp = float(fp.sum())
    total_fn = float(fn.sum())
    total_iou = float(tp_iou.sum())
    total_rq = total_tp / (total_tp + 0.5 * total_fp + 0.5 * total_fn + 1e-6)
    total_sq = total_iou / (total_tp + 1e-6)
    return {
        "pq": total_rq * total_sq * 100.0,
        "rq": total_rq * 100.0,
        "sq": total_sq * 100.0,
        "tp": int(total_tp),
        "fp": int(total_fp),
        "fn": int(total_fn),
        "matched_iou_sum": total_iou,
        "per_class": [
            {
                "class_id": i,
                "pq": float(pq[i] * 100.0),
                "rq": float(rq[i] * 100.0),
                "sq": float(sq[i] * 100.0),
                "tp": int(tp[i]),
                "fp": int(fp[i]),
                "fn": int(fn[i]),
                "matched_iou_sum": float(tp_iou[i]),
            }
            for i in range(len(tp))
        ],
    }


def _new_evaluator(num_classes):
    return InstanceEval_Hungarian(
        num_classes=num_classes,
        ignore_label=num_classes,
        class_names=[f"class_{i}" for i in range(num_classes)],
        thing_class=list(range(num_classes)),
        stuff_class=[],
        gpu_num=1,
        min_obj_score=0.1,
        iou_thres=0.5,
    )


def _derive_industry_gt_with_agv(stage_a_path, num_segments):
    """Recover zero-based instance labels from Stage-A primitive metadata."""
    stage_a = np.load(stage_a_path, allow_pickle=True)
    seg_prim = np.asarray(stage_a["scene_segment_primitive_ids"], dtype=np.int64)
    prim_sem = np.asarray(stage_a["scene_primitive_semantic_ids"], dtype=np.int32)
    prim_inst = np.asarray(stage_a["scene_primitive_instance_ids"], dtype=np.int32)
    if seg_prim.size != num_segments:
        raise ValueError(
            f"Stage-A/line-level segment mismatch: {seg_prim.size} != {num_segments}"
        )
    seg_inst = prim_inst[seg_prim]
    seg_sem = np.full(num_segments, -1, dtype=np.int32)
    foreground = seg_inst >= 0
    # Older Stage-A files omit the offset and store class IDs minus one.
    offset = int(stage_a.get("scene_semantic_offset", 1))
    seg_sem[foreground] = prim_sem[seg_prim[foreground]] + offset
    instance_ids = np.unique(seg_inst[foreground])
    masks = np.zeros((instance_ids.size, num_segments), dtype=bool)
    labels = np.zeros(instance_ids.size, dtype=np.int32)
    for row, instance_id in enumerate(instance_ids):
        mask = seg_inst == instance_id
        masks[row] = mask
        labels[row] = int(seg_sem[mask][0])
    return masks, labels


def _load_and_update(
    path, evaluator, num_classes, stage_a_dir=None, fix_industry_agv_gt=False
):
    data = np.load(path, allow_pickle=True)
    pred_masks = np.asarray(data["pred_instance_masks"], dtype=bool)
    pred_labels = np.asarray(data["pred_instance_labels"], dtype=np.int64)
    gt_masks = np.asarray(data["gt_instance_masks"], dtype=bool)
    gt_labels = np.asarray(data["gt_instance_labels"], dtype=np.int64)
    lengths = np.asarray(data["seg_lengths"], dtype=np.float64)

    if fix_industry_agv_gt:
        if stage_a_dir is None:
            raise ValueError("stage_a_dir is required for the Industry AGV GT fix")
        stage_a_path = os.path.join(stage_a_dir, os.path.basename(path))
        if not os.path.isfile(stage_a_path):
            raise FileNotFoundError(stage_a_path)
        gt_masks, gt_labels = _derive_industry_gt_with_agv(stage_a_path, lengths.size)

    if pred_masks.ndim != 2 or gt_masks.ndim != 2:
        raise ValueError(
            f"instance masks must be 2-D: pred={pred_masks.shape}, gt={gt_masks.shape}"
        )
    if pred_masks.shape[0] != pred_labels.size or gt_masks.shape[0] != gt_labels.size:
        raise ValueError("mask/label instance counts do not match")
    if pred_masks.shape[1] != lengths.size or gt_masks.shape[1] != lengths.size:
        raise ValueError(
            f"segment counts do not match: pred={pred_masks.shape}, gt={gt_masks.shape}, "
            f"lengths={lengths.shape}"
        )

    pred_labels = pred_labels.copy()
    gt_labels = gt_labels.copy()
    pred_labels[(pred_labels < 0) | (pred_labels >= num_classes)] = num_classes
    gt_labels[(gt_labels < 0) | (gt_labels >= num_classes)] = num_classes
    instances = [
        {"masks": pred_masks[i], "labels": int(pred_labels[i]), "scores": 1.0}
        for i in range(pred_labels.size)
    ]
    # Reference evaluator applies np.asarray(target["masks"]).T internally.
    target = {"masks": gt_masks.T, "labels": gt_labels}
    evaluator.update(instances, target, lengths)
    return int(pred_labels.size), int(gt_labels.size)
