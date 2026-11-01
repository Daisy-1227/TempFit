import numpy as np
import torch
import torch.distributed as dist
from scipy.optimize import linear_sum_assignment


class InstanceEval_Hungarian(object):
    """Hungarian-matching variant of panoptic instance evaluator.

    Per-class bipartite optimal matching (max total IoU under IoU>thres),
    so each GT matches at most one Pred and vice versa.
    """

    def __init__(
        self,
        num_classes,
        ignore_label,
        class_names,
        thing_class=None,
        stuff_class=None,
        gpu_num=1,
        min_obj_score=0.1,
        iou_thres=0.5,
    ):
        self.ignore_label = ignore_label
        self._num_classes = num_classes
        self._class_names = class_names
        self.gpu_num = gpu_num
        self.min_obj_score = min_obj_score
        self.IoU_thres = iou_thres

        self.tp_classes = np.zeros(num_classes)
        self.tp_classes_values = np.zeros(num_classes)
        self.fp_classes = np.zeros(num_classes)
        self.fn_classes = np.zeros(num_classes)
        self.thing_class = (
            list(range(num_classes)) if thing_class is None else list(thing_class)
        )
        self.stuff_class = [] if stuff_class is None else list(stuff_class)

    @staticmethod
    def _iou(src_mask, tgt_mask, lengths):
        inter = np.sum(lengths[np.logical_and(src_mask, tgt_mask)])
        union = np.sum(lengths[np.logical_or(src_mask, tgt_mask)])
        return inter / (union + 1e-6)

    def update(self, instances, target, lengths):
        if torch.is_tensor(lengths):
            lengths = lengths.cpu().numpy()
        lengths = np.round(np.log(1 + lengths), 3)

        tgt_labels = target["labels"]
        tgt_masks = target["masks"]
        if torch.is_tensor(tgt_labels):
            tgt_labels = tgt_labels.cpu().numpy()
        tgt_labels = np.asarray(tgt_labels).tolist()
        if torch.is_tensor(tgt_masks):
            tgt_masks = tgt_masks.transpose(0, 1).cpu().numpy()
        else:
            tgt_masks = np.asarray(tgt_masks).T  # [N_tgt, N_points]

        valid_instances = [
            {"labels": ins["labels"], "scores": ins["scores"], "masks": ins["masks"]}
            for ins in instances
            if ins["labels"] != self.ignore_label
            and ins["scores"] >= self.min_obj_score
        ]
        valid_gt_indices = [
            i for i, lab in enumerate(tgt_labels) if lab != self.ignore_label
        ]

        num_preds = len(valid_instances)
        num_gts = len(valid_gt_indices)

        if num_preds == 0 or num_gts == 0:
            for gt_idx in valid_gt_indices:
                self.fn_classes[tgt_labels[gt_idx]] += 1
            for ins in valid_instances:
                self.fp_classes[ins["labels"]] += 1
            return

        iou_matrix = np.zeros((num_gts, num_preds), dtype=np.float32)
        for i, gt_idx in enumerate(valid_gt_indices):
            tgt_mask = tgt_masks[gt_idx]
            for j, ins in enumerate(valid_instances):
                iou_matrix[i, j] = self._iou(ins["masks"], tgt_mask, lengths)

        for class_id in range(self._num_classes):
            gt_in_cls = [
                i for i in range(num_gts) if tgt_labels[valid_gt_indices[i]] == class_id
            ]
            pred_in_cls = [
                j for j in range(num_preds) if valid_instances[j]["labels"] == class_id
            ]

            if not gt_in_cls and not pred_in_cls:
                continue
            if not gt_in_cls:
                self.fp_classes[class_id] += len(pred_in_cls)
                continue
            if not pred_in_cls:
                self.fn_classes[class_id] += len(gt_in_cls)
                continue

            sub = iou_matrix[np.ix_(gt_in_cls, pred_in_cls)]
            valid = sub > self.IoU_thres
            cost = -sub.copy()
            cost[~valid] = 1e6
            row_ind, col_ind = linear_sum_assignment(cost)

            class_tp = 0
            class_tp_iou = 0.0
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < 1e5:
                    class_tp += 1
                    class_tp_iou += sub[r, c]

            self.tp_classes[class_id] += class_tp
            self.tp_classes_values[class_id] += class_tp_iou
            self.fn_classes[class_id] += len(gt_in_cls) - class_tp
            self.fp_classes[class_id] += len(pred_in_cls) - class_tp

    def get_eval(self, logger):
        if self.gpu_num > 1:
            t = torch.from_numpy(
                np.stack(
                    [
                        self.tp_classes,
                        self.tp_classes_values,
                        self.fp_classes,
                        self.fn_classes,
                    ]
                )
            ).to("cuda")
            tlist = [torch.full_like(t, 0) for _ in range(self.gpu_num)]
            dist.barrier()
            dist.all_gather(tlist, t)
            agg = sum(tlist).cpu().numpy()
            (
                self.tp_classes,
                self.tp_classes_values,
                self.fp_classes,
                self.fn_classes,
            ) = agg

        tp, tpv, fp, fn = (
            self.tp_classes,
            self.tp_classes_values,
            self.fp_classes,
            self.fn_classes,
        )
        RQ = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-6)
        SQ = tpv / (tp + 1e-6)
        PQ = RQ * SQ

        def _agg(idx):
            if not idx:
                return 0.0, 0.0, 0.0
            r = sum(tp[idx]) / (
                sum(tp[idx]) + 0.5 * sum(fp[idx]) + 0.5 * sum(fn[idx]) + 1e-6
            )
            s = sum(tpv[idx]) / (sum(tp[idx]) + 1e-6)
            return r * s, r, s

        thing_PQ, thing_RQ, thing_SQ = _agg(self.thing_class)
        stuff_PQ, stuff_RQ, stuff_SQ = _agg(self.stuff_class)

        sRQ = sum(tp) / (sum(tp) + 0.5 * sum(fp) + 0.5 * sum(fn) + 1e-6)
        sSQ = sum(tpv) / (sum(tp) + 1e-6)
        sPQ = sRQ * sSQ

        for i, name in enumerate(self._class_names[: self._num_classes]):
            logger.info(
                "Class_{}  PQ / RQ / SQ : {:.3f} / {:.3f} / {:.3f}".format(
                    name, PQ[i] * 100, RQ[i] * 100, SQ[i] * 100
                )
            )
        logger.info(
            "PQ / RQ / SQ : {:.3f} / {:.3f} / {:.3f}".format(
                sPQ * 100, sRQ * 100, sSQ * 100
            )
        )
        logger.info(
            "thing PQ / RQ / SQ : {:.3f} / {:.3f} / {:.3f}".format(
                thing_PQ * 100, thing_RQ * 100, thing_SQ * 100
            )
        )
        if self.stuff_class:
            logger.info(
                "stuff PQ / RQ / SQ : {:.3f} / {:.3f} / {:.3f}".format(
                    stuff_PQ * 100, stuff_RQ * 100, stuff_SQ * 100
                )
            )
        return sPQ * 100, sRQ * 100, sSQ * 100
