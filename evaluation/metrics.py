from pathlib import Path
import numpy as np
from evaluation.geometry import (
    center_assignment,
    load_template,
    transform_template,
    uniform_support,
    support_chamfer,
)

METRICS = [
    "attr_sum",
    "eos_sum",
    "recon_sum",
    "assignment_sum",
    "n",
    "ne",
    "nr",
    "scenes",
]


def summarize(vector, wr=1.0, we=1.0, ws=1.0):
    v = dict(zip(METRICS, [float(x) for x in vector]))
    v.update(
        attributes=v["attr_sum"] / max(v["n"], 1),
        eos=v["eos_sum"] / max(v["ne"], 1),
        recon=v["recon_sum"] / max(v["nr"], 1),
        assignment=v["assignment_sum"] / max(v["n"], 1),
    )
    v["original_loss"] = v["attributes"] + we * v["eos"] + wr * v["recon"]
    v["total"] = v["original_loss"] + ws * v["assignment"]
    return v


def truth(sample):
    def arr(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    return {
        k: arr(sample[k]).tolist()
        for k in (
            "semantic_id",
            "primitive_id",
            "translation",
            "scale",
            "rotation",
            "flip_mode",
            "size",
            "recon_is_partial",
        )
    }


class ARScorer:
    def __init__(self, template_root, scene_bound=30000):
        self.root = Path(template_root)
        self.cache = {}
        self.bound = scene_bound

    def template(self, p, s, r, t, f):
        return transform_template(
            load_template(self.root, int(p), self.cache),
            np.asarray(s).reshape(2),
            float(np.asarray(r).reshape(-1)[0]),
            np.asarray(t).reshape(2),
            int(f),
            self.bound,
        )

    def score(self, pred, g):
        if pred["shape_key"] != "scale":
            raise ValueError("Historical AR scorer requires scale predictions")
        links = center_assignment(
            np.asarray(pred["translation"]).reshape(-1, 2),
            np.asarray(g["translation"]).reshape(-1, 2),
            0.1,
        )
        mapped = {j: (i, d) for i, j, d in links}
        out = []
        for j in range(len(g["primitive_id"])):
            row = dict(
                slot=j,
                center_match=j in mapped,
                semantic_correct=False,
                primitive_correct=False,
                identity_correct=False,
                strict=False,
            )
            if j in mapped:
                i, d = mapped[j]
                sem = pred["semantic"][i] == g["semantic_id"][j]
                prim = pred["primitive"][i] == g["primitive_id"][j]
                gt = self.template(
                    g["primitive_id"][j],
                    g["scale"][j],
                    g["rotation"][j],
                    g["translation"][j],
                    g["flip_mode"][j],
                )
                pr = self.template(
                    pred["primitive"][i],
                    pred["shape"][i],
                    pred["rotation"][i],
                    pred["translation"][i],
                    pred["flip"][i],
                )
                cd = float(
                    support_chamfer(uniform_support(pr, 256), uniform_support(gt, 256))
                )
                row.update(
                    pred_slot=i,
                    center_distance=float(d),
                    semantic_correct=sem,
                    primitive_correct=prim,
                    identity_correct=sem and prim,
                    strict=sem and prim and cd < 0.03,
                    geometry_distance=cd,
                )
            out.append(row)
        return out


def ar_summary(rows, predictions):
    n = len(rows)
    return dict(
        gt_instances=n,
        pred_instances=sum(p["pred"]["n_pred"] for p in predictions),
        scenes=len(predictions),
        hit_cap_scenes=sum(p["pred"]["hit_cap"] for p in predictions),
        **{
            k + "_rate": sum(r[k] for r in rows) / n if n else None
            for k in (
                "center_match",
                "semantic_correct",
                "primitive_correct",
                "identity_correct",
                "strict",
            )
        }
    )
