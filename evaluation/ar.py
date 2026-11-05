from pathlib import Path
import gc
import torch
from util.io import save, seed
from evaluation.metrics import METRICS, summarize, truth, ARScorer, ar_summary
from evaluation.geometry import tensor_records


@torch.no_grad()
def evaluate(
    raw, cfg, output, guard, limit=None, with_ar=True, beta=None, teacher_forced=True
):
    from datasets.instance_dropout import SceneDataset, batch
    from models.batched import BatchedPrompt

    dest = Path(output)
    dest.mkdir(parents=True, exist_ok=False)
    ds = SceneDataset(cfg, "val")
    n = len(ds) if limit is None else min(limit, len(ds))
    from evaluation.seeds import indices

    seed_indices = indices(ds.names)
    if n <= 0:
        raise ValueError("No validation scenes")
    device = next(raw.parameters()).device
    raw.eval()
    raw.clear_cache()
    scorer = ARScorer(cfg["paths"]["template_root"])
    predictions = []
    instances = []
    totals = [0.0] * len(METRICS)
    beta = cfg["soft_guidance"]["beta_max"] if beta is None else beta
    for i in range(n):
        if guard.boundary():
            raise RuntimeError("Evaluation stopped at resource/user boundary")
        d = ds[i]
        b, c, t = batch(d, device)
        den = torch.tensor(raw.counts(d["sample"]), device=device, dtype=torch.float32)
        scene_seed = cfg["train"]["seed"] + 100000 + seed_indices[ds.names[i]]
        row = dict(
            filename=ds.names[i], seed=scene_seed, seed_index=seed_indices[ds.names[i]]
        )
        if teacher_forced:
            seed(scene_seed)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = raw(b, c, t, den, beta=beta, audit=True)
            for j, k in enumerate(METRICS):
                totals[j] += float(out["metrics"][k])
            row["metrics"] = {k: float(v) for k, v in out["metrics"].items()}
            del out
        if with_ar:
            # Only input VecData and its token coordinates enter this branch.
            seed(scene_seed)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                encoded = raw.model._encode_shape_for_generate(b["vec_data"])
                with BatchedPrompt(raw.model, raw.selector, c, beta=beta, mode="ar"):
                    p = raw.model._generate_attr_decouple_batched(
                        *encoded, max_primitives=128
                    )
            count = int(p["pred_mask"].sum())
            record = tensor_records(p, None, count, "ar")
            record.update(n_pred=count, hit_cap=count == 128)
            gt = truth(d["sample"])
            rows = scorer.score(record, gt)
            from evaluation.export import export_input

            (dest / "results_val").mkdir(exist_ok=True)
            export_input(
                d["sample"],
                gt,
                record,
                dest / "results_val" / (Path(ds.names[i]).stem + ".npz"),
            )
            predictions.append(dict(filename=ds.names[i], pred=record))
            instances.extend(dict(filename=ds.names[i], **r) for r in rows)
            row.update(prediction=record, instances=rows)
            del encoded, p
        save(dest / "scenes" / f"{i:04d}.json", row)
        print(f"Validation {i+1}/{n}", flush=True)
        del b, c, t, d
        raw.clear_cache()
        gc.collect()
        torch.cuda.empty_cache()
    summary = (
        summarize(
            totals,
            raw.model.loss_weight["reconstruction"] * raw.model.recon_weight_factor(),
            raw.model.loss_weight["eos"],
            raw.assignment_weight,
        )
        if teacher_forced
        else {}
    )
    population = cfg["release"].get("evaluation_population", "explicit manifest")
    summary.update(
        scenes=n,
        population=population,
        complete_population=n == len(ds),
        with_ar=with_ar,
        teacher_forced_first=teacher_forced,
        beta=beta,
        protocol="AR center gate 0.1; 256 support points; normalized symmetric CD < 0.03; identity requires semantic AND primitive",
        data_version=cfg["release"]["data_version"],
    )
    from util.io import sha

    save(
        dest / "EVALUATED_MANIFEST.json",
        dict(
            population=population,
            complete_population=n == len(ds),
            source_manifest_sha256=sha(cfg["paths"]["test_manifest"]),
            filenames=list(ds.names[:n]),
            seed_rule="train.seed + 100000 + per-scene seed_index",
            seed_indices={name: seed_indices[name] for name in ds.names[:n]},
            preprocessing_seeds={
                name: ds.preprocessing_seeds[name] for name in ds.names[:n]
            },
        ),
    )
    if with_ar:
        summary.update(ar=ar_summary(instances, predictions))
        save(dest / "predictions.json", predictions)
        save(dest / "instances.json", instances)
    save(dest / "SUMMARY.json", summary)
    raw.train()
    raw.clear_cache()
    return summary
