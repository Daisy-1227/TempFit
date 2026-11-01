import gc
import math
import time
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader

from datasets.instance_dropout import batch
from datasets.metadata import identity, schedule
from datasets.scene_mix import mix_group, summarize_mix
from datasets.transport import decode, transport_collate
from evaluation.metrics import METRICS, summarize
from util.guard import ProtectedStop
from util.io import seed
from util.logger import log_update


@dataclass
class TrainingState:
    epoch: int = 0
    offset: int = 0
    step: int = 0
    totals: list = field(default_factory=lambda: [0.0] * len(METRICS))
    best: float | None = None


def train_one_epoch(
    wrapped,
    model,
    dataset,
    optimizer,
    scheduler,
    accelerator,
    cfg,
    state,
    output,
    guard,
    save_latest,
    max_updates=None,
    initial_step=0,
):
    epoch = state.epoch
    train_cfg = cfg["train"]
    batch_size = train_cfg["batch_size"]
    accumulation = train_cfg["grad_accum_every"]
    microbatches = math.ceil(len(dataset) / batch_size)
    dataset.epoch = epoch
    model.model.set_recon_epoch(cfg["initialization"]["source_epoch"] + epoch)
    indices = schedule(len(dataset), epoch, cfg["experiment"]["data_seed"], 1, 0)
    if not 0 <= state.offset <= microbatches:
        raise ValueError("Resume micro offset invalid")
    workers = cfg["distributed"]["workers_per_rank"]
    transport = cfg.get("runtime_optimization", {}).get("enabled", False)
    options = (
        dict(prefetch_factor=cfg["runtime_optimization"].get("prefetch_factor", 2))
        if workers
        else {}
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=indices[state.offset * batch_size :],
        num_workers=workers,
        collate_fn=transport_collate if transport else identity,
        generator=torch.Generator().manual_seed(train_cfg["seed"] + epoch),
        **options,
    )
    iterator = iter(loader)
    while state.offset < microbatches:
        reason = guard.boundary()
        if reason:
            save_latest()
            raise ProtectedStop(reason)
        started = time.perf_counter()
        source = [
            next(iterator)
            for _ in range(min(accumulation, microbatches - state.offset))
        ]
        if transport:
            source = decode(source)
        mixed = [
            mix_group(group, cfg, epoch, state.offset + j)
            for j, group in enumerate(source)
        ]
        window = [item[0] for item in mixed]
        counts = [[model.counts(item["sample"]) for item in group] for group in window]
        # Normalize over all real instances in the accumulation window.
        denominators = torch.tensor(
            [
                sum(
                    count[k] * item["real"]
                    for group_counts, group in zip(counts, window)
                    for count, item in zip(group_counts, group)
                )
                for k in range(3)
            ],
            device=accelerator.device,
            dtype=torch.float32,
        )
        optimizer.zero_grad(set_to_none=True)
        model.clear_cache()
        update = [0.0] * len(METRICS)
        beta = cfg["soft_guidance"]["beta_max"] * min(
            1.0, (state.step + 1) / cfg["soft_guidance"]["beta_warmup_updates"]
        )
        for j, group in enumerate(window):
            guard.check()
            seed(train_cfg["seed"] + epoch * 10000000 + state.offset + j)
            inputs, coords, targets = batch(group, accelerator.device)
            with accelerator.autocast():
                result = wrapped(
                    inputs,
                    coords,
                    targets,
                    denominators,
                    beta=beta,
                    audit=max_updates is not None,
                )
                loss = result["objective"]
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                accelerator.backward(loss)
            for k, name in enumerate(METRICS):
                update[k] += float(result["metrics"][name])
            del inputs, coords, targets, result, loss
        norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite gradient norm")
        gradient_groups = {}
        if max_updates is not None:
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    continue
                if name.startswith("selector."):
                    group = "selector"
                elif name.startswith("model.conditioner."):
                    group = "backbone"
                elif name.startswith("model.decoder."):
                    group = "decoder"
                else:
                    group = "other_model"
                gradient_groups[group] = gradient_groups.get(group, 0.0) + float(
                    parameter.grad.detach().float().square().sum()
                )
            if any(
                gradient_groups.get(k, 0.0) <= 0
                for k in ("selector", "backbone", "decoder", "other_model")
            ):
                raise FloatingPointError("Missing expected training gradient group")
        optimizer.step()
        scheduler.step()
        state.step += 1
        state.offset += len(window)
        state.totals = [a + b for a, b in zip(state.totals, update)]
        row = dict(
            step=state.step,
            epoch=epoch + 1,
            micro_offset=state.offset,
            lr=optimizer.param_groups[0]["lr"],
            beta=beta,
            grad_norm=float(norm),
            seconds=time.perf_counter() - started,
            peak_memory_gib=torch.cuda.max_memory_allocated() / 2**30,
            scene_mix=summarize_mix(
                [event for item in mixed for event in item[1]],
                sum(item["real"] for group in source for item in group),
            ),
            **summarize(
                update,
                model.model.loss_weight["reconstruction"]
                * model.model.recon_weight_factor(),
                model.model.loss_weight["eos"],
                model.assignment_weight,
            ),
        )
        if max_updates is not None:
            row["gradient_square_by_group"] = gradient_groups
        log_update(output, row)
        del source, mixed, window, counts, denominators
        model.clear_cache()
        gc.collect()
        torch.cuda.empty_cache()
        if max_updates and state.step - initial_step >= max_updates:
            return True
        if state.step == 1 or state.step % train_cfg["checkpoint_every_updates"] == 0:
            save_latest()
    return False
