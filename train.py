import argparse
import math
import os
from datetime import timedelta
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs

from datasets import build_dataset
from util.engine import TrainingState, train_one_epoch
from evaluation.ar import evaluate
from evaluation.metrics import summarize
from models import build_model
from util import checkpoint
from util.config import load, configure_training
from util.guard import ProtectedStop, ResourceGuard
from util.io import save, seed, sha
from util.speed import install
from util.distributed import initialize, on_main, gather_objects, check_resources


def get_args_parser():
    parser = argparse.ArgumentParser("IndustryCAD training")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--output_dir", "--output", dest="output_dir", required=True)
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument("--resume", default=None, help="Restore the full training state")
    weights.add_argument(
        "--pretrained", default=None,
        help="Load model and selector weights (.pt or .safetensors) for a new training run",
    )
    parser.add_argument("--max_updates", "--max-updates", dest="max_updates", type=int)
    parser.add_argument("--batch_size", type=int, help="Source scenes per GPU")
    parser.add_argument(
        "--grad_accum_every", type=int,
        help="Accumulation steps; by default preserve the configured global batch size",
    )
    parser.add_argument(
        "--check_config", "--check-config", dest="check_config", action="store_true"
    )
    return parser


def main(args):
    if args.max_updates is not None and args.max_updates < 1:
        raise ValueError("--max_updates must be positive")
    cfg = load(args.config)
    if args.check_config:
        cfg = configure_training(
            cfg, int(os.environ.get("WORLD_SIZE", "1")), args.batch_size, args.grad_accum_every
        )
        print(f"Configuration valid: {cfg['distributed']['world_size']} processes, "
              f"batch={cfg['train']['batch_size']}, accumulation={cfg['train']['grad_accum_every']}, "
              f"global batch={cfg['distributed']['effective_batch']}")
        return

    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    accelerator = Accelerator(
        mixed_precision="bf16",
        kwargs_handlers=[
            DistributedDataParallelKwargs(
                find_unused_parameters=cfg["distributed"]["find_unused_parameters"],
            ),
            InitProcessGroupKwargs(
                backend=None,
                timeout=timedelta(minutes=cfg["distributed"].get("timeout_minutes", 15)),
            ),
        ],
    )
    cfg = configure_training(
        cfg, accelerator.num_processes, args.batch_size, args.grad_accum_every
    )
    initialize()
    output = Path(args.output_dir)
    guard = None
    try:
        def prepare_output():
            if output.exists() and not args.resume:
                raise FileExistsError("Use a new output directory or --resume")
            output.mkdir(parents=True, exist_ok=True)
            return ResourceGuard(output)

        guard = on_main(accelerator, prepare_output)

        identity = on_main(accelerator, lambda: checkpoint.binding(cfg), broadcast_result=True)
        seed(cfg["train"]["seed"])
        model = build_model(cfg)
        if cfg["distributed"]["sync_batchnorm"]:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.to(accelerator.device)
        if args.pretrained:
            metadata = checkpoint.load_model(model, args.pretrained, cfg)
            on_main(accelerator, lambda: save(
                output / "INITIALIZATION.json",
                dict(
                    mode="pretrained",
                    checkpoint=str(Path(args.pretrained).resolve()),
                    sha256=sha(args.pretrained),
                    source_step=metadata.get("step"),
                    source_epoch=metadata.get("epoch"),
                    optimizer_scheduler_reset=True,
                ),
            ))
            del metadata
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg["train"]["lr"],
            eps=1e-7,
            weight_decay=0.0,
        )
        dataset = build_dataset(cfg, "train")
        train_cfg = cfg["train"]
        local_samples = math.ceil(len(dataset) / accelerator.num_processes)
        microbatches = math.ceil(local_samples / train_cfg["batch_size"])
        total_steps = train_cfg["epochs"] * math.ceil(
            microbatches / train_cfg["grad_accum_every"]
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=train_cfg["max_lr"],
            total_steps=total_steps,
            pct_start=train_cfg["pct_start"],
            anneal_strategy=train_cfg["anneal_strategy"],
            div_factor=train_cfg["div_factor"],
            final_div_factor=train_cfg["final_div_factor"],
        )
        state = TrainingState()
        resume_rng = None
        if args.resume:
            state, resume_rng = checkpoint.restore_training(
                model, optimizer, scheduler, args.resume, identity,
                rank=accelerator.process_index, world_size=accelerator.num_processes,
            )

        install(model, cfg)
        wrapped, optimizer = accelerator.prepare(model, optimizer)
        model = accelerator.unwrap_model(wrapped)
        if resume_rng is not None:
            checkpoint.restore_rng(resume_rng)
            if checkpoint.object_digest(
                checkpoint.rng_state()
            ) != checkpoint.object_digest(resume_rng):
                raise ValueError("RNG restore mismatch")
            on_main(accelerator, lambda: save(
                output / "RESUME_RESTORE.json",
                dict(
                    passed=True,
                    step=state.step,
                    epoch=state.epoch,
                    micro_offset=state.offset,
                    model_optimizer_scheduler_rng_exact=True,
                ),
            ))
        on_main(accelerator, lambda: save(output / "config.json", cfg))
        on_main(accelerator, lambda: save(
            output / "SETUP.json",
            dict(
                binding=identity,
                effective_source_batch=train_cfg["batch_size"]
                * train_cfg["grad_accum_every"] * accelerator.num_processes,
                world_size=accelerator.num_processes,
                microbatch=train_cfg["batch_size"],
                accumulation=train_cfg["grad_accum_every"],
                total_scheduler_steps=total_steps,
                mixed_precision=accelerator.mixed_precision,
                max_updates=args.max_updates,
            ),
        ))
        initial_step = state.step
        model.train()

        def save_latest():
            check_resources(accelerator, guard)
            rng_by_rank = gather_objects(checkpoint.rng_state())
            on_main(accelerator, lambda: checkpoint.write(
                output / "latest.pt",
                model,
                optimizer,
                scheduler,
                state.step,
                state.epoch,
                state.offset,
                state.totals,
                identity,
                state.best,
                rng_by_rank=rng_by_rank,
            ))

        for epoch in range(state.epoch, train_cfg["epochs"]):
            state.epoch = epoch
            bounded = train_one_epoch(
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
                max_updates=args.max_updates,
                initial_step=initial_step,
            )
            if bounded:
                save_latest()
                on_main(accelerator, lambda: save(
                    output / "BOUNDED_COMPLETE.json",
                    dict(
                        step=state.step,
                        updates=state.step - initial_step,
                        formal_training=False,
                    ),
                ))
                return

            if (epoch + 1) % train_cfg["val_every_epoch"] == 0:
                result = on_main(accelerator, lambda: evaluate(
                    model,
                    cfg,
                    output / f"val_{state.step:06d}",
                    guard,
                    with_ar=train_cfg["val_native_ar"],
                ), broadcast_result=True)
                if state.best is None or result["original_loss"] < state.best:
                    state.best = result["original_loss"]
                    on_main(accelerator, lambda: checkpoint.save_inference(
                        output / "best.pt", model, epoch + 1, state.step, identity
                    ))
            on_main(accelerator, lambda: save(
                output / f"epoch_{epoch + 1:03d}.json",
                summarize(
                    state.totals,
                    model.model.loss_weight["reconstruction"]
                    * model.model.recon_weight_factor(),
                    model.model.loss_weight["eos"],
                    model.assignment_weight,
                ),
            ))
            state.totals = [0.0] * len(state.totals)
            state.offset = 0
            state.epoch = epoch + 1
            save_latest()
            if (epoch + 1) % train_cfg["checkpoint_every_epoch"] == 0:
                on_main(accelerator, lambda: checkpoint.save_inference(
                    output / f"epoch_{epoch + 1:03d}.pt",
                    model,
                    epoch + 1,
                    state.step,
                    identity,
                ))

        on_main(accelerator, lambda: save(
            output / "COMPLETE.json",
            dict(step=state.step, epochs=train_cfg["epochs"], passed=True),
        ))
    except ProtectedStop as error:
        on_main(accelerator, lambda: save(
            output / "STOPPED.json",
            dict(reason=str(error), checkpoint_may_precede_stop=True),
        ))
        raise
    finally:
        if guard is not None:
            guard.close()
        accelerator.end_training()


if __name__ == "__main__":
    main(get_args_parser().parse_args())
