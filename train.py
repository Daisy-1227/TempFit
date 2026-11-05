import argparse
import math
from pathlib import Path

import torch
from accelerate import Accelerator

from datasets import build_dataset
from util.engine import TrainingState, train_one_epoch
from evaluation.ar import evaluate
from evaluation.metrics import summarize
from models import build_model
from util import checkpoint
from util.config import load
from util.guard import ProtectedStop, ResourceGuard
from util.io import save, seed, sha
from util.speed import install


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
    parser.add_argument(
        "--check_config", "--check-config", dest="check_config", action="store_true"
    )
    return parser


def main(args):
    if args.max_updates is not None and args.max_updates < 1:
        raise ValueError("--max_updates must be positive")
    cfg = load(args.config)
    if args.check_config:
        print("Configuration valid")
        return

    output = Path(args.output_dir)
    if output.exists() and not args.resume:
        raise FileExistsError("Use a new output directory or --resume")
    output.mkdir(parents=True, exist_ok=True)
    guard = ResourceGuard(output)
    torch.set_num_threads(2)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required")
        accelerator = Accelerator(mixed_precision="bf16")
        if accelerator.num_processes != 1:
            raise ValueError("Only single GPU training is supported")

        identity = checkpoint.binding(cfg)
        seed(cfg["train"]["seed"])
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(build_model(cfg)).to(
            accelerator.device
        )
        if args.pretrained:
            metadata = checkpoint.load_model(model, args.pretrained, cfg)
            save(
                output / "INITIALIZATION.json",
                dict(
                    mode="pretrained",
                    checkpoint=str(Path(args.pretrained).resolve()),
                    sha256=sha(args.pretrained),
                    source_step=metadata.get("step"),
                    source_epoch=metadata.get("epoch"),
                    optimizer_scheduler_reset=True,
                ),
            )
            del metadata
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg["train"]["lr"],
            eps=1e-7,
            weight_decay=0.0,
        )
        dataset = build_dataset(cfg, "train")
        train_cfg = cfg["train"]
        microbatches = math.ceil(len(dataset) / train_cfg["batch_size"])
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
                model, optimizer, scheduler, args.resume, identity
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
            save(
                output / "RESUME_RESTORE.json",
                dict(
                    passed=True,
                    step=state.step,
                    epoch=state.epoch,
                    micro_offset=state.offset,
                    model_optimizer_scheduler_rng_exact=True,
                ),
            )
        save(output / "config.json", cfg)
        save(
            output / "SETUP.json",
            dict(
                binding=identity,
                effective_source_batch=train_cfg["batch_size"]
                * train_cfg["grad_accum_every"],
                microbatch=train_cfg["batch_size"],
                accumulation=train_cfg["grad_accum_every"],
                total_scheduler_steps=total_steps,
                mixed_precision=accelerator.mixed_precision,
                max_updates=args.max_updates,
            ),
        )
        initial_step = state.step
        model.train()

        def save_latest():
            guard.check()
            checkpoint.write(
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
            )

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
                save(
                    output / "BOUNDED_COMPLETE.json",
                    dict(
                        step=state.step,
                        updates=state.step - initial_step,
                        formal_training=False,
                    ),
                )
                return

            if (epoch + 1) % train_cfg["val_every_epoch"] == 0:
                result = evaluate(
                    model,
                    cfg,
                    output / f"val_{state.step:06d}",
                    guard,
                    with_ar=train_cfg["val_native_ar"],
                )
                if state.best is None or result["original_loss"] < state.best:
                    state.best = result["original_loss"]
                    checkpoint.save_inference(
                        output / "best.pt", model, epoch + 1, state.step, identity
                    )
            save(
                output / f"epoch_{epoch + 1:03d}.json",
                summarize(
                    state.totals,
                    model.model.loss_weight["reconstruction"]
                    * model.model.recon_weight_factor(),
                    model.model.loss_weight["eos"],
                    model.assignment_weight,
                ),
            )
            state.totals = [0.0] * len(state.totals)
            state.offset = 0
            state.epoch = epoch + 1
            save_latest()
            if (epoch + 1) % train_cfg["checkpoint_every_epoch"] == 0:
                checkpoint.save_inference(
                    output / f"epoch_{epoch + 1:03d}.pt",
                    model,
                    epoch + 1,
                    state.step,
                    identity,
                )

        save(
            output / "COMPLETE.json",
            dict(step=state.step, epochs=train_cfg["epochs"], passed=True),
        )
        accelerator.end_training()
    except ProtectedStop as error:
        save(
            output / "STOPPED.json",
            dict(reason=str(error), checkpoint_may_precede_stop=True),
        )
        raise
    finally:
        guard.close()


if __name__ == "__main__":
    main(get_args_parser().parse_args())
