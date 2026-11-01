import argparse
from pathlib import Path

import torch

from evaluation.ar import evaluate
from models import build_model
from util.checkpoint import load_model
from util.config import load
from util.guard import ResourceGuard
from util.io import save, seed, sha


def get_args_parser():
    parser = argparse.ArgumentParser("IndustryCAD AR inference")
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", "--output", dest="output_dir", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--beta", type=float)
    parser.add_argument(
        "--check_config", "--check-config", dest="check_config", action="store_true"
    )
    return parser


def main(args):
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    cfg = load(args.config)
    if args.check_config:
        print("Configuration valid")
        return
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(output)
    guard = ResourceGuard(output.parent / "resource-monitor")
    torch.set_num_threads(2)
    try:
        seed(cfg["train"]["seed"])
        model = build_model(cfg)
        metadata = load_model(model, args.checkpoint, cfg)
        step = metadata.get("step")
        del metadata
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
        model.model.set_recon_epoch(cfg["model"]["recon_warmup_epochs"])
        result = evaluate(
            model,
            cfg,
            output,
            guard,
            limit=args.limit,
            beta=args.beta,
            teacher_forced=False,
        )
        save(output / "config.json", cfg)
        save(output / "CHECKPOINT.json", dict(sha256=sha(args.checkpoint), step=step))
        print(result["ar"])
    finally:
        guard.close()


if __name__ == "__main__":
    main(get_args_parser().parse_args())
