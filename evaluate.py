import argparse

from evaluation.scoring import evaluate_predictions
from util.config import load


def get_args_parser():
    parser = argparse.ArgumentParser("IndustryCAD AR evaluation")
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output_dir", "--output", dest="output_dir", required=True)
    parser.add_argument(
        "--allow_partial", "--allow-partial", dest="allow_partial", action="store_true"
    )
    return parser


def main(args):
    cfg = load(args.config)
    summary = evaluate_predictions(
        cfg, args.predictions, args.output_dir, args.allow_partial
    )
    for name, value in summary["ar"].items():
        print(
            f"{name}: {value:.6%}"
            if name.endswith("_rate") and value is not None
            else f"{name}: {value}"
        )


if __name__ == "__main__":
    main(get_args_parser().parse_args())
