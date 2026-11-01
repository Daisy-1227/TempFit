import importlib, importlib.metadata as md, json, platform, sys


def check(gpu=False):
    result = dict(
        python=sys.version,
        executable=sys.executable,
        platform=platform.platform(),
        packages={},
        errors=[],
    )
    for name in [
        "numpy",
        "scipy",
        "yaml",
        "torch",
        "torchvision",
        "accelerate",
        "x_transformers",
        "gateloop_transformer",
        "spconv.pytorch",
        "torch_scatter",
        "pytorch3d",
        "flash_attn",
    ]:
        try:
            module = importlib.import_module(name)
            result["packages"][name] = dict(
                path=module.__file__, version=getattr(module, "__version__", None)
            )
        except Exception as e:
            result["errors"].append(dict(module=name, error=str(e)))
    if not result["errors"]:
        from x_transformers import PrefixDecoder
        import torch

        result["torch_cuda"] = torch.version.cuda
        result["cuda_available"] = torch.cuda.is_available()
        if gpu:
            from util.guard import linux_memory, windows_commit

            result["memory"] = linux_memory()
            if "microsoft" in platform.release().lower():
                result["windows_memory"] = windows_commit()
            if not torch.cuda.is_available():
                result["errors"].append(dict(module="cuda", error="CUDA unavailable"))
            else:
                result["gpu"] = dict(
                    name=torch.cuda.get_device_name(),
                    total_bytes=torch.cuda.get_device_properties(0).total_memory,
                )
    result["passed"] = not result["errors"]
    return result


if __name__ == "__main__":
    import argparse
    from util.io import save

    parser = argparse.ArgumentParser("IndustryCAD environment check")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = check(args.gpu)
    if args.output:
        save(args.output, result)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
