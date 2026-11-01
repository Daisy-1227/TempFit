def indices(manifest):
    names = manifest["val"]
    mapping = manifest.get("evaluation_seed_indices")
    if mapping is None:
        return dict(zip(names, range(len(names))))
    if set(mapping) != set(names):
        raise ValueError("evaluation_seed_indices must cover exactly the val filenames")
    values = list(mapping.values())
    if any(type(v) is not int or v < 0 for v in values) or len(set(values)) != len(
        values
    ):
        raise ValueError("evaluation_seed_indices requires unique nonnegative integers")
    return mapping


def preprocessing(manifest):
    from pathlib import Path
    from util.io import scene_seed

    names = manifest["val"]
    mapping = manifest.get("preprocessing_seeds")
    if mapping is None:
        return {name: scene_seed(Path(name).stem) for name in names}
    if set(mapping) != set(names):
        raise ValueError("preprocessing_seeds must cover exactly the val filenames")
    if any(
        type(value) is not int or not 0 <= value < 2**32 for value in mapping.values()
    ):
        raise ValueError("preprocessing_seeds requires unsigned 32-bit integers")
    return mapping
