def indices(names):
    return dict(zip(names, range(len(names))))


def preprocessing(names):
    from pathlib import Path
    from util.io import scene_seed

    return {name: scene_seed(Path(name).stem) for name in names}
