def build_dataset(cfg, split="train"):
    from .instance_dropout import SceneDataset

    return SceneDataset(cfg, split)
