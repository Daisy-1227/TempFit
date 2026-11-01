__all__ = ["build_dataset", "register_dataset"]

import json
import yaml
import logging
import importlib
from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple

from torch.utils.data import Dataset

logger = logging.getLogger("transformers")


class DatasetRegistry:
    """Dataset registry for automatic dataset registration and building"""

    _registry: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a new dataset builder"""

        def wrapper(build_fn: Callable):
            if name in cls._registry:
                logger.warning(f"Dataset {name} already registered, overwriting...")
            cls._registry[name] = build_fn
            return build_fn

        return wrapper

    @classmethod
    def get_build_fn(cls, name: str) -> Callable:
        """get the build function of the registered dataset"""
        if name not in cls._registry:
            raise ValueError(f"Dataset {name} not found in registry")
        return cls._registry[name]


register_dataset = DatasetRegistry.register


@dataclass
class DatasetSplits:
    train: Dataset
    val: Dataset
    test: Dataset


@dataclass
class DataArguments:
    dataset_name: str = field(default="")
    dataset_args: dict = field(default_factory=dict)

    def __str__(self):
        return json.dumps(self.__dict__, indent=4)

    def __repr__(self):
        return self.__str__()


def build_dataset(data_args_path: str) -> Tuple[DatasetSplits, Callable]:
    with open(data_args_path, "r") as f:
        data_args = yaml.safe_load(f)
    data_args = DataArguments(**data_args)
    logger.info(f"Data Arguments: {data_args}")

    try:
        importlib.import_module("data." + data_args.dataset_name)
        build_fn = DatasetRegistry.get_build_fn(data_args.dataset_name)
        return build_fn(data_args.dataset_args)
    except ImportError as e:
        raise ImportError(f"Failed to import dataset module: {e}")
