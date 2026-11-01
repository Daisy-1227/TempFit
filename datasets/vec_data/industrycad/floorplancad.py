import os
import json
from typing import Dict, Any

import torch
from torch.utils.data import Dataset
from datasets.vec_data.industrycad.dataclass_define import (
    SVGData,
    VecData,
    VecDataTransformArgs,
)
from datasets.vec_data.industrycad.transform_utils import (
    to_tensor,
    norm_coords,
    augment_line_args,
    to_vec_data,
)

import os
import glob
import xml.etree.ElementTree as ET

import numpy as np


def scan_dir(dir_path: str, suffix: str, recursive: bool = True) -> list[str]:
    """
    Scan directory for files with specific suffix
    Args:
        dir_path: directory path
        suffix: suffix of the file, e.g. "png, svg, jpg, etc."
        recursive: whether to scan recursively
    Returns:
        list[str]: List of file paths relative to dir_path, containing only files with the specified suffix
    """
    if suffix.startswith("."):
        suffix = suffix[1:]
    if recursive:
        files = glob.glob(os.path.join(dir_path, "**", f"*.{suffix}"), recursive=True)
    else:
        files = glob.glob(os.path.join(dir_path, f"*.{suffix}"))
    return [os.path.relpath(f, dir_path) for f in files]


class IndustryCAD(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str,
        train_transform_args: Dict[str, Any],
        eval_transform_args: Dict[str, Any],
    ):
        self.root_dir = root_dir
        self.split = split
        self.train_transform_args = train_transform_args
        self.eval_transform_args = eval_transform_args
        self.data_dir = os.path.join(root_dir, split)
        self.data_paths = scan_dir(self.data_dir, suffix=".json")
        ### filter 长度太长的样本？ ###
        with open("invalid_files_industry.json", "r") as f:
            invalid_files = json.load(f)
        self.data_paths = [x for x in self.data_paths if x not in invalid_files]
        print(f"{split} dataset length: {len(self.data_paths)}")

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        # ------------- load origin json data ------------ #
        data_path = os.path.join(self.data_dir, self.data_paths[idx])
        json_data = json.load(open(data_path, "r"))
        # ------ dict to data class for easy access ------ #
        svg_data = SVGData(**json_data)
        ### modify the color into zero
        svg_data.colors = [[0, 0, 0] for _ in range(len(svg_data.colors))]
        ### 对semantic_id加1
        svg_data.semantic_ids = [x + 1 for x in svg_data.semantic_ids]
        # -------- transform to line data (tensor) ------- #
        transform_args = self._get_transform_args()
        vec_data = self._transform(svg_data, VecDataTransformArgs(**transform_args))
        vec_data.data_path = data_path
        return vec_data

    def _get_transform_args(self) -> Dict[str, Any]:
        if self.split == "train":
            return self.train_transform_args
        elif self.split == "val" or self.split == "test":
            return self.eval_transform_args
        else:
            raise ValueError(f"Invalid split: {self.split}")

    def _transform(
        self, data: SVGData, transform_args: VecDataTransformArgs
    ) -> VecData:
        # -------------- transform to tensor ------------- #
        svg_data_tensor = to_tensor(data)
        # transform all line coords from [N * [x1, y1, x2, y2]] to [N * 2 * [x, y]] for easier processing
        if svg_data_tensor.coords.shape[-1] == 4:
            svg_data_tensor.coords = svg_data_tensor.coords.reshape(-1, 2, 2)
        # ---------------- normalize lines --------------- #
        svg_data_tensor.coords = norm_coords(
            coords=svg_data_tensor.coords,
            bbox=svg_data_tensor.viewBox,
            min_val=transform_args.norm_range[0],
            max_val=transform_args.norm_range[1],
        )
        # ----------------- augment lines ---------------- #
        svg_data_tensor.coords = augment_line_args(
            coords=svg_data_tensor.coords,
            min_val=transform_args.norm_range[0],
            max_val=transform_args.norm_range[1],
            transform_args=transform_args,
        )
        # ------------ transform to line data ------------ #
        # transform all line coords back to [N * [x1, y1, x2, y2]]
        if len(svg_data_tensor.coords.shape) == 3:
            svg_data_tensor.coords = svg_data_tensor.coords.reshape(-1, 4)
        vec_data = to_vec_data(svg_data_tensor)
        return vec_data

    @staticmethod
    def collate_fn(batch: list[VecData]) -> Dict[str, torch.Tensor]:
        """
        Collate function for FloorPlanCAD dataset, concatenate variable length sequences in the batch.

        Args:

            `batch`: List of `VecData` objects with variable length sequences

        Returns:
            Dict containing concatenated tensors:
                # Inputs
                `coords`: (N1+N2+..., coords_dim) - Concatenated coordinates
                `feats`: (N1+N2+..., feats_dim) - Concatenated features
                `prim_ids`: (N1+N2+...,) - Concatenated primitive IDs
                `layer_ids`: (N1+N2+...,) - Concatenated layer IDs
                `cu_seqlens`: (B+1,) - The cumulative sequence lengths of the sequences in the batch
                # Labels
                `sem_ids`: (N1+N2+...,) - Concatenated semantic IDs
                `inst_ids`: (N1+N2+...,) - Concatenated instance IDs
                `prim_lengths`: (N1+N2+...,) - Concatenated primitive lengths
                `cu_numprims`: (B+1,) - The cumulative number of primitives in each sequence
            B is the batch size,
            N1, N2, ... are lengths of sequences of each element in the batch.

        Raises:
            ValueError: If batch is empty or contains invalid data
        """
        if not batch:
            raise ValueError("Batch cannot be empty")

        # Define fields to extract and pad
        fields = [
            "coords",
            "feats",
            "prim_ids",
            "layer_ids",
            "sem_ids",
            "inst_ids",
            "prim_lengths",
        ]

        # Extract sequences for each field
        sequences = {
            field: [getattr(item, field) for item in batch] for field in fields
        }

        # Calculate the cumulative sequence lengths and the number of primitives
        seq_lens = torch.tensor(
            [len(seq) for seq in sequences["coords"]], dtype=torch.int32
        )
        cu_seqlens = torch.cat(
            [
                torch.tensor([0], dtype=torch.int32),
                torch.cumsum(seq_lens, dim=0, dtype=torch.int32),
            ]
        )
        num_prim = torch.tensor(
            [len(seq) for seq in sequences["sem_ids"]], dtype=torch.int32
        )
        cu_numprims = torch.cat(
            [
                torch.tensor([0], dtype=torch.int32),
                torch.cumsum(num_prim, dim=0, dtype=torch.int32),
            ]
        )

        # Concatenate all sequences
        concat_data = {
            # Inputs
            "coords": torch.cat(sequences["coords"], dim=0),
            "feats": torch.cat(sequences["feats"], dim=0),
            "prim_ids": torch.cat(sequences["prim_ids"], dim=0),
            "layer_ids": torch.cat(sequences["layer_ids"], dim=0),
            "cu_seqlens": cu_seqlens,
            # Labels
            "sem_ids": torch.cat(sequences["sem_ids"], dim=0),
            "inst_ids": torch.cat(sequences["inst_ids"], dim=0),
            "prim_lengths": torch.cat(sequences["prim_lengths"], dim=0),
            "cu_numprims": cu_numprims,
            # Data paths
            "data_paths": [item.data_path for item in batch],
        }

        return concat_data


class FloorPlanCAD(Dataset):

    def __init__(
        self,
        root_dir: str,
        split: str,
        train_transform_args: Dict[str, Any],
        eval_transform_args: Dict[str, Any],
    ):
        self.root_dir = root_dir
        self.split = split
        self.train_transform_args = train_transform_args
        self.eval_transform_args = eval_transform_args
        self.data_dir = os.path.join(root_dir, split)
        self.data_paths = scan_dir(self.data_dir, suffix=".json")
        ### filter 长度太长的样本？ ###
        with open("invalid_files.json", "r") as f:
            invalid_files = json.load(f)
        self.data_paths = [x for x in self.data_paths if x not in invalid_files]
        print(f"{split} dataset length: {len(self.data_paths)}")

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        # ------------- load origin json data ------------ #
        data_path = os.path.join(self.data_dir, self.data_paths[idx])
        json_data = json.load(open(data_path, "r"))
        # ------ dict to data class for easy access ------ #
        svg_data = SVGData(**json_data)
        # -------- transform to line data (tensor) ------- #
        transform_args = self._get_transform_args()
        vec_data = self._transform(svg_data, VecDataTransformArgs(**transform_args))
        vec_data.data_path = data_path
        return vec_data

    def _get_transform_args(self) -> Dict[str, Any]:
        if self.split == "train":
            return self.train_transform_args
        elif self.split == "val" or self.split == "test":
            return self.eval_transform_args
        else:
            raise ValueError(f"Invalid split: {self.split}")

    def _transform(
        self, data: SVGData, transform_args: VecDataTransformArgs
    ) -> VecData:
        # -------------- transform to tensor ------------- #
        svg_data_tensor = to_tensor(data)
        # transform all line coords from [N * [x1, y1, x2, y2]] to [N * 2 * [x, y]] for easier processing
        if svg_data_tensor.coords.shape[-1] == 4:
            svg_data_tensor.coords = svg_data_tensor.coords.reshape(-1, 2, 2)
        # ---------------- normalize lines --------------- #
        svg_data_tensor.coords = norm_coords(
            coords=svg_data_tensor.coords,
            bbox=svg_data_tensor.viewBox,
            min_val=transform_args.norm_range[0],
            max_val=transform_args.norm_range[1],
        )
        # ----------------- augment lines ---------------- #
        svg_data_tensor.coords = augment_line_args(
            coords=svg_data_tensor.coords,
            min_val=transform_args.norm_range[0],
            max_val=transform_args.norm_range[1],
            transform_args=transform_args,
        )
        # ------------ transform to line data ------------ #
        # transform all line coords back to [N * [x1, y1, x2, y2]]
        if len(svg_data_tensor.coords.shape) == 3:
            svg_data_tensor.coords = svg_data_tensor.coords.reshape(-1, 4)
        vec_data = to_vec_data(svg_data_tensor)
        return vec_data

    @staticmethod
    def collate_fn(batch: list[VecData]) -> Dict[str, torch.Tensor]:
        """
        Collate function for FloorPlanCAD dataset, concatenate variable length sequences in the batch.

        Args:

            `batch`: List of `VecData` objects with variable length sequences

        Returns:
            Dict containing concatenated tensors:
                # Inputs
                `coords`: (N1+N2+..., coords_dim) - Concatenated coordinates
                `feats`: (N1+N2+..., feats_dim) - Concatenated features
                `prim_ids`: (N1+N2+...,) - Concatenated primitive IDs
                `layer_ids`: (N1+N2+...,) - Concatenated layer IDs
                `cu_seqlens`: (B+1,) - The cumulative sequence lengths of the sequences in the batch
                # Labels
                `sem_ids`: (N1+N2+...,) - Concatenated semantic IDs
                `inst_ids`: (N1+N2+...,) - Concatenated instance IDs
                `prim_lengths`: (N1+N2+...,) - Concatenated primitive lengths
                `cu_numprims`: (B+1,) - The cumulative number of primitives in each sequence
            B is the batch size,
            N1, N2, ... are lengths of sequences of each element in the batch.

        Raises:
            ValueError: If batch is empty or contains invalid data
        """
        if not batch:
            raise ValueError("Batch cannot be empty")

        # Define fields to extract and pad
        fields = [
            "coords",
            "feats",
            "prim_ids",
            "layer_ids",
            "sem_ids",
            "inst_ids",
            "prim_lengths",
        ]

        # Extract sequences for each field
        sequences = {
            field: [getattr(item, field) for item in batch] for field in fields
        }

        # Calculate the cumulative sequence lengths and the number of primitives
        seq_lens = torch.tensor(
            [len(seq) for seq in sequences["coords"]], dtype=torch.int32
        )
        cu_seqlens = torch.cat(
            [
                torch.tensor([0], dtype=torch.int32),
                torch.cumsum(seq_lens, dim=0, dtype=torch.int32),
            ]
        )
        num_prim = torch.tensor(
            [len(seq) for seq in sequences["sem_ids"]], dtype=torch.int32
        )
        cu_numprims = torch.cat(
            [
                torch.tensor([0], dtype=torch.int32),
                torch.cumsum(num_prim, dim=0, dtype=torch.int32),
            ]
        )

        # Concatenate all sequences
        concat_data = {
            # Inputs
            "coords": torch.cat(sequences["coords"], dim=0),
            "feats": torch.cat(sequences["feats"], dim=0),
            "prim_ids": torch.cat(sequences["prim_ids"], dim=0),
            "layer_ids": torch.cat(sequences["layer_ids"], dim=0),
            "cu_seqlens": cu_seqlens,
            # Labels
            "sem_ids": torch.cat(sequences["sem_ids"], dim=0),
            "inst_ids": torch.cat(sequences["inst_ids"], dim=0),
            "prim_lengths": torch.cat(sequences["prim_lengths"], dim=0),
            "cu_numprims": cu_numprims,
            # Data paths
            "data_paths": [item.data_path for item in batch],
        }

        return concat_data
