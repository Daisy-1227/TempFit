import copy
import json
import os

import numpy as np
from scipy.linalg import polar
from scipy.spatial.transform import Rotation
import open3d as o3d
import torch
from torch.utils.data import Dataset

from models.primitive.utils import exists
from models.primitive.utils.logger import print_log

typeid2type = {
    1101002001034001: 0,
    1101002001034010: 1,
    1101002001034002: 2,
}

typeid2name = {
    1101002001034001: "SM_GR_BS_CubeBevel_001.ply",
    1101002001034010: "SM_GR_BS_SphereSharp_001.ply",
    1101002001034002: "SM_GR_BS_CylinderSharp_001.ply",
}


def create_dataset(cfg_dataset):
    kwargs = cfg_dataset
    name = kwargs.pop("name")
    dataset = get_dataset(name)(**kwargs)
    print_log(f"Dataset '{name}' init: kwargs={kwargs}, len={len(dataset)}")
    return dataset


def get_dataset(name):
    return {
        "base": TrainDataset,
    }[name]


SHAPE_CODE = {
    "CubeBevel": 0,
    "SphereSharp": 1,
    "CylinderSharp": 2,
}


class TrainDataset(Dataset):
    def __init__(
        self,
        pc_dir,
        bs_dir,
        json_dir,
        max_length=144,
        range_scale=[0, 1],
        range_rotation=[-180, 180],
        range_translation=[-1, 1],
        rotation_type="euler",
        pc_format="pc",
        max_primitive_len=144,
        pad_id=-1,
    ):
        self.data_filename = os.listdir(pc_dir)

        self.pc_dir = pc_dir

        #### Add for GT primitive
        self.json_dir = json_dir

        self.max_length = max_length
        self.range_scale = range_scale
        self.range_rotation = range_rotation
        self.range_translation = range_translation
        self.rotation_type = rotation_type
        self.pc_format = pc_format

        with open(
            os.path.join(bs_dir, "basic_shapes.json"), "r", encoding="utf-8"
        ) as f:
            basic_shapes = json.load(f)

        self.typeid_map = {
            1101002001034001: "CubeBevel",
            1101002001034010: "SphereSharp",
            1101002001034002: "CylinderSharp",
        }
        self.typeid2name = typeid2name
        self.typeid2type = typeid2type
        self.max_primitive_len = max_primitive_len
        self.pad_id = pad_id

    def __len__(self):
        return len(self.data_filename)

    def __getitem__(self, idx):
        ### Load the input pc
        pc_file = os.path.join(self.pc_dir, self.data_filename[idx])
        pc = o3d.io.read_point_cloud(pc_file)

        model_data = {}

        points = torch.from_numpy(np.asarray(pc.points)).float()
        colors = torch.from_numpy(np.asarray(pc.colors)).float()
        normals = torch.from_numpy(np.asarray(pc.normals)).float()
        if self.pc_format == "pc":
            model_data["pc"] = torch.concatenate([points, colors], dim=-1).T
        elif self.pc_format == "pn":
            model_data["pc"] = torch.concatenate([points, normals], dim=-1)
        elif self.pc_format == "pcn":
            model_data["pc"] = torch.concatenate([points, colors, normals], dim=-1)
        else:
            raise ValueError(f"invalid pc_format: {self.pc_format}")

        ### Load the GT primitive
        json_file = os.path.join(
            self.json_dir, self.data_filename[idx].replace(".ply", ".json")
        )
        with open(json_file, "r", encoding="utf-8") as f:
            primitives_json = json.load(f)
        primitives = primitives_json["group"]
        primitive_types = []
        primitive_scale = []
        primitive_translation = []
        primitive_rotation = []
        for primitive in primitives:
            primitive_types.append(
                torch.tensor(self.typeid2type[primitive["type_id"]]).long()
            )
            primitive_scale.append(torch.tensor(primitive["data"]["scale"]).float())
            primitive_translation.append(
                torch.tensor(primitive["data"]["location"]).float()
            )
            primitive_rotation.append(
                torch.tensor(primitive["data"]["rotation_euler"]).float()
            )
        primitive_types = torch.stack(primitive_types, dim=0)
        primitive_scale = torch.stack(primitive_scale, dim=0)
        primitive_translation = torch.stack(primitive_translation, dim=0)
        primitive_rotation = torch.stack(primitive_rotation, dim=0)

        model_data["primitive_type"] = primitive_types
        model_data["primitive_scale"] = primitive_scale
        model_data["primitive_translation"] = primitive_translation
        model_data["primitive_rotation"] = primitive_rotation

        return model_data
