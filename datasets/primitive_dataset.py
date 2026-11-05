import copy
import hashlib
import json
import os
import pickle
import time
import numpy as np
import math

try:
    import open3d as o3d
except ImportError:
    o3d = None
import torch
from PIL import Image
from scipy.linalg import polar
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from torch import is_tensor
from torch.nn.utils.rnn import pad_sequence
from hilbertcurve.hilbertcurve import HilbertCurve  # pip install hilbertcurve
from models.primitive.utils import exists, discretize
from models.primitive.utils.logger import print_log
from functools import partial
from tqdm import tqdm
from datasets.vec_data.industrycad.dataclass_define import VecData, VecDataTransformArgs

# from .vec_data.floorplancad.dataclass_define import SVGData
from datasets.dataclass import SVGData
from datasets.assets import template_id
from typing import Dict, Any
from datasets.transform_utils import (
    to_tensor,
    norm_coords,
    augment_line_args,
    to_vec_data,
    svg_to_instance,
    instance_to_svg,
)
from datasets.dataclass import InstanceData
from collections.abc import Sequence, Mapping
from typing import List


# Legacy compatibility is restricted to the exact verified template catalog.
_LEGACY_MAPPING_SHA256 = (
    "8cfceb0c2a6248419a5046e124712bc237f0097ce6bf210682d6c1830149a968"
)
_LEGACY_FLIP_VARIANT_IDS = (
    1,
    2,
    4,
    7,
    8,
    10,
    11,
    12,
    13,
    14,
    17,
    18,
    19,
    20,
    22,
    23,
    24,
    25,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    39,
    40,
    42,
    46,
    53,
    54,
    56,
    58,
    63,
    64,
    65,
    67,
    70,
    72,
)

flip_invariant_primitive_id = []


def load_flip_invariant_ids(mapping_path):
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    ids = sorted(mapping["primitive_name2id"].values())
    if any(type(i) is not int for i in ids) or ids != list(range(len(ids))):
        raise ValueError("Primitive IDs must be unique and contiguous from zero")
    if mapping["primitive_id2name"] != {
        str(i): name for name, i in mapping["primitive_name2id"].items()
    }:
        raise ValueError("Primitive name mappings are not inverse bijections")
    variants = mapping.get("flip_variant_primitive_ids")
    if variants is None:
        canonical = json.dumps(
            mapping, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        if hashlib.sha256(canonical.encode()).hexdigest() != _LEGACY_MAPPING_SHA256:
            raise ValueError("Template mapping must specify flip_variant_primitive_ids")
        variants = _LEGACY_FLIP_VARIANT_IDS
    if (
        not isinstance(variants, (list, tuple))
        or any(type(i) is not int or i not in ids for i in variants)
        or len(set(variants)) != len(variants)
    ):
        raise ValueError("Invalid flip_variant_primitive_ids")
    return [i for i in ids if i not in variants]


class ToTensor(object):
    def __call__(self, data):
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, str):
            # note that str is also a kind of sequence, judgement should before sequence
            return data
        elif isinstance(data, int):
            return torch.LongTensor([data])
        elif isinstance(data, float):
            return torch.FloatTensor([data])
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, bool):
            return torch.from_numpy(data)
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.integer):
            return torch.from_numpy(data).long()
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.floating):
            return torch.from_numpy(data).float()
        elif isinstance(data, Mapping):
            result = {sub_key: self(item) for sub_key, item in data.items()}
            return result
        elif isinstance(data, Sequence):
            result = [self(item) for item in data]
            return result
        else:
            raise TypeError(f"type {type(data)} cannot be converted to tensor.")


def normalize_translation(
    translation_value, original_range=(0, 140), target_range=(-1, 1)
):
    """
    将translation值从原始范围映射到目标范围

    Args:
        translation_value: 原始translation值，可以是单个值、list或包含translation值的数组/张量
        original_range: 原始值域范围，默认为(0, 140)
        target_range: 目标值域范围，默认为(-1, 1)

    Returns:
        映射后的值，保持输入的数据结构
    """
    min_orig, max_orig = original_range
    min_target, max_target = target_range

    # 处理list类型的输入
    if isinstance(translation_value, list):
        normalized = [
            (val - min_orig) / (max_orig - min_orig) * (max_target - min_target)
            + min_target
            for val in translation_value
        ]
        return normalized

    # 处理numpy数组或torch张量
    if hasattr(translation_value, "__iter__") and not isinstance(
        translation_value, str
    ):
        # 线性映射公式
        normalized = (translation_value - min_orig) / (max_orig - min_orig) * (
            max_target - min_target
        ) + min_target
        return normalized

    # 处理单个数值
    normalized = (translation_value - min_orig) / (max_orig - min_orig) * (
        max_target - min_target
    ) + min_target
    return normalized


def denormalize_translation(
    normalized_value, original_range=(0, 140), target_range=(-1, 1)
):
    """
    将normalized translation值从目标范围映射回原始范围（逆映射）

    Args:
        normalized_value: 已归一化的translation值，可以是单个值、list或包含归一化值的数组/张量
        original_range: 原始值域范围，默认为(0, 140)
        target_range: 目标值域范围，默认为(-1, 1)

    Returns:
        逆映射后的原始范围值，保持输入的数据结构
    """
    min_orig, max_orig = original_range
    min_target, max_target = target_range

    # 处理list类型的输入
    if isinstance(normalized_value, list):
        original = [
            (val - min_target) / (max_target - min_target) * (max_orig - min_orig)
            + min_orig
            for val in normalized_value
        ]
        return original

    # 处理numpy数组或torch张量
    if hasattr(normalized_value, "__iter__") and not isinstance(normalized_value, str):
        # 逆映射公式
        original = (normalized_value - min_target) / (max_target - min_target) * (
            max_orig - min_orig
        ) + min_orig
        return original

    # 处理单个数值
    original = (normalized_value - min_target) / (max_target - min_target) * (
        max_orig - min_orig
    ) + min_orig
    return original


def create_dataset(cfg_dataset):
    kwargs = cfg_dataset
    name = kwargs.pop("name")
    dataset = get_dataset(name)(**kwargs)
    print_log(f"Dataset '{name}' init: kwargs={kwargs}, len={len(dataset)}")
    return dataset


def get_dataset(name):
    return {
        "base": PrimitiveDataset,
    }[name]


SHAPE_CODE = {
    "CubeBevel": 0,
    "SphereSharp": 1,
    "CylinderSharp": 2,
}


def load_primitive_mapping(primitive_mapping_file: str):
    with open(primitive_mapping_file, "rb") as f:
        primitive_mapping = pickle.load(f)
    return primitive_mapping


# primitive_mapping = load_primitive_mapping("<external-path-required>")
# path_to_id = primitive_mapping['path_to_id']
# #### get flip_invariant primitive_id
# with open("<external-path-required>", "rb") as f:
#     primitive_order_dict = pickle.load(f)
# flip_invariant_primitive_id = []
# flip_invariant_primitive_files = []
# for semantic_id in ori_semantic_lst:
#     sub_primitive_order_dict = primitive_order_dict[f"semantic_{semantic_id}"]
#     for primitive_file in sub_primitive_order_dict.keys():
#         flip_invariant = sub_primitive_order_dict[primitive_file]['flip_invariant']
#         if flip_invariant:
#             flip_invariant_primitive_id.append(path_to_id[primitive_file])
#             flip_invariant_primitive_files.append(primitive_file)


def hilbert_order(xy, n_bits=10):
    # xy: (N,2) in [0,1]
    grid = 2**n_bits - 1
    ix = np.clip(np.floor(xy[:, 0] * grid), 0, grid).astype(int)
    iy = np.clip(np.floor(xy[:, 1] * grid), 0, grid).astype(int)
    hc = HilbertCurve(n_bits, 2)
    h = np.array(hc.distances_from_points(np.stack([ix, iy], 1).tolist()))
    return np.argsort(h)


def load_svg_w_bg_semantic(
    json_file: str,
    normalize_coord: bool,
    bg_pred=False,
    scene_bound=20000.0,
    keep_ratio=0.1,
    min_keep=100,
):
    data = json.load(open(json_file))
    #### directly handle with primitive_input
    primitive_args = data["primitive_args"]
    primitive_lengths = data["primitive_lengths"]
    # primitive_commands = data['primitive_commands']
    # primitive_commands = np.zeros((len(primitive_args)), dtype=np.int32)
    primitive_num = len(primitive_args)
    primitive_coords = []
    primitive_feats = []
    primitive_offsets = []
    primitive_bboxes = []
    start_idx = 0
    # primitive_offsets.append(start_idx)
    for _ in range(primitive_num):
        primitive_args_i = np.concatenate(primitive_args[_], axis=0).reshape(-1, 2)
        ### 这里的范围就是[0, 20000]
        if normalize_coord:
            # primitive_args_i = primitive_args_i.clip(0, scene_bound)
            primitive_args_i = primitive_args_i
            #### 不做clipping，保持完整的instance
        else:
            primitive_args_i = primitive_args_i.clip(0, scene_bound) / scene_bound

        primitive_lengths_i = primitive_lengths[_]
        # primitive_commands_i = primitive_commands[_]
        primitive_commands_i = np.zeros((len(primitive_args_i)), dtype=np.int32)
        primitive_num_i = len(primitive_args_i)
        primitive_coords_i = np.zeros((primitive_num_i, 3))
        primitive_feats_i = np.zeros((primitive_num_i, 6))
        # primitive_coords_i[:, 0] = np.mean(primitive_args_i[:, 0::2], axis=1)
        # primitive_coords_i[:, 1] = np.mean(primitive_args_i[:, 1::2], axis=1)
        primitive_coords_i[:, 0] = primitive_args_i[:, 0]
        primitive_coords_i[:, 1] = primitive_args_i[:, 1]
        primitive_coords_i[:, 2] = np.zeros((primitive_num_i,))
        primitive_feats_i[:, 0] = (
            np.arctan(primitive_coords_i[:, 1] / (primitive_coords_i[:, 0] + 1e-8))
            / math.pi
        )
        # primitive_feats_i[:, 1] = np.array(primitive_lengths_i)[:, 0].clip(0, scene_bound) / scene_bound
        primitive_feats_i[:, 1] = np.array(primitive_lengths_i)[:, 0] / scene_bound
        primitive_feats_i[:, 2:] = np.eye(4)[primitive_commands_i]
        primitive_coords_i = primitive_coords_i.astype(np.float32)
        primitive_feats_i = primitive_feats_i.astype(np.float32)

        # ######### 对太大的primitive_args进行降采样;
        # sub_sample_num = int(primitive_num_i * keep_ratio)
        # if sub_sample_num <= 1:
        #     primitive_num_i = primitive_num_i
        #     primitive_coords_i = primitive_coords_i
        #     primitive_feats_i = primitive_feats_i
        # else:
        #     idxs = np.random.choice(primitive_num_i, sub_sample_num, replace=False)
        #     primitive_num_i = sub_sample_num
        #     primitive_coords_i = primitive_coords_i[idxs]
        #     primitive_feats_i = primitive_feats_i[idxs]

        primitive_boxes_w = np.max(primitive_coords_i[:, 0]) - np.min(
            primitive_coords_i[:, 0]
        )
        primitive_boxes_h = np.max(primitive_coords_i[:, 1]) - np.min(
            primitive_coords_i[:, 1]
        )
        primitive_boxes = np.array([primitive_boxes_w, primitive_boxes_h])
        primitive_bboxes.append(primitive_boxes)

        primitive_coords.append(primitive_coords_i)
        primitive_feats.append(primitive_feats_i)
        primitive_offsets.append([start_idx, start_idx + primitive_num_i])

        start_idx += primitive_num_i

    return primitive_coords, primitive_feats, primitive_offsets, primitive_bboxes


def load_svg_w_bg(json_file: str, normalize_coord: bool):
    data = json.load(open(json_file))
    ### first handle bg_inputs
    bg_args = data["bg_args"]
    bg_lengths = data["bg_lengths"]
    bg_commands = data["bg_commands"]
    bg_args = np.concatenate(bg_args, axis=0).reshape(-1, 8)
    if normalize_coord:
        bg_args = bg_args.clip(0, 140)
    else:
        bg_args = bg_args.clip(0, 140) / 140.0

    num = bg_args.shape[0]
    # 坐标处理
    bg_coord = np.zeros((num, 3))
    bg_coord_x = np.mean(bg_args[:, 0::2], axis=1)
    bg_coord_y = np.mean(bg_args[:, 1::2], axis=1)
    bg_coord_z = np.zeros((num,))
    bg_coord[:num, 0] = bg_coord_x
    bg_coord[:num, 1] = bg_coord_y
    bg_coord[:num, 2] = bg_coord_z
    # feat 处理
    bg_feat = np.zeros((num, 6))
    bg_arc = np.arctan(bg_coord_y / (bg_coord_x + 1e-8)) / math.pi
    bg_lens = np.array(bg_lengths).clip(0, 140) / 140.0
    bg_ctype = np.eye(4)[bg_commands]
    bg_feat[:num, 0] = bg_arc
    bg_feat[:num, 1] = bg_lens
    bg_feat[:num, 2:] = bg_ctype

    bg_coord = bg_coord.astype(np.float32)
    bg_feat = bg_feat.astype(np.float32)

    ### then handle primitive_inputs
    primitive_args = data["primitive_args"]
    primitive_lengths = data["primitive_lengths"]
    primitive_commands = data["primitive_commands"]
    primitive_num = len(primitive_args)
    primitive_coords = []
    primitive_feats = []
    primitive_offsets = []
    start_idx = 0
    # primitive_offsets.append(start_idx)
    for _ in range(primitive_num):
        primitive_args_i = np.concatenate(primitive_args[_], axis=0).reshape(-1, 8)
        if normalize_coord:
            primitive_args_i = primitive_args_i.clip(0, 140)
        else:
            primitive_args_i = primitive_args_i.clip(0, 140) / 140.0

        primitive_lengths_i = primitive_lengths[_]
        primitive_commands_i = primitive_commands[_]
        primitive_num_i = len(primitive_args_i)

        primitive_coords_i = np.zeros((primitive_num_i, 3))
        primitive_feats_i = np.zeros((primitive_num_i, 6))
        primitive_coords_i[:, 0] = np.mean(primitive_args_i[:, 0::2], axis=1)
        primitive_coords_i[:, 1] = np.mean(primitive_args_i[:, 1::2], axis=1)
        primitive_coords_i[:, 2] = np.zeros((primitive_num_i,))
        primitive_feats_i[:, 0] = (
            np.arctan(primitive_coords_i[:, 1] / (primitive_coords_i[:, 0] + 1e-8))
            / math.pi
        )
        primitive_feats_i[:, 1] = np.array(primitive_lengths_i).clip(0, 140) / 140.0
        primitive_feats_i[:, 2:] = np.eye(4)[primitive_commands_i]
        primitive_coords_i = primitive_coords_i.astype(np.float32)
        primitive_feats_i = primitive_feats_i.astype(np.float32)
        primitive_coords.append(primitive_coords_i)
        primitive_feats.append(primitive_feats_i)
        primitive_offsets.append([start_idx, start_idx + primitive_num_i])
        start_idx += primitive_num_i

    return bg_coord, bg_feat, primitive_coords, primitive_feats, primitive_offsets


def load_svg(json_file: str):
    data = json.load(open(json_file))

    data["args"] = np.concatenate(data["args"], axis=0)

    args = np.array(data["args"]).reshape(-1, 8) / 140
    num = args.shape[0]

    # 坐标处理
    coord = np.zeros((num, 3))
    coord_x = np.mean(args[:, 0::2], axis=1)
    coord_y = np.mean(args[:, 1::2], axis=1)
    coord_z = np.zeros((num,))

    coord[:num, 0] = coord_x
    coord[:num, 1] = coord_y
    coord[:num, 2] = coord_z

    # 特征处理
    feat = np.zeros((num, 6))
    arc = np.arctan(coord_y / (coord_x + 1e-8)) / math.pi
    lens = np.array(data["lengths"]).clip(0, 140) / 140
    ctype = np.eye(4)[data["commands"]]

    feat[:num, 0] = arc
    feat[:num, 1] = lens
    feat[:num, 2:] = ctype

    # 语义标签处理
    semanticIds = np.full_like(coord[:, 0], 35)  # bg sem id = 35
    seg = np.array(data["semanticIds"])
    semanticIds[:num] = seg
    semanticIds = semanticIds.astype(np.int32)

    # 实例标签处理
    instanceIds = np.full_like(coord[:, 0], -1)  # stuff id = -1
    ins = np.array(data["instanceIds"])
    instanceIds[:num] = ins
    instanceIds = instanceIds.astype(np.int32)

    return coord, feat, semanticIds, instanceIds


def load_transformation(transformation_file: str):
    with open(transformation_file, "r", encoding="utf-8") as f:
        transformation = json.load(f)
    return transformation


class PrimitiveDataset(Dataset):
    def __init__(
        self,
        svg_dir: str,
        gt_dir: str,
        primitive_dir: str,
        range_scale=[0, 1],
        range_rotation=[-180, 180],
        range_translation=[-1, 1],
        split="train",
        rotation_type="euler",
        transform=None,
        instance_transform=None,  # 新增: instance-level 增强配置
        tuning=False,
        normalize_coord=False,
        reparam_rotation=False,
        scale_inv=False,
        filter_empty=False,
        repeat=1,
        hilbert_sort=False,
        bg_pred=False,
        scene_bound=30000.0,
        ##### Add for primitive_lib downsample
        keep_ratio=0.1,
        min_keep=100,
        grid_size=0.002,
        debug_list=None,
        continuous_range_translation=[-1, 1],
        num_discrete_translation=None,
        bbox_sort=False,
        train_transform_args: Dict[str, Any] = None,
        eval_transform_args: Dict[str, Any] = None,
        filter_translation: bool = False,
        return_size: bool = False,
        adjust_coords: bool = False,
        filter_size: bool = False,
        use_cache_pkl: bool = False,
        cache_dir: str = None,
        invalid_scene_file: str = None,
        shuffle: bool = True,
        demo: bool = False,
        # Read-only diagnostics for the Industry instance-scene reconstruction
        # contract. Disabled by default so existing training samples/collation
        # are byte-for-byte unchanged.
        recon_audit: bool = False,
        # Opt-in production payload for the Industry full-instance
        # reconstruction loss.  This reuses the audit bookkeeping internally,
        # but returns only bounded post-grid source segments and compact masks.
        return_recon_data: bool = False,
        recon_source_limit: int = 96,
        recon_translation_tolerance_bins: float = 0.5,
        recon_include_partial: bool = False,
        # Preserve each retained instance's geometry immediately before
        # VecClipping.  By default this is enabled whenever partial instances
        # participate in reconstruction, so full and partial objects share the
        # same pre-clip/full-geometry target.  The payload remains loss-only
        # and is never exposed to the conditioner.
        recon_preclip_full_source: bool | None = None,
        # Mirror of the model-side `no_color_layer` flag. When True, both the
        # primitive_lib loader and __getitem__'s scene loader zero out colors
        # and layer_ids at the data source — so any downstream code (CopyPaste,
        # transforms, vec_data) sees zeroed metadata. Combined with the
        # model-side flag this gives belt-and-suspenders zero coverage.
        no_color_layer: bool = False,
        # Controlled experiments must not silently replace a failed crop with
        # its successor.  Training may retry the same stochastic augmentation;
        # deterministic evaluation should use "raise" and a valid manifest.
        invalid_sample_policy: str = "successor",
        invalid_sample_max_retries: int = 8,
        # Historical GT uses a template-relative multiplier.  The opt-in
        # absolute representation predicts the complete object's local x/y
        # extent in normalized scene coordinates.  Conversion happens after
        # every augmentation, so existing transform semantics remain intact.
        scale_parameterization: str = "relative",
    ):
        from models.primitive.scale_parameterization import (
            validate_scale_parameterization,
        )

        self.scale_parameterization = validate_scale_parameterization(
            scale_parameterization
        )
        self.no_color_layer = no_color_layer
        self.train_trnasform_args = train_transform_args
        self.eval_transform_args = eval_transform_args
        self.return_size = return_size
        self.adjust_coords = adjust_coords
        self.keep_ratio = keep_ratio
        self.min_keep = min_keep
        self.grid_size = grid_size
        self.bbox_sort = bbox_sort
        self.scene_bound = scene_bound
        self.demo = demo
        self.recon_audit = bool(recon_audit)
        self.return_recon_data = bool(return_recon_data)
        self.recon_source_limit = int(recon_source_limit)
        self.recon_translation_tolerance_bins = float(recon_translation_tolerance_bins)
        self.recon_include_partial = bool(recon_include_partial)
        if recon_preclip_full_source is None:
            recon_preclip_full_source = (
                self.return_recon_data and self.recon_include_partial
            )
        self.recon_preclip_full_source = bool(recon_preclip_full_source)
        if self.recon_preclip_full_source and not self.return_recon_data:
            raise ValueError(
                "recon_preclip_full_source requires return_recon_data=true"
            )
        if invalid_sample_policy not in {
            "successor",
            "retry_same",
            "retry_same_then_deterministic",
            "raise",
        }:
            raise ValueError(
                "invalid_sample_policy must be successor/retry_same/"
                "retry_same_then_deterministic/raise, "
                f"got {invalid_sample_policy!r}"
            )
        if int(invalid_sample_max_retries) < 0:
            raise ValueError("invalid_sample_max_retries must be non-negative")
        self.invalid_sample_policy = str(invalid_sample_policy)
        self.invalid_sample_max_retries = int(invalid_sample_max_retries)
        if self.recon_source_limit <= 0:
            raise ValueError("recon_source_limit must be positive")
        if self.recon_translation_tolerance_bins < 0:
            raise ValueError("recon_translation_tolerance_bins must be non-negative")
        self._svg_dir_root = svg_dir
        self._gt_dir_root = gt_dir
        self._cache_dir_root = cache_dir if cache_dir is not None else gt_dir
        self._demo_per_file_split = None

        _gt_subdir = "gt"

        def _filenames_for_split(sp: str):
            _svg = os.path.join(svg_dir, sp, "svg")
            _gt = os.path.join(gt_dir, sp, _gt_subdir)
            if cache_dir is not None:
                _cache = os.path.join(cache_dir, sp, "cache_pkl")
            else:
                _cache = os.path.join(gt_dir, sp, "cache_pkl")
            if not os.path.isdir(_gt):
                return []
            gt_files_sp = {x for x in os.listdir(_gt) if x.endswith(".json")}
            svg_files_sp = (
                {x for x in os.listdir(_svg) if x.endswith(".json")}
                if os.path.isdir(_svg)
                else set()
            )
            if self.use_cache_pkl and os.path.isdir(_cache):
                cache_json_names = {
                    x.replace(".pkl", ".json")
                    for x in os.listdir(_cache)
                    if x.endswith(".pkl")
                }
                available_names = cache_json_names.union(svg_files_sp)
                return sorted(list(gt_files_sp.intersection(available_names)))
            return (
                sorted(list(gt_files_sp.intersection(svg_files_sp)))
                if len(svg_files_sp) > 0
                else sorted(list(gt_files_sp))
            )

        # self.svg_dir = f"{svg_dir}/{split}/svg_instance_lines"
        self.svg_dir = f"{svg_dir}/{split}/svg"
        self.gt_dir = f"{gt_dir}/{split}/gt"
        self.use_cache_pkl = use_cache_pkl
        if cache_dir is not None:
            self.cache_dir = f"{cache_dir}/{split}/cache_pkl"
        else:
            self.cache_dir = f"{gt_dir}/{split}/cache_pkl"

        self.primitive_dir = primitive_dir
        if self.demo:
            self._demo_per_file_split = []
            self.data_filename = []
            for sp in ("train", "test"):
                for name in _filenames_for_split(sp):
                    self.data_filename.append(name)
                    self._demo_per_file_split.append(sp)
            print(f"[demo] merged train+test samples: n={len(self.data_filename)}")
            if self.use_cache_pkl:
                print(
                    f"[cache_pkl] enabled; per-sample split resolved in __getitem__, root={self._cache_dir_root}"
                )
        else:
            gt_files = {x for x in os.listdir(self.gt_dir) if x.endswith(".json")}
            svg_files = (
                {x for x in os.listdir(self.svg_dir) if x.endswith(".json")}
                if os.path.isdir(self.svg_dir)
                else set()
            )
            if self.use_cache_pkl and os.path.isdir(self.cache_dir):
                cache_json_names = {
                    x.replace(".pkl", ".json")
                    for x in os.listdir(self.cache_dir)
                    if x.endswith(".pkl")
                }
                available_names = cache_json_names.union(svg_files)
                self.data_filename = sorted(
                    list(gt_files.intersection(available_names))
                )
                print(
                    f"[cache_pkl] enabled, cache_dir={self.cache_dir}, matched_files={len(self.data_filename)}"
                )
            else:
                self.data_filename = (
                    sorted(list(gt_files.intersection(svg_files)))
                    if len(svg_files) > 0
                    else sorted(list(gt_files))
                )
        self.filter_empty = filter_empty
        if self.filter_empty:
            print(f"filter before: {len(self.data_filename)}")
            empty_set = []
            if self.demo and self._demo_per_file_split is not None:
                for json_file, sp in zip(self.data_filename, self._demo_per_file_split):
                    _gt = os.path.join(gt_dir, sp, _gt_subdir)
                    with open(os.path.join(_gt, json_file), "r", encoding="utf-8") as f:
                        primitives_json = json.load(f)
                    if len(primitives_json) == 0:
                        empty_set.append((json_file, sp))
                keep = [
                    (f, s)
                    for f, s in zip(self.data_filename, self._demo_per_file_split)
                    if (f, s) not in empty_set
                ]
                self.data_filename = [f for f, s in keep]
                self._demo_per_file_split = [s for f, s in keep]
            else:
                for json_file in tqdm(self.data_filename):
                    with open(
                        os.path.join(self.gt_dir, json_file), "r", encoding="utf-8"
                    ) as f:
                        primitives_json = json.load(f)
                    if len(primitives_json) == 0:
                        empty_set.append(json_file)
                self.data_filename = [
                    x for x in self.data_filename if x not in empty_set
                ]
            print(f"filter after: {len(self.data_filename)}")
        self.invalid_scene_file = invalid_scene_file
        if self.invalid_scene_file is not None:
            with open(self.invalid_scene_file, "r", encoding="utf-8") as f:
                invalid_scene_names = f.read().splitlines()
            invalid_set = set(invalid_scene_names)
            if self.demo and self._demo_per_file_split is not None:
                keep = [
                    (f, s)
                    for f, s in zip(self.data_filename, self._demo_per_file_split)
                    if f not in invalid_set
                ]
                self.data_filename = [f for f, s in keep]
                self._demo_per_file_split = [s for f, s in keep]
            else:
                self.data_filename = [
                    x for x in self.data_filename if x not in invalid_set
                ]
            print(f"filter after: {len(self.data_filename)}")
        self.reparam_rotation = reparam_rotation
        self.scale_inv = scale_inv
        self.hilbert_sort = hilbert_sort
        # self.flip_invariant_primitive_id = flip_invariant_primitive_id
        # self.flip_variant_primitive_id = flip_variant_primitive_id
        self.flip_invariant_primitive_id = load_flip_invariant_ids(
            os.path.join(svg_dir, "primitive_mapping.json")
        )
        if (
            flip_invariant_primitive_id
            and flip_invariant_primitive_id != self.flip_invariant_primitive_id
        ):
            raise ValueError(
                "Different primitive mappings in one process are not supported"
            )
        flip_invariant_primitive_id[:] = self.flip_invariant_primitive_id
        self.bg_pred = bg_pred
        self.filter_translation = filter_translation
        self.filter_size = filter_size
        ##### Adjust for model tuning
        len_data = len(self.data_filename) * 0.2
        if tuning:
            n_keep = int(len_data)
            self.data_filename = self.data_filename[:n_keep]
            if self.demo and self._demo_per_file_split is not None:
                self._demo_per_file_split = self._demo_per_file_split[:n_keep]
        if shuffle:
            if self.demo and self._demo_per_file_split is not None:
                _pairs = list(zip(self.data_filename, self._demo_per_file_split))
                np.random.shuffle(_pairs)
                if _pairs:
                    self.data_filename, self._demo_per_file_split = map(
                        list, zip(*_pairs)
                    )
                else:
                    self.data_filename = []
                    self._demo_per_file_split = []
            else:
                np.random.shuffle(self.data_filename)

        self.repeat = repeat
        self.range_scale = range_scale
        self.range_rotation = range_rotation
        self.range_translation = range_translation
        self.rotation_type = rotation_type

        self.normalize_coord = normalize_coord
        self.split = split  # 保存 split 用于判断是否应用增强
        ##### Load primitive_json
        self.load_primitive_lib()
        # self.primitive_mapping = load_primitive_mapping(primitive_mapping_file)
        if transform is not None:
            from datasets.transform import Compose

            self.transform = Compose(transform)
        else:
            self.transform = None

        # 初始化 instance-level 增强
        if instance_transform is not None:
            from datasets.transform import InstanceCompose

            self.instance_transform = InstanceCompose(instance_transform)
        else:
            self.instance_transform = None

        self.to_tensor = ToTensor()

        num_primitive = len(self.primitive_lib.keys())
        self.primitive_ids = [x for x in range(num_primitive)]
        self.primitive_weights = np.ones(num_primitive)

        primitive_mapping_path = f"{svg_dir}/primitive_mapping.json"
        with open(primitive_mapping_path, "r", encoding="utf-8") as f:
            primitive_mapping = json.load(f)
        primitive_to_semantic = primitive_mapping["primitive_id2semantic_id"]
        self.primitive_to_semantic = primitive_to_semantic  # 保存供后续使用
        for template_id, template in self.primitive_lib.items():
            template["sem_ids"] = np.full_like(
                template["sem_ids"], int(primitive_to_semantic[str(template_id)])
            )

        if self.transform is not None:
            self.transform.import_primitive_lib(
                self.primitive_lib,
                self.primitive_ids,
                self.primitive_weights,
                primitive_to_semantic,
            )

        # 将 primitive_lib 导入到 instance_transform
        if self.instance_transform is not None:
            self.instance_transform.import_primitive_lib(
                self.primitive_lib,
                self.primitive_ids,
                self.primitive_weights,
                primitive_to_semantic,
            )
        self.num_discrete_translation = num_discrete_translation
        self.continuous_range_translation = continuous_range_translation
        if self.num_discrete_translation is not None:
            self.discretize_trans = partial(
                discretize,
                num_discrete=num_discrete_translation,
                continuous_range=continuous_range_translation,
            )
        else:
            self.discretize_trans = None

    def _primitive_normalized(self, svg_data: SVGData, scene_bound: float):
        # svg_data_tensor = to_tensor(data)
        # transform all line coords from [N * [x1, y1, x2, y2]] to [N * 2 * [x, y]] for easier processing
        svg_data.coords = np.array(svg_data.coords)
        if svg_data.coords.shape[-1] == 4:
            svg_data.coords = svg_data.coords.reshape(-1, 2, 2)
        # # ---------------- normalize lines --------------- #
        svg_data.coords = svg_data.coords - (scene_bound / 2)
        svg_data.coords = svg_data.coords / (scene_bound / 2)
        if len(svg_data.coords.shape) == 3:
            svg_data.coords = svg_data.coords.reshape(-1, 4)
        return svg_data

    def load_primitive_lib(self):
        """
        加载 primitive 库，并转换为包含 coords 和 feats 的字典格式
        用于 CopyPaste 和 CutMix 增强
        """
        from models.primitive.scale_parameterization import scene_template_extent

        self.primitive_lib = {}
        self.primitive_template_extent_scene = {}
        primitive_files_lst = sorted(
            [x for x in os.listdir(self.primitive_dir) if x.endswith(".json")]
        )
        for file in primitive_files_lst:
            json_data = json.load(open(os.path.join(self.primitive_dir, file)))
            svg_data = SVGData(**json_data)
            # ── no_color_layer ablation: zero colors / layer_ids at primitive_lib
            #    load time. Any downstream use (CopyPasteInstance pasting these
            #    primitives into a scene) will then carry zeros instead of the
            #    original CAD layer/color metadata. ──────────────────────────
            if getattr(self, "no_color_layer", False):
                if hasattr(svg_data, "colors") and svg_data.colors is not None:
                    svg_data.colors = np.zeros_like(np.asarray(svg_data.colors))
                if hasattr(svg_data, "layer_ids") and svg_data.layer_ids is not None:
                    svg_data.layer_ids = np.zeros_like(np.asarray(svg_data.layer_ids))
            # 对 primitive 进行坐标归一化
            # svg_data = self._primitive_normalized(svg_data, self.scene_bound)
            # 转换为 tensor 再调用 to_vec_data 获取 feats
            svg_data_tensor = to_tensor(svg_data)
            vec_data = to_vec_data(svg_data_tensor)

            primitive_id = int(file.split(".json")[0])
            self.primitive_template_extent_scene[primitive_id] = scene_template_extent(
                svg_data.coords, self.scene_bound
            ).astype(np.float32)

            # Build the per-primitive dict with feats already produced by
            # to_vec_data (which slices feats[..., :7] so colors are dropped).
            # If no_color_layer=True we ALSO zero the stored colors / layer_ids
            # arrays themselves so any consumer reading them sees zeros.
            colors_arr = np.array(svg_data.colors)
            layer_ids_arr = vec_data.layer_ids.numpy()
            feats_np = vec_data.feats.numpy()
            if getattr(self, "no_color_layer", False):
                colors_arr = np.zeros_like(colors_arr)
                layer_ids_arr = np.zeros_like(layer_ids_arr)
                # Zero color channels when they are included in the feature array.
                if feats_np.shape[-1] > 7:
                    feats_np = feats_np.copy()
                    feats_np[..., 7:] = 0.0

            # 存储为字典格式，包含 coords, feats 和其他信息
            # 注意：primitive_lib 中每个 key 本质上是一个 instance 级别的元素
            self.primitive_lib[primitive_id] = {
                # Point-level 数据
                "coords": vec_data.coords.numpy(),  # shape: (N, 2) 线段中心坐标
                "feats": feats_np,  # shape: (N, feat_dim)
                "raw_coords": np.array(svg_data.coords),  # shape: (N, 4) 原始线段坐标
                "colors": colors_arr,  # shape: (N, 3)
                "widths": np.array(svg_data.widths),  # shape: (N,)
                "prim_ids": vec_data.prim_ids.numpy(),  # shape: (N,) 点属于哪个子 primitive
                "layer_ids": layer_ids_arr,  # shape: (N,)
                # Primitive-level 数据 (该 instance 内各子 primitive 的属性)
                "sem_ids": np.array(
                    svg_data.semantic_ids
                ),  # shape: (P,) 各子 primitive 的 semantic_id
                "inst_ids": np.array(
                    svg_data.instance_ids
                ),  # shape: (P,) 各子 primitive 的 instance_id
                "primitive_lengths": np.array(
                    svg_data.primitive_lengths
                ),  # shape: (P,) 各子 primitive 的长度
            }

    def _reparameterize_scale_labels(self, model_data):
        """Convert augmented relative scales to absolute local extents."""
        if self.scale_parameterization == "relative":
            return
        primitive_id = model_data["primitive_id"]
        scale = model_data["scale"]
        if isinstance(primitive_id, torch.Tensor):
            primitive_id_np = primitive_id.detach().cpu().numpy()
        else:
            primitive_id_np = np.asarray(primitive_id)
        if primitive_id_np.size == 0:
            return
        extents = np.stack(
            [
                self.primitive_template_extent_scene[int(pid)]
                for pid in primitive_id_np.reshape(-1)
            ],
            axis=0,
        ).reshape(*primitive_id_np.shape, 2)
        if isinstance(scale, torch.Tensor):
            factor = torch.as_tensor(extents, dtype=scale.dtype, device=scale.device)
            model_data["scale"] = scale * factor
        else:
            model_data["scale"] = np.asarray(scale) * extents.astype(
                np.asarray(scale).dtype, copy=False
            )

    def __len__(self):
        return len(self.data_filename) * self.repeat

    def _get_transform_args(self) -> Dict[str, Any]:
        if self.split == "train":
            return self.train_transform_args
        elif self.split == "val" or self.split == "test":
            return self.eval_transform_args
        else:
            raise ValueError(f"Invalid split: {self.split}")

    def _apply_instance_transforms_vectorized(
        self, svg_data: SVGData, gt_data: Dict[str, Any]
    ):
        """
        向量化的 Instance-level 增强，直接在 SVGData 上操作，
        避免 svg_to_instance / instance_to_svg 的拆解-重组开销。
        按 transform 类型分派：支持 InstanceJitterTranslate、PrimitiveDropoutInstance
        、CopyPasteInstance 和 SourceInstanceCutMix。
        """
        for t in self.instance_transform.transforms:
            if not hasattr(t, "p") or np.random.rand() > t.p:
                continue

            t_name = type(t).__name__

            # CopyPasteInstance 在 num_instances == 0 时也应执行（填充空场景）
            if t_name == "CopyPasteInstance":
                svg_data, gt_data = self._apply_copypaste_vectorized(
                    svg_data, gt_data, t
                )
                continue

            if t_name == "SourceInstanceCutMix":
                svg_data, gt_data = self._apply_source_cutmix_vectorized(
                    svg_data, gt_data, t
                )
                continue

            num_instances = len(gt_data["translation"])
            if num_instances == 0:
                continue

            if t_name == "PrimitiveDropoutInstance":
                svg_data, gt_data = self._apply_dropout_vectorized(svg_data, gt_data, t)
            elif t_name == "InstanceJitterTranslate":
                svg_data, gt_data = self._apply_jitter_vectorized(svg_data, gt_data, t)

        return svg_data, gt_data

    def _apply_source_cutmix_vectorized(
        self, svg_data: SVGData, gt_data: Dict[str, Any], t
    ):
        """Paste real Train source instances while preserving GT alignment."""
        t._ensure_loaded()
        sampled_requested = max(1, int(np.random.poisson(t.primitive_num_mean)))
        if t.placement_mode == "hard_density":
            t._ensure_loaded()
            target_minimum = 10
            target_maximum = 24
            if t.profile is not None:
                target_minimum = int(t.profile["target"]["instance_count"][0])
                target_maximum = int(t.profile["target"]["instance_count"][1])
            if len(gt_data["translation"]) >= target_maximum:
                return svg_data, gt_data
            sampled_requested = max(
                sampled_requested,
                target_minimum - len(gt_data["translation"]),
            )
            sampled_requested = min(
                sampled_requested,
                target_maximum - len(gt_data["translation"]),
            )
        requested = min(t.max_added_instances, max(1, sampled_requested))
        coords = np.asarray(svg_data.coords, dtype=np.float32).reshape(-1, 4)
        colors = np.asarray(svg_data.colors)
        widths = np.asarray(svg_data.widths, dtype=np.float32)
        point_primitive = np.asarray(svg_data.primitive_ids, dtype=np.int32)
        layers = np.asarray(svg_data.layer_ids, dtype=np.int32)
        primitive_semantic = np.asarray(svg_data.semantic_ids, dtype=np.int32)
        primitive_instance = np.asarray(svg_data.instance_ids, dtype=np.int32)
        primitive_lengths = np.asarray(svg_data.primitive_lengths, dtype=np.float32)

        half = float(t.scene_bound) / 2.0
        grid_width = max(float(self.grid_size) * half, 1e-6)
        centers = coords.reshape(-1, 2, 2).mean(axis=1) - half
        if len(centers):
            grid_keys = np.floor(centers / grid_width).astype(np.int64)
            estimated_postgrid = len(np.unique(grid_keys, axis=0))
        else:
            estimated_postgrid = 0

        # Existing foreground bboxes in adjusted [0, scene_bound] coordinates.
        existing_boxes = []
        if len(point_primitive) and len(primitive_instance):
            point_instance = primitive_instance[point_primitive]
            for instance_id in np.unique(point_instance[point_instance >= 0]):
                instance_coords = coords[point_instance == instance_id].reshape(-1, 2)
                if len(instance_coords):
                    existing_boxes.append(
                        np.concatenate(
                            [instance_coords.min(axis=0), instance_coords.max(axis=0)]
                        )
                    )

        def bbox_iou(left, right):
            inter_low = np.maximum(left[:2], right[:2])
            inter_high = np.minimum(left[2:], right[2:])
            inter_size = np.maximum(inter_high - inter_low, 0.0)
            inter = float(np.prod(inter_size))
            left_area = float(np.prod(np.maximum(left[2:] - left[:2], 0.0)))
            right_area = float(np.prod(np.maximum(right[2:] - right[:2], 0.0)))
            return inter / max(left_area + right_area - inter, 1e-12)

        def valid_box(candidate):
            if t.require_full_inside and (
                np.any(candidate[:2] < 0.0)
                or np.any(candidate[2:] > float(t.scene_bound))
            ):
                return False
            expanded = candidate.copy()
            expanded[:2] -= t.min_bbox_gap
            expanded[2:] += t.min_bbox_gap
            return all(
                bbox_iou(expanded, box) <= t.max_bbox_iou for box in existing_boxes
            )

        new_coords = []
        new_colors = []
        new_widths = []
        new_point_primitives = []
        new_layers = []
        new_primitive_semantics = []
        new_primitive_instances = []
        new_primitive_lengths = []
        new_gt = {
            "primitive_id": [],
            "semantic_id": [],
            "translation": [],
            "rotation": [],
            "scale": [],
            "flip_mode": [],
            "flip_invariant_primitive": [],
            "size": [],
        }
        primitive_offset = len(primitive_semantic)
        foreground_instances = primitive_instance[primitive_instance >= 0]
        instance_offset = (
            int(foreground_instances.max()) + 1 if len(foreground_instances) else 0
        )
        layer_offset = int(layers.max()) + 1 if len(layers) else 0
        present_identities = set(
            zip(
                np.asarray(gt_data["semantic_id"], dtype=np.int32).tolist(),
                np.asarray(gt_data["primitive_id"], dtype=np.int32).tolist(),
            )
        )
        bundle = None
        bundle_indices = None
        bundle_cursor = 0
        bundle_failures = 0
        if t.sampling_mode == "target_anchor_scene_bundle":
            bundle = t.sample_target_anchor_scene_bundle(
                np.asarray(gt_data["semantic_id"], dtype=np.int32).tolist(),
                requested,
            )
            if bundle is None:
                if bool(gt_data.get("_recon_audit_enabled", False)):
                    audit = gt_data.setdefault("_recon_audit", {})
                    audit["source_cutmix_requested_instances"] = int(requested)
                    audit["source_cutmix_added_instances"] = 0
                    audit["source_cutmix_attempts"] = 0
                    audit["source_cutmix_bundle_available"] = False
                return svg_data, gt_data
            bundle_indices = bundle["donor_indices"]
            # A source scene may contain fewer companions than the Poisson
            # request.  This is the actual upper bound for this invocation.
            requested = min(requested, len(bundle_indices))

        def reject_bundle_candidate():
            nonlocal bundle_cursor, bundle_failures
            if bundle_indices is None:
                return
            bundle_failures += 1
            # Move to another donor from the same source scene instead of
            # spending the full global retry budget on one infeasible object.
            if bundle_failures >= 8:
                bundle_cursor += 1
                bundle_failures = 0

        accepted = 0
        attempted = 0
        max_attempts = max(32, requested * 32)
        while accepted < requested and attempted < max_attempts:
            attempted += 1
            if bundle_indices is not None:
                if bundle_cursor >= len(bundle_indices):
                    break
                donor_index = bundle_indices[bundle_cursor]
            else:
                donor_index = t.sample_donor(present_identities)
            donor = t.donor(donor_index)
            if (
                estimated_postgrid + donor["estimated_postgrid_lines"]
                > t.max_postgrid_segments
            ):
                reject_bundle_candidate()
                continue
            local_bbox = donor["local_bbox"].astype(np.float64)
            valid_low = -half - local_bbox[:2]
            valid_high = half - local_bbox[2:]
            if np.any(valid_low > valid_high):
                reject_bundle_candidate()
                continue
            if t.placement_mode == "hard_density" and existing_boxes:
                anchor_box = existing_boxes[int(np.random.randint(len(existing_boxes)))]
                anchor_center = 0.5 * (anchor_box[:2] + anchor_box[2:]) - half
                extent = np.maximum(local_bbox[2:] - local_bbox[:2], 1.0)
                target_translation = (
                    anchor_center + np.random.uniform(-1.5, 1.5, 2) * extent
                )
                target_translation = np.clip(target_translation, valid_low, valid_high)
            else:
                target_translation = np.random.uniform(valid_low, valid_high).astype(
                    np.float64
                )
            candidate_bbox = local_bbox + np.tile(target_translation + half, 2)
            if not valid_box(candidate_bbox):
                reject_bundle_candidate()
                continue

            donor_coords = donor["coords_local"].reshape(-1, 2, 2)
            donor_coords = donor_coords + target_translation.reshape(1, 1, 2) + half
            donor_coords = donor_coords.reshape(-1, 4).astype(np.float32)
            donor_point_primitive = donor["point_primitive_ids"].astype(np.int32)
            donor_primitive_semantic = donor["primitive_semantic_ids"].astype(np.int32)
            donor_primitive_count = len(donor_primitive_semantic)
            donor_layers = donor["layer_ids"].astype(np.int32)
            unique_layers, inverse_layers = np.unique(donor_layers, return_inverse=True)

            new_coords.append(donor_coords)
            new_colors.append(donor["colors"].astype(colors.dtype, copy=False))
            new_widths.append(donor["widths"].astype(np.float32, copy=False))
            new_point_primitives.append(donor_point_primitive + primitive_offset)
            new_layers.append(inverse_layers.astype(np.int32) + layer_offset)
            new_primitive_semantics.append(donor_primitive_semantic)
            new_primitive_instances.append(
                np.full(donor_primitive_count, instance_offset, dtype=np.int32)
            )
            new_primitive_lengths.append(
                donor["primitive_lengths"].astype(np.float32, copy=False)
            )
            new_gt["primitive_id"].append(donor["primitive_id"])
            new_gt["semantic_id"].append(donor["semantic_id"])
            new_gt["translation"].append(target_translation)
            new_gt["rotation"].append(donor["rotation"])
            new_gt["scale"].append(donor["scale"])
            new_gt["flip_mode"].append(donor["flip"])
            new_gt["flip_invariant_primitive"].append(
                donor["primitive_id"] in self.flip_invariant_primitive_id
            )
            new_gt["size"].append(donor["size"])

            primitive_offset += donor_primitive_count
            instance_offset += 1
            layer_offset += len(unique_layers)
            estimated_postgrid += donor["estimated_postgrid_lines"]
            existing_boxes.append(candidate_bbox)
            present_identities.add((donor["semantic_id"], donor["primitive_id"]))
            accepted += 1
            if bundle_indices is not None:
                bundle_cursor += 1
                bundle_failures = 0

        if accepted == 0:
            if bool(gt_data.get("_recon_audit_enabled", False)):
                audit = gt_data.setdefault("_recon_audit", {})
                audit["source_cutmix_requested_instances"] = int(requested)
                audit["source_cutmix_added_instances"] = 0
                audit["source_cutmix_attempts"] = int(attempted)
            return svg_data, gt_data

        svg_data.coords = np.concatenate([coords] + new_coords, axis=0)
        svg_data.colors = np.concatenate([colors] + new_colors, axis=0)
        svg_data.widths = np.concatenate([widths] + new_widths, axis=0)
        svg_data.primitive_ids = np.concatenate(
            [point_primitive] + new_point_primitives, axis=0
        )
        svg_data.layer_ids = np.concatenate([layers] + new_layers, axis=0)
        svg_data.semantic_ids = np.concatenate(
            [primitive_semantic] + new_primitive_semantics, axis=0
        )
        svg_data.instance_ids = np.concatenate(
            [primitive_instance] + new_primitive_instances, axis=0
        )
        svg_data.primitive_lengths = np.concatenate(
            [primitive_lengths] + new_primitive_lengths, axis=0
        )

        instance_specs = {
            "primitive_id": np.int32,
            "semantic_id": np.int32,
            "translation": np.float32,
            "rotation": np.float32,
            "scale": np.float32,
            "flip_mode": np.int32,
            "flip_invariant_primitive": np.bool_,
            "size": np.float32,
        }
        for key, dtype in instance_specs.items():
            if key == "size" and (key not in gt_data or gt_data[key] is None):
                continue
            addition = np.asarray(new_gt[key], dtype=dtype)
            gt_data[key] = np.concatenate(
                [np.asarray(gt_data[key], dtype=dtype), addition], axis=0
            )

        if bool(gt_data.get("_recon_audit_enabled", False)):
            audit = gt_data.setdefault("_recon_audit", {})
            audit["source_cutmix_requested_instances"] = int(requested)
            audit["source_cutmix_added_instances"] = int(accepted)
            audit["source_cutmix_attempts"] = int(attempted)
            if bundle is not None:
                audit["source_cutmix_bundle_available"] = True
                audit["source_cutmix_bundle_anchor_semantic"] = int(
                    bundle["anchor_semantic"]
                )
            if "instance_origin_kind" in audit:
                audit["instance_origin_kind"] = np.concatenate(
                    [
                        np.asarray(audit["instance_origin_kind"], dtype=np.int8),
                        np.full(accepted, 2, dtype=np.int8),
                    ]
                )
            if "instance_jittered" in audit:
                audit["instance_jittered"] = np.concatenate(
                    [
                        np.asarray(audit["instance_jittered"], dtype=np.bool_),
                        np.zeros(accepted, dtype=np.bool_),
                    ]
                )
        return svg_data, gt_data

    def _apply_copypaste_vectorized(
        self, svg_data: SVGData, gt_data: Dict[str, Any], t
    ):
        """
        向量化的 CopyPasteInstance：直接在 SVGData + gt_data 上追加新 instance，
        无需经过 svg_to_instance / instance_to_svg 的拆解-重组。

        数据层级说明：
          - point-level  : coords / colors / widths / primitive_ids / layer_ids
          - primitive-level: semantic_ids / instance_ids / primitive_lengths
          - instance-level : gt_data 中各字段
        primitive_ids (point级) 用作索引，指向 primitive-level 数组。
        新增 primitive 的全局偏移 = 现有 primitive 数量 (len(svg_data.semantic_ids))。
        """
        if t.primitive_lib is None:
            return svg_data, gt_data

        primitive_num = max(1, np.random.poisson(t.primitive_num_mean))
        selected_ids = np.random.choice(
            t.primitive_ids, size=primitive_num, replace=True
        )

        # 缓存现有 point-level / primitive-level 数组（numpy 化，供后续 concatenate）
        exist_point_prim_ids = np.asarray(
            svg_data.primitive_ids, dtype=np.int32
        )  # point→prim LUT
        exist_prim_inst_ids = np.asarray(
            svg_data.instance_ids, dtype=np.int32
        )  # primitive-level
        exist_layer_ids = np.asarray(svg_data.layer_ids, dtype=np.int32)  # point-level

        # 新增数据起始偏移
        current_prim_offset = len(svg_data.semantic_ids)  # 现有 primitive 总数
        fg_inst = exist_prim_inst_ids[exist_prim_inst_ids >= 0]
        current_inst_id = int(fg_inst.max()) + 1 if len(fg_inst) > 0 else 0
        max_layer_id = int(exist_layer_ids.max()) if len(exist_layer_ids) > 0 else -1
        new_shared_layer_id = max_layer_id + 1

        # 新增数据临时列表
        new_pt_coords, new_pt_colors, new_pt_widths = [], [], []
        new_pt_prim_ids, new_pt_layer_ids = [], []
        new_pr_sem_ids, new_pr_inst_ids, new_pr_lens = [], [], []
        new_gt_prim_ids, new_gt_sem_ids = [], []
        new_gt_trans, new_gt_rots, new_gt_scales = [], [], []
        new_gt_flip_modes, new_gt_flip_invs = [], []
        new_gt_sizes = []

        for primitive_id in selected_ids:
            if primitive_id not in t.primitive_lib:
                continue
            prim_data = t.primitive_lib[primitive_id]

            raw_coords = prim_data["raw_coords"].copy()  # (N, 4)
            colors = prim_data["colors"].copy()  # (N, 3)
            widths = prim_data["widths"].copy()  # (N,)
            prim_ids = prim_data["prim_ids"].copy()  # (N,) within-instance prim idx
            prim_sem = prim_data["sem_ids"].copy()  # (P,)
            prim_lens = prim_data["primitive_lengths"].copy()  # (P,)

            n_points = len(raw_coords)
            num_prims_in_inst = len(prim_sem)

            # 随机变换参数
            angle = np.random.choice([0, 90, 180, 270])
            delta = np.random.uniform(
                t.range_translation[0], t.range_translation[1], 2
            ).astype(np.float32)

            # 应用变换（复用 CopyPasteInstance._apply_transform，其内部已支持向量化）
            raw_coords, gt_translation = t._apply_transform(
                raw_coords,
                angle,
                delta,
                return_origin_translation=True,
            )
            raw_coords += t.scene_bound / 2

            # 计算 bounding box size
            all_x = raw_coords[:, [0, 2]]
            all_y = raw_coords[:, [1, 3]]

            # Point-level 数据
            new_pt_coords.append(raw_coords)
            new_pt_colors.append(colors)
            new_pt_widths.append(widths)
            new_pt_prim_ids.append(prim_ids + current_prim_offset)
            new_pt_layer_ids.append(
                np.full(n_points, new_shared_layer_id, dtype=np.int32)
            )

            # Primitive-level 数据
            new_pr_sem_ids.append(prim_sem.astype(np.int32))
            new_pr_inst_ids.append(
                np.full(num_prims_in_inst, current_inst_id, dtype=np.int32)
            )
            new_pr_lens.append(prim_lens.astype(np.float32))

            # GT instance-level 数据
            semantic_id = t.primitive_to_semantic.get(str(primitive_id), 0)
            new_gt_prim_ids.append(primitive_id)
            new_gt_sem_ids.append(semantic_id)
            # CopyPaste keeps the source rotating around its own bbox centre,
            # while model/reconstruction attributes use an origin-based
            # transform. ``gt_translation`` includes the exact pivot
            # compensation c - R c, so both geometries are identical.
            new_gt_trans.append(gt_translation)
            new_gt_rots.append(float(angle))
            new_gt_scales.append([1.0, 1.0])
            new_gt_flip_modes.append(0)
            new_gt_flip_invs.append(primitive_id in self.flip_invariant_primitive_id)
            new_gt_sizes.append([all_x.max() - all_x.min(), all_y.max() - all_y.min()])

            current_prim_offset += num_prims_in_inst
            current_inst_id += 1

        if not new_gt_prim_ids:
            return svg_data, gt_data

        if bool(gt_data.get("_recon_audit_enabled", False)):
            audit = gt_data.setdefault("_recon_audit", {})
            audit["copypaste_added_instances"] = int(len(new_gt_prim_ids))
            audit["instance_origin_kind"] = np.concatenate(
                [
                    np.asarray(audit["instance_origin_kind"], dtype=np.int8),
                    np.ones(len(new_gt_prim_ids), dtype=np.int8),
                ]
            )
            audit["instance_jittered"] = np.concatenate(
                [
                    np.asarray(audit["instance_jittered"], dtype=np.bool_),
                    np.zeros(len(new_gt_prim_ids), dtype=np.bool_),
                ]
            )

        # ── 一次性 concatenate，追加到 SVGData（Point-level） ──────────────────
        new_coords_arr = np.concatenate(new_pt_coords, axis=0)
        new_colors_arr = np.concatenate(new_pt_colors, axis=0)
        new_widths_arr = np.concatenate(new_pt_widths, axis=0)
        new_ppids_arr = np.concatenate(new_pt_prim_ids, axis=0)
        new_lids_arr = np.concatenate(new_pt_layer_ids, axis=0)

        svg_data.coords = np.concatenate(
            [np.asarray(svg_data.coords, dtype=np.float32), new_coords_arr], axis=0
        )
        svg_data.colors = np.concatenate(
            [np.asarray(svg_data.colors, dtype=np.int32), new_colors_arr], axis=0
        )
        svg_data.widths = np.concatenate(
            [np.asarray(svg_data.widths, dtype=np.float32), new_widths_arr], axis=0
        )
        svg_data.primitive_ids = np.concatenate(
            [exist_point_prim_ids, new_ppids_arr], axis=0
        )
        svg_data.layer_ids = np.concatenate([exist_layer_ids, new_lids_arr], axis=0)

        # ── 一次性 concatenate，追加到 SVGData（Primitive-level） ─────────────
        new_psem_arr = np.concatenate(new_pr_sem_ids, axis=0)
        new_pinst_arr = np.concatenate(new_pr_inst_ids, axis=0)
        new_plens_arr = np.concatenate(new_pr_lens, axis=0)

        svg_data.semantic_ids = np.concatenate(
            [np.asarray(svg_data.semantic_ids, dtype=np.int32), new_psem_arr], axis=0
        )
        svg_data.instance_ids = np.concatenate(
            [exist_prim_inst_ids, new_pinst_arr], axis=0
        )
        svg_data.primitive_lengths = np.concatenate(
            [np.asarray(svg_data.primitive_lengths, dtype=np.float32), new_plens_arr],
            axis=0,
        )

        # ── 一次性 concatenate，追加到 gt_data（Instance-level） ─────────────
        gt_data["primitive_id"] = np.concatenate(
            [
                np.asarray(gt_data["primitive_id"], dtype=np.int32),
                np.array(new_gt_prim_ids, dtype=np.int32),
            ]
        )
        gt_data["semantic_id"] = np.concatenate(
            [
                np.asarray(gt_data["semantic_id"], dtype=np.int32),
                np.array(new_gt_sem_ids, dtype=np.int32),
            ]
        )
        gt_data["translation"] = np.concatenate(
            [
                np.asarray(gt_data["translation"], dtype=np.float32),
                np.array(new_gt_trans, dtype=np.float32),
            ],
            axis=0,
        )
        gt_data["rotation"] = np.concatenate(
            [
                np.asarray(gt_data["rotation"], dtype=np.float32),
                np.array(new_gt_rots, dtype=np.float32),
            ]
        )
        gt_data["scale"] = np.concatenate(
            [
                np.asarray(gt_data["scale"], dtype=np.float32),
                np.array(new_gt_scales, dtype=np.float32),
            ],
            axis=0,
        )
        gt_data["flip_mode"] = np.concatenate(
            [
                np.asarray(gt_data["flip_mode"], dtype=np.int32),
                np.array(new_gt_flip_modes, dtype=np.int32),
            ]
        )
        gt_data["flip_invariant_primitive"] = np.concatenate(
            [
                np.asarray(gt_data["flip_invariant_primitive"], dtype=np.bool_),
                np.array(new_gt_flip_invs, dtype=np.bool_),
            ]
        )
        if "size" in gt_data and gt_data["size"] is not None:
            gt_data["size"] = np.concatenate(
                [
                    np.asarray(gt_data["size"], dtype=np.float32),
                    np.array(new_gt_sizes, dtype=np.float32),
                ],
                axis=0,
            )

        return svg_data, gt_data

    def _apply_dropout_vectorized(self, svg_data: SVGData, gt_data: Dict[str, Any], t):
        """
        向量化的 PrimitiveDropoutInstance：随机丢弃一部分 instance，
        同步更新 SVGData 的所有层级属性和 GT instance-level 属性。
        """
        num_instances = len(gt_data["translation"])
        if num_instances <= 1:
            return svg_data, gt_data

        keep_ratio = np.random.uniform(t.min_ratio, t.max_ratio)
        keep_num = max(1, int(keep_ratio * num_instances))
        keep_indices = np.sort(np.random.choice(num_instances, keep_num, replace=False))

        prim_ids = np.asarray(svg_data.primitive_ids, dtype=np.int32)
        inst_ids_prim = np.asarray(svg_data.instance_ids, dtype=np.int32)
        point_inst = inst_ids_prim[prim_ids]

        point_keep = np.isin(point_inst, keep_indices) | (point_inst < 0)

        if point_keep.all() and keep_num == num_instances:
            return svg_data, gt_data

        # Point-level 过滤
        svg_data.coords = np.asarray(svg_data.coords)[point_keep]
        svg_data.colors = np.asarray(svg_data.colors)[point_keep]
        svg_data.widths = np.asarray(svg_data.widths)[point_keep]
        svg_data.layer_ids = np.asarray(svg_data.layer_ids)[point_keep]

        # Primitive-level 过滤 + LUT 重映射
        surviving_prim_ids = prim_ids[point_keep]
        surviving_prims = np.unique(surviving_prim_ids)

        if len(surviving_prims) < len(inst_ids_prim):
            svg_data.semantic_ids = np.asarray(svg_data.semantic_ids)[surviving_prims]
            svg_data.instance_ids = inst_ids_prim[surviving_prims]
            svg_data.primitive_lengths = np.asarray(svg_data.primitive_lengths)[
                surviving_prims
            ]

        if len(surviving_prims) > 0:
            max_prim = int(surviving_prims.max()) + 1
            prim_lut = np.empty(max_prim, dtype=np.int32)
            prim_lut[surviving_prims] = np.arange(len(surviving_prims), dtype=np.int32)
            svg_data.primitive_ids = prim_lut[surviving_prim_ids]
        else:
            svg_data.primitive_ids = surviving_prim_ids

        # Instance-level 重映射（连续编号 0..K-1）
        updated_inst = np.asarray(svg_data.instance_ids, dtype=np.int32)
        fg_mask = updated_inst >= 0
        if fg_mask.any():
            present = np.unique(updated_inst[fg_mask])
            max_inst = int(present.max()) + 1
            inst_lut = np.empty(max_inst, dtype=np.int32)
            inst_lut[present] = np.arange(len(present), dtype=np.int32)
            new_inst = updated_inst.copy()
            new_inst[fg_mask] = inst_lut[updated_inst[fg_mask]]
            svg_data.instance_ids = new_inst

        # GT instance-level 属性按 keep_indices 过滤
        instance_level_keys = [
            "semantic_id",
            "primitive_id",
            "scale",
            "rotation",
            "flip_mode",
            "translation",
            "flip_invariant_primitive",
            "size",
        ]
        for key_name in instance_level_keys:
            if key_name in gt_data and gt_data[key_name] is not None:
                arr = gt_data[key_name]
                if hasattr(arr, "__len__") and len(arr) == num_instances:
                    gt_data[key_name] = arr[keep_indices]

        return svg_data, gt_data

    def _apply_jitter_vectorized(self, svg_data: SVGData, gt_data: Dict[str, Any], t):
        """
        向量化的 InstanceJitterTranslate：随机平移选中的 instance。
        """
        translations = gt_data["translation"]
        num_instances = len(translations)

        prim_ids = np.asarray(svg_data.primitive_ids, dtype=np.int32)
        inst_ids_prim = np.asarray(svg_data.instance_ids, dtype=np.int32)
        coords = np.asarray(svg_data.coords, dtype=np.float32)
        if coords.ndim == 2 and coords.shape[1] == 4:
            pass
        elif coords.ndim == 3 and coords.shape[1] == 2 and coords.shape[2] == 2:
            coords = coords.reshape(-1, 4)
            svg_data.coords = coords

        point_inst = inst_ids_prim[prim_ids]

        k = max(1, int(t.sample_ratio * num_instances))
        selected = np.random.choice(num_instances, k, replace=False)

        if getattr(t, "random_translate", False):
            targets = np.random.uniform(
                t.trans_range[0], t.trans_range[1], size=(k, 2)
            ).astype(np.float32)
            # ``translation`` and the unnormalised SVG coordinates use the
            # same raw coordinate unit.  Move the instance by the exact GT
            # transform delta instead of aligning its *visible bbox centre*
            # to the target.  Bbox alignment silently changes the template's
            # local geometry whenever the template is off-centre or clipped.
            # The transform-delta form preserves this invariant exactly:
            #
            #   coords_new - half_bound - translation_new
            #     == coords_old - half_bound - translation_old
            #
            # This is also valid for partial instances because it does not
            # infer an object centre from the currently visible fragments.
            offsets = np.zeros((num_instances, 2), dtype=np.float32)
            offsets[selected] = targets - translations[selected, :2]

            apply_mask = np.isin(point_inst, selected)
            pt_offsets = offsets[point_inst[apply_mask]]
            coords[apply_mask, 0] += pt_offsets[:, 0]
            coords[apply_mask, 1] += pt_offsets[:, 1]
            coords[apply_mask, 2] += pt_offsets[:, 0]
            coords[apply_mask, 3] += pt_offsets[:, 1]

            translations[selected, :2] = targets
            if bool(gt_data.get("_recon_audit_enabled", False)):
                audit = gt_data.setdefault("_recon_audit", {})
                jittered = np.asarray(audit["instance_jittered"], dtype=np.bool_)
                jittered[selected] = True
                audit["instance_jittered"] = jittered
                audit["jitter_applied_instances"] = int(len(selected))
        else:
            deltas = np.random.uniform(-t.max_delta, t.max_delta, size=(k, 2)).astype(
                np.float32
            )
            shifted_trans = translations[selected, :2] + deltas * (t.scene_bound / 2.0)
            valid = (
                (shifted_trans[:, 0] >= t.trans_range[0])
                & (shifted_trans[:, 0] <= t.trans_range[1])
                & (shifted_trans[:, 1] >= t.trans_range[0])
                & (shifted_trans[:, 1] <= t.trans_range[1])
            )
            valid_sel = selected[valid]
            valid_deltas = deltas[valid]
            if len(valid_sel) > 0:
                coord_deltas = valid_deltas * (t.scene_bound / 2.0)
                offsets = np.zeros((num_instances, 2), dtype=np.float32)
                offsets[valid_sel] = coord_deltas

                apply_mask = np.isin(point_inst, valid_sel)
                pt_offsets = offsets[point_inst[apply_mask]]
                coords[apply_mask, 0] += pt_offsets[:, 0]
                coords[apply_mask, 1] += pt_offsets[:, 1]
                coords[apply_mask, 2] += pt_offsets[:, 0]
                coords[apply_mask, 3] += pt_offsets[:, 1]

                # ``translations`` is still expressed in the raw scene coordinate
                # system here.  The source geometry was moved by ``coord_deltas``
                # (not by normalized ``valid_deltas``), so the GT transform must
                # receive the same raw-coordinate displacement before both are
                # normalized in ``_transform``.
                translations[valid_sel, :2] += coord_deltas

                if bool(gt_data.get("_recon_audit_enabled", False)):
                    audit = gt_data.setdefault("_recon_audit", {})
                    jittered = np.asarray(audit["instance_jittered"], dtype=np.bool_)
                    jittered[valid_sel] = True
                    audit["instance_jittered"] = jittered
                    audit["jitter_applied_instances"] = int(len(valid_sel))

        svg_data.coords = coords
        gt_data["translation"] = translations
        return svg_data, gt_data

    def _transform(
        self,
        svg_data: SVGData,
        gt_data: Dict[str, Any],
        crop_window: List[float],
        *,
        deterministic_fallback: bool = False,
    ):
        """
        对 SVGData 进行变换处理（优化版：全程 numpy，无 list 转换）

        流程:
        1. 坐标偏移校正
        2. Instance-level 增强（向量化，不做 SVGData 拆解/重组）
        3. 坐标归一化
        4. Point-level 增强链
        5. 转换为 VecData
        """
        coords = np.asarray(svg_data.coords, dtype=np.float32)
        if self.adjust_coords:
            coords = coords.reshape(-1, 2, 2)
            coords = coords - np.asarray(crop_window[0:2], dtype=np.float32)
            coords = coords.reshape(-1, 4)
        svg_data.coords = coords

        # ============== Step 1: Instance-level 增强（向量化） ============== #
        if (
            self.instance_transform is not None
            and self.split == "train"
            and not deterministic_fallback
        ):
            svg_data, gt_data = self._apply_instance_transforms_vectorized(
                svg_data, gt_data
            )

        # ============== Step 2: 坐标归一化 ============== #
        coords = np.asarray(svg_data.coords, dtype=np.float32)
        half_bound = self.scene_bound / 2.0
        if coords.shape[-1] == 4:
            coords = coords.reshape(-1, 2, 2)
        coords = (coords - half_bound) / half_bound
        svg_data.coords = coords
        gt_data["translation"] = gt_data["translation"] / half_bound
        if self.return_size:
            gt_data["size"] = gt_data["size"] / half_bound

        # ============== Step 3: Point-level 增强 ============== #
        gt_data["svg_data"] = svg_data
        if self.transform is not None:
            if deterministic_fallback:
                # Preserve sample identity while guaranteeing that a valid base
                # crop cannot be killed by an unlucky sequence of stochastic
                # scene augmentations.  The full deterministic audit uses the
                # same transform suffix.
                deterministic_names = {"VecClipping", "VecGridSample", "ToTensor"}
                for transform in self.transform.transforms:
                    if type(transform).__name__ not in deterministic_names:
                        continue
                    if type(transform).__name__ == "VecGridSample":
                        original_mode = transform.mode
                        try:
                            transform.mode = "test"
                            gt_data = transform(gt_data)
                        finally:
                            transform.mode = original_mode
                    else:
                        gt_data = transform(gt_data)
            else:
                gt_data = self.transform(gt_data)
        svg_data = gt_data["svg_data"]

        # ============== Step 4: 转换为 VecData ============== #
        svg_data_tensor = to_tensor(svg_data)
        if len(svg_data_tensor.coords.shape) == 3:
            svg_data_tensor.coords = svg_data_tensor.coords.reshape(-1, 4)
        vec_data = to_vec_data(svg_data_tensor)
        return svg_data_tensor, vec_data, gt_data

    def _handle_invalid_sample(
        self,
        idx: int,
        retry_count: int,
        reason: str,
        *,
        retryable: bool,
    ):
        if self.invalid_sample_policy == "successor":
            return self.__getitem__((idx + 1) % len(self), 0)
        if (
            self.invalid_sample_policy
            in {"retry_same", "retry_same_then_deterministic"}
            and retryable
            and retry_count < self.invalid_sample_max_retries
        ):
            return self.__getitem__(idx, retry_count + 1)
        if (
            self.invalid_sample_policy == "retry_same_then_deterministic"
            and retryable
            and retry_count == self.invalid_sample_max_retries
        ):
            return self.__getitem__(idx, retry_count + 1)
        data_file = self.data_filename[idx % len(self.data_filename)]
        raise RuntimeError(
            "Invalid PrimitiveDataset sample: "
            f"filename={data_file!r}, index={idx}, reason={reason}, "
            f"policy={self.invalid_sample_policy!r}, retries={retry_count}"
        )

    def __getitem__(self, idx, _invalid_retry_count=0):

        #### 版本1: 直接用一样的svg_data格式，不加任何primitive-level的数据增强，只对整体进行加强 ####
        ## io测试1:数据读取 ##
        t_getitem_start = time.perf_counter()
        t_io1_start = time.perf_counter()

        data_file = self.data_filename[idx % len(self.data_filename)]
        if self.demo and self._demo_per_file_split is not None:
            sp_item = self._demo_per_file_split[idx % len(self.data_filename)]
            gt_sub_item = "gt"
            svg_dir_item = os.path.join(self._svg_dir_root, sp_item, "svg")
            gt_dir_item = os.path.join(self._gt_dir_root, sp_item, gt_sub_item)
            cache_dir_item = os.path.join(self._cache_dir_root, sp_item, "cache_pkl")
        else:
            svg_dir_item = self.svg_dir
            gt_dir_item = self.gt_dir
            cache_dir_item = self.cache_dir

        svg_file = os.path.join(svg_dir_item, data_file)
        gt_file = os.path.join(gt_dir_item, data_file)

        model_data = {}
        model_data["filename"] = data_file
        # model_data['bg_coord'] = bg_coord
        # model_data['bg_feat'] = bg_feat

        cache_loaded = False
        if self.use_cache_pkl:
            cache_file = os.path.join(
                cache_dir_item, data_file.replace(".json", ".pkl")
            )
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "rb") as f:
                        cache_obj = pickle.load(f)
                    primitives_json = cache_obj.get("gt_raw", None)
                    json_data = cache_obj.get("svg_raw", None)
                    if isinstance(primitives_json, list) and isinstance(
                        json_data, dict
                    ):
                        cache_loaded = True
                except Exception:
                    cache_loaded = False

        if not cache_loaded:
            with open(gt_file, "r", encoding="utf-8") as f:
                primitives_json = json.load(f)
        crop_window = primitives_json[0]["crop_window"]
        primitives = primitives_json
        primitive_sem_id = []
        primitive_prim_id = []
        primitive_scale = []
        primitive_rotation = []
        primitive_flip = []
        primitive_translate = []
        primitive_flip_invariant = []
        if self.return_size:
            primitive_size = []
        else:
            primitive_size = None
        # global_translate = []
        # crop_windows = []
        # primitive_flip_invariant_primitive = []
        ### 若len(primitives) == 0, 则返回empty?
        for primitive in primitives:
            primitive_sem_id.append(np.array(primitive["semantic_id"], dtype=np.int32))
            primitive_prim_id.append(
                np.array(template_id(primitive), dtype=np.int32)
            )
            primitive_scale.append(np.array(primitive["scale"], dtype=np.float32))
            primitive_rotation.append(np.array(primitive["rotate"], dtype=np.float32))
            primitive_translate.append(
                np.array(primitive["translation"], dtype=np.float32)
            )
            if template_id(primitive) in self.flip_invariant_primitive_id:
                primitive_flip_invariant.append(True)
                assert (
                    primitive["flip"] == 0
                ), "flip_mode should be 0 for flip_invariant_primitive"
            else:
                primitive_flip_invariant.append(False)
            primitive_flip.append(np.array(primitive["flip"], dtype=np.int32))
            if self.return_size:
                primitive_size.append(np.array(primitive["size"], dtype=np.float32))

        if len(primitive_sem_id) == 0:
            primitive_sem_id = np.empty((0,), dtype=np.int32)
            primitive_prim_id = np.empty((0,), dtype=np.int32)
            primitive_scale = np.empty((0, 2), dtype=np.float32)
            primitive_rotation = np.empty((0,), dtype=np.float32)
            primitive_flip = np.empty((0,), dtype=np.int32)
            primitive_translate = np.empty((0, 2), dtype=np.float32)
            primitive_flip_invariant = np.empty((0,), dtype=np.bool)
            if self.return_size:
                primitive_size = np.empty((0, 2), dtype=np.float32)
        else:
            primitive_sem_id = np.stack(primitive_sem_id, axis=0)  ### (N, )
            primitive_prim_id = np.stack(primitive_prim_id, axis=0)  ### (N, )
            primitive_scale = np.stack(primitive_scale, axis=0)  ### (N, )
            primitive_rotation = np.stack(primitive_rotation, axis=0)  ### (N, )
            primitive_flip = np.stack(primitive_flip, axis=0)  ### (N, )
            primitive_translate = np.stack(primitive_translate, axis=0)  ### (N, 2)
            # primitive_translate = primitive_translate / (self.scene_bound / 2)
            primitive_flip_invariant = np.stack(
                primitive_flip_invariant, axis=0
            )  ### (N, )
            if self.return_size:
                primitive_size = np.stack(primitive_size, axis=0)  ### (N, 2)
        model_data["semantic_id"] = primitive_sem_id
        model_data["primitive_id"] = primitive_prim_id
        model_data["scale"] = primitive_scale
        model_data["rotation"] = primitive_rotation
        model_data["flip_mode"] = primitive_flip
        model_data["translation"] = primitive_translate
        model_data["flip_invariant_primitive"] = primitive_flip_invariant
        if self.return_size:
            model_data["size"] = primitive_size
        #### Modify the 输入格式
        # primitive_coords, primitive_feats, primitive_offsets, primitive_bboxes = load_svg_w_bg_semantic(svg_file, normalize_coord=self.normalize_coord, bg_pred=self.bg_pred, scene_bound=self.scene_bound, keep_ratio=self.keep_ratio, min_keep=self.min_keep)
        if not cache_loaded:
            json_data = json.load(open(svg_file, "r"))
        """
            将list形式转化为最终格式
        """
        viewBox = json_data["viewBox"]
        coords = json_data["coords"]
        colors = json_data["colors"]
        widths = json_data["widths"]
        primitive_ids = json_data["primitive_ids"]
        layer_ids = json_data["layer_ids"]
        semantic_ids = json_data["semantic_ids"]
        instance_ids = json_data["instance_ids"]
        primitive_lengths = json_data["primitive_lengths"]
        svg_data = SVGData(
            viewBox=viewBox,
            coords=np.array(coords, dtype=np.float32),
            colors=np.array(colors, dtype=np.int32),
            widths=np.array(widths, dtype=np.float32),
            primitive_ids=np.array(primitive_ids, dtype=np.int32),
            layer_ids=np.array(layer_ids, dtype=np.int32),
            semantic_ids=np.array(semantic_ids, dtype=np.int32),
            instance_ids=np.array(instance_ids, dtype=np.int32),
            primitive_lengths=np.array(primitive_lengths, dtype=np.float32),
        )
        # ── no_color_layer ablation: zero scene-level colors / layer_ids at
        #    the data source so all downstream transforms / vec_data conversion
        #    operate on zeros. Mirrors the same zeroing in load_primitive_lib. ──
        if getattr(self, "no_color_layer", False):
            svg_data.colors = np.zeros_like(np.asarray(svg_data.colors))
            svg_data.layer_ids = np.zeros_like(np.asarray(svg_data.layer_ids))
        ## io测试1:数据读取 ##
        t_io1_end = time.perf_counter()

        ## io测试2:预处理过滤 ##
        t_io2_start = time.perf_counter()

        # ============== 预处理: 对齐并重编号 instance_ids ============== #
        # 目标：
        # 1) 支持 svg instance_ids 非连续；
        # 2) 支持 0-based 或 1-based（当前数据常见从 1 开始）；
        # 3) 对齐到 GT 的 0-based 索引后，再压缩为连续 0..K-1。
        if len(model_data["translation"]) > 0:
            prim_instance_ids_arr = np.array(svg_data.instance_ids, dtype=np.int32)
            fg_mask = prim_instance_ids_arr >= 0
            if fg_mask.any():
                present_instance_ids = np.unique(prim_instance_ids_arr[fg_mask])
                present_instance_ids = np.sort(present_instance_ids)
                num_gt_instances = len(model_data["translation"])

                # 候选映射 A: 0-based (id -> id)
                gt_ids_zero = [
                    int(x)
                    for x in present_instance_ids
                    if 0 <= int(x) < num_gt_instances
                ]
                # 候选映射 B: 1-based (id -> id-1)
                gt_ids_one = [
                    int(x) - 1
                    for x in present_instance_ids
                    if 1 <= int(x) <= num_gt_instances
                ]

                # 选择覆盖更多 GT 的映射；若并列且最小 id >= 1，优先 1-based
                if (len(gt_ids_one) > len(gt_ids_zero)) or (
                    len(gt_ids_one) == len(gt_ids_zero)
                    and len(present_instance_ids) > 0
                    and int(present_instance_ids.min()) >= 1
                ):
                    base = 1
                    mapped_gt_ids = sorted(set(gt_ids_one))
                else:
                    base = 0
                    mapped_gt_ids = sorted(set(gt_ids_zero))

                keep_mask = np.zeros(num_gt_instances, dtype=bool)
                if len(mapped_gt_ids) > 0:
                    keep_mask[np.array(mapped_gt_ids, dtype=np.int32)] = True

                # 过滤 model_data，只保留 SVG 中存在且可映射到 GT 的 instance
                model_data["semantic_id"] = model_data["semantic_id"][keep_mask]
                model_data["primitive_id"] = model_data["primitive_id"][keep_mask]
                model_data["scale"] = model_data["scale"][keep_mask]
                model_data["rotation"] = model_data["rotation"][keep_mask]
                model_data["flip_mode"] = model_data["flip_mode"][keep_mask]
                model_data["translation"] = model_data["translation"][keep_mask]
                model_data["flip_invariant_primitive"] = model_data[
                    "flip_invariant_primitive"
                ][keep_mask]
                if self.return_size:
                    model_data["size"] = model_data["size"][keep_mask]

                # old_svg_inst_id -> gt_idx（0-based）
                if base == 1:
                    old_to_gt = {
                        int(old_id): int(old_id) - 1
                        for old_id in present_instance_ids
                        if 1 <= int(old_id) <= num_gt_instances
                    }
                else:
                    old_to_gt = {
                        int(old_id): int(old_id)
                        for old_id in present_instance_ids
                        if 0 <= int(old_id) < num_gt_instances
                    }

                # gt_idx -> new_compact_idx（连续 0..K-1）
                gt_to_new = {
                    int(gt_id): new_idx for new_idx, gt_id in enumerate(mapped_gt_ids)
                }

                # 更新 svg_data.instance_ids：
                # - 背景(-1)保持不变
                # - 无法映射到 GT 的前景 id 置为 -1，避免后续过滤错位
                inst_ids_raw = np.asarray(svg_data.instance_ids, dtype=np.int32)
                new_inst_ids = np.full_like(inst_ids_raw, -1)
                max_old_id = int(inst_ids_raw.max()) + 1 if len(inst_ids_raw) > 0 else 0
                if max_old_id > 0:
                    lut = np.full(max_old_id, -1, dtype=np.int32)
                    for old_id, gt_idx in old_to_gt.items():
                        if gt_idx in gt_to_new and old_id < max_old_id:
                            lut[old_id] = gt_to_new[gt_idx]
                    fg = inst_ids_raw >= 0
                    valid_fg = fg & (inst_ids_raw < max_old_id)
                    new_inst_ids[valid_fg] = lut[inst_ids_raw[valid_fg]]
                svg_data.instance_ids = new_inst_ids

        inst_ids_arr = np.array(svg_data.instance_ids, dtype=np.int32)
        inst_num_from_scene = np.unique(inst_ids_arr[inst_ids_arr >= 0]).shape[0]
        inst_num_from_gt = model_data["semantic_id"].shape[0]
        # assert inst_num_from_scene == inst_num_from_gt, f"inst_num_from_scene != inst_num_from_gt, {inst_num_from_scene} != {inst_num_from_gt}, error_file_name: {filename}"
        if inst_num_from_scene != inst_num_from_gt:
            print(
                f"inst_num_from_scene != inst_num_from_gt, {inst_num_from_scene} != {inst_num_from_gt}, error_file_name: {data_file}"
            )
            return self._handle_invalid_sample(
                idx,
                _invalid_retry_count,
                "pre_transform_instance_count_mismatch",
                retryable=False,
            )

        # ============== 过滤 translation 超过 [-15000, 15000] 范围的 instance ============== #
        translation_bound = 15000.0
        if self.filter_translation and len(model_data["translation"]) > 0:
            #### TODO: 存在BUG
            # 检查每个 instance 的 translation 是否在范围内（x 和 y 都需要在范围内）
            valid_instance_mask = np.all(
                np.abs(model_data["translation"]) <= translation_bound, axis=1
            )  # (N_instances,)

            # 如果所有 instance 都被过滤了，返回下一个样本
            if not np.any(valid_instance_mask):
                return self._handle_invalid_sample(
                    idx,
                    _invalid_retry_count,
                    "all_instances_filtered_by_translation",
                    retryable=False,
                )

            if not np.all(valid_instance_mask):
                old_num_instances = int(valid_instance_mask.shape[0])
                kept_instance_ids = np.flatnonzero(valid_instance_mask).astype(
                    np.int32
                )  # old instance ids
                num_kept_instances = int(kept_instance_ids.shape[0])

                # 过滤 model_data 中的 GT 数据（前景 instance 维度）
                model_data["semantic_id"] = model_data["semantic_id"][
                    valid_instance_mask
                ]
                model_data["primitive_id"] = model_data["primitive_id"][
                    valid_instance_mask
                ]
                model_data["scale"] = model_data["scale"][valid_instance_mask]
                model_data["rotation"] = model_data["rotation"][valid_instance_mask]
                model_data["flip_mode"] = model_data["flip_mode"][valid_instance_mask]
                model_data["translation"] = model_data["translation"][
                    valid_instance_mask
                ]
                model_data["flip_invariant_primitive"] = model_data[
                    "flip_invariant_primitive"
                ][valid_instance_mask]
                if self.return_size:
                    model_data["size"] = model_data["size"][valid_instance_mask]

                # old instance id -> new compact instance id（连续 0..num_kept_instances-1）
                map_inst = np.full((old_num_instances,), -1, dtype=np.int32)
                map_inst[kept_instance_ids] = np.arange(
                    num_kept_instances, dtype=np.int32
                )

                # ============== 过滤 Primitive-level 数据（按实例归属过滤） ==============
                prim_instance_ids = np.asarray(
                    svg_data.instance_ids, dtype=np.int32
                )  # (N_primitives,)
                prim_semantic_ids = np.asarray(
                    svg_data.semantic_ids, dtype=np.int32
                )  # (N_primitives,)
                prim_lengths = np.asarray(
                    svg_data.primitive_lengths, dtype=np.float32
                )  # (N_primitives,)

                # 数据集假设无背景：因此 prim_keep_mask 仅由“实例是否保留”决定
                prim_keep_mask = np.isin(prim_instance_ids, kept_instance_ids)
                prim_keep_indices = np.flatnonzero(prim_keep_mask).astype(np.int32)

                # old primitive idx -> new primitive idx（连续 0..P_kept-1）
                map_prim = np.full((prim_instance_ids.shape[0],), -1, dtype=np.int32)
                map_prim[prim_keep_indices] = np.arange(
                    prim_keep_indices.shape[0], dtype=np.int32
                )

                # ============== 过滤 Point-level 数据（按 primitive 过滤并重编号） ==============
                point_coords = np.asarray(svg_data.coords)
                point_colors = np.asarray(svg_data.colors)
                point_widths = np.asarray(svg_data.widths)
                point_prim_ids = np.asarray(svg_data.primitive_ids, dtype=np.int32)
                point_layer_ids = np.asarray(svg_data.layer_ids, dtype=np.int32)

                point_keep_mask = map_prim[point_prim_ids] >= 0
                new_coords = point_coords[point_keep_mask]
                new_colors = point_colors[point_keep_mask]
                new_widths = point_widths[point_keep_mask]
                new_layer_ids = point_layer_ids[point_keep_mask]
                new_prim_ids = map_prim[point_prim_ids[point_keep_mask]]

                # ============== 更新 svg_data（primitive-level + point-level） ==============
                svg_data.coords = new_coords
                svg_data.colors = new_colors
                svg_data.widths = new_widths
                svg_data.primitive_ids = new_prim_ids
                svg_data.layer_ids = new_layer_ids
                svg_data.semantic_ids = prim_semantic_ids[prim_keep_mask]
                svg_data.instance_ids = map_inst[prim_instance_ids[prim_keep_mask]]
                svg_data.primitive_lengths = prim_lengths[prim_keep_mask]

        # ============== 过滤 size 超过 [0, 40000] 范围的 instance ============== #
        size_bound = 40000.0
        # if self.return_size and len(model_data['size']) > 0 and self.filter_size:
        if self.return_size and len(model_data["size"]) > 0 and self.filter_size:

            # 检查每个 instance 的 size 是否在范围内（width 和 height 都需要在 [0, size_bound] 范围内）
            valid_size_mask = np.all(
                (model_data["size"] >= 0) & (model_data["size"] <= size_bound), axis=1
            )  # (N_instances,)

            # 如果所有 instance 都被过滤了，返回下一个样本
            if not np.any(valid_size_mask):
                return self._handle_invalid_sample(
                    idx,
                    _invalid_retry_count,
                    "all_instances_filtered_by_size",
                    retryable=False,
                )

            if not np.all(valid_size_mask):
                old_num_instances = int(valid_size_mask.shape[0])
                kept_instance_ids = np.flatnonzero(valid_size_mask).astype(
                    np.int32
                )  # old instance ids
                num_kept_instances = int(kept_instance_ids.shape[0])

                # 过滤 model_data 中的 GT 数据（前景 instance 维度）
                model_data["semantic_id"] = model_data["semantic_id"][valid_size_mask]
                model_data["primitive_id"] = model_data["primitive_id"][valid_size_mask]
                model_data["scale"] = model_data["scale"][valid_size_mask]
                model_data["rotation"] = model_data["rotation"][valid_size_mask]
                model_data["flip_mode"] = model_data["flip_mode"][valid_size_mask]
                model_data["translation"] = model_data["translation"][valid_size_mask]
                model_data["flip_invariant_primitive"] = model_data[
                    "flip_invariant_primitive"
                ][valid_size_mask]
                model_data["size"] = model_data["size"][valid_size_mask]

                # old instance id -> new compact instance id
                map_inst = np.full((old_num_instances,), -1, dtype=np.int32)
                map_inst[kept_instance_ids] = np.arange(
                    num_kept_instances, dtype=np.int32
                )

                # ============== 过滤 Primitive-level 数据 ==============
                prim_instance_ids = np.asarray(svg_data.instance_ids, dtype=np.int32)
                prim_semantic_ids = np.asarray(svg_data.semantic_ids, dtype=np.int32)
                prim_lengths = np.asarray(svg_data.primitive_lengths, dtype=np.float32)

                prim_keep_mask = np.isin(prim_instance_ids, kept_instance_ids)
                prim_keep_indices = np.flatnonzero(prim_keep_mask).astype(np.int32)

                map_prim = np.full((prim_instance_ids.shape[0],), -1, dtype=np.int32)
                map_prim[prim_keep_indices] = np.arange(
                    prim_keep_indices.shape[0], dtype=np.int32
                )

                # ============== 过滤 Point-level 数据 ==============
                point_coords = np.asarray(svg_data.coords)
                point_colors = np.asarray(svg_data.colors)
                point_widths = np.asarray(svg_data.widths)
                point_prim_ids = np.asarray(svg_data.primitive_ids, dtype=np.int32)
                point_layer_ids = np.asarray(svg_data.layer_ids, dtype=np.int32)

                point_keep_mask = map_prim[point_prim_ids] >= 0
                new_coords = point_coords[point_keep_mask]
                new_colors = point_colors[point_keep_mask]
                new_widths = point_widths[point_keep_mask]
                new_layer_ids = point_layer_ids[point_keep_mask]
                new_prim_ids = map_prim[point_prim_ids[point_keep_mask]]

                # ============== 更新 svg_data ==============
                svg_data.coords = new_coords
                svg_data.colors = new_colors
                svg_data.widths = new_widths
                svg_data.primitive_ids = new_prim_ids
                svg_data.layer_ids = new_layer_ids
                svg_data.semantic_ids = prim_semantic_ids[prim_keep_mask]
                svg_data.instance_ids = map_inst[prim_instance_ids[prim_keep_mask]]
                svg_data.primitive_lengths = prim_lengths[prim_keep_mask]

        #### 确保输入的svg_data对应的instance数目和GT对应的instance相对应 ####
        inst_ids_arr = np.array(svg_data.instance_ids, dtype=np.int32)
        inst_num_from_scene = np.unique(inst_ids_arr[inst_ids_arr >= 0]).shape[0]
        inst_num_from_gt = model_data["semantic_id"].shape[0]
        #### 假设该数据集没有背景类别，也就是inst_num_from_scene 一定 = inst_num_from_gt
        assert (
            inst_num_from_scene == inst_num_from_gt
        ), f"inst_num_from_scene != inst_num_from_gt, {inst_num_from_scene} != {inst_num_from_gt}, error_file_name: {data_file}"
        # if inst_num_from_scene != inst_num_from_gt:
        #     print(f"inst_num_from_scene != inst_num_from_gt, {inst_num_from_scene} != {inst_num_from_gt}, error_file_name: {data_file}")
        #     return self.__getitem__((idx + 1) % len(self))

        ## io测试2:预处理过滤 ##
        t_io2_end = time.perf_counter()

        ## io测试3:数据增强处理 ##
        t_io3_start = time.perf_counter()
        if self.recon_audit or self.return_recon_data:
            initial_num_instances = int(len(model_data["translation"]))
            model_data["_recon_audit_enabled"] = True
            model_data["_recon_audit"] = {
                "initial_num_instances": initial_num_instances,
                "instance_origin_kind": np.zeros(initial_num_instances, dtype=np.int8),
                "instance_jittered": np.zeros(initial_num_instances, dtype=np.bool_),
            }
            if self.recon_preclip_full_source:
                model_data["_recon_capture_preclip_source"] = True
        deterministic_fallback = (
            self.invalid_sample_policy == "retry_same_then_deterministic"
            and _invalid_retry_count > self.invalid_sample_max_retries
        )
        svg_data, vec_data, model_data = self._transform(
            svg_data,
            model_data,
            crop_window,
            deterministic_fallback=deterministic_fallback,
        )
        # 若 VecClipping 判定该样本在 clip_range 下无有效线段，则丢弃该样本并取下一个
        if bool(model_data.get("_drop_sample", False)):
            return self._handle_invalid_sample(
                idx,
                _invalid_retry_count,
                "vec_clipping_empty",
                retryable=self.split == "train",
            )
        # Keep every augmentation in the historical relative-template space,
        # then perform one exact label-only conversion for the decoder/recon.
        self._reparameterize_scale_labels(model_data)
        t_io3_end = time.perf_counter()
        ## io测试3:数据增强处理 ##
        #### 二次检查：确保输入的svg_data对应的instance数目和GT对应的instance相对应 ####
        inst_ids_arr = np.array(svg_data.instance_ids, dtype=np.int32)
        inst_num_from_scene = np.unique(inst_ids_arr[inst_ids_arr >= 0]).shape[0]
        inst_num_from_gt = model_data["semantic_id"].shape[0]
        #### 假设该数据集没有背景类别，也就是inst_num_from_scene 一定 = inst_num_from_gt
        # assert inst_num_from_scene == inst_num_from_gt, f"inst_num_from_scene != inst_num_from_gt, {inst_num_from_scene} != {inst_num_from_gt}, error_file_name: {data_file}"
        if inst_num_from_scene != inst_num_from_gt:
            print(
                f"inst_num_from_scene != inst_num_from_gt, {inst_num_from_scene} != {inst_num_from_gt}, error_file_name: {data_file}"
            )
            return self._handle_invalid_sample(
                idx,
                _invalid_retry_count,
                "post_transform_instance_count_mismatch",
                retryable=self.split == "train",
            )

        model_data["svg_data"] = svg_data
        model_data["vec_data"] = vec_data
        model_data["filename"] = data_file

        if self.hilbert_sort:
            #### first_normalize the translation into [0-1]
            if self.discretize_trans is not None:
                sort_discrete_translation = self.discretize_trans(
                    model_data["translation"]
                )
                normalize_translation = sort_discrete_translation.numpy() / (
                    self.num_discrete_translation - 1
                )
                normalize_translation = np.clip(normalize_translation, 0.0, 1.0)
                sort_indices = hilbert_order(normalize_translation)
            else:
                normalize_translation = copy.deepcopy(model_data["translation"].numpy())
                normalize_translation = (
                    normalize_translation * (self.scene_bound / 2)
                    + (self.scene_bound / 2)
                ) / self.scene_bound
                normalize_translation = np.clip(normalize_translation, 0.0, 1.0)
                sort_indices = hilbert_order(normalize_translation)
        else:
            if self.discretize_trans is not None:
                sort_discrete_translation = self.discretize_trans(
                    model_data["translation"]
                )
                sort_indices = np.lexsort(
                    (
                        sort_discrete_translation[:, 1].numpy(),
                        sort_discrete_translation[:, 0].numpy(),
                    )
                )
            else:
                sort_indices = np.lexsort(
                    (
                        model_data["translation"][:, 1].numpy(),
                        model_data["translation"][:, 0].numpy(),
                    )
                )

        # 保存原始顺序的逆映射：sort_indices[i]表示第i个排序后的元素在原始序列中的位置
        # original_indices[j]表示原始序列第j个元素在排序后的位置
        original_indices = np.argsort(sort_indices)

        if self.recon_audit or self.return_recon_data:
            # Build the proposed V1 post-grid membership from the final scene,
            # before XY_SORT changes the decoder-slot order. This is deliberately
            # O(N) grouping, not a geometric template/source matching operation.
            coords_audit = np.asarray(
                (
                    svg_data.coords.detach().cpu().numpy()
                    if isinstance(svg_data.coords, torch.Tensor)
                    else svg_data.coords
                ),
                dtype=np.float32,
            ).reshape(-1, 2, 2)
            point_primitive_ids = np.asarray(
                (
                    svg_data.primitive_ids.detach().cpu().numpy()
                    if isinstance(svg_data.primitive_ids, torch.Tensor)
                    else svg_data.primitive_ids
                ),
                dtype=np.int64,
            ).reshape(-1)
            primitive_instance_ids = np.asarray(
                (
                    svg_data.instance_ids.detach().cpu().numpy()
                    if isinstance(svg_data.instance_ids, torch.Tensor)
                    else svg_data.instance_ids
                ),
                dtype=np.int64,
            ).reshape(-1)
            line_instance_ids = primitive_instance_ids[point_primitive_ids]
            num_instances_audit = int(len(model_data["translation"]))
            raw_counts = np.asarray(
                [
                    np.count_nonzero(line_instance_ids == i)
                    for i in range(num_instances_audit)
                ],
                dtype=np.int64,
            )
            audit = model_data.get("_recon_audit", {})
            # ToTensor may already have converted the nested dictionary.
            audit["final_num_instances"] = num_instances_audit
            audit["final_num_scene_segments"] = int(len(coords_audit))
            audit["final_foreground_segment_counts_raw"] = raw_counts
            audit["final_foreground_segment_counts_sorted"] = raw_counts[sort_indices]
            audit["final_present_raw_instance_ids"] = np.unique(
                line_instance_ids[line_instance_ids >= 0]
            ).astype(np.int64)
            audit["sort_indices"] = np.asarray(sort_indices, dtype=np.int64)
            audit["original_indices"] = np.asarray(original_indices, dtype=np.int64)
            model_data["_recon_audit"] = audit
            model_data.pop("_recon_audit_enabled", None)
            model_data.pop("_recon_capture_preclip_source", None)

            if self.return_recon_data:

                def _as_numpy(value, dtype):
                    if isinstance(value, torch.Tensor):
                        value = value.detach().cpu().numpy()
                    return np.asarray(value, dtype=dtype)

                clip_ratio = _as_numpy(
                    audit.get(
                        "clip_count_ratio",
                        np.ones(num_instances_audit, dtype=np.float32),
                    ),
                    np.float32,
                ).reshape(-1)
                clip_length_ratio = _as_numpy(
                    audit.get(
                        "clip_length_ratio",
                        np.ones(num_instances_audit, dtype=np.float32),
                    ),
                    np.float32,
                ).reshape(-1)
                full_translation = _as_numpy(
                    audit.get("full_translation_pre_clip", model_data["translation"]),
                    np.float32,
                ).reshape(-1, 2)
                visible_translation = _as_numpy(
                    model_data["translation"], np.float32
                ).reshape(-1, 2)
                source_segments = np.zeros(
                    (
                        num_instances_audit,
                        self.recon_source_limit,
                        2,
                        2,
                    ),
                    dtype=np.float32,
                )
                source_mask = np.zeros(
                    (num_instances_audit, self.recon_source_limit), dtype=np.bool_
                )
                source_raw_count = np.zeros(num_instances_audit, dtype=np.int64)

                def _pack_recon_segments(all_segments, all_instance_ids):
                    packed = np.zeros(
                        (
                            num_instances_audit,
                            self.recon_source_limit,
                            2,
                            2,
                        ),
                        dtype=np.float32,
                    )
                    packed_mask = np.zeros(
                        (num_instances_audit, self.recon_source_limit),
                        dtype=np.bool_,
                    )
                    raw = np.zeros(num_instances_audit, dtype=np.int64)
                    for raw_instance_id in range(num_instances_audit):
                        segments = all_segments[all_instance_ids == raw_instance_id]
                        raw[raw_instance_id] = len(segments)
                        if len(segments) > self.recon_source_limit:
                            lengths = np.linalg.norm(
                                segments[:, 1] - segments[:, 0], axis=1
                            )
                            length_order = np.argsort(lengths, kind="stable")
                            ranks = (
                                np.linspace(
                                    0,
                                    len(length_order) - 1,
                                    self.recon_source_limit,
                                )
                                .round()
                                .astype(np.int64)
                            )
                            segments = segments[length_order[ranks]]
                        count = min(len(segments), self.recon_source_limit)
                        if count:
                            packed[raw_instance_id, :count] = segments[:count]
                            packed_mask[raw_instance_id, :count] = True
                    return packed, packed_mask, raw

                source_segments, source_mask, source_raw_count = _pack_recon_segments(
                    coords_audit, line_instance_ids
                )

                translation_bin = (
                    float(self.continuous_range_translation[1])
                    - float(self.continuous_range_translation[0])
                ) / max(int(self.num_discrete_translation or 2) - 1, 1)
                translation_delta = np.linalg.norm(
                    visible_translation - full_translation, axis=1
                )
                translation_consistent = translation_delta <= (
                    self.recon_translation_tolerance_bins * translation_bin + 1e-8
                )
                recon_valid = (source_raw_count > 0) & translation_consistent
                if not self.recon_include_partial:
                    recon_valid &= clip_ratio >= 1.0 - 1e-7
                model_data["recon_source_segments"] = torch.from_numpy(
                    source_segments[sort_indices]
                )
                model_data["recon_source_mask"] = torch.from_numpy(
                    source_mask[sort_indices]
                )
                model_data["recon_valid_mask"] = torch.from_numpy(
                    recon_valid[sort_indices]
                )
                model_data["recon_full_translation"] = torch.from_numpy(
                    full_translation[sort_indices]
                )
                model_data["recon_clip_count_ratio"] = torch.from_numpy(
                    clip_ratio[sort_indices]
                )
                model_data["recon_clip_length_ratio"] = torch.from_numpy(
                    clip_length_ratio[sort_indices]
                )
                model_data["recon_is_partial"] = torch.from_numpy(
                    (clip_ratio[sort_indices] < 1.0 - 1e-7)
                )
                model_data["recon_translation_delta"] = torch.from_numpy(
                    translation_delta[sort_indices].astype(np.float32)
                )
                model_data["recon_source_raw_count"] = torch.from_numpy(
                    source_raw_count[sort_indices]
                )
                if self.recon_preclip_full_source:
                    if not all(
                        key in audit
                        for key in (
                            "preclip_full_coords",
                            "preclip_full_line_instance_ids",
                        )
                    ):
                        raise RuntimeError(
                            "VecClipping did not emit preclip full-source data"
                        )
                    preclip_coords = _as_numpy(
                        audit["preclip_full_coords"], np.float32
                    ).reshape(-1, 2, 2)
                    preclip_ids = _as_numpy(
                        audit["preclip_full_line_instance_ids"], np.int64
                    ).reshape(-1)
                    (
                        full_source_segments,
                        full_source_mask,
                        full_source_raw_count,
                    ) = _pack_recon_segments(preclip_coords, preclip_ids)
                    recon_valid &= full_source_raw_count > 0
                    model_data["recon_valid_mask"] = torch.from_numpy(
                        recon_valid[sort_indices]
                    )
                    model_data["recon_preclip_source_segments"] = torch.from_numpy(
                        full_source_segments[sort_indices]
                    )
                    model_data["recon_preclip_source_mask"] = torch.from_numpy(
                        full_source_mask[sort_indices]
                    )
                    model_data["recon_preclip_source_raw_count"] = torch.from_numpy(
                        full_source_raw_count[sort_indices]
                    )
                if not self.recon_audit:
                    model_data.pop("_recon_audit", None)

        model_data["sort_indices"] = torch.from_numpy(sort_indices)  # 保存排序索引
        model_data["original_indices"] = torch.from_numpy(
            original_indices
        )  # 保存原始位置索引
        model_data["semantic_id"] = model_data["semantic_id"][sort_indices]
        model_data["primitive_id"] = model_data["primitive_id"][sort_indices]
        model_data["scale"] = model_data["scale"][sort_indices]
        model_data["rotation"] = model_data["rotation"][sort_indices]
        model_data["flip_mode"] = model_data["flip_mode"][sort_indices]
        model_data["translation"] = model_data["translation"][sort_indices]
        model_data["flip_invariant_primitive"] = model_data["flip_invariant_primitive"][
            sort_indices
        ]
        if self.return_size:
            model_data["size"] = model_data["size"][sort_indices]
        t_getitem_end = time.perf_counter()
        model_data["io_timing"] = {
            "io_test1_read": t_io1_end - t_io1_start,
            "io_test2_filter": t_io2_end - t_io2_start,
            "io_test3_transform": t_io3_end - t_io3_start,
            "getitem_total": t_getitem_end - t_getitem_start,
        }
        return model_data
        # coord = self.normalize_coordinates(coord)

        # model_data = {}
        # model_data['coord'] = np.array(coord, dtype=np.float32)
        # model_data['feat'] = np.array(feat, dtype=np.float32)
        # model_data['semanticIds'] = np.array(semanticIds, dtype=np.int32)
        # model_data['instanceIds'] = np.array(instanceIds, dtype=np.int32)

        # ### load img files
        # img_file = os.path.join(self.img_dir, self.data_filename[idx].replace('.json', '_reg.png'))
        # img = Image.open(img_file)
        # img = np.array(img, dtype=np.float32)
        # model_data['img'] = img

        # ### Load the GT primitive
        # json_file = os.path.join(self.primitive_dir, self.data_filename[idx].replace('.svg', '.json'))
        # with open(json_file, 'r', encoding='utf-8') as f:
        #     primitives_json = json.load(f)
        # primitives = primitives_json
        # primitive_sem_id = []
        # primitive_prim_id = []
        # primitive_scale = []
        # primitive_rotation = []
        # primitive_flip = []
        # primitive_translate = []
        # for primitive in primitives:
        #     primitive_sem_id.append(np.array(primitive['semantic_id'], dtype=np.int32))
        #     primitive_prim_id.append(np.array(primitive['primitive_id'], dtype=np.int32))
        #     primitive_scale.append(np.array(primitive['scale'], dtype=np.float32))
        #     primitive_rotation.append(np.array(primitive['rotation'], dtype=np.float32))
        #     primitive_flip.append(np.array(primitive['flip_mode'], dtype=np.int32))
        #     primitive_translate.append(np.array(primitive['translation'], dtype=np.float32))
        # primitive_sem_id = np.stack(primitive_sem_id, axis=0)
        # primitive_prim_id = np.stack(primitive_prim_id, axis=0)
        # primitive_scale = np.stack(primitive_scale, axis=0)
        # primitive_rotation = np.stack(primitive_rotation, axis=0)
        # primitive_flip = np.stack(primitive_flip, axis=0)
        # primitive_translate = np.stack(primitive_translate, axis=0)

        # #### Mapping coordinate from [-70, 70] into [-1, 1]
        # primitive_translate = primitive_translate / 70.0

        # model_data['semantic_id'] = primitive_sem_id
        # model_data['primitive_id'] = primitive_prim_id
        # model_data['scale'] = primitive_scale
        # model_data['rotation'] = primitive_rotation
        # model_data['flip_mode'] = primitive_flip
        # model_data['translation'] = primitive_translate.reshape(-1, 2)

        # if self.transform is not None:
        #     model_data = self.transform(model_data)

        # sort_indices = np.lexsort((model_data['translation'][:, 1].numpy(), model_data['translation'][:, 0].numpy()))
        # model_data['semantic_id'] = model_data['semantic_id'][sort_indices]
        # model_data['primitive_id'] = model_data['primitive_id'][sort_indices]
        # model_data['scale'] = model_data['scale'][sort_indices]
        # model_data['rotation'] = model_data['rotation'][sort_indices]
        # model_data['flip_mode'] = model_data['flip_mode'][sort_indices]
        # model_data['translation'] = model_data['translation'][sort_indices]

        # return model_data

        # primitive_types = []
        # primitive_scale = []
        # primitive_translation = []
        # primitive_rotation = []
        # for primitive in primitives:
        #     primitive_types.append(np.array(primitive['type_code'], dtype=np.int32))
        #     primitive_scale.append(np.array(primitive['scale'], dtype=np.float32))
        #     # 对translation进行归一化处理，从[0,140]映射到[-1,1]
        #     normalized_translate = normalize_translation(primitive['translate'])
        #     primitive_translation.append(np.array(normalized_translate, dtype=np.float32))
        #     primitive_rotation.append(np.array(primitive['rotate'], dtype=np.float32))
        # primitive_types = np.stack(primitive_types, axis=0)
        # primitive_scale = np.stack(primitive_scale, axis=0)
        # primitive_translation = np.stack(primitive_translation, axis=0)
        # primitive_rotation = np.stack(primitive_rotation, axis=0)

        # model_data['type_code'] = primitive_types
        # model_data['scale'] = primitive_scale
        # model_data['translation'] = primitive_translation
        # model_data['rotation'] = primitive_rotation

        # # 应用数据增强
        # if self.transform is not None:
        #     model_data = self.transform(model_data)

        # return model_data


def first(it):
    return it[0]


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
    sequences = {field: [getattr(item, field) for item in batch] for field in fields}

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


def custom_collate(data, pad_id=-1):
    from collections.abc import Sequence, Mapping
    from torch.utils.data.dataloader import default_collate

    # 定义需要padding处理的primitive相关keys
    primitive_keys = {
        "semantic_id",
        "primitive_id",
        "scale",
        "translation",
        "rotation",
        "flip_mode",
        "original_indices",
        "global_translation",
        "primitive_bboxes",
        "flip_invariant_primitive",
        "size",
        "recon_source_segments",
        "recon_source_mask",
        "recon_valid_mask",
        "recon_full_translation",
        "recon_clip_count_ratio",
        "recon_clip_length_ratio",
        "recon_is_partial",
        "recon_translation_delta",
        "recon_source_raw_count",
        "recon_preclip_source_segments",
        "recon_preclip_source_mask",
        "recon_preclip_source_raw_count",
    }
    recon_zero_pad_keys = {
        "recon_source_segments",
        "recon_source_mask",
        "recon_valid_mask",
        "recon_full_translation",
        "recon_clip_count_ratio",
        "recon_clip_length_ratio",
        "recon_is_partial",
        "recon_translation_delta",
        "recon_source_raw_count",
        "recon_preclip_source_segments",
        "recon_preclip_source_mask",
        "recon_preclip_source_raw_count",
    }
    img_keys = {"img"}
    vec_data_keys = {"vec_data"}  # 需要调用 collate_fn 处理的 keys

    # 定义需要特殊处理的点云相关keys
    pointcloud_keys = {
        key
        for key in data[0].keys()
        if key not in primitive_keys
        and key not in img_keys
        and key not in vec_data_keys
        and key not in ["filename", "sort_indices", "svg_data"]
    }

    def pointcloud_collate_fn(batch):
        """点云数据的collate逻辑"""
        if not isinstance(batch, Sequence):
            raise TypeError(f"{batch.dtype} is not supported.")

        if isinstance(batch[0], torch.Tensor):
            return torch.cat(list(batch))
        elif isinstance(batch[0], str):
            return list(batch)
        elif isinstance(batch[0], Sequence):
            for data in batch:
                data.append(torch.tensor([data[0].shape[0]]))
            batch = [pointcloud_collate_fn(samples) for samples in zip(*batch)]
            batch[-1] = torch.cumsum(batch[-1], dim=0).int()
            return batch
        elif isinstance(batch[0], Mapping):
            batch = {
                key: (
                    pointcloud_collate_fn([d[key] for d in batch])
                    if "offset" not in key
                    else torch.cumsum(
                        pointcloud_collate_fn(
                            [d[key].diff(prepend=torch.tensor([0])) for d in batch]
                        ),
                        dim=0,
                    )
                )
                for key in batch[0]
            }
            return batch
        else:
            return default_collate(batch)

    def primitive_collate_fn(datum):
        """primitive数据的collate逻辑（原来的padding逻辑）"""
        if is_tensor(first(datum)):
            return pad_sequence(datum, batch_first=True, padding_value=pad_id)
        else:
            return list(datum)

    def img_collate_fn(datum):
        """img数据的collate逻辑"""
        if is_tensor(first(datum)):
            return torch.stack(datum, dim=0).permute(0, 3, 1, 2)
        else:
            return list(datum)

    def vec_data_collate_fn(batch_vec_data):
        """vec_data的collate逻辑，调用collate_fn的处理方式"""
        if not batch_vec_data:
            raise ValueError("Batch cannot be empty")

        # Define fields to extract
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
            field: [getattr(item, field) for item in batch_vec_data] for field in fields
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
            "data_paths": [item.data_path for item in batch_vec_data],
        }

        return concat_data

    # 主处理逻辑
    is_dict = isinstance(first(data), dict)

    if not is_dict:
        # 如果不是字典格式，使用原来的逻辑
        output = []
        for datum in zip(*data):
            if is_tensor(first(datum)):
                datum = pad_sequence(datum, batch_first=True, padding_value=pad_id)
            else:
                datum = list(datum)
            output.append(datum)
        return tuple(output)

    # 字典格式的处理
    keys = first(data).keys()
    output = {}

    for key in keys:
        key_data = [d[key] for d in data]
        if key == "index_valid_keys":
            continue
        if key in pointcloud_keys:
            # 对点云相关的key使用点云collate逻辑
            if key == "offset":
                # offset需要特殊处理
                output[key] = torch.cumsum(
                    torch.cat([d.diff(prepend=torch.tensor([0])) for d in key_data]),
                    dim=0,
                )
            elif key == "primitive_index":
                #### 不做任何处理
                output[key] = key_data
            elif key == "instance_masks":
                #### 不做任何处理
                output[key] = key_data
            else:
                # coord, feat, semanticIds, instanceIds使用拼接
                if isinstance(key_data[0], torch.Tensor):
                    output[key] = torch.cat(key_data, dim=0)
                else:
                    output[key] = pointcloud_collate_fn(key_data)
        elif key in primitive_keys:
            # 对primitive相关的key使用padding逻辑
            if key in recon_zero_pad_keys and is_tensor(first(key_data)):
                output[key] = pad_sequence(key_data, batch_first=True, padding_value=0)
            else:
                output[key] = primitive_collate_fn(key_data)
        elif key in img_keys:
            # 对img相关的key使用默认逻辑
            output[key] = img_collate_fn(key_data)
        elif key in vec_data_keys:
            # 对vec_data使用collate_fn的逻辑
            output[key] = vec_data_collate_fn(key_data)
        elif key == "primitive_lib":
            output[key] = key_data[0]
        elif key == "svg_data":
            output[key] = key_data
        else:
            # 其他key使用默认逻辑
            if is_tensor(first(key_data)):
                output[key] = pad_sequence(
                    key_data, batch_first=True, padding_value=pad_id
                )
            else:
                output[key] = list(key_data)

    return output
