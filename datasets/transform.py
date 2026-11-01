"""
3D point cloud augmentation

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import random
import numbers
from tkinter import EW
import scipy
import scipy.ndimage
import scipy.interpolate
import scipy.stats
import numpy as np
import torch
import copy
import math
import json
from collections.abc import Sequence, Mapping

from models.primitive.utils.registry import Registry
from datasets.primitive_dataset import flip_invariant_primitive_id
from datasets.dataclass import InstanceData, SVGData

# flip_invariant_primitive_id = []

TRANSFORMS = Registry("transforms")


def apply_transform_to_primitives(data_dict, transform_matrix, center=None):
    """
    将几何变换应用到primitive参数上

    Args:
        data_dict: 包含primitive参数的数据字典
        transform_matrix: 4x4变换矩阵（numpy数组）
        center: 变换中心点，用于平移
    """
    if "translation" in data_dict and "rotation" in data_dict:
        translation = data_dict["translation"].clone()
        rotation = data_dict["rotation"].clone()

        # 处理平移：先减去中心点，应用变换，再加回中心点
        if center is not None:
            center_tensor = torch.tensor(
                center[:2], dtype=translation.dtype, device=translation.device
            )
            translation = translation - center_tensor

        # 将平移向量变换（只处理2D部分）
        transform_tensor = torch.tensor(
            transform_matrix[:2, :2], dtype=translation.dtype, device=translation.device
        )
        translation_transformed = torch.mm(translation, transform_tensor.T)

        if center is not None:
            translation_transformed = translation_transformed + center_tensor

        # 处理旋转：提取旋转角度的变化
        # 计算变换矩阵对应的旋转角度
        transform_angle = torch.atan2(transform_tensor[1, 0], transform_tensor[0, 0])

        # 将变换角度加到现有旋转上
        new_rotation = rotation + transform_angle

        data_dict["translation"] = translation_transformed
        data_dict["rotation"] = new_rotation

    return data_dict


def index_operator(data_dict, index, duplicate=False):
    # index selection operator for keys in "index_valid_keys"
    # custom these keys by "Update" transform in config
    if "index_valid_keys" not in data_dict:
        data_dict["index_valid_keys"] = [
            "coord",
            "color",
            "normal",
            "superpoint",
            "strength",
            "segment",
            "instance",
            "feat",
            "semanticIds",
            "instanceIds",
        ]
    if not duplicate:
        for key in data_dict["index_valid_keys"]:
            if key in data_dict:
                data_dict[key] = data_dict[key][index]
        return data_dict
    else:
        data_dict_ = dict()
        for key in data_dict.keys():
            if key in data_dict["index_valid_keys"]:
                data_dict_[key] = data_dict[key][index]
            elif key == "index_valid_keys":
                data_dict_[key] = copy.copy(data_dict[key])
            else:
                data_dict_[key] = data_dict[key]
        return data_dict_


#### 修改translation的范围 ####
#### random translate the whole input
@TRANSFORMS.register_module()
class RandomSceneShift(object):
    """
    对部分实例做小幅平移；同步更新 primitive_coords 和 translation
    Args:
        p: 触发概率
        sample_ratio: 被选中的实例比例
        max_delta: 每轴平移幅度（与坐标同量纲，eg. 归一化到[-1,1]则给0.02之类）
        axis_bias: None / 'x' / 'y' （若你想假设“墙横平竖直”并沿某轴更常见，可稍后加权）
    """

    def __init__(
        self,
        p=0.0,
        sample_ratio=0.3,
        max_delta=0.1,
        axis_bias=None,
        scene_bound=20000.0,
        clip_bound=None,
    ):
        assert 0 <= p <= 1, f"p should be in [0, 1], got {p}"
        self.p = p
        self.max_delta = max_delta
        self.axis_bias = axis_bias
        self.scene_bound = scene_bound
        self.clip_bound = clip_bound

    def __call__(self, data_dict):
        if "coord" not in data_dict and "svg_data" not in data_dict:
            return data_dict
        if self.p <= 0.0 or (self.p < 1.0 and np.random.rand() >= self.p):
            return data_dict

        # 检查 coord 是否有点在 [-1, 1] 范围内
        # coord = data_dict['coord']
        # has_valid_points = ((coord[:, 0] >= -1) & (coord[:, 0] <= 1) &
        #                     (coord[:, 1] >= -1) & (coord[:, 1] <= 1)).any()

        # # 如果没有有效点，强制进行 shift；否则按概率 p 决定
        # if has_valid_points and np.random.rand() > self.p:
        #     return data_dict

        # coords_list = data_dict['primitive_coords']
        # feats_list = data_dict['primitive_feats']
        # translation = data_dict['translation']          # [N,2] 或 [N,3](用前两维)
        # N = len(coords_list)

        # dx = np.random.uniform(-self.max_delta, self.max_delta)
        # dy = np.random.uniform(-self.max_delta, self.max_delta)
        # if self.axis_bias == 'x': dy *= 0.5
        # if self.axis_bias == 'y': dx *= 0.5
        # delta = np.array([dx, dy], dtype=np.float32)

        # # 更新坐标
        # for i in range(N):
        #     coords_list[i][:, :2] += delta * (self.scene_bound / 2)
        #     feats_list[i][:, 0] = np.arctan(coords_list[i][:, 1] / (coords_list[i][:, 0] + 1e-8)) / math.pi
        #     # 更新属性
        #     translation[i, :2] += delta

        # data_dict['primitive_coords'] = coords_list
        # data_dict['primitive_feats'] = feats_list
        # data_dict['translation'] = translation
        if "coord" in data_dict:
            coord = data_dict["coord"]
            coord_min, coord_max = np.min(coord[:, :2], axis=0), np.max(
                coord[:, :2], axis=0
            )
            # feat = data_dict['feat']
            translation = data_dict["translation"]
            if self.clip_bound is not None:
                # min_delta = np.array([self.clip_bound, self.clip_bound]) - coord_max
                # max_delta = -np.array([self.clip_bound, self.clip_bound]) - coord_min
                min_delta = -np.array([self.clip_bound, self.clip_bound]) - coord_max
                max_delta = np.array([self.clip_bound, self.clip_bound]) - coord_min
            else:
                min_delta = -self.max_delta
                max_delta = self.max_delta
            # dx = np.random.uniform(min_delta, max_delta)
            # dy = np.random.uniform(min_delta, max_delta)
            # if self.axis_bias == 'x': dy *= 0.5
            # if self.axis_bias == 'y': dx *= 0.5
            # delta = np.array([dx, dy], dtype=np.float32)
            delta = np.random.uniform(min_delta, max_delta)

            #### 执行平移 ####
            coord = coord + delta
            # feat[:, 0] = np.arctan(coord[:, 1] / (coord[:, 0] + 1e-8)) / math.pi
            translation += delta
            data_dict["coord"] = coord
            # data_dict['feat'] = feat
            data_dict["translation"] = translation
        elif "svg_data" in data_dict:
            svg_data = data_dict["svg_data"]
            coord = svg_data.coords
            coord_min, coord_max = np.min(coord[:, :2], axis=0), np.max(
                coord[:, :2], axis=0
            )
            # feat = data_dict['feat']
            translation = data_dict["translation"]
            if self.clip_bound is not None:
                # min_delta = np.array([self.clip_bound, self.clip_bound]) - coord_max
                # max_delta = -np.array([self.clip_bound, self.clip_bound]) - coord_min
                min_delta = -np.array([self.clip_bound, self.clip_bound]) - coord_max
                max_delta = np.array([self.clip_bound, self.clip_bound]) - coord_min
            else:
                min_delta = -self.max_delta
                max_delta = self.max_delta
            # dx = np.random.uniform(min_delta, max_delta)
            # dy = np.random.uniform(min_delta, max_delta)
            # if self.axis_bias == 'x': dy *= 0.5
            # if self.axis_bias == 'y': dx *= 0.5
            # delta = np.array([dx, dy], dtype=np.float32)
            delta = np.random.uniform(min_delta, max_delta)

            #### 执行平移 ####
            coord = coord + delta
            translation += delta
            data_dict["svg_data"].coords = coord
            data_dict["translation"] = translation
        if bool(data_dict.get("_recon_audit_enabled", False)):
            audit = data_dict.setdefault("_recon_audit", {})
            audit["global_shift"] = np.asarray(delta, dtype=np.float32)
        return data_dict


##### Random scale the whole scene
@TRANSFORMS.register_module()
class RandomSceneScale(object):
    """
    对整个场景做小幅尺度扰动；同步更新 primitive_coords 和 scale
    Args:
        p: 触发概率
        sample_ratio: 被选中的实例比例
        max_delta: 每轴平移幅度（与坐标同量纲，eg. 归一化到[-1,1]则给0.02之类）
        axis_bias: None / 'x' / 'y' （若你想假设“墙横平竖直”并沿某轴更常见，可稍后加权）
    """

    def __init__(self, p=0.0, scale_range=[0.9, 1.1]):
        self.p = p
        self.scale_range = scale_range

    def __call__(self, data_dict):
        if np.random.rand() > self.p or "scale" not in data_dict:
            return data_dict
        if "coord" in data_dict:
            coord = data_dict["coord"]
            # feat = data_dict['feat']
            scale = data_dict["scale"]
            translation = data_dict["translation"]
            random_scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
            coord = coord * random_scale
            # feat[:, 0] = np.arctan(coord[:, 1] / (coord[:, 0] + 1e-8)) / math.pi
            # feat = feat * random_scale
            scale = scale * random_scale
            translation = translation * random_scale
            ### update for primitive_bboxes: 根据primitive_indexs重新计算primitive_bboxes
            # assert 'primitive_index' in data_dict, "primitive_index is not in data_dict"
            # primitive_index = data_dict['primitive_index']
            # update_primitive_bboxes = []
            # for prim_idx in primitive_index:
            #     primitive_coords = data_dict['coord'][prim_idx[0]:prim_idx[1]]
            #     primitive_w = np.max(primitive_coords[:, 0]) - np.min(primitive_coords[:, 0])
            #     primitive_h = np.max(primitive_coords[:, 1]) - np.min(primitive_coords[:, 1])
            #     primitive_bbox = np.array([primitive_w, primitive_h], dtype=np.float32)
            #     update_primitive_bboxes.append(primitive_bbox)
            # data_dict['primitive_bboxes'] = np.array(update_primitive_bboxes, dtype=np.float32)
            data_dict["coord"] = coord
            # data_dict['feat'] = feat
            data_dict["scale"] = scale
            data_dict["translation"] = translation
            if "size" in data_dict.keys():
                data_dict["size"] = data_dict["size"] * random_scale
        elif "svg_data" in data_dict:
            svg_data = data_dict["svg_data"]
            coord = svg_data.coords
            scale = data_dict["scale"]
            translation = data_dict["translation"]
            random_scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
            coord = coord * random_scale
            scale = scale * random_scale
            translation = translation * random_scale
            data_dict["svg_data"].coords = coord
            data_dict["scale"] = scale
            data_dict["translation"] = translation
            if "size" in data_dict.keys():
                data_dict["size"] = data_dict["size"] * random_scale
        if bool(data_dict.get("_recon_audit_enabled", False)):
            audit = data_dict.setdefault("_recon_audit", {})
            audit["global_scale"] = float(random_scale)
        return data_dict


########################### Add for primitive attributes jitter
@TRANSFORMS.register_module()
class PrimitiveJitterTranslate(object):
    """
    对部分实例做小幅平移；同步更新 primitive_coords 和 translation
    Args:
        p: 触发概率
        sample_ratio: 被选中的实例比例
        max_delta: 每轴平移幅度（与坐标同量纲，eg. 归一化到[-1,1]则给0.02之类）
        axis_bias: None / 'x' / 'y' （若你想假设“墙横平竖直”并沿某轴更常见，可稍后加权）
    """

    def __init__(
        self,
        p=0.0,
        sample_ratio=0.3,
        max_delta=0.1,
        axis_bias=None,
        scene_bound=20000.0,
    ):
        self.p = p
        self.sample_ratio = sample_ratio
        self.max_delta = max_delta
        self.axis_bias = axis_bias
        self.scene_bound = scene_bound

    def __call__(self, data_dict):
        if np.random.rand() > self.p or "primitive_coords" not in data_dict:
            return data_dict
        """
            coords范围: [0, 140]
            translation范围: [-1, 1]
        """
        coords_list = data_dict["primitive_coords"]
        feats_list = data_dict["primitive_feats"]
        translation = data_dict["translation"]  # [N,2] 或 [N,3](用前两维)
        N = len(coords_list)
        k = max(1, int(self.sample_ratio * N))
        idxs = np.random.choice(N, k, replace=False)

        for i in idxs:
            dx = np.random.uniform(-self.max_delta, self.max_delta)
            dy = np.random.uniform(-self.max_delta, self.max_delta)
            if self.axis_bias == "x":
                dy *= 0.5
            if self.axis_bias == "y":
                dx *= 0.5
            delta = np.array([dx, dy], dtype=np.float32)

            # 检查更新后 coord（归一化后）是否仍有点在 [-1, 1] 范围内
            shifted_coord = coords_list[i][:, :2] + delta * (self.scene_bound / 2)
            # normalized_coord = shifted_coord / (self.scene_bound / 2)  # 归一化到 [-1, 1] 尺度
            normalized_coord = shifted_coord - (self.scene_bound / 2)
            normalized_coord = normalized_coord / (self.scene_bound / 2)
            valid_mask = (
                (normalized_coord[:, 0] >= -1)
                & (normalized_coord[:, 0] <= 1)
                & (normalized_coord[:, 1] >= -1)
                & (normalized_coord[:, 1] <= 1)
            )

            # 检查更新后 translation 是否在 [-2, 2] 范围内
            shifted_trans = translation[i, :2] + delta
            trans_valid = (
                (shifted_trans[0] >= -2)
                & (shifted_trans[0] <= 2)
                & (shifted_trans[1] >= -2)
                & (shifted_trans[1] <= 2)
            )

            if not valid_mask.any() or not trans_valid:
                # 更新后会导致问题，跳过该 instance
                continue

            # 更新坐标
            coords_list[i][:, :2] += delta * (self.scene_bound / 2)
            feats_list[i][:, 0] = (
                np.arctan(coords_list[i][:, 1] / (coords_list[i][:, 0] + 1e-8))
                / math.pi
            )
            # 更新属性
            translation[i, :2] += delta

        data_dict["primitive_coords"] = coords_list
        data_dict["primitive_feats"] = feats_list
        data_dict["translation"] = translation
        ##### jitter translate对box大小不影响
        return data_dict


@TRANSFORMS.register_module()
class PrimitiveJitterScale(object):
    """
    对部分实例做小幅尺度扰动；绕实例自身中心缩放；同步更新 primitive_coords 与 scale
    Args:
        p, sample_ratio
        sigma: 对数域噪声强度；各轴 s' = s * exp(N(0, sigma))；若各向同性则两轴相同
        anisotropic: 是否各向异性
    """

    def __init__(self, p=0.5, sample_ratio=0.3, sigma=0.05, anisotropic=True):
        self.p = p
        self.sample_ratio = sample_ratio
        self.sigma = sigma
        self.anisotropic = anisotropic

    def __call__(self, data_dict):
        if np.random.rand() > self.p or "primitive_coords" not in data_dict:
            return data_dict

        coords_list = data_dict["primitive_coords"]
        feats_list = data_dict["primitive_feats"]
        scale = data_dict["scale"]  # [N,2]（若为[ N ]也兼容：广播）
        N = len(coords_list)
        k = max(1, int(self.sample_ratio * N))
        idxs = np.random.choice(N, k, replace=False)

        for i in idxs:
            # 取中心
            c = coords_list[i][:, :2].mean(axis=0, keepdims=True)
            # 采样缩放因子
            if self.anisotropic:
                # g = np.random.normal(0.0, self.sigma, size=(2,))
                g = np.random.normal(0.0, self.sigma, size=(1,))
                s = np.exp(g).astype(np.float32)
            else:
                g = float(np.random.normal(0.0, self.sigma))
                # s = np.array([np.exp(g), np.exp(g)], dtype=np.float32)
                s = np.array([np.exp(g)], dtype=np.float32)

            # 坐标绕中心缩放（z不变）
            coords_list[i][:, :2] = (coords_list[i][:, :2] - c) * s + c
            feats_list[i][:, 0] = (
                np.arctan(coords_list[i][:, 1] / (coords_list[i][:, 0] + 1e-8))
                / math.pi
            )
            # 属性尺度更新（若原来是标量/一维，会自动广播）
            scale[i, ...] = scale[i, ...] * s

        data_dict["primitive_coords"] = coords_list
        data_dict["primitive_feats"] = feats_list
        data_dict["scale"] = scale
        return data_dict


@TRANSFORMS.register_module()
class PrimitiveJitterRotate(object):
    """
    对部分实例做小角度扰动；绕实例自身中心旋转；同步更新 primitive_coords 与 rotation（度制）
    Args:
        p, sample_ratio
        max_deg: 角度扰动幅度（例如 10 表示 [-10,10]°）
        snap90_prob: 以一定概率吸附到 {0,90,180,270}，利于贴墙/对齐先验
    """

    def __init__(self, p=0.5, sample_ratio=0.3, max_deg=90.0, snap90_prob=0.0):
        self.p = p
        self.sample_ratio = sample_ratio
        self.max_deg = max_deg
        self.snap90_prob = snap90_prob

    def __call__(self, data_dict):
        if np.random.rand() > self.p or "primitive_coords" not in data_dict:
            return data_dict

        coords_list = data_dict["primitive_coords"]
        feats_list = data_dict["primitive_feats"]
        rotation = data_dict["rotation"]  # [N] or [N,1]
        N = len(coords_list)
        k = max(1, int(self.sample_ratio * N))
        idxs = np.random.choice(N, k, replace=False)

        for i in idxs:
            # 中心
            # c = coords_list[i][:, :2].mean(axis=0, keepdims=True)
            c = np.array(
                (coords_list[i][:, :2].min(axis=0) + coords_list[i][:, :2].max(axis=0))
                / 2
            )
            # 角度
            delta = np.random.uniform(-self.max_deg, self.max_deg)
            if np.random.rand() < self.snap90_prob:
                delta = float(np.random.choice([0, 90, 180, 270]))
            th = np.deg2rad(delta).astype(np.float32)
            R = np.array(
                [[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], dtype=np.float32
            )

            # 绕中心旋转
            xy = coords_list[i][:, :2] - c
            coords_list[i][:, :2] = xy @ R.T + c
            feats_list[i][:, 0] = (
                np.arctan(coords_list[i][:, 1] / (coords_list[i][:, 0] + 1e-8))
                / math.pi
            )
            # 属性更新（度制、取模）
            rotation[i] = (rotation[i] + delta) % 360.0

        data_dict["primitive_coords"] = coords_list
        data_dict["primitive_feats"] = feats_list
        data_dict["rotation"] = rotation
        return data_dict


@TRANSFORMS.register_module()
class PrimitiveRandomFlip(object):
    """
    对部分实例做镜像翻转；绕实例自身中心翻转；同步更新 primitive_coords / rotation / flip_mode / translation
    Args:
        p: 触发概率
        sample_ratio: 被选中的实例比例
        p_x, p_y: 轴向翻转概率（可同时为真 => 等价于加180°）
    角度更新：
        x翻转: θ -> (-θ) mod 360
        y翻转: θ -> (180 - θ) mod 360
        x&y翻转: θ -> (θ + 180) mod 360
    """

    def __init__(self, p=0.5, sample_ratio=0.3, p_x=0.5, p_y=0.5):
        self.p = p
        self.sample_ratio = sample_ratio
        self.p_x = p_x
        self.p_y = p_y

    def __call__(self, data_dict):
        if np.random.rand() > self.p or "primitive_coords" not in data_dict:
            return data_dict

        coords_list = data_dict["primitive_coords"]
        feats_list = data_dict["primitive_feats"]
        rotation = data_dict.get("rotation", None)
        translation = data_dict.get("translation", None)
        flip_mode = data_dict.get("flip_mode", None)
        flip_invariant = data_dict.get(
            "flip_invariant_primitive", None
        )  # 若提供则跳过对称类
        N = len(coords_list)
        k = max(1, int(self.sample_ratio * N))
        idxs = np.random.choice(N, k, replace=False)

        for i in idxs:
            fx = np.random.rand() < self.p_x
            fy = np.random.rand() < self.p_y
            if not (fx or fy):
                continue

            # 绕实例中心翻转（不改变 translation 的含义）
            c = coords_list[i][:, :2].mean(axis=0, keepdims=True)
            xy = coords_list[i][:, :2] - c
            if fx:
                xy[:, 0] = -xy[:, 0]
            if fy:
                xy[:, 1] = -xy[:, 1]
            coords_list[i][:, :2] = xy + c
            feats_list[i][:, 0] = (
                np.arctan(coords_list[i][:, 1] / (coords_list[i][:, 0] + 1e-8))
                / math.pi
            )
            # rotation
            if rotation is not None:
                if fx and fy:
                    rotation[i] = (rotation[i] + 180.0) % 360.0
                elif fx:
                    rotation[i] = (-rotation[i]) % 360.0
                elif fy:
                    rotation[i] = (180.0 - rotation[i]) % 360.0

            # translation（绕中心翻转不改平移；若你希望“全局镜像”再同时取负）
            if translation is not None:
                pass

            # flip_mode
            if flip_mode is not None:
                if flip_invariant[i]:
                    assert (
                        flip_mode[i] == 0
                    ), "flip_mode should be 0 for flip_invariant_primitive"
                    continue
                else:
                    if fx or fy:
                        flip_mode[i] = 1 - flip_mode[i]

        data_dict["primitive_coords"] = coords_list
        data_dict["primitive_feats"] = feats_list
        if rotation is not None:
            data_dict["rotation"] = rotation
        if translation is not None:
            data_dict["translation"] = translation
        if flip_mode is not None:
            data_dict["flip_mode"] = flip_mode
        return data_dict


@TRANSFORMS.register_module()
class Collect(object):
    def __init__(self, keys, offset_keys_dict=None, **kwargs):
        """
        e.g. Collect(keys=[coord], feat_keys=[coord, color])
        """
        if offset_keys_dict is None:
            offset_keys_dict = dict(offset="coord")
        self.keys = keys
        self.offset_keys = offset_keys_dict
        self.kwargs = kwargs

    def __call__(self, data_dict):
        data = dict()
        if isinstance(self.keys, str):
            self.keys = [self.keys]
        for key in self.keys:
            data[key] = data_dict[key]
        for key, value in self.offset_keys.items():
            data[key] = torch.tensor([data_dict[value].shape[0]])
        for name, keys in self.kwargs.items():
            name = name.replace("_keys", "")
            assert isinstance(keys, Sequence)
            data[name] = torch.cat([data_dict[key].float() for key in keys], dim=1)
        return data


@TRANSFORMS.register_module()
class AddOffset(object):
    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            data_dict["offset"] = torch.tensor([data_dict["coord"].shape[0]])
        return data_dict


@TRANSFORMS.register_module()
class Copy(object):
    def __init__(self, keys_dict=None):
        if keys_dict is None:
            keys_dict = dict(coord="origin_coord", segment="origin_segment")
        self.keys_dict = keys_dict

    def __call__(self, data_dict):
        for key, value in self.keys_dict.items():
            if isinstance(data_dict[key], np.ndarray):
                data_dict[value] = data_dict[key].copy()
            elif isinstance(data_dict[key], torch.Tensor):
                data_dict[value] = data_dict[key].clone().detach()
            else:
                data_dict[value] = copy.deepcopy(data_dict[key])
        return data_dict


@TRANSFORMS.register_module()
class Update(object):
    def __init__(self, keys_dict=None):
        if keys_dict is None:
            keys_dict = dict()
        self.keys_dict = keys_dict

    def __call__(self, data_dict):
        for key, value in self.keys_dict.items():
            data_dict[key] = value
        return data_dict


@TRANSFORMS.register_module()
class SceneClip(object):
    """
    对场景坐标进行裁剪，移除超出 clip_range 范围的点

    同时对 instance 属性进行过滤：如果某个 instance 在 clip_range 内的点
    不存在或只剩余不超过 min_keep_ratio 的部分，则过滤掉该 instance 的属性

    Args:
        clip_range: [min, max] 裁剪范围（归一化坐标，默认 [-1, 1]）
        min_keep_ratio: 最小保留比例，低于该比例的 instance 会被过滤（默认 0.3）

    数据要求:
        - coord: 点坐标 [N, 2, 2] 或 [N, 2]
        - point_primitive_ids: 点级别的 primitive_id [N,]
        - translation, rotation, scale, flip_mode 等 instance 级别属性
    """

    def __init__(self, clip_range=[-1.0, 1.0], min_keep_ratio=0.3):
        self.clip_range = clip_range
        self.min_keep_ratio = min_keep_ratio

    def __call__(self, data_dict):
        if "coord" not in data_dict:
            return data_dict

        coord = data_dict["coord"]
        original_shape = coord.shape

        # 将坐标展平为 [N*2, 2] 或 [N, 2] 形式以便处理
        if len(coord.shape) == 3:
            # [N, 2, 2] -> [N*2, 2]
            coord_flat = coord.reshape(-1, 2)
            is_line_mode = True
        else:
            coord_flat = coord
            is_line_mode = False

        # 判断每个点是否在 clip_range 内
        in_range = (
            (coord_flat[:, 0] >= self.clip_range[0])
            & (coord_flat[:, 0] <= self.clip_range[1])
            & (coord_flat[:, 1] >= self.clip_range[0])
            & (coord_flat[:, 1] <= self.clip_range[1])
        )

        if is_line_mode:
            # 对于线段，两个端点都在范围内才保留
            in_range_lines = in_range.reshape(-1, 2).all(axis=1)
            point_keep_mask = in_range_lines
        else:
            point_keep_mask = in_range

        # 如果 point_primitive_ids 存在，计算每个 instance 的保留比例
        if "point_primitive_ids" in data_dict:
            point_prim_ids = data_dict["point_primitive_ids"]

            if is_line_mode:
                # 线段模式：每条线段有一个 primitive_id
                prim_ids_per_line = point_prim_ids  # [N,]
            else:
                prim_ids_per_line = point_prim_ids

            # 获取所有唯一的 instance id
            unique_prim_ids = np.unique(prim_ids_per_line)

            # 计算每个 instance 的保留比例
            instance_keep_mask = []
            prim_id_to_new_id = {}
            new_id = 0

            for prim_id in unique_prim_ids:
                mask = prim_ids_per_line == prim_id
                total_points = mask.sum()
                kept_points = (mask & point_keep_mask).sum()

                if total_points > 0:
                    keep_ratio = kept_points / total_points
                else:
                    keep_ratio = 0

                # 如果保留比例大于阈值，保留该 instance
                if keep_ratio > self.min_keep_ratio:
                    instance_keep_mask.append(True)
                    prim_id_to_new_id[prim_id] = new_id
                    new_id += 1
                else:
                    instance_keep_mask.append(False)

            instance_keep_mask = np.array(instance_keep_mask)

            # 更新点级别的过滤：只保留属于被保留 instance 的点
            valid_prim_ids = set(unique_prim_ids[instance_keep_mask])
            prim_id_valid_mask = np.isin(prim_ids_per_line, list(valid_prim_ids))

            # 最终的点保留 mask：在 clip_range 内 且 属于保留的 instance
            final_point_mask = point_keep_mask & prim_id_valid_mask

            # 更新 point_primitive_ids（重新编号）
            new_prim_ids = np.zeros_like(prim_ids_per_line)
            for old_id, new_id in prim_id_to_new_id.items():
                new_prim_ids[prim_ids_per_line == old_id] = new_id
            data_dict["point_primitive_ids"] = new_prim_ids[final_point_mask]

            # 过滤 instance 级别的属性
            # 需要根据原始 primitive_id 顺序来确定哪些 instance 被保留
            # unique_prim_ids 是排序后的，但 gt_data 中的 instance 是按原始顺序的
            # 我们需要根据 unique_prim_ids 中被保留的 id 来过滤

            # 假设 instance 级别属性的索引与 primitive_id 一一对应
            # 但实际上 primitive_id 可能不是连续的 0, 1, 2, ...
            # 需要使用 unique_prim_ids 和 instance_keep_mask 来确定

            # 创建从原始 primitive_id 到 instance 索引的映射
            # 注意：这里假设 unique_prim_ids 的顺序与 instance 属性的顺序一致
            # 如果不一致，需要根据实际情况调整

            instance_level_keys = [
                "semantic_id",
                "primitive_id",
                "scale",
                "rotation",
                "flip_mode",
                "translation",
                "flip_invariant_primitive",
            ]

            for key in instance_level_keys:
                if key in data_dict and data_dict[key] is not None:
                    arr = data_dict[key]
                    if len(arr) == len(unique_prim_ids):
                        # 直接使用 instance_keep_mask
                        data_dict[key] = arr[instance_keep_mask]
                    elif len(arr) > 0:
                        # 如果长度不匹配，尝试根据有效的 instance 数量过滤
                        # 这种情况可能需要更复杂的处理
                        pass
        else:
            final_point_mask = point_keep_mask

        # 过滤坐标
        if is_line_mode:
            coord_filtered = coord[final_point_mask]
        else:
            coord_filtered = coord_flat[final_point_mask]

        data_dict["coord"] = coord_filtered

        # 同步过滤其他点级别属性（如果存在）
        point_level_keys = [
            "color",
            "width",
            "layer_id",
            "semantic_id_point",
            "instance_id_point",
        ]
        for key in point_level_keys:
            if key in data_dict and data_dict[key] is not None:
                arr = data_dict[key]
                if len(arr) == len(final_point_mask):
                    data_dict[key] = arr[final_point_mask]

        return data_dict


@TRANSFORMS.register_module()
class ToTensor(object):
    def __call__(self, data):
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, SVGData):
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


@TRANSFORMS.register_module()
class NormalizeColor(object):
    def __call__(self, data_dict):
        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"] / 255
        return data_dict


# @TRANSFORMS.register_module()
# class NormalizeCoord(object):
#     def __call__(self, data_dict):
#         if "coord" in data_dict.keys():
#             # modified from pointnet2
#             centroid = np.mean(data_dict["coord"], axis=0)
#             data_dict["coord"] -= centroid
#             m = np.max(np.sqrt(np.sum(data_dict["coord"] ** 2, axis=1)))
#             data_dict["coord"] = data_dict["coord"] / m
#         return data_dict


@TRANSFORMS.register_module()
class PositiveShift(object):
    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            coord_min = np.min(data_dict["coord"], 0)
            data_dict["coord"] -= coord_min
        return data_dict


@TRANSFORMS.register_module()
class CenterShift(object):
    def __init__(self, apply_z=True):
        self.apply_z = apply_z

    ### 偏移点云中心到原点处
    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            if isinstance(data_dict["coord"], torch.Tensor):
                x_min, y_min, z_min = data_dict["coord"].min(dim=0)[0]
                x_max, y_max, _ = data_dict["coord"].max(dim=0)[0]
                if self.apply_z:
                    shift = torch.tensor(
                        [(x_min + x_max) / 2, (y_min + y_max) / 2, z_min],
                        dtype=data_dict["coord"].dtype,
                        device=data_dict["coord"].device,
                    )
                else:
                    shift = torch.tensor(
                        [(x_min + x_max) / 2, (y_min + y_max) / 2, 0],
                        dtype=data_dict["coord"].dtype,
                        device=data_dict["coord"].device,
                    )
                data_dict["coord"] -= shift

                # # 对primitive的translation进行相应调整
                # if "translation" in data_dict:
                #     shift_2d = shift[:2]  # 只取x,y分量
                #     data_dict["translation"] -= shift_2d
            else:
                x_min, y_min, z_min = data_dict["coord"].min(axis=0)
                x_max, y_max, _ = data_dict["coord"].max(axis=0)
                if self.apply_z:
                    shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, z_min]
                else:
                    shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, 0]
                data_dict["coord"] -= shift

                # # 对primitive的translation进行相应调整
                # if "translation" in data_dict:
                #     shift_2d = torch.tensor(shift[:2], dtype=data_dict["translation"].dtype, device=data_dict["translation"].device)
                #     data_dict["translation"] -= shift_2d
        return data_dict


@TRANSFORMS.register_module()
class RandomShift(object):
    def __init__(self, shift=((-0.2, 0.2), (-0.2, 0.2), (0, 0))):
        self.shift = shift

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            shift_x = np.random.uniform(self.shift[0][0], self.shift[0][1])
            shift_y = np.random.uniform(self.shift[1][0], self.shift[1][1])
            shift_z = np.random.uniform(self.shift[2][0], self.shift[2][1])
            data_dict["coord"] += [shift_x, shift_y, shift_z]
        return data_dict


@TRANSFORMS.register_module()
class PointClip(object):
    def __init__(self, point_cloud_range=(-80, -80, -3, 80, 80, 1)):
        self.point_cloud_range = point_cloud_range

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            data_dict["coord"] = np.clip(
                data_dict["coord"],
                a_min=self.point_cloud_range[:3],
                a_max=self.point_cloud_range[3:],
            )
        return data_dict


@TRANSFORMS.register_module()
class RandomDropout(object):
    def __init__(self, dropout_ratio=0.2, dropout_application_ratio=0.5):
        """
        upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        self.dropout_ratio = dropout_ratio
        self.dropout_application_ratio = dropout_application_ratio

    def __call__(self, data_dict):
        if random.random() < self.dropout_application_ratio:
            n = len(data_dict["coord"])
            idx = np.random.choice(n, int(n * (1 - self.dropout_ratio)), replace=False)
            if "sampled_index" in data_dict:
                # for ScanNet data efficient, we need to make sure labeled point is sampled.
                idx = np.unique(np.append(idx, data_dict["sampled_index"]))
                mask = np.zeros_like(data_dict["segment"]).astype(bool)
                mask[data_dict["sampled_index"]] = True
                data_dict["sampled_index"] = np.where(mask[idx])[0]
            data_dict = index_operator(data_dict, idx)
        return data_dict


# @TRANSFORMS.register_module()
# class RandomRotate(object):
#     def __init__(self, angle=None, center=None, axis="z", always_apply=False, p=0.5):
#         self.angle = [-1, 1] if angle is None else angle
#         self.axis = axis
#         self.always_apply = always_apply
#         self.p = p if not self.always_apply else 1
#         self.center = center

#     def __call__(self, data_dict):
#         if random.random() > self.p:
#             return data_dict
#         angle = np.random.uniform(self.angle[0], self.angle[1]) * np.pi
#         rot_cos, rot_sin = np.cos(angle), np.sin(angle)
#         if self.axis == "x":
#             rot_t = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
#         elif self.axis == "y":
#             rot_t = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
#         elif self.axis == "z":
#             rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
#         else:
#             raise NotImplementedError
#         if "coord" in data_dict.keys():
#             if self.center is None:
#                 x_min, y_min, z_min = data_dict["coord"].min(dim=0)[0]
#                 x_max, y_max, z_max = data_dict["coord"].max(dim=0)[0]
#                 center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
#             else:
#                 center = self.center
#             data_dict["coord"] -= center
#             data_dict["coord"] = np.dot(data_dict["coord"], np.transpose(rot_t))
#             data_dict["coord"] += center
#         if "normal" in data_dict.keys():
#             data_dict["normal"] = np.dot(data_dict["normal"], np.transpose(rot_t))

#         # 对primitive参数进行相应变换
#         if "translation" in data_dict and "rotation" in data_dict:
#             # 构建4x4变换矩阵
#             transform_matrix = np.eye(4)
#             transform_matrix[:3, :3] = rot_t
#             data_dict = apply_transform_to_primitives(data_dict, transform_matrix, center)

#         return data_dict


@TRANSFORMS.register_module()
class RandomRotateTargetAngle(object):
    def __init__(
        self, angle=(1 / 2, 1, 3 / 2), center=None, axis="z", always_apply=False, p=0.75
    ):
        self.angle = angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        angle = np.random.choice(self.angle) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        if self.axis == "x":
            rot_t = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
        elif self.axis == "y":
            rot_t = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
        elif self.axis == "z":
            rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        else:
            raise NotImplementedError
        if "coord" in data_dict.keys():
            if self.center is None:
                x_min, y_min, z_min = data_dict["coord"].min(axis=0)
                x_max, y_max, z_max = data_dict["coord"].max(axis=0)
                center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
            else:
                center = self.center
            data_dict["coord"] -= center
            data_dict["coord"] = np.dot(data_dict["coord"], np.transpose(rot_t))
            data_dict["coord"] += center
        if "normal" in data_dict.keys():
            data_dict["normal"] = np.dot(data_dict["normal"], np.transpose(rot_t))

        # 对primitive参数进行相应变换
        if "translation" in data_dict and "rotation" in data_dict:
            # 构建4x4变换矩阵
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rot_t
            data_dict = apply_transform_to_primitives(
                data_dict, transform_matrix, center
            )

        return data_dict


@TRANSFORMS.register_module()
class RandomScale(object):
    def __init__(self, scale=None, anisotropic=False):
        self.scale = scale if scale is not None else [0.95, 1.05]
        self.anisotropic = anisotropic

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            scale = np.random.uniform(
                self.scale[0], self.scale[1], 3 if self.anisotropic else 1
            )
            data_dict["coord"] *= scale

            # 对primitive的scale和translation进行相应调整
            if "scale" in data_dict:
                # primitive的scale需要相应缩放
                if self.anisotropic:
                    scale_2d = torch.tensor(
                        scale[:2],
                        dtype=data_dict["scale"].dtype,
                        device=data_dict["scale"].device,
                    )
                else:
                    scale_2d = torch.tensor(
                        [scale, scale],
                        dtype=data_dict["scale"].dtype,
                        device=data_dict["scale"].device,
                    )
                data_dict["scale"] *= scale_2d

            if "translation" in data_dict:
                # primitive的translation也需要相应缩放
                if self.anisotropic:
                    scale_2d = torch.tensor(
                        scale[:2],
                        dtype=data_dict["translation"].dtype,
                        device=data_dict["translation"].device,
                    )
                else:
                    scale_2d = torch.tensor(
                        [scale, scale],
                        dtype=data_dict["translation"].dtype,
                        device=data_dict["translation"].device,
                    )
                data_dict["translation"] *= scale_2d
        return data_dict


# @TRANSFORMS.register_module()
# class RandomFlip(object):
#     def __init__(self, p=0.5):
#         self.p = p

#     def __call__(self, data_dict):
#         flip_x = np.random.rand() < self.p
#         flip_y = np.random.rand() < self.p

#         if flip_x:
#             if "coord" in data_dict.keys():
#                 data_dict["coord"][:, 0] = -data_dict["coord"][:, 0]
#             if "normal" in data_dict.keys():
#                 data_dict["normal"][:, 0] = -data_dict["normal"][:, 0]
#             # 对primitive参数进行相应调整
#             if "translation" in data_dict:
#                 data_dict["translation"][:, 0] = -data_dict["translation"][:, 0]
#             if "rotation" in data_dict:
#                 # 翻转x会改变旋转方向
#                 data_dict["rotation"] = -data_dict["rotation"]

#         if flip_y:
#             if "coord" in data_dict.keys():
#                 data_dict["coord"][:, 1] = -data_dict["coord"][:, 1]
#             if "normal" in data_dict.keys():
#                 data_dict["normal"][:, 1] = -data_dict["normal"][:, 1]
#             # 对primitive参数进行相应调整
#             if "translation" in data_dict:
#                 data_dict["translation"][:, 1] = -data_dict["translation"][:, 1]
#             if "rotation" in data_dict:
#                 # 翻转y会改变旋转方向
#                 if not flip_x:  # 如果x已经翻转了，则不需要再次改变旋转
#                     data_dict["rotation"] = -data_dict["rotation"]


#         return data_dict
# ----------------------------------------------------------------------------------
# 1. 修改 RandomFlip —— 同步处理 img、flip_mode，并保证 rotation 方向正确
# ----------------------------------------------------------------------------------
@TRANSFORMS.register_module()
class RandomFlip(object):
    """
    随机沿 x 轴 / y 轴镜像翻转

    Args:
        p_x (float): x轴翻转概率，默认0.5
        p_y (float): y轴翻转概率，默认0.5

    支持的数据类型：
        - coord: 点云坐标 [N, 3]
        - normal: 点云法向量 [N, 3]
        - translation: primitive位移 [M, 2]
        - rotation: primitive旋转角度 [M] (度数制)
        - flip_mode: 翻转模式标记 [M]
        - img: 图像数据 [H, W, C]

    角度变换规则：
        - 仅x轴翻转: θ -> 180° - θ (水平镜像)
        - 仅y轴翻转: θ -> -θ (垂直镜像)
        - x和y同时翻转: θ -> θ + 180° (点对称)
    """

    def __init__(self, p_x=0.5, p_y=0.5, reparam_rotation=False):
        assert 0 <= p_x <= 1, f"p_x应在[0,1]范围内，得到: {p_x}"
        assert 0 <= p_y <= 1, f"p_y应在[0,1]范围内，得到: {p_y}"
        self.p_x = p_x
        self.p_y = p_y
        self.reparam_rotation = reparam_rotation

    def __call__(self, data_dict):
        flip_x = np.random.rand() < self.p_x
        flip_y = np.random.rand() < self.p_y
        if bool(data_dict.get("_recon_audit_enabled", False)):
            audit = data_dict.setdefault("_recon_audit", {})
            audit["global_flip_x"] = bool(flip_x)
            audit["global_flip_y"] = bool(flip_y)

        # --- 点云与法向 ---
        if "coord" in data_dict and (flip_x or flip_y):
            ### coords: [N, 2, 2]
            if flip_x:
                data_dict["coord"][:, :, 0] = -data_dict["coord"][:, :, 0]
            if flip_y:
                data_dict["coord"][:, :, 1] = -data_dict["coord"][:, :, 1]
            # data_dict["feat"][:, 0] = np.arctan(data_dict["coord"][:, 1] / (data_dict["coord"][:, 0] + 1e-8)) / math.pi

        if "svg_data" in data_dict and (flip_x or flip_y):
            coords = data_dict["svg_data"].coords
            if flip_x:
                coords[:, :, 0] = -coords[:, :, 0]
            if flip_y:
                coords[:, :, 1] = -coords[:, :, 1]
            data_dict["svg_data"].coords = coords

        if "normal" in data_dict and (flip_x or flip_y):
            if flip_x:
                data_dict["normal"][:, 0] = -data_dict["normal"][:, 0]
            if flip_y:
                data_dict["normal"][:, 1] = -data_dict["normal"][:, 1]

        # --- primitive translation ---
        if "translation" in data_dict and (flip_x or flip_y):
            if flip_x:
                data_dict["translation"][:, 0] = -data_dict["translation"][:, 0]
            if flip_y:
                data_dict["translation"][:, 1] = -data_dict["translation"][:, 1]

        ##### TODO: 修正rotation/flip数据增强
        # --- primitive rotation: 镜像翻转时正确处理角度变换 ---
        if "rotation" in data_dict and (flip_x or flip_y):
            rotation = data_dict["rotation"]
            #### 顺时针旋转逻辑
            if flip_x and flip_y:
                # 同时翻转x和y轴 = 旋转180度
                rotation = (rotation + 180) % 360  #
            elif flip_x:
                ### 顺时针旋转360-rotation度 --- 逆时针旋转360-(360-rotation) = rotation度
                rotation = (-rotation) % 360
            elif flip_y:
                ### 顺时针旋转180-rotation度 --- 逆时针旋转360-(180-rotation) = 180+rotation度
                rotation = (180 - rotation) % 360
            #### 逆时针旋转逻辑
            # if flip_x and flip_y:
            #     # 同时翻转x和y轴 = 旋转180度
            #     rotation = (rotation - 180) % 360
            # elif flip_x:
            #     rotation = (180-rotation) % 360
            # elif flip_y:
            #     rotation = (-rotation) % 360

            data_dict["rotation"] = rotation

        if "flip_mode" in data_dict:
            flip_mode = data_dict["flip_mode"]
            flip_invariant_primitive = data_dict["flip_invariant_primitive"]
            for i in range(len(flip_mode)):
                if flip_invariant_primitive[i]:
                    assert (
                        flip_mode[i] == 0
                    ), "flip_mode should be 0 for flip_invariant_primitive"
                    continue
                else:
                    if flip_x and flip_y:
                        continue
                    elif flip_x:
                        flip_mode[i] = 1 - flip_mode[i]
                    elif flip_y:
                        flip_mode[i] = 1 - flip_mode[i]
                    # if flip_x or flip_y:
                    #     flip_mode[i] = 1 - flip_mode[i]
            data_dict["flip_mode"] = flip_mode

        # ### update for primitive_bboxes: 根据primitive_indexs重新计算primitive_bboxes
        # assert 'primitive_index' in data_dict, "primitive_index is not in data_dict"
        # primitive_index = data_dict['primitive_index']
        # update_primitive_bboxes = []
        # for prim_idx in primitive_index:
        #     primitive_coords = data_dict['coord'][prim_idx[0]:prim_idx[1]]
        #     primitive_w = np.max(primitive_coords[:, 0]) - np.min(primitive_coords[:, 0])
        #     primitive_h = np.max(primitive_coords[:, 1]) - np.min(primitive_coords[:, 1])
        #     primitive_bbox = np.array([primitive_w, primitive_h], dtype=np.float32)
        #     update_primitive_bboxes.append(primitive_bbox)
        # data_dict['primitive_bboxes'] = np.array(update_primitive_bboxes, dtype=np.float32)

        # # --- flip_mode: 记录 x 轴是否已经翻过 ---
        # if "flip_mode" in data_dict and flip_x:
        #     data_dict["flip_mode"] = 1 - data_dict["flip_mode"]  # 0↔1 取反

        # if "flip_mode" in data_dict and self.reparam_rotation:
        #     for i in range(len(data_dict["flip_mode"])):
        #         if data_dict["flip_mode"][i] == 1:
        #             data_dict["flip_mode"][i] = 0
        #             data_dict["rotation"][i] = (360 - data_dict["rotation"][i]) % 360

        # --- 同步翻转图像 ---
        if "img" in data_dict and (flip_x or flip_y):
            img = data_dict["img"]
            if flip_x:
                img = np.flip(img, axis=1)  # 水平
            if flip_y:
                img = np.flip(img, axis=0)  # 垂直
            data_dict["img"] = img.copy()

        return data_dict


@TRANSFORMS.register_module()
class RandomRotate(object):
    """
    随机选择一个角度，绕 z 轴旋转

    Args:
        p (float): 应用旋转的概率，默认1.0（总是旋转）

    支持的数据类型：
        - coord: 点云坐标 [N, 3]
        - normal: 点云法向量 [N, 3]
        - translation: primitive位移 [M, 2] -> 应用2D旋转变换
        - rotation: primitive旋转角度 [M] (度数制) -> 累加旋转角度
        - img: 图像数据 [H, W, C] -> 使用np.rot90旋转

    变换细节：
        - 旋转矩阵：标准2D旋转，逆时针为正
        - translation: 先补零到3D，旋转后取前2维
        - rotation: 直接累加角度并取模360°
        - 图像：使用numpy.rot90进行90°的整数倍旋转
    """

    def __init__(self, p=1.0, angle_range=(0, 360)):
        assert 0 <= p <= 1, f"p应在[0,1]范围内，得到: {p}"
        self.p = p
        self.angle_range = angle_range

    def __call__(self, data_dict):
        if np.random.rand() > self.p:
            return data_dict

        # angle_deg = np.random.uniform(self.angle_range[0], self.angle_range[1])
        ###### 工业数据集基本上都是横平竖直的
        # angle_deg = np.random.choice([0, 90, 180, 270])
        angle_deg = np.random.choice([90, 180, 270])
        if bool(data_dict.get("_recon_audit_enabled", False)):
            audit = data_dict.setdefault("_recon_audit", {})
            audit["global_rotation_deg"] = int(angle_deg)
        if angle_deg == 0:
            return data_dict  # 不做旋转

        # ---- 构造旋转矩阵 (逆时针为正) ----
        theta = np.deg2rad(angle_deg)
        rot_mat = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        # 2D 旋转矩阵，用于 [N, 2] 形状的点
        rot_mat_2d = rot_mat[:2, :2]

        # ---- 点云 ----
        if "coord" in data_dict:
            # data_dict["coord"] 形状为 [N, 2, 2]，[:, 0] 和 [:, 1] 各是 [N, 2] 的 2D 点
            data_dict["coord"][:, 0] = data_dict["coord"][:, 0] @ rot_mat_2d.T
            data_dict["coord"][:, 1] = data_dict["coord"][:, 1] @ rot_mat_2d.T
            # data_dict["feat"][:, 0] = np.arctan(data_dict["coord"][:, 1] / (data_dict["coord"][:, 0] + 1e-8)) / math.pi

        if "svg_data" in data_dict:
            coords = data_dict["svg_data"].coords
            coords[:, 0] = coords[:, 0] @ rot_mat_2d.T
            coords[:, 1] = coords[:, 1] @ rot_mat_2d.T
            data_dict["svg_data"].coords = coords

        if "normal" in data_dict:
            data_dict["normal"] = data_dict["normal"] @ rot_mat.T

        # ---- primitive translation ----
        if "translation" in data_dict:
            # pad z=0 以便与 3×3 矩阵相乘
            trans_pad = np.concatenate(
                [
                    data_dict["translation"],
                    np.zeros_like(data_dict["translation"][:, :1]),
                ],
                axis=1,
            )
            trans_rot = trans_pad @ rot_mat.T
            data_dict["translation"] = trans_rot[:, :2]

        # ---- primitive rotation ----
        if "rotation" in data_dict:
            data_dict["rotation"] = (data_dict["rotation"] + angle_deg) % 360
            # data_dict["rotation"] = (data_dict["rotation"] - angle_deg) % 360

        # ---- 图像 ----
        if "img" in data_dict:
            k = angle_deg // 90  # 0 →0, 90→1, 180→2, 270→3
            data_dict["img"] = np.rot90(data_dict["img"], k).copy()

        if "size" in data_dict.keys():
            if angle_deg in [90, 270]:
                data_dict["size"] = data_dict["size"][:, [1, 0]]

        # ### update for primitive_bboxes: 根据primitive_indexs重新计算primitive_bboxes
        # assert 'primitive_index' in data_dict, "primitive_index is not in data_dict"
        # primitive_index = data_dict['primitive_index']
        # update_primitive_bboxes = []
        # for prim_idx in primitive_index:
        #     primitive_coords = data_dict['coord'][prim_idx[0]:prim_idx[1]]
        #     primitive_w = np.max(primitive_coords[:, 0]) - np.min(primitive_coords[:, 0])
        #     primitive_h = np.max(primitive_coords[:, 1]) - np.min(primitive_coords[:, 1])
        #     primitive_bbox = np.array([primitive_w, primitive_h], dtype=np.float32)
        #     update_primitive_bboxes.append(primitive_bbox)
        # data_dict['primitive_bboxes'] = np.array(update_primitive_bboxes, dtype=np.float32)

        return data_dict


# ----------------------------------------------------------------------------------
# 2. 新增 RandomRotate90 —— 在 {0, 90, 180, 270}° 之间离散旋转
# ----------------------------------------------------------------------------------
@TRANSFORMS.register_module()
class RandomRotate90(object):
    """
    在 {0°, 90°, 180°, 270°} 上随机选择一个角度，绕 z 轴旋转

    Args:
        p (float): 应用旋转的概率，默认1.0（总是旋转）

    支持的数据类型：
        - coord: 点云坐标 [N, 3]
        - normal: 点云法向量 [N, 3]
        - translation: primitive位移 [M, 2] -> 应用2D旋转变换
        - rotation: primitive旋转角度 [M] (度数制) -> 累加旋转角度
        - img: 图像数据 [H, W, C] -> 使用np.rot90旋转

    变换细节：
        - 旋转矩阵：标准2D旋转，逆时针为正
        - translation: 先补零到3D，旋转后取前2维
        - rotation: 直接累加角度并取模360°
        - 图像：使用numpy.rot90进行90°的整数倍旋转
    """

    def __init__(self, p=1.0):
        assert 0 <= p <= 1, f"p应在[0,1]范围内，得到: {p}"
        self.p = p
        self.candidates = [0, 90, 180, 270]

    def __call__(self, data_dict):
        if np.random.rand() > self.p:
            return data_dict

        angle_deg = np.random.choice(self.candidates)
        if angle_deg == 0:
            return data_dict  # 不做旋转

        # ---- 构造旋转矩阵 (逆时针为正) ----
        theta = np.deg2rad(angle_deg)
        rot_mat = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

        # ---- 点云 ----
        if "coord" in data_dict:
            data_dict["coord"] = data_dict["coord"] @ rot_mat.T

        if "normal" in data_dict:
            data_dict["normal"] = data_dict["normal"] @ rot_mat.T

        # ---- primitive translation ----
        if "translation" in data_dict:
            # pad z=0 以便与 3×3 矩阵相乘
            trans_pad = np.concatenate(
                [
                    data_dict["translation"],
                    np.zeros_like(data_dict["translation"][:, :1]),
                ],
                axis=1,
            )
            trans_rot = trans_pad @ rot_mat.T
            data_dict["translation"] = trans_rot[:, :2]

        # ---- primitive rotation ----
        if "rotation" in data_dict:
            data_dict["rotation"] = (data_dict["rotation"] + angle_deg) % 360

        # ---- 图像 ----
        if "img" in data_dict:
            k = angle_deg // 90  # 0 →0, 90→1, 180→2, 270→3
            data_dict["img"] = np.rot90(data_dict["img"], k).copy()

        return data_dict


@TRANSFORMS.register_module()
class RandomJitter(object):
    def __init__(self, sigma=0.01, clip=0.05):
        assert clip > 0
        self.sigma = sigma
        self.clip = clip

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            jitter = np.clip(
                self.sigma * np.random.randn(data_dict["coord"].shape[0], 3),
                -self.clip,
                self.clip,
            )
            data_dict["coord"] += jitter
        return data_dict


@TRANSFORMS.register_module()
class ClipGaussianJitter(object):
    def __init__(self, scalar=0.02, store_jitter=False):
        self.scalar = scalar
        self.mean = np.mean(3)
        self.cov = np.identity(3)
        self.quantile = 1.96
        self.store_jitter = store_jitter

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            jitter = np.random.multivariate_normal(
                self.mean, self.cov, data_dict["coord"].shape[0]
            )
            jitter = self.scalar * np.clip(jitter / 1.96, -1, 1)
            data_dict["coord"] += jitter
            if self.store_jitter:
                data_dict["jitter"] = jitter
        return data_dict


@TRANSFORMS.register_module()
class ChromaticAutoContrast(object):
    def __init__(self, p=0.2, blend_factor=None):
        self.p = p
        self.blend_factor = blend_factor

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            lo = np.min(data_dict["color"], 0, keepdims=True)
            hi = np.max(data_dict["color"], 0, keepdims=True)
            scale = 255 / (hi - lo)
            contrast_feat = (data_dict["color"][:, :3] - lo) * scale
            blend_factor = (
                np.random.rand() if self.blend_factor is None else self.blend_factor
            )
            data_dict["color"][:, :3] = (1 - blend_factor) * data_dict["color"][
                :, :3
            ] + blend_factor * contrast_feat
        return data_dict


@TRANSFORMS.register_module()
class ChromaticTranslation(object):
    def __init__(self, p=0.95, ratio=0.05):
        self.p = p
        self.ratio = ratio

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * self.ratio
            data_dict["color"][:, :3] = np.clip(tr + data_dict["color"][:, :3], 0, 255)
        return data_dict


@TRANSFORMS.register_module()
class ChromaticJitter(object):
    def __init__(self, p=0.95, std=0.005):
        self.p = p
        self.std = std

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            noise = np.random.randn(data_dict["color"].shape[0], 3)
            noise *= self.std * 255
            data_dict["color"][:, :3] = np.clip(
                noise + data_dict["color"][:, :3], 0, 255
            )
        return data_dict


@TRANSFORMS.register_module()
class RandomColorGrayScale(object):
    def __init__(self, p):
        self.p = p

    @staticmethod
    def rgb_to_grayscale(color, num_output_channels=1):
        if color.shape[-1] < 3:
            raise TypeError(
                "Input color should have at least 3 dimensions, but found {}".format(
                    color.shape[-1]
                )
            )

        if num_output_channels not in (1, 3):
            raise ValueError("num_output_channels should be either 1 or 3")

        r, g, b = color[..., 0], color[..., 1], color[..., 2]
        gray = (0.2989 * r + 0.587 * g + 0.114 * b).astype(color.dtype)
        gray = np.expand_dims(gray, axis=-1)

        if num_output_channels == 3:
            gray = np.broadcast_to(gray, color.shape)

        return gray

    def __call__(self, data_dict):
        if np.random.rand() < self.p:
            data_dict["color"] = self.rgb_to_grayscale(data_dict["color"], 3)
        return data_dict


@TRANSFORMS.register_module()
class RandomColorJitter(object):
    """
    Random Color Jitter for 3D point cloud (refer torchvision)
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=0.95):
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(
            hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False
        )
        self.p = p

    @staticmethod
    def _check_input(
        value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True
    ):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    "If {} is a single number, it must be non negative.".format(name)
                )
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError(
                "{} should be a single number or a list/tuple with length 2.".format(
                    name
                )
            )

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def blend(color1, color2, ratio):
        ratio = float(ratio)
        bound = 255.0
        return (
            (ratio * color1 + (1.0 - ratio) * color2)
            .clip(0, bound)
            .astype(color1.dtype)
        )

    @staticmethod
    def rgb2hsv(rgb):
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb, axis=-1)
        minc = np.min(rgb, axis=-1)
        eqc = maxc == minc
        cr = maxc - minc
        s = cr / (np.ones_like(maxc) * eqc + maxc * (1 - eqc))
        cr_divisor = np.ones_like(maxc) * eqc + cr * (1 - eqc)
        rc = (maxc - r) / cr_divisor
        gc = (maxc - g) / cr_divisor
        bc = (maxc - b) / cr_divisor

        hr = (maxc == r) * (bc - gc)
        hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
        hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
        h = hr + hg + hb
        h = (h / 6.0 + 1.0) % 1.0
        return np.stack((h, s, maxc), axis=-1)

    @staticmethod
    def hsv2rgb(hsv):
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = np.floor(h * 6.0)
        f = (h * 6.0) - i
        i = i.astype(np.int32)

        p = np.clip((v * (1.0 - s)), 0.0, 1.0)
        q = np.clip((v * (1.0 - s * f)), 0.0, 1.0)
        t = np.clip((v * (1.0 - s * (1.0 - f))), 0.0, 1.0)
        i = i % 6
        mask = np.expand_dims(i, axis=-1) == np.arange(6)

        a1 = np.stack((v, q, p, p, t, v), axis=-1)
        a2 = np.stack((t, v, v, q, p, p), axis=-1)
        a3 = np.stack((p, p, t, v, v, q), axis=-1)
        a4 = np.stack((a1, a2, a3), axis=-1)

        return np.einsum("...na, ...nab -> ...nb", mask.astype(hsv.dtype), a4)

    def adjust_brightness(self, color, brightness_factor):
        if brightness_factor < 0:
            raise ValueError(
                "brightness_factor ({}) is not non-negative.".format(brightness_factor)
            )

        return self.blend(color, np.zeros_like(color), brightness_factor)

    def adjust_contrast(self, color, contrast_factor):
        if contrast_factor < 0:
            raise ValueError(
                "contrast_factor ({}) is not non-negative.".format(contrast_factor)
            )
        mean = np.mean(RandomColorGrayScale.rgb_to_grayscale(color))
        return self.blend(color, mean, contrast_factor)

    def adjust_saturation(self, color, saturation_factor):
        if saturation_factor < 0:
            raise ValueError(
                "saturation_factor ({}) is not non-negative.".format(saturation_factor)
            )
        gray = RandomColorGrayScale.rgb_to_grayscale(color)
        return self.blend(color, gray, saturation_factor)

    def adjust_hue(self, color, hue_factor):
        if not (-0.5 <= hue_factor <= 0.5):
            raise ValueError(
                "hue_factor ({}) is not in [-0.5, 0.5].".format(hue_factor)
            )
        orig_dtype = color.dtype
        hsv = self.rgb2hsv(color / 255.0)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        h = (h + hue_factor) % 1.0
        hsv = np.stack((h, s, v), axis=-1)
        color_hue_adj = (self.hsv2rgb(hsv) * 255.0).astype(orig_dtype)
        return color_hue_adj

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        fn_idx = torch.randperm(4)
        b = (
            None
            if brightness is None
            else np.random.uniform(brightness[0], brightness[1])
        )
        c = None if contrast is None else np.random.uniform(contrast[0], contrast[1])
        s = (
            None
            if saturation is None
            else np.random.uniform(saturation[0], saturation[1])
        )
        h = None if hue is None else np.random.uniform(hue[0], hue[1])
        return fn_idx, b, c, s, h

    def __call__(self, data_dict):
        (
            fn_idx,
            brightness_factor,
            contrast_factor,
            saturation_factor,
            hue_factor,
        ) = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

        for fn_id in fn_idx:
            if (
                fn_id == 0
                and brightness_factor is not None
                and np.random.rand() < self.p
            ):
                data_dict["color"] = self.adjust_brightness(
                    data_dict["color"], brightness_factor
                )
            elif (
                fn_id == 1 and contrast_factor is not None and np.random.rand() < self.p
            ):
                data_dict["color"] = self.adjust_contrast(
                    data_dict["color"], contrast_factor
                )
            elif (
                fn_id == 2
                and saturation_factor is not None
                and np.random.rand() < self.p
            ):
                data_dict["color"] = self.adjust_saturation(
                    data_dict["color"], saturation_factor
                )
            elif fn_id == 3 and hue_factor is not None and np.random.rand() < self.p:
                data_dict["color"] = self.adjust_hue(data_dict["color"], hue_factor)
        return data_dict


@TRANSFORMS.register_module()
class HueSaturationTranslation(object):
    @staticmethod
    def rgb_to_hsv(rgb):
        # Translated from source of colorsys.rgb_to_hsv
        # r,g,b should be a numpy arrays with values between 0 and 255
        # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
        rgb = rgb.astype("float")
        hsv = np.zeros_like(rgb)
        # in case an RGBA array was passed, just copy the A channel
        hsv[..., 3:] = rgb[..., 3:]
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb[..., :3], axis=-1)
        minc = np.min(rgb[..., :3], axis=-1)
        hsv[..., 2] = maxc
        mask = maxc != minc
        hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
        rc = np.zeros_like(r)
        gc = np.zeros_like(g)
        bc = np.zeros_like(b)
        rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
        gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
        bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
        hsv[..., 0] = np.select(
            [r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc
        )
        hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
        return hsv

    @staticmethod
    def hsv_to_rgb(hsv):
        # Translated from source of colorsys.hsv_to_rgb
        # h,s should be a numpy arrays with values between 0.0 and 1.0
        # v should be a numpy array with values between 0.0 and 255.0
        # hsv_to_rgb returns an array of uints between 0 and 255.
        rgb = np.empty_like(hsv)
        rgb[..., 3:] = hsv[..., 3:]
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = (h * 6.0).astype("uint8")
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
        rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
        rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
        rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
        return rgb.astype("uint8")

    def __init__(self, hue_max=0.5, saturation_max=0.2):
        self.hue_max = hue_max
        self.saturation_max = saturation_max

    def __call__(self, data_dict):
        if "color" in data_dict.keys():
            # Assume color[:, :3] is rgb
            hsv = HueSaturationTranslation.rgb_to_hsv(data_dict["color"][:, :3])
            hue_val = (np.random.rand() - 0.5) * 2 * self.hue_max
            sat_ratio = 1 + (np.random.rand() - 0.5) * 2 * self.saturation_max
            hsv[..., 0] = np.remainder(hue_val + hsv[..., 0] + 1, 1)
            hsv[..., 1] = np.clip(sat_ratio * hsv[..., 1], 0, 1)
            data_dict["color"][:, :3] = np.clip(
                HueSaturationTranslation.hsv_to_rgb(hsv), 0, 255
            )
        return data_dict


@TRANSFORMS.register_module()
class RandomColorDrop(object):
    def __init__(self, p=0.2, color_augment=0.0):
        self.p = p
        self.color_augment = color_augment

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            data_dict["color"] *= self.color_augment
        return data_dict

    def __repr__(self):
        return "RandomColorDrop(color_augment: {}, p: {})".format(
            self.color_augment, self.p
        )


@TRANSFORMS.register_module()
class ElasticDistortion(object):
    def __init__(self, distortion_params=None):
        self.distortion_params = (
            [[0.2, 0.4], [0.8, 1.6]] if distortion_params is None else distortion_params
        )

    @staticmethod
    def elastic_distortion(coords, granularity, magnitude):
        """
        Apply elastic distortion on sparse coordinate space.
        pointcloud: numpy array of (number of points, at least 3 spatial dims)
        granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
        magnitude: noise multiplier
        """
        blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
        blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
        blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
        coords_min = coords.min(0)

        # Create Gaussian noise tensor of the size given by granularity.
        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        # Smoothing.
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(
                noise, blurx, mode="constant", cval=0
            )
            noise = scipy.ndimage.filters.convolve(
                noise, blury, mode="constant", cval=0
            )
            noise = scipy.ndimage.filters.convolve(
                noise, blurz, mode="constant", cval=0
            )

        # Trilinear interpolate noise filters for each spatial dimensions.
        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(
                coords_min - granularity,
                coords_min + granularity * (noise_dim - 2),
                noise_dim,
            )
        ]
        interp = scipy.interpolate.RegularGridInterpolator(
            ax, noise, bounds_error=False, fill_value=0
        )
        coords += interp(coords) * magnitude
        return coords

    def __call__(self, data_dict):
        if "coord" in data_dict.keys() and self.distortion_params is not None:
            if random.random() < 0.95:
                for granularity, magnitude in self.distortion_params:
                    data_dict["coord"] = self.elastic_distortion(
                        data_dict["coord"], granularity, magnitude
                    )
        return data_dict


#### 增加一个对coord相关属性进行clipping的函数 ###
@TRANSFORMS.register_module()
class ClipCoord(object):
    """
    对 coord 进行 clipping，去除超出指定范围的点，并同步更新相关属性。
    """

    def __init__(self, coord_min=None, coord_max=None, coord_bound=None):
        """
        Args:
            coord_min: 坐标最小值，可以是标量或 [x_min, y_min] / [x_min, y_min, z_min]
            coord_max: 坐标最大值，可以是标量或 [x_max, y_max] / [x_max, y_max, z_max]
            coord_bound: 对称边界，等价于 coord_min=-coord_bound, coord_max=coord_bound
        """
        if coord_bound is not None:
            # 使用对称边界
            if isinstance(coord_bound, (int, float)):
                self.coord_min = -coord_bound
                self.coord_max = coord_bound
            else:
                self.coord_min = [-b for b in coord_bound]
                self.coord_max = list(coord_bound)
        else:
            self.coord_min = coord_min
            self.coord_max = coord_max

    def __call__(self, data_dict):
        coord = data_dict["coord"]

        # 构建有效点的 mask
        if isinstance(coord, np.ndarray):
            valid_mask = np.ones(coord.shape[0], dtype=bool)
        else:
            valid_mask = torch.ones(coord.shape[0], dtype=torch.bool)

        # 检查坐标下界
        if self.coord_min is not None:
            if isinstance(self.coord_min, (int, float)):
                # 标量，对所有维度使用相同的下界
                for dim in range(coord.shape[1]):
                    if isinstance(coord, np.ndarray):
                        valid_mask = valid_mask & (coord[:, dim] >= self.coord_min)
                    else:
                        valid_mask = valid_mask & (coord[:, dim] >= self.coord_min)
            else:
                # 列表/数组，每个维度有不同的下界
                for dim, min_val in enumerate(self.coord_min):
                    if min_val is not None and dim < coord.shape[1]:
                        if isinstance(coord, np.ndarray):
                            valid_mask = valid_mask & (coord[:, dim] >= min_val)
                        else:
                            valid_mask = valid_mask & (coord[:, dim] >= min_val)

        # 检查坐标上界
        if self.coord_max is not None:
            if isinstance(self.coord_max, (int, float)):
                # 标量，对所有维度使用相同的上界
                for dim in range(coord.shape[1]):
                    if isinstance(coord, np.ndarray):
                        valid_mask = valid_mask & (coord[:, dim] <= self.coord_max)
                    else:
                        valid_mask = valid_mask & (coord[:, dim] <= self.coord_max)
            else:
                # 列表/数组，每个维度有不同的上界
                for dim, max_val in enumerate(self.coord_max):
                    if max_val is not None and dim < coord.shape[1]:
                        if isinstance(coord, np.ndarray):
                            valid_mask = valid_mask & (coord[:, dim] <= max_val)
                        else:
                            valid_mask = valid_mask & (coord[:, dim] <= max_val)

        # 如果所有点都有效，直接返回
        if isinstance(valid_mask, np.ndarray):
            if valid_mask.all():
                return data_dict
            valid_indices = np.where(valid_mask)[0]
        else:
            if valid_mask.all():
                return data_dict
            valid_indices = torch.where(valid_mask)[0]

        # 使用 index_valid_keys 同步更新相关属性
        if "index_valid_keys" not in data_dict:
            data_dict["index_valid_keys"] = [
                "coord",
                "color",
                "normal",
                "superpoint",
                "strength",
                "segment",
                "instance",
                "feat",
                "semanticIds",
                "instanceIds",
            ]

        for key in data_dict["index_valid_keys"]:
            if key in data_dict and data_dict[key] is not None:
                data_dict[key] = data_dict[key][valid_indices]

        # 更新 offset（如果存在）
        if "offset" in data_dict:
            data_dict["offset"] = torch.tensor([data_dict["coord"].shape[0]])

        # 更新 sampled_index（如果存在）
        if "sampled_index" in data_dict:
            if isinstance(valid_indices, np.ndarray):
                data_dict["sampled_index"] = np.arange(0, data_dict["coord"].shape[0])
            else:
                data_dict["sampled_index"] = torch.arange(
                    0, data_dict["coord"].shape[0]
                )

        return data_dict


#### 对点云进行体素化并进行下采样
@TRANSFORMS.register_module()
class GridSample(object):
    def __init__(
        self,
        grid_size=0.05,
        hash_type="fnv",
        mode="train",
        return_inverse=False,
        return_grid_coord=False,
        return_min_coord=False,
        return_displacement=False,
        project_displacement=False,
        clip_bound=None,  # 新增：clipping 边界，如 2.0 表示 [-2, 2]
    ):
        self.grid_size = grid_size
        self.hash = self.fnv_hash_vec if hash_type == "fnv" else self.ravel_hash_vec
        assert mode in ["train", "test"]
        self.mode = mode
        self.return_inverse = return_inverse
        self.return_grid_coord = return_grid_coord
        self.return_min_coord = return_min_coord
        self.return_displacement = return_displacement
        self.project_displacement = project_displacement
        self.clip_bound = clip_bound

    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()

        ### BUG: 可能出现clip窗口内没有的点 ###
        # ====== 先进行 Clipping ======
        if self.clip_bound is not None:
            coord = data_dict["coord"]
            data_dict["coord_bk"] = coord.copy()

            # 检查每个点是否在 clipping 范围内（只检查 x, y 维度）
            valid_mask = (
                (coord[:, 0] >= -self.clip_bound)
                & (coord[:, 0] <= self.clip_bound)
                & (coord[:, 1] >= -self.clip_bound)
                & (coord[:, 1] <= self.clip_bound)
            )

            # 如果有超出范围的点，进行过滤
            if not valid_mask.all():
                valid_indices = np.where(valid_mask)[0]
                # valid_indices = []
                # 如果所有点都被过滤掉了，设置 offset 为 0，让 Dataset 跳过该样本
                if len(valid_indices) == 0:
                    data_dict["coord"] = np.array([]).reshape(0, 3)
                    # data_dict['offset'] = np.array([0])
                    return data_dict

                valid_indices_set = set(valid_indices)

                #### 检查哪些 instance 被完全 clip 掉 ####
                if "primitive_index" in data_dict:
                    old_primitive_index = data_dict["primitive_index"]
                    instance_valid_mask = []  # 每个 instance 是否有保留的点

                    for start, end in old_primitive_index:
                        # 检查该 instance 的点范围 [start, end) 内是否有任何点在 valid_indices 中
                        has_valid_point = any(
                            i in valid_indices_set for i in range(start, end)
                        )
                        instance_valid_mask.append(has_valid_point)

                    instance_valid_mask = np.array(instance_valid_mask)

                    # 如果有 instance 被完全 clip 掉，需要过滤 instance 级别属性
                    if not instance_valid_mask.all():
                        instance_valid_indices = np.where(instance_valid_mask)[0]

                        # instance 级别属性列表（包含 primitive_index）
                        instance_keys = [
                            "translation",
                            "rotation",
                            "scale",
                            "semantic_id",
                            "primitive_id",
                            "flip_mode",
                            "primitive_bboxes",
                            "instance_masks",
                            "primitive_index",
                        ]

                        for key in instance_keys:
                            if key in data_dict and data_dict[key] is not None:
                                data_dict[key] = data_dict[key][instance_valid_indices]

                    # 对于 partial clip 或未 clip 的 instance，保留完整的原始 primitive_index，不更新偏移量

                # 使用 index_operator 同步更新所有点级别的属性
                data_dict = index_operator(data_dict, valid_indices)
                data_dict["sampled_index"] = np.arange(0, data_dict["coord"].shape[0])
        else:
            coord = data_dict["coord"]
            data_dict["coord_bk"] = coord.copy()

        # ====== 然后进行 GridSample ======
        scaled_coord = data_dict["coord"] / np.array(self.grid_size)
        grid_coord = np.floor(scaled_coord).astype(int)
        min_coord = grid_coord.min(0)
        grid_coord -= min_coord
        scaled_coord -= min_coord
        min_coord = min_coord * np.array(self.grid_size)
        key = self.hash(grid_coord)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
        if self.mode == "train":  # train mode
            idx_select = (
                np.cumsum(np.insert(count, 0, 0)[0:-1])
                + np.random.randint(0, count.max(), count.size) % count
            )
            idx_unique = idx_sort[idx_select]
            if "sampled_index" in data_dict:
                # for ScanNet data efficient, we need to make sure labeled point is sampled.
                sampled_idx = data_dict["sampled_index"]
                # idx_unique = np.unique(
                #     np.append(idx_unique, data_dict["sampled_index"])
                # )
                idx_part_unique = np.setdiff1d(
                    idx_unique, sampled_idx, assume_unique=False
                )
                idx_unique = np.concatenate([idx_part_unique, sampled_idx])
                # mask = np.zeros_like(data_dict["segment"]).astype(bool)
                # mask[data_dict["sampled_index"]] = True
                # data_dict["sampled_index"] = np.where(mask[idx_unique])[0]
            data_dict = index_operator(data_dict, idx_unique)
            if self.return_inverse:
                data_dict["inverse"] = np.zeros_like(inverse)
                data_dict["inverse"][idx_sort] = inverse
            if self.return_grid_coord:
                data_dict["grid_coord"] = grid_coord[idx_unique]
                if "grid_coord" not in data_dict["index_valid_keys"]:
                    data_dict["index_valid_keys"].append("grid_coord")
            if self.return_min_coord:
                data_dict["min_coord"] = min_coord.reshape([1, 3])
            if self.return_displacement:
                displacement = (
                    scaled_coord - grid_coord - 0.5
                )  # [0, 1] -> [-0.5, 0.5] displacement to center
                if self.project_displacement:
                    displacement = np.sum(
                        displacement * data_dict["normal"], axis=-1, keepdims=True
                    )
                data_dict["displacement"] = displacement[idx_unique]
                if "displacement" not in data_dict["index_valid_keys"]:
                    data_dict["index_valid_keys"].append("displacement")
            return data_dict

        elif self.mode == "test":  # test mode
            data_part_list = []
            for i in range(count.max()):
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                if "sampled_index" in data_dict:
                    # 方案2：使用setdiff1d去除重复元素，然后添加sampled_index
                    sampled_idx = data_dict["sampled_index"]
                    # setdiff1d返回在idx_part中但不在sampled_idx中的元素
                    idx_part_unique = np.setdiff1d(
                        idx_part, sampled_idx, assume_unique=False
                    )
                    idx_part = np.concatenate([idx_part_unique, sampled_idx])

                data_part = index_operator(data_dict, idx_part, duplicate=True)
                data_part["index"] = idx_part

                if self.return_inverse:
                    data_part["inverse"] = np.zeros_like(inverse)
                    data_part["inverse"][idx_sort] = inverse
                if self.return_grid_coord:
                    data_part["grid_coord"] = grid_coord[idx_part]
                    if "grid_coord" not in data_part["index_valid_keys"]:
                        data_part["index_valid_keys"].append("grid_coord")
                if self.return_min_coord:
                    data_part["min_coord"] = min_coord.reshape([1, 3])
                if self.return_displacement:
                    displacement = (
                        scaled_coord - grid_coord - 0.5
                    )  # [0, 1] -> [-0.5, 0.5] displacement to center
                    if self.project_displacement:
                        displacement = np.sum(
                            displacement * data_dict["normal"], axis=-1, keepdims=True
                        )
                    data_part["displacement"] = displacement[idx_part]
                    if "displacement" not in data_part["index_valid_keys"]:
                        data_part["index_valid_keys"].append("displacement")
                data_part_list.append(data_part)
            return data_part_list[0]
        else:
            raise NotImplementedError

    @staticmethod
    def ravel_hash_vec(arr):
        """
        Ravel the coordinates after subtracting the min coordinates.
        """
        assert arr.ndim == 2
        arr = arr.copy()
        arr -= arr.min(0)
        arr = arr.astype(np.uint64, copy=False)
        arr_max = arr.max(0).astype(np.uint64) + 1

        keys = np.zeros(arr.shape[0], dtype=np.uint64)
        # Fortran style indexing
        for j in range(arr.shape[1] - 1):
            keys += arr[:, j]
            keys *= arr_max[j + 1]
        keys += arr[:, -1]
        return keys

    @staticmethod
    def fnv_hash_vec(arr):
        """
        FNV64-1A
        """
        assert arr.ndim == 2
        # Floor first for negative coordinates
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(
            arr.shape[0], dtype=np.uint64
        )
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr


#### 通过某种采样方式来确保处理的点云的点数量小于一定阈值 (point_max)；
@TRANSFORMS.register_module()
class SphereCrop(object):
    def __init__(self, point_max=80000, sample_rate=None, mode="random"):
        self.point_max = point_max
        self.sample_rate = sample_rate
        assert mode in ["random", "center", "all"]
        self.mode = mode

    def __call__(self, data_dict):
        point_max = (
            int(self.sample_rate * data_dict["coord"].shape[0])
            if self.sample_rate is not None
            else self.point_max
        )

        assert "coord" in data_dict.keys()
        if data_dict["coord"].shape[0] > point_max:
            if self.mode == "random":
                center = data_dict["coord"][
                    np.random.randint(data_dict["coord"].shape[0])
                ]
            elif self.mode == "center":
                center = data_dict["coord"][data_dict["coord"].shape[0] // 2]
            else:
                raise NotImplementedError
            idx_crop = np.argsort(np.sum(np.square(data_dict["coord"] - center), 1))[
                :point_max
            ]
            data_dict = index_operator(data_dict, idx_crop)
        return data_dict


@TRANSFORMS.register_module()
class ShufflePoint(object):
    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        shuffle_index = np.arange(data_dict["coord"].shape[0])
        np.random.shuffle(shuffle_index)
        data_dict = index_operator(data_dict, shuffle_index)
        return data_dict


@TRANSFORMS.register_module()
class CropBoundary(object):
    def __call__(self, data_dict):
        assert "segment" in data_dict
        segment = data_dict["segment"].flatten()
        mask = (segment != 0) * (segment != 1)
        data_dict = index_operator(data_dict, mask)
        return data_dict


@TRANSFORMS.register_module()
class ContrastiveViewsGenerator(object):
    def __init__(
        self,
        view_keys=("coord", "color", "normal", "origin_coord"),
        view_trans_cfg=None,
    ):
        self.view_keys = view_keys
        self.view_trans = Compose(view_trans_cfg)

    def __call__(self, data_dict):
        view1_dict = dict()
        view2_dict = dict()
        for key in self.view_keys:
            view1_dict[key] = data_dict[key].copy()
            view2_dict[key] = data_dict[key].copy()
        view1_dict = self.view_trans(view1_dict)
        view2_dict = self.view_trans(view2_dict)
        for key, value in view1_dict.items():
            data_dict["view1_" + key] = value
        for key, value in view2_dict.items():
            data_dict["view2_" + key] = value
        return data_dict


@TRANSFORMS.register_module()
class MultiViewGenerator(object):
    def __init__(
        self,
        global_view_num=2,
        global_view_scale=(0.4, 1.0),
        local_view_num=4,
        local_view_scale=(0.1, 0.4),
        global_shared_transform=None,
        global_transform=None,
        local_transform=None,
        max_size=65536,
        center_height_scale=(0, 1),
        shared_global_view=False,
        view_keys=("coord", "origin_coord", "color", "normal"),
    ):
        self.global_view_num = global_view_num
        self.global_view_scale = global_view_scale
        self.local_view_num = local_view_num
        self.local_view_scale = local_view_scale
        self.global_shared_transform = Compose(global_shared_transform)
        self.global_transform = Compose(global_transform)
        self.local_transform = Compose(local_transform)
        self.max_size = max_size
        self.center_height_scale = center_height_scale
        self.shared_global_view = shared_global_view
        self.view_keys = view_keys
        assert "coord" in view_keys

    def get_view(self, point, center, scale):
        coord = point["coord"]
        max_size = min(self.max_size, coord.shape[0])
        size = int(np.random.uniform(*scale) * max_size)
        index = np.argsort(np.sum(np.square(coord - center), axis=-1))[:size]
        view = dict(index=index)
        for key in point.keys():
            if key in self.view_keys:
                view[key] = point[key][index]

        if "index_valid_keys" in point.keys():
            # inherit index_valid_keys from point
            view["index_valid_keys"] = point["index_valid_keys"]
        return view

    def __call__(self, data_dict):
        coord = data_dict["coord"]
        point = self.global_shared_transform(copy.deepcopy(data_dict))
        z_min = coord[:, 2].min()
        z_max = coord[:, 2].max()
        z_min_ = z_min + (z_max - z_min) * self.center_height_scale[0]
        z_max_ = z_min + (z_max - z_min) * self.center_height_scale[1]
        center_mask = np.logical_and(coord[:, 2] >= z_min_, coord[:, 2] <= z_max_)
        # get major global view
        major_center = coord[np.random.choice(np.where(center_mask)[0])]
        major_view = self.get_view(point, major_center, self.global_view_scale)
        major_coord = major_view["coord"]
        # get global views: restrict the center of left global view within the major global view
        if not self.shared_global_view:
            global_views = [
                self.get_view(
                    point=point,
                    center=major_coord[np.random.randint(major_coord.shape[0])],
                    scale=self.global_view_scale,
                )
                for _ in range(self.global_view_num - 1)
            ]
        else:
            global_views = [
                {key: value.copy() for key, value in major_view.items()}
                for _ in range(self.global_view_num - 1)
            ]

        global_views = [major_view] + global_views

        # get local views: restrict the center of local view within the major global view
        cover_mask = np.zeros_like(major_view["index"], dtype=bool)
        local_views = []
        for i in range(self.local_view_num):
            if sum(~cover_mask) == 0:
                # reset cover mask if all points are sampled
                cover_mask[:] = False
            local_view = self.get_view(
                point=data_dict,
                center=major_coord[np.random.choice(np.where(~cover_mask)[0])],
                scale=self.local_view_scale,
            )
            local_views.append(local_view)
            cover_mask[np.isin(major_view["index"], local_view["index"])] = True

        # augmentation and concat
        view_dict = {}
        for global_view in global_views:
            global_view.pop("index")
            global_view = self.global_transform(global_view)
            for key in self.view_keys:
                if f"global_{key}" in view_dict.keys():
                    view_dict[f"global_{key}"].append(global_view[key])
                else:
                    view_dict[f"global_{key}"] = [global_view[key]]
        view_dict["global_offset"] = np.cumsum(
            [data.shape[0] for data in view_dict["global_coord"]]
        )
        for local_view in local_views:
            local_view.pop("index")
            local_view = self.local_transform(local_view)
            for key in self.view_keys:
                if f"local_{key}" in view_dict.keys():
                    view_dict[f"local_{key}"].append(local_view[key])
                else:
                    view_dict[f"local_{key}"] = [local_view[key]]
        view_dict["local_offset"] = np.cumsum(
            [data.shape[0] for data in view_dict["local_coord"]]
        )
        for key in view_dict.keys():
            if "offset" not in key:
                view_dict[key] = np.concatenate(view_dict[key], axis=0)
        data_dict.update(view_dict)
        return data_dict


@TRANSFORMS.register_module()
class InstanceParser(object):
    def __init__(self, segment_ignore_index=(-1, 0, 1), instance_ignore_index=-1):
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index

    def __call__(self, data_dict):
        coord = data_dict["coord"]
        segment = data_dict["segment"]
        instance = data_dict["instance"]
        mask = ~np.in1d(segment, self.segment_ignore_index)
        # mapping ignored instance to ignore index
        instance[~mask] = self.instance_ignore_index
        # reorder left instance
        unique, inverse = np.unique(instance[mask], return_inverse=True)
        instance_num = len(unique)
        instance[mask] = inverse
        # init instance information
        centroid = np.ones((coord.shape[0], 3)) * self.instance_ignore_index
        bbox = np.ones((instance_num, 8)) * self.instance_ignore_index
        vacancy = [
            index for index in self.segment_ignore_index if index >= 0
        ]  # vacate class index

        for instance_id in range(instance_num):
            mask_ = instance == instance_id
            coord_ = coord[mask_]
            bbox_min = coord_.min(0)
            bbox_max = coord_.max(0)
            bbox_centroid = coord_.mean(0)
            bbox_center = (bbox_max + bbox_min) / 2
            bbox_size = bbox_max - bbox_min
            bbox_theta = np.zeros(1, dtype=coord_.dtype)
            bbox_class = np.array([segment[mask_][0]], dtype=coord_.dtype)
            # shift class index to fill vacate class index caused by segment ignore index
            bbox_class -= np.greater(bbox_class, vacancy).sum()

            centroid[mask_] = bbox_centroid
            bbox[instance_id] = np.concatenate(
                [bbox_center, bbox_size, bbox_theta, bbox_class]
            )  # 3 + 3 + 1 + 1 = 8
        data_dict["instance"] = instance
        data_dict["instance_centroid"] = centroid
        data_dict["bbox"] = bbox
        return data_dict


@TRANSFORMS.register_module()
class PrimitiveRandomRotate(object):
    def __init__(self, p=0.5, sample_ratio=0.2, rotate_range=(-180, 180)):
        self.p = p
        self.sample_ratio = sample_ratio
        self.rotate_range = rotate_range

    def __call__(self, data_dict):
        if np.random.rand() < self.p:
            primitive_coords = data_dict["primitive_coords"]
            primitive_feats = data_dict["primitive_feats"]

            # 复制一份以避免修改原数据
            rotated_primitive_coords = [coord.copy() for coord in primitive_coords]

            updated_rotation = data_dict["rotation"].copy()

            #### sample sample_ratio of the primitive_coords to rotate
            sample_num = int(self.sample_ratio * len(primitive_coords))
            if sample_num > 0:
                sample_indices = np.random.choice(
                    len(primitive_coords), sample_num, replace=False
                )

                # 对每个选中的primitive进行旋转
                for idx in sample_indices:
                    coord = rotated_primitive_coords[idx]  # shape: [M, 3]

                    # 计算primitive的中心点（只考虑x, y坐标）
                    center_x = np.mean(coord[:, 0])
                    center_y = np.mean(coord[:, 1])
                    center = np.array([center_x, center_y])

                    # 随机生成旋转角度
                    angle = np.random.uniform(
                        self.rotate_range[0], self.rotate_range[1]
                    )
                    angle_rad = np.radians(angle)

                    # 创建2D旋转矩阵
                    cos_angle = np.cos(angle_rad)
                    sin_angle = np.sin(angle_rad)
                    rotation_matrix = np.array(
                        [[cos_angle, -sin_angle], [sin_angle, cos_angle]]
                    )

                    # 提取x, y坐标
                    xy_coords = coord[:, :2]  # shape: [M, 2]

                    # 平移到以中心为原点
                    centered_coords = xy_coords - center  # shape: [M, 2]

                    # 应用旋转
                    rotated_coords = (
                        centered_coords @ rotation_matrix.T
                    )  # shape: [M, 2]

                    # 平移回原来的位置
                    final_coords = rotated_coords + center  # shape: [M, 2]

                    # 更新坐标（保持z坐标不变）
                    rotated_primitive_coords[idx][:, :2] = final_coords

                    updated_rotation[idx] += angle
                    # 确保角度在[0, 360]范围内
                    updated_rotation[idx] = updated_rotation[idx] % 360

            #### update the data_dict
            data_dict["primitive_coords"] = rotated_primitive_coords
            data_dict["primitive_feats"] = primitive_feats
            data_dict["rotation"] = updated_rotation

        return data_dict


@TRANSFORMS.register_module()
class PrimitiveRandomTranslate(object):
    def __init__(self, p=0.5, sample_ratio=0.2, translate_range=(-0.1, 0.1)):
        self.p = p
        self.sample_ratio = sample_ratio
        self.translate_range = translate_range

    def __call__(self, data_dict):
        if np.random.rand() < self.p:
            primitive_coords = data_dict["primitive_coords"]
            primitive_feats = data_dict["primitive_feats"]

            # 复制一份以避免修改原数据
            translated_primitive_coords = [coord.copy() for coord in primitive_coords]

            updated_translation = data_dict["translation"].copy()

            #### sample sample_ratio of the primitive_coords to translate
            sample_num = int(self.sample_ratio * len(primitive_coords))
            if sample_num > 0:
                sample_indices = np.random.choice(
                    len(primitive_coords), sample_num, replace=False
                )

                # 对每个选中的primitive进行平移
                for idx in sample_indices:
                    coord = translated_primitive_coords[idx]  # shape: [M, 3]

                    # 随机生成平移量 (只在x, y方向)
                    delta_x = np.random.uniform(
                        self.translate_range[0], self.translate_range[1]
                    )
                    delta_y = np.random.uniform(
                        self.translate_range[0], self.translate_range[1]
                    )
                    delta = np.array([delta_x, delta_y])

                    # 检查平移后的translation是否会超出[-1, 1]范围
                    new_translation = updated_translation[idx] + delta / 70.0
                    # if np.any(new_translation < -1.0) or np.any(new_translation > 1.0):
                    #     # 如果超出范围，跳过这个primitive
                    #     continue

                    # 应用平移到坐标
                    translated_primitive_coords[idx][:, :2] += delta

                    # 更新translation属性
                    updated_translation[idx] = new_translation

            #### update the data_dict
            data_dict["primitive_coords"] = translated_primitive_coords
            data_dict["primitive_feats"] = primitive_feats
            data_dict["translation"] = updated_translation

        return data_dict


@TRANSFORMS.register_module()
class PrimitiveRandomDropout(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data_dict):
        if np.random.rand() < self.p:
            num_primitives = len(data_dict["primitive_coords"])

            if num_primitives == 0:
                return data_dict

            sample_ratio = np.random.uniform(0.1, 0.9)
            # 计算要丢弃的primitive数量
            drop_num = int(sample_ratio * num_primitives)
            if drop_num == 0:
                return data_dict

            # 确保至少保留一个primitive
            if drop_num >= num_primitives:
                drop_num = num_primitives - 1

            # 随机选择要丢弃的primitive索引
            drop_indices = np.random.choice(num_primitives, drop_num, replace=False)
            # 计算要保留的索引
            keep_indices = np.setdiff1d(np.arange(num_primitives), drop_indices)

            # 丢弃primitive_coords和primitive_feats（如果存在）
            if "primitive_coords" in data_dict:
                data_dict["primitive_coords"] = [
                    data_dict["primitive_coords"][i] for i in keep_indices
                ]

            if "primitive_feats" in data_dict:
                data_dict["primitive_feats"] = [
                    data_dict["primitive_feats"][i] for i in keep_indices
                ]

            # ### add for primitive_bboxes
            # if 'primitive_bboxes' in data_dict:
            #     data_dict['primitive_bboxes'] = [data_dict['primitive_bboxes'][i] for i in keep_indices]

            # 丢弃primitive属性（处理numpy数组和tensor）
            primitive_keys = [
                "semantic_id",
                "primitive_id",
                "scale",
                "rotation",
                "flip_mode",
                "translation",
                "flip_invariant_primitive",
                "primitive_bboxes",
            ]

            for key in primitive_keys:
                if key in data_dict:
                    data = data_dict[key]
                    data_dict[key] = data[keep_indices]

            # 重新计算primitive_offsets
            if "primitive_offsets" in data_dict:
                # 获取保留的primitive的原始offsets
                original_offsets = data_dict["primitive_offsets"]
                kept_offsets = [original_offsets[i] for i in keep_indices]

                # 重新计算offsets
                new_offsets = []
                start_idx = 0

                for offset_pair in kept_offsets:
                    primitive_length = offset_pair[1] - offset_pair[0]
                    new_offsets.append([start_idx, start_idx + primitive_length])
                    start_idx += primitive_length

                data_dict["primitive_offsets"] = new_offsets

        return data_dict


@TRANSFORMS.register_module()
class CombineCoord(object):

    def __call__(self, data_dict):
        primitive_coords = data_dict["primitive_coords"]
        primitive_feats = data_dict["primitive_feats"]
        if len(primitive_coords) == 0:
            primitive_coords = np.empty((0, 3), dtype=np.float32)
            primitive_feats = np.empty((0, 6), dtype=np.float32)
        else:
            primitive_coords = np.concatenate(primitive_coords, axis=0)
            primitive_feats = np.concatenate(primitive_feats, axis=0)
        coord = primitive_coords
        feat = primitive_feats
        data_dict["coord"] = coord
        data_dict["feat"] = feat
        ##### save the labeled primitive offsets
        data_dict["sampled_index"] = np.arange(0, coord.shape[0])
        primitive_index = np.array(data_dict["primitive_offsets"])
        data_dict["primitive_index"] = primitive_index
        ##### 这一步可以直接获取每个Instance对应的mask
        instance_masks = []
        for primitive_offset in primitive_index:
            primitive_mask = np.zeros((coord.shape[0]), dtype=np.float32)
            primitive_mask[primitive_offset[0] : primitive_offset[1]] = 1.0
            instance_masks.append(primitive_mask)
        if len(instance_masks) == 0:
            data_dict["instance_masks"] = np.empty(
                (0, coord.shape[0]), dtype=np.float32
            )
        else:
            data_dict["instance_masks"] = np.stack(instance_masks, axis=0)
        del data_dict["primitive_coords"]
        del data_dict["primitive_feats"]
        del data_dict["primitive_offsets"]
        return data_dict


@TRANSFORMS.register_module()
class ShuffleCoord(object):

    def __call__(self, data_dict):
        assert "coord" in data_dict
        coord = data_dict["coord"]
        feat = data_dict["feat"]

        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat = coord[shuf_idx], feat[shuf_idx]

        data_dict["coord"] = coord
        data_dict["feat"] = feat

        return data_dict


@TRANSFORMS.register_module()
class NormalizeCoord(object):
    def __init__(self, normalize_coord=False, scene_bound=20000.0):
        self.normalize_coord = normalize_coord
        self.scene_bound = scene_bound

    def __call__(self, data_dict):
        assert "coord" in data_dict
        coord = data_dict["coord"]
        if self.normalize_coord:
            coord = coord - (self.scene_bound / 2)
            coord = coord / (self.scene_bound / 2)
            coord[:, 2] = 0.0
        else:
            coord = coord - np.mean(coord, 0)

        # valid_mask = (coord[:, 0] >= -1) & (coord[:, 0] <= 1) & \
        #     (coord[:, 1] >= -1) & (coord[:, 1] <= 1)
        # assert valid_mask.any(), f"Before normalize: no points in [-1, 1] range, {data_dict['filename']}"

        #### Normalize box --- 归一化至同样的量纲
        if "primitive_bboxes" in data_dict:
            primitive_bboxes = data_dict["primitive_bboxes"]
            primitive_bboxes = primitive_bboxes / (self.scene_bound / 2)
            data_dict["primitive_bboxes"] = primitive_bboxes

        data_dict["coord"] = coord
        return data_dict


@TRANSFORMS.register_module()
class FilterInvalidTranslation(object):
    def __init__(self, translation_bound=2.0):
        self.translation_bound = translation_bound

    def __call__(self, data_dict):
        translation = data_dict["translation"]

        # 检查每个 instance 的 translation 是否在 [-bound, bound] 范围内
        valid_mask = (np.abs(translation[:, 0]) <= self.translation_bound) & (
            np.abs(translation[:, 1]) <= self.translation_bound
        )
        if translation.shape[1] > 2:
            valid_mask = valid_mask & (
                translation[:, 2].abs() <= self.translation_bound
            )

        # 如果所有 instance 都有效，直接返回
        if valid_mask.all():
            return data_dict

        valid_indices = np.where(valid_mask)[0]

        # 如果没有有效的 instance，返回空数据
        if len(valid_indices) == 0:
            data_dict["translation"] = translation[:0]
            data_dict["semantic_id"] = data_dict["semantic_id"][:0]
            data_dict["primitive_id"] = data_dict["primitive_id"][:0]
            data_dict["scale"] = data_dict["scale"][:0]
            data_dict["rotation"] = data_dict["rotation"][:0]
            data_dict["flip_mode"] = data_dict["flip_mode"][:0]
            if "primitive_bboxes" in data_dict:
                data_dict["primitive_bboxes"] = data_dict["primitive_bboxes"][:0]
            if "instance_masks" in data_dict:
                data_dict["instance_masks"] = data_dict["instance_masks"][:0]
            if "primitive_index" in data_dict:
                data_dict["primitive_index"] = data_dict["primitive_index"][:0]
                data_dict["coord"] = data_dict["coord"][:0]
                if "feat" in data_dict and data_dict["feat"] is not None:
                    data_dict["feat"] = data_dict["feat"][:0]
                data_dict["offset"] = torch.tensor([0])
            return data_dict

        # 过滤 instance 级别的数据
        data_dict["translation"] = translation[valid_indices]
        data_dict["semantic_id"] = data_dict["semantic_id"][valid_indices]
        data_dict["primitive_id"] = data_dict["primitive_id"][valid_indices]
        data_dict["scale"] = data_dict["scale"][valid_indices]
        data_dict["rotation"] = data_dict["rotation"][valid_indices]
        data_dict["flip_mode"] = data_dict["flip_mode"][valid_indices]

        if "primitive_bboxes" in data_dict:
            data_dict["primitive_bboxes"] = data_dict["primitive_bboxes"][valid_indices]
        ##### instance_masks的大小可能存在BUG ####
        if "instance_masks" in data_dict:
            data_dict["instance_masks"] = data_dict["instance_masks"][valid_indices]

        # 过滤 coord 和 feat：根据 primitive_index 过滤对应的点
        if "primitive_index" in data_dict:
            old_primitive_index = data_dict["primitive_index"]

            # 构建有效点的 mask
            valid_point_mask = torch.zeros(
                data_dict["coord"].shape[0], dtype=torch.bool
            )
            new_primitive_index = []
            current_offset = 0

            for idx in valid_indices:
                start, end = old_primitive_index[idx]
                point_count = end - start
                valid_point_mask[start:end] = True
                new_primitive_index.append(
                    torch.tensor([current_offset, current_offset + point_count])
                )
                current_offset += point_count

            # 过滤点云数据
            data_dict["coord"] = data_dict["coord"][valid_point_mask]
            if "feat" in data_dict and data_dict["feat"] is not None:
                data_dict["feat"] = data_dict["feat"][valid_point_mask]

            # 更新 primitive_index
            if len(new_primitive_index) > 0:
                data_dict["primitive_index"] = torch.stack(new_primitive_index)
            else:
                data_dict["primitive_index"] = torch.empty((0, 2), dtype=torch.long)

            # 更新 offset
            data_dict["offset"] = torch.tensor([data_dict["coord"].shape[0]])
            # 更新sampled_index
            data_dict["sampled_index"] = np.arange(0, data_dict["coord"].shape[0])

        return data_dict


@TRANSFORMS.register_module()
class CutMix(object):
    def __init__(self, p=0.5, group_dir=None, scene_bound=30000.0, scale_inv=False):
        self.p = p
        self.group_dir = group_dir
        self.valid_group_lst = [
            os.path.join(self.group_dir, x)
            for x in os.listdir(self.group_dir)
            if x.endswith(".json")
        ]
        self.scene_bound = scene_bound
        self.scale_inv = scale_inv

    def import_primitive_lib(
        self, primitive_lib, primitive_ids, primitive_weights, primitive_to_semantic
    ):
        self.primitive_lib = primitive_lib
        self.primitive_ids = primitive_ids
        self.primitive_weights = primitive_weights
        self.primitive_to_semantic = primitive_to_semantic

    def __call__(self, data_dict):
        num_primitives = len(data_dict["primitive_coords"])
        if np.random.rand() < self.p or num_primitives == 0:
            #### 随机选取一个场景
            group_file = np.random.choice(self.valid_group_lst)
            with open(group_file, "r") as f:
                group_data = json.load(f)
            ### 随机对应场景的一个group进行拼接
            select_group_idx = np.random.choice(len(group_data))
            select_group = group_data[select_group_idx]

            select_group_bbox = select_group["group_bbox"]
            select_group_center = np.array(
                [
                    (select_group_bbox[2] + select_group_bbox[0]) / 2,
                    (select_group_bbox[3] + select_group_bbox[1]) / 2,
                ],
                dtype=np.float32,
            )
            select_group_center += self.scene_bound / 2

            select_group_primitive_ids = select_group["primitive_ids"]
            select_group_semantic_ids = select_group["semantic_ids"]
            select_group_translations = select_group["translations"]
            select_group_rotations = select_group["rotations"]
            select_group_scales = select_group["scales"]
            select_group_flips = select_group["flips"]

            # ### 确认落点：在已有场景的bbox外随机选择一个位置
            # primitive_coords = data_dict['primitive_coords']
            # allPA = np.concatenate([pc[:, :2] for pc in primitive_coords], axis=0)
            # bbA = np.array([allPA[:,0].min(), allPA[:,1].min(), allPA[:,0].max(), allPA[:,1].max()], dtype=np.float32)
            # # 外扩一点便于贴外围
            # pad = 0.1 * max(bbA[2]-bbA[0], bbA[3]-bbA[1])
            # x_rng = (bbA[0]-pad, bbA[2]+pad); y_rng = (bbA[1]-pad, bbA[3]+pad)
            # select_center = np.array([np.random.uniform(*x_rng), np.random.uniform(*y_rng)], dtype=np.float32)
            # ######### 如上代码可能会导致越界

            #### 随机放置
            # select_center = select_group_center + np.random.uniform(-0.1 * self.scene_bound, 0.1 * self.scene_bound, 2)
            select_center = np.random.uniform(
                0.2 * self.scene_bound, 0.8 * self.scene_bound, 2
            )

            relative_translation = select_center - select_group_center

            #### tgt translation = src_translation + relative_translation

            #### 随后把相应的group放进去
            for primitive_id, semantic_id, translation, rotation, scale, flip in zip(
                select_group_primitive_ids,
                select_group_semantic_ids,
                select_group_translations,
                select_group_rotations,
                select_group_scales,
                select_group_flips,
            ):
                # if scale[0] > 2.0 or scale[0] < 0.5 or scale[1] > 2.0 or scale[1] < 0.5:
                #     continue
                scale = np.clip(scale, 0.5, 2.0)
                coords = self.primitive_lib[primitive_id]["coords"].copy()
                feats = self.primitive_lib[primitive_id]["feats"].copy()
                center = (
                    np.min(coords[:, :2], axis=0) + np.max(coords[:, :2], axis=0)
                ) / 2

                rotate_rad = np.radians(rotation)
                cos_angle = np.cos(rotate_rad)
                sin_angle = np.sin(rotate_rad)
                rotation_matrix = np.array(
                    [[cos_angle, -sin_angle], [sin_angle, cos_angle]]
                )

                #### 考虑变换: scale, flip, rotation
                xy_coords = coords[:, :2]  # shape: [M, 2]
                centered_coords = xy_coords - center  # shape: [M, 2]
                if self.scale_inv:
                    scale = 1.0
                    centered_coords = centered_coords * scale
                else:
                    centered_coords = centered_coords * scale
                if flip:
                    centered_coords[:, 0] = -centered_coords[:, 0]
                rotated_coords = centered_coords @ rotation_matrix.T  # shape: [M, 2]
                final_coords = rotated_coords + center  # shape: [M, 2]
                coords[:, :2] = final_coords

                #### 考虑变换translation
                update_translation = translation + relative_translation

                coords[:, :2] += update_translation
                coords[:, 2] = 0.0
                feats[:, 0] = np.arctan(coords[:, 1] / (coords[:, 0] + 1e-8)) / math.pi

                # sub_sample_num = int(coords.shape[0] * self.keep_ratio)
                # if sub_sample_num <= 1:
                #     coords = coords
                #     feats = feats
                # else:
                #     idxs = np.random.choice(coords.shape[0], sub_sample_num, replace=False)
                #     coords = coords[idxs]
                #     feats = feats[idxs]

                ### add for primitive_bboxes
                primitive_w = np.max(coords[:, 0]) - np.min(coords[:, 0])
                primitive_h = np.max(coords[:, 1]) - np.min(coords[:, 1])
                primitive_bbox = np.array([primitive_w, primitive_h])
                data_dict["primitive_bboxes"] = np.concatenate(
                    [data_dict["primitive_bboxes"], np.array([primitive_bbox])], axis=0
                )

                ### 将相应primitive加入到相应结果中;
                data_dict["primitive_coords"].append(coords)
                data_dict["primitive_feats"].append(feats)
                data_dict["semantic_id"] = np.concatenate(
                    [data_dict["semantic_id"], np.array([semantic_id])], axis=0
                )
                data_dict["primitive_id"] = np.concatenate(
                    [data_dict["primitive_id"], np.array([primitive_id])], axis=0
                )
                data_dict["translation"] = np.concatenate(
                    [
                        data_dict["translation"],
                        np.array([update_translation / (self.scene_bound / 2)]),
                    ],
                    axis=0,
                )
                data_dict["rotation"] = np.concatenate(
                    [data_dict["rotation"], np.array([rotation])], axis=0
                )
                data_dict["scale"] = np.concatenate(
                    [data_dict["scale"], np.array([scale])], axis=0
                )
                data_dict["flip_mode"] = np.concatenate(
                    [data_dict["flip_mode"], np.array([flip])], axis=0
                )
                if len(data_dict["primitive_offsets"]) > 0:
                    start_idx = data_dict["primitive_offsets"][-1][1]
                    data_dict["primitive_offsets"].append(
                        [start_idx, start_idx + coords.shape[0]]
                    )
                else:
                    data_dict["primitive_offsets"].append([0, coords.shape[0]])

        return data_dict

    # def __call__(self, data_dict):
    #     num_primitives = len(data_dict['primitive_coords'])
    #     if np.random.rand() < self.p or num_primitives == 0:
    #         #### 随机选取一个场景
    #         group_file = np.random.choice(self.valid_group_lst)
    #         with open(group_file, 'r') as f:
    #             group_data = json.load(f)
    #         ### 随机对应场景的一个group进行拼接
    #         select_group_idx = np.random.choice(len(group_data))
    #         select_group = group_data[select_group_idx]

    #         select_group_bbox = select_group['group_bbox']
    #         select_group_center = np.array([(select_group_bbox[2] + select_group_bbox[0]) / 2, (select_group_bbox[3] + select_group_bbox[1]) / 2], dtype=np.float32)
    #         select_group_center += (self.scene_bound / 2)

    #         select_group_primitive_coords = select_group['primitive_coords']
    #         select_group_primitive_feats = select_group['primitive_feats']
    #         select_group_primitive_ids = select_group['primitive_ids']
    #         select_group_semantic_ids = select_group['semantic_ids']
    #         select_group_translations = select_group['translations']
    #         select_group_rotations = select_group['rotations']
    #         select_group_scales = select_group['scales']
    #         select_group_flips = select_group['flips']

    #         #### 随机放置
    #         # select_center = select_group_center + np.random.uniform(-0.1 * self.scene_bound, 0.1 * self.scene_bound, 2)
    #         select_center = np.random.uniform(0.2*self.scene_bound, 0.8*self.scene_bound, 2)

    #         relative_translation = select_center - select_group_center

    #         #### tgt translation = src_translation + relative_translation

    #         #### 随后把相应的group放进去
    #         for primitive_id, semantic_id, translation, rotation, scale, flip, primitive_coord, primitive_feat in zip(select_group_primitive_ids, select_group_semantic_ids, select_group_translations, select_group_rotations, select_group_scales, select_group_flips, select_group_primitive_coords, select_group_primitive_feats):
    #             if scale[0] > 2.0 or scale[0] < 0.5 or scale[1] > 2.0 or scale[1] < 0.5:
    #                 continue
    #             primitive_coord = np.array(primitive_coord)
    #             coords = np.zeros((primitive_coord.shape[0], 3), dtype=np.float32)
    #             coords[:, :2] = primitive_coord[:, :2]
    #             feats = np.array(primitive_feat)
    #             coords[:, :2] += relative_translation
    #             feats[:,0] = np.arctan(coords[:,1]/(coords[:,0] + 1e-8)) / math.pi
    #             update_translation = translation + relative_translation

    #             # #### 直接primitive_coords
    #             # coords = self.primitive_lib[primitive_id]['coords'].copy()
    #             # feats = self.primitive_lib[primitive_id]['feats'].copy()
    #             # center = (np.min(coords[:, :2], axis=0) + np.max(coords[:, :2], axis=0)) / 2
    #             # rotate_rad = np.radians(rotation)
    #             # cos_angle = np.cos(rotate_rad)
    #             # sin_angle = np.sin(rotate_rad)
    #             # rotation_matrix = np.array([
    #             #     [cos_angle, -sin_angle],
    #             #     [sin_angle, cos_angle]
    #             # ])
    #             # #### 考虑变换: scale, flip, rotation
    #             # xy_coords = coords[:, :2]  # shape: [M, 2]
    #             # centered_coords = xy_coords - center  # shape: [M, 2]
    #             # if self.scale_inv:
    #             #     scale = 1.0
    #             #     centered_coords = centered_coords * scale
    #             # else:
    #             #     centered_coords = centered_coords * scale
    #             # if flip:
    #             #     centered_coords[:, 0] = -centered_coords[:, 0]
    #             # rotated_coords = centered_coords @ rotation_matrix.T  # shape: [M, 2]
    #             # final_coords = rotated_coords + center  # shape: [M, 2]
    #             # coords[:, :2] = final_coords

    #             # #### 考虑变换translation
    #             # update_translation = translation + relative_translation

    #             # coords[:, :2] += update_translation
    #             # coords[:, 2] = 0.0
    #             # feats[:,0] = np.arctan(coords[:,1]/(coords[:,0] + 1e-8)) / math.pi

    #             # sub_sample_num = int(coords.shape[0] * self.keep_ratio)
    #             # if sub_sample_num <= 1:
    #             #     coords = coords
    #             #     feats = feats
    #             # else:
    #             #     idxs = np.random.choice(coords.shape[0], sub_sample_num, replace=False)
    #             #     coords = coords[idxs]
    #             #     feats = feats[idxs]

    #             ## add for primitive_bboxes
    #             primitive_w = np.max(coords[:, 0]) - np.min(coords[:, 0])
    #             primitive_h = np.max(coords[:, 1]) - np.min(coords[:, 1])
    #             primitive_bbox = np.array([primitive_w, primitive_h])
    #             data_dict['primitive_bboxes'] = np.concatenate([data_dict['primitive_bboxes'], np.array([primitive_bbox])], axis=0)

    #             ### 将相应primitive加入到相应结果中;
    #             data_dict['primitive_coords'].append(coords)
    #             data_dict['primitive_feats'].append(feats)
    #             data_dict['semantic_id'] = np.concatenate([data_dict['semantic_id'], np.array([semantic_id])], axis=0)
    #             data_dict['primitive_id'] = np.concatenate([data_dict['primitive_id'], np.array([primitive_id])], axis=0)
    #             data_dict['translation'] = np.concatenate([data_dict['translation'], np.array([update_translation / (self.scene_bound / 2)])], axis=0)
    #             data_dict['rotation'] = np.concatenate([data_dict['rotation'], np.array([rotation])], axis=0)
    #             data_dict['scale'] = np.concatenate([data_dict['scale'], np.array([scale])], axis=0)
    #             data_dict['flip_mode'] = np.concatenate([data_dict['flip_mode'], np.array([flip])], axis=0)
    #             if len(data_dict['primitive_offsets']) > 0:
    #                 start_idx = data_dict['primitive_offsets'][-1][1]
    #                 data_dict['primitive_offsets'].append([start_idx, start_idx + coords.shape[0]])
    #             else:
    #                 data_dict['primitive_offsets'].append([0, coords.shape[0]])
    #             # if primitive_id in flip_invariant_primitive_id:
    #             #     data_dict['flip_invariant_primitive'] = np.concatenate([data_dict['flip_invariant_primitive'], np.array([True])], axis=0)
    #             # else:
    #             #     data_dict['flip_invariant_primitive'] = np.concatenate([data_dict['flip_invariant_primitive'], np.array([False])], axis=0)
    #     return data_dict


@TRANSFORMS.register_module()
class CopyPaste(object):
    """
    primitive_ids: 需要添加的primitive_id (Tail_classes_id)
    primitive_min_num: 需要添加的primitive_id的最小数量
    primitive_max_num: 需要添加的primitive_id的最大数量
    range_translation: 添加primitive的translation范围
    range_rotation: 添加primitive的rotation范围
    range_scale: 添加primitive的scale范围
    """

    def __init__(
        self,
        p=0.5,
        primitive_num_mean=8,
        range_translation=(-0.8, 0.8),
        range_rotation=(0, 360),
        normalize_coord=False,
        scene_bound=20000.0,
    ):
        self.p = p
        self.primitive_num_mean = primitive_num_mean
        self.range_translation = range_translation
        self.range_rotation = range_rotation
        self.normalize_coord = normalize_coord
        self.scene_bound = scene_bound

    def import_primitive_lib(
        self, primitive_lib, primitive_ids, primitive_weights, primitive_to_semantic
    ):
        self.primitive_lib = primitive_lib
        self.primitive_ids = primitive_ids
        self.primitive_weights = primitive_weights
        self.primitive_to_semantic = primitive_to_semantic

    def __call__(self, data_dict):
        #### 如果对应数据没有instance，则强制进行copypaste
        num_primitives = len(data_dict["primitive_coords"])

        if np.random.rand() < self.p or num_primitives == 0:
            #### 随机插入若干个primitive_id对应的primitive到相应的data_dict中；
            #### 首先确定选择primitive的数量，以及对应的primitive_id
            primitive_num = np.random.poisson(self.primitive_num_mean)
            primitive_num = max(1, primitive_num)

            #### 这一步挑选的primitive_ids需要考究
            # primitive_ids = np.random.choice(self.primitive_ids, size=primitive_num, p=self.primitive_weights, replace=True)
            #### Modify for all classes, not the tail classes
            primitive_ids = np.random.choice(
                self.primitive_ids, size=primitive_num, replace=True
            )

            #### 更新相应data_dict的primitive_coords, primitive_feats, (直接append就行) translation, rotation, scale
            for idx, primitive_id in enumerate(primitive_ids):
                coords = self.primitive_lib[primitive_id]["coords"].copy()
                feats = self.primitive_lib[primitive_id]["feats"].copy()

                ### 对相应primitive进行变换
                ### 对相应primitive进行旋转
                # center_x = np.mean(coords[:, 0])
                # center_y = np.mean(coords[:, 1])
                # center = np.array([center_x, center_y])
                #### assert the center is [0, 0]
                # center = np.array([0.0, 0.0])
                center = (
                    np.min(coords[:, :2], axis=0) + np.max(coords[:, :2], axis=0)
                ) / 2

                # 随机生成旋转角度
                angle = np.random.choice([0, 90, 180, 270])
                # angle = np.random.uniform(self.range_rotation[0], self.range_rotation[1])
                angle_rad = np.radians(angle)

                # 创建2D旋转矩阵
                cos_angle = np.cos(angle_rad)
                sin_angle = np.sin(angle_rad)
                rotation_matrix = np.array(
                    [[cos_angle, -sin_angle], [sin_angle, cos_angle]]
                )

                # 提取x, y坐标
                xy_coords = coords[:, :2]  # shape: [M, 2]
                # 平移到以中心为原点
                centered_coords = xy_coords - center  # shape: [M, 2]
                # 应用旋转
                rotated_coords = centered_coords @ rotation_matrix.T  # shape: [M, 2]
                # 平移回原来的位置
                final_coords = rotated_coords + center  # shape: [M, 2]
                coords[:, :2] = final_coords

                #### 相应primitive的coords中心应当是0.5
                delta_x = np.random.uniform(
                    self.range_translation[0], self.range_translation[1]
                )
                delta_y = np.random.uniform(
                    self.range_translation[0], self.range_translation[1]
                )
                delta = np.array([delta_x, delta_y])

                # if np.any(coords[:, :2] + delta / 2 < 0.0) or np.any(coords[:, :2] + delta / 2 > 1.0):
                #     # 如果超出范围，跳过这个primitive
                #     continue
                if self.normalize_coord:
                    coords[:, :2] += delta
                    coords[:, 2] = 0.0
                    feats[:, 0] = (
                        np.arctan(coords[:, 1] / (coords[:, 0] + 1e-8)) / math.pi
                    )
                else:
                    coords[:, :2] += delta / 140.0
                    coords[:, 2] = 0.0
                    feats[:, 0] = (
                        np.arctan(coords[:, 1] / (coords[:, 0] + 1e-8)) / math.pi
                    )

                sub_sample_num = int(coords.shape[0] * self.keep_ratio)
                if sub_sample_num < self.min_keep:
                    coords = coords
                    feats = feats
                else:
                    idxs = np.random.choice(
                        coords.shape[0], sub_sample_num, replace=False
                    )
                    coords = coords[idxs]
                    feats = feats[idxs]

                ### 将相应primitive加入到相应结果中;
                data_dict["primitive_coords"].append(coords)
                data_dict["primitive_feats"].append(feats)
                data_dict["semantic_id"] = np.concatenate(
                    [
                        data_dict["semantic_id"],
                        np.array([self.primitive_to_semantic[str(primitive_id)]]),
                    ],
                    axis=0,
                )
                data_dict["primitive_id"] = np.concatenate(
                    [data_dict["primitive_id"], np.array([primitive_id])], axis=0
                )
                data_dict["translation"] = np.concatenate(
                    [
                        data_dict["translation"],
                        np.array([delta / (self.scene_bound / 2)]),
                    ],
                    axis=0,
                )
                # data_dict['rotation'] = np.concatenate([data_dict['rotation'], np.array([angle])], axis=0)
                data_dict["rotation"] = np.concatenate(
                    [data_dict["rotation"], np.array([(360 - angle) % 360])], axis=0
                )
                data_dict["scale"] = np.concatenate(
                    [data_dict["scale"], np.array([1.0])], axis=0
                )
                data_dict["flip_mode"] = np.concatenate(
                    [data_dict["flip_mode"], np.array([0])], axis=0
                )
                # if primitive_id in flip_invariant_primitive_id:
                #     data_dict['flip_invariant_primitive'] = np.concatenate([data_dict['flip_invariant_primitive'], np.array([True])], axis=0)
                # else:
                #     data_dict['flip_invariant_primitive'] = np.concatenate([data_dict['flip_invariant_primitive'], np.array([False])], axis=0)
                if len(data_dict["primitive_offsets"]) > 0:
                    start_idx = data_dict["primitive_offsets"][-1][1]
                    data_dict["primitive_offsets"].append(
                        [start_idx, start_idx + coords.shape[0]]
                    )
                else:
                    data_dict["primitive_offsets"].append([0, coords.shape[0]])

        return data_dict


class Compose(object):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.transforms = []
        for t_cfg in self.cfg:
            self.transforms.append(TRANSFORMS.build(t_cfg))

    def import_primitive_lib(
        self, primitive_lib, primitive_ids, primitive_weights, primitive_to_semantic
    ):
        for t in self.transforms:
            if isinstance(t, CopyPaste):
                t.import_primitive_lib(
                    primitive_lib,
                    primitive_ids,
                    primitive_weights,
                    primitive_to_semantic,
                )
            elif isinstance(t, CutMix):
                t.import_primitive_lib(
                    primitive_lib,
                    primitive_ids,
                    primitive_weights,
                    primitive_to_semantic,
                )

    def __call__(self, data_dict):
        for t in self.transforms:
            data_dict = t(data_dict)
        return data_dict


# ==================== Instance-level 数据增强类 ==================== #


@TRANSFORMS.register_module()
class PrimitiveDropoutInstance(object):
    """
    适配 InstanceData 的 Primitive Dropout 增强

    随机丢弃一部分 instance，模拟场景中元素缺失的情况

    Args:
        p: 执行增强的概率
        min_ratio: 保留的最小比例
        max_ratio: 保留的最大比例
    """

    def __init__(self, p=0.5, min_ratio=0.1, max_ratio=0.9):
        self.p = p
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def __call__(self, instance_data: InstanceData) -> InstanceData:
        """
        对 InstanceData 执行随机丢弃增强

        Args:
            instance_data: 输入的 InstanceData

        Returns:
            丢弃部分 instance 后的 InstanceData
        """
        if np.random.rand() >= self.p:
            return instance_data

        num_instances = instance_data.num_instances()

        if num_instances <= 1:
            # 如果只有一个或没有 instance，不进行丢弃
            return instance_data

        # 计算要保留的 instance 数量
        keep_ratio = np.random.uniform(self.min_ratio, self.max_ratio)
        keep_num = max(1, int(keep_ratio * num_instances))  # 至少保留一个

        # 随机选择要保留的 instance 索引
        keep_indices = np.random.choice(num_instances, keep_num, replace=False)
        keep_indices = np.sort(keep_indices)  # 保持顺序

        # 更新 Per-instance 的点级数据 (List)
        instance_data.instance_coords = [
            instance_data.instance_coords[i] for i in keep_indices
        ]
        instance_data.instance_colors = [
            instance_data.instance_colors[i] for i in keep_indices
        ]
        instance_data.instance_widths = [
            instance_data.instance_widths[i] for i in keep_indices
        ]
        instance_data.instance_prim_ids = [
            instance_data.instance_prim_ids[i] for i in keep_indices
        ]
        instance_data.instance_layer_ids = [
            instance_data.instance_layer_ids[i] for i in keep_indices
        ]

        # 更新 Per-instance 的 primitive-level 数据 (List)
        instance_data.instance_semantic_ids = [
            instance_data.instance_semantic_ids[i] for i in keep_indices
        ]
        instance_data.instance_instance_ids = [
            instance_data.instance_instance_ids[i] for i in keep_indices
        ]
        instance_data.instance_primitive_lengths = [
            instance_data.instance_primitive_lengths[i] for i in keep_indices
        ]

        # 更新 GT 的 instance-level 属性 (numpy arrays)
        if instance_data.primitive_ids is not None:
            instance_data.primitive_ids = instance_data.primitive_ids[keep_indices]
        if instance_data.semantic_ids is not None:
            instance_data.semantic_ids = instance_data.semantic_ids[keep_indices]
        if instance_data.translations is not None:
            instance_data.translations = instance_data.translations[keep_indices]
        if instance_data.rotations is not None:
            instance_data.rotations = instance_data.rotations[keep_indices]
        if instance_data.scales is not None:
            instance_data.scales = instance_data.scales[keep_indices]
        if instance_data.flip_modes is not None:
            instance_data.flip_modes = instance_data.flip_modes[keep_indices]
        if instance_data.flip_invariant_primitives is not None:
            instance_data.flip_invariant_primitives = (
                instance_data.flip_invariant_primitives[keep_indices]
            )
        if instance_data.sizes is not None:
            instance_data.sizes = instance_data.sizes[keep_indices]
        return instance_data


@TRANSFORMS.register_module()
class CopyPasteInstance(object):
    """
    适配 InstanceData 的 CopyPaste 增强

    从 primitive_lib 中随机选择 primitive 并粘贴到场景中

    Args:
        p: 执行增强的概率
        primitive_num_mean: 粘贴 primitive 数量的泊松分布均值
        range_translation: 平移范围 (min, max)
        range_rotation: 旋转范围（角度列表或范围）
        scene_bound: 场景边界，用于归一化
    """

    def __init__(
        self,
        p=0.5,
        primitive_num_mean=8,
        range_translation=(-0.8, 0.8),
        range_rotation=(0, 360),
        scene_bound=30000.0,
    ):
        self.p = p
        self.primitive_num_mean = primitive_num_mean
        self.range_translation = range_translation
        self.range_rotation = range_rotation
        self.scene_bound = scene_bound

        self.primitive_lib = None
        self.primitive_ids = None
        self.primitive_weights = None
        self.primitive_to_semantic = None

    def import_primitive_lib(
        self, primitive_lib, primitive_ids, primitive_weights, primitive_to_semantic
    ):
        """导入 primitive 库"""
        self.primitive_lib = primitive_lib
        self.primitive_ids = primitive_ids
        self.primitive_weights = primitive_weights
        self.primitive_to_semantic = primitive_to_semantic

    def _apply_transform(
        self,
        coords,
        angle,
        translation,
        *,
        return_origin_translation=False,
    ):
        """
        对 primitive 坐标应用旋转和平移变换

        Args:
            coords: (N, 4) 线段坐标 [x1, y1, x2, y2]
            angle: 旋转角度（度）
            translation: (2,) 平移量

        Args:
            return_origin_translation: 同时返回与模板原点参数化严格等价的
                translation。CopyPaste 为了保持物体绕自身 bbox 中心旋转，source
                使用 ``R(p-c)+c+delta``；模型/recon 的统一参数化是
                ``R p+t``，因此对应 ``t=delta+c-Rc``。

        Returns:
            默认只返回变换后的坐标；开启 ``return_origin_translation`` 时返回
            ``(coords, origin_translation)``。
        """
        # 计算中心
        center = (np.min(coords[:, :2], axis=0) + np.max(coords[:, :2], axis=0)) / 2

        # 旋转变换
        angle_rad = np.radians(angle)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])

        # Preserve the existing source geometry (rotation about its bbox
        # centre), but expose the exactly equivalent translation for the
        # origin-based transform used by GT attributes and reconstruction.
        # Coordinates are row vectors below, hence ``center @ R.T``.
        origin_translation = (
            np.asarray(translation, dtype=np.float64)
            + center
            - center @ rotation_matrix.T
        )

        # 改进2：将 (N, 4) reshape 成 (2N, 2)，一次矩阵乘法同时处理两端点
        all_xy = coords.reshape(-1, 2) - center  # (2N, 2)
        all_xy_rotated = all_xy @ rotation_matrix.T  # (2N, 2)
        coords[:] = (all_xy_rotated + center).reshape(-1, 4)

        # 平移变换
        coords[:, [0, 2]] += translation[0]
        coords[:, [1, 3]] += translation[1]

        if return_origin_translation:
            return coords, origin_translation.astype(np.float32)
        return coords

    def __call__(self, instance_data: InstanceData) -> InstanceData:
        """
        对 InstanceData 执行 CopyPaste 增强

        注意: primitive_lib 中每个 primitive 的数据结构:
        - Point-level: raw_coords, colors, widths, prim_ids, layer_ids
        - 对于单个 primitive，所有点都属于同一个 primitive
        - instance_semantic_ids 需要存储 primitive-level 的 semantic_id

        Args:
            instance_data: 输入的 InstanceData

        Returns:
            增强后的 InstanceData
        """
        if self.primitive_lib is None:
            return instance_data

        num_instances = instance_data.num_instances()

        # 如果没有 instance 或随机触发增强
        if np.random.rand() < self.p or num_instances == 0:
            primitive_num = max(1, np.random.poisson(self.primitive_num_mean))
            selected_ids = np.random.choice(
                self.primitive_ids, size=primitive_num, replace=True
            )

            # 计算当前最大的 layer_id，为本次新增的所有 instance 分配一个新的共享 layer_id
            max_layer_id = 0
            for layer_arr in instance_data.instance_layer_ids:
                if len(layer_arr) > 0:
                    max_layer_id = max(max_layer_id, np.max(layer_arr))
            if (
                instance_data.bg_layer_ids is not None
                and len(instance_data.bg_layer_ids) > 0
            ):
                max_layer_id = max(max_layer_id, np.max(instance_data.bg_layer_ids))
            new_shared_layer_id = max_layer_id + 1

            # 改进3：循环前记录初始 instance 索引，循环内手动递增，避免反复调用 num_instances()
            current_instance_idx = instance_data.num_instances()

            # 改进1：临时收集 instance-level 标量数组，循环结束后统一做一次 concatenate
            new_primitive_ids = []
            new_semantic_ids = []
            new_translations = []
            new_rotations = []
            new_scales = []
            new_flip_modes = []
            new_flip_invariants = []
            new_sizes = []

            for primitive_id in selected_ids:
                if primitive_id not in self.primitive_lib:
                    continue

                prim_data = self.primitive_lib[primitive_id]

                # 复制 primitive 数据 (Point-level)
                raw_coords = prim_data["raw_coords"].copy()  # (N, 4)
                colors = prim_data["colors"].copy()  # (N, 3)
                widths = prim_data["widths"].copy()  # (N,)
                # 使用本次新增 instance 共享的新 layer_id
                n_points = len(raw_coords)
                layer_ids = np.full(n_points, new_shared_layer_id, dtype=np.int32)
                # prim_ids = np.zeros(n_points, dtype=np.int32)
                prim_ids = prim_data["prim_ids"].copy()

                # 随机生成变换参数
                angle = np.random.choice([0, 90, 180, 270])
                delta = np.random.uniform(
                    self.range_translation[0], self.range_translation[1], 2
                ).astype(np.float32)

                # 应用变换
                raw_coords, gt_translation = self._apply_transform(
                    raw_coords,
                    angle,
                    delta,
                    return_origin_translation=True,
                )
                raw_coords += self.scene_bound / 2
                # Append Point-level 数据到 instance_data（Python list.append，O(1)，无需优化）
                instance_data.instance_coords.append(raw_coords)
                instance_data.instance_colors.append(colors)
                instance_data.instance_widths.append(widths)
                instance_data.instance_prim_ids.append(prim_ids)
                instance_data.instance_layer_ids.append(layer_ids)

                # Append Primitive-level 数据 (直接使用 primitive_lib 中保存的数据)
                # 注意：primitive_lib 中每个 key 本质上是一个 instance 级别的元素
                prim_sem_ids = prim_data[
                    "sem_ids"
                ].copy()  # 该 instance 内各子 primitive 的 semantic_id
                prim_inst_ids = prim_data[
                    "inst_ids"
                ].copy()  # 该 instance 内各子 primitive 的 instance_id
                prim_lengths = prim_data[
                    "primitive_lengths"
                ].copy()  # 该 instance 内各子 primitive 的长度

                instance_data.instance_semantic_ids.append(
                    prim_sem_ids.astype(np.int32)
                )
                # 改进3：直接使用局部计数器，避免重复调用 num_instances()
                instance_data.instance_instance_ids.append(
                    np.full(len(prim_sem_ids), current_instance_idx, dtype=np.int32)
                )
                current_instance_idx += 1
                instance_data.instance_primitive_lengths.append(
                    prim_lengths.astype(np.float32)
                )

                # 改进1：收集 GT instance-level 属性到临时列表，延迟 concatenate
                semantic_id = self.primitive_to_semantic.get(str(primitive_id), 0)
                new_primitive_ids.append(primitive_id)
                new_semantic_ids.append(semantic_id)
                new_translations.append(gt_translation)
                new_rotations.append(float(angle))
                new_scales.append([1.0, 1.0])
                new_flip_modes.append(0)
                new_flip_invariants.append(primitive_id in flip_invariant_primitive_id)

                if instance_data.sizes is not None:
                    all_x = raw_coords[:, [0, 2]]
                    all_y = raw_coords[:, [1, 3]]
                    new_sizes.append(
                        [all_x.max() - all_x.min(), all_y.max() - all_y.min()]
                    )

            # 改进1：循环结束后统一做一次 concatenate，O(k) 而非 O(k²)
            if new_primitive_ids:
                instance_data.primitive_ids = np.concatenate(
                    [
                        instance_data.primitive_ids,
                        np.array(new_primitive_ids, dtype=np.int32),
                    ]
                )
                instance_data.semantic_ids = np.concatenate(
                    [
                        instance_data.semantic_ids,
                        np.array(new_semantic_ids, dtype=np.int32),
                    ]
                )
                instance_data.translations = np.concatenate(
                    [
                        instance_data.translations,
                        np.array(new_translations, dtype=np.float32),
                    ],
                    axis=0,
                )
                instance_data.rotations = np.concatenate(
                    [instance_data.rotations, np.array(new_rotations, dtype=np.float32)]
                )
                instance_data.scales = np.concatenate(
                    [instance_data.scales, np.array(new_scales, dtype=np.float32)],
                    axis=0,
                )
                instance_data.flip_modes = np.concatenate(
                    [instance_data.flip_modes, np.array(new_flip_modes, dtype=np.int32)]
                )
                instance_data.flip_invariant_primitives = np.concatenate(
                    [
                        instance_data.flip_invariant_primitives,
                        np.array(new_flip_invariants, dtype=np.bool_),
                    ]
                )
                if instance_data.sizes is not None and new_sizes:
                    instance_data.sizes = np.concatenate(
                        [instance_data.sizes, np.array(new_sizes, dtype=np.float32)],
                        axis=0,
                    )

        return instance_data


@TRANSFORMS.register_module()
class SourceInstanceCutMix(object):
    """Read-only real-source donor bank used by the vectorized dataset path."""

    def __init__(
        self,
        p=0.2,
        donor_bank=None,
        sampling_mode="random_identity",
        placement_mode="uniform_valid",
        primitive_num_mean=4,
        max_added_instances=8,
        max_postgrid_segments=7000,
        require_full_inside=True,
        min_bbox_gap=0.0,
        max_bbox_iou=0.02,
        target_profile=None,
        seed_offset=0,
        scene_bound=30000.0,
    ):
        self.p = float(p)
        self.donor_bank = donor_bank
        self.sampling_mode = str(sampling_mode)
        self.placement_mode = str(placement_mode)
        self.primitive_num_mean = float(primitive_num_mean)
        self.max_added_instances = int(max_added_instances)
        self.max_postgrid_segments = int(max_postgrid_segments)
        self.require_full_inside = bool(require_full_inside)
        self.min_bbox_gap = float(min_bbox_gap)
        self.max_bbox_iou = float(max_bbox_iou)
        self.target_profile = target_profile
        self.seed_offset = int(seed_offset)
        self.scene_bound = float(scene_bound)
        if self.donor_bank is None:
            raise ValueError("SourceInstanceCutMix requires donor_bank")
        if self.sampling_mode not in {
            "random_identity",
            "random_donor",
            "composition_deficit",
            "target_anchor_scene_bundle",
        }:
            raise ValueError(
                f"unsupported SourceInstanceCutMix sampling_mode={self.sampling_mode!r}"
            )
        if self.placement_mode not in {"uniform_valid", "hard_density"}:
            raise ValueError(
                f"unsupported SourceInstanceCutMix placement_mode={self.placement_mode!r}"
            )
        if self.max_added_instances <= 0:
            raise ValueError("max_added_instances must be positive")
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        bank = os.path.abspath(os.path.expanduser(self.donor_bank))
        metadata_path = os.path.join(bank, "metadata.json")
        index_path = os.path.join(bank, "donor_index.npz")
        if not os.path.isfile(metadata_path) or not os.path.isfile(index_path):
            raise FileNotFoundError(f"incomplete source-instance donor bank: {bank!r}")
        with open(metadata_path, "r", encoding="utf-8") as handle:
            self.metadata = json.load(handle)
        self.profile = None
        if self.target_profile is not None:
            profile_path = os.path.abspath(os.path.expanduser(self.target_profile))
            with open(profile_path, "r", encoding="utf-8") as handle:
                self.profile = json.load(handle)
        index = np.load(index_path, allow_pickle=False)
        self.index = {key: index[key] for key in index.files}
        self.coords = np.load(os.path.join(bank, "coords.npy"), mmap_mode="r")
        self.colors = np.load(os.path.join(bank, "colors.npy"), mmap_mode="r")
        self.widths = np.load(os.path.join(bank, "widths.npy"), mmap_mode="r")
        self.point_primitive_ids = np.load(
            os.path.join(bank, "point_primitive_ids.npy"), mmap_mode="r"
        )
        self.layer_ids = np.load(os.path.join(bank, "layer_ids.npy"), mmap_mode="r")
        self.primitive_semantic_ids = np.load(
            os.path.join(bank, "primitive_semantic_ids.npy"), mmap_mode="r"
        )
        self.primitive_lengths = np.load(
            os.path.join(bank, "primitive_lengths.npy"), mmap_mode="r"
        )
        required = {
            "line_offsets",
            "primitive_offsets",
            "semantic_id",
            "primitive_id",
            "rotation",
            "scale",
            "flip",
            "size",
            "local_bbox",
            "estimated_postgrid_lines",
        }
        missing = required.difference(self.index)
        if missing:
            raise ValueError(f"donor_index.npz missing keys: {sorted(missing)}")
        self.num_donors = len(self.index["semantic_id"])
        if self.num_donors == 0:
            raise ValueError("source-instance donor bank is empty")
        identities = np.stack(
            [self.index["semantic_id"], self.index["primitive_id"]], axis=1
        )
        self.identity_to_donors = {}
        for identity in np.unique(identities, axis=0):
            key = (int(identity[0]), int(identity[1]))
            self.identity_to_donors[key] = np.flatnonzero(
                np.all(identities == identity[None], axis=1)
            )
        self.identity_keys = sorted(self.identity_to_donors)
        self.scene_to_donors = {}
        self.semantic_to_bundle_scenes = {}
        if self.sampling_mode == "target_anchor_scene_bundle":
            provenance = self.metadata.get("provenance")
            if not isinstance(provenance, list) or len(provenance) != self.num_donors:
                raise ValueError(
                    "target_anchor_scene_bundle requires one provenance record "
                    "per donor in metadata.json"
                )
            for donor_index, record in enumerate(provenance):
                scene = str(record.get("scene", ""))
                if not scene:
                    raise ValueError(
                        "target_anchor_scene_bundle found donor provenance without scene"
                    )
                self.scene_to_donors.setdefault(scene, []).append(donor_index)
            # A valid bundle scene must contain an anchor plus at least one
            # companion.  Match anchors by semantic (not exact primitive) so
            # the prior is about scene composition rather than template ID.
            semantic_to_scenes = {}
            for scene, donor_indices in self.scene_to_donors.items():
                if len(donor_indices) < 2:
                    continue
                scene_semantics = np.unique(
                    self.index["semantic_id"][np.asarray(donor_indices, dtype=np.int64)]
                )
                for semantic_id in scene_semantics.tolist():
                    semantic_to_scenes.setdefault(int(semantic_id), []).append(scene)
            self.semantic_to_bundle_scenes = semantic_to_scenes
        self._loaded = True

    def sample_donor(self, present_identities=None):
        self._ensure_loaded()
        if self.sampling_mode == "random_donor":
            return int(np.random.randint(self.num_donors))
        if (
            self.sampling_mode == "composition_deficit"
            and present_identities is not None
        ):
            present = {tuple(map(int, identity)) for identity in present_identities}
            absent = [
                identity for identity in self.identity_keys if identity not in present
            ]
            pool = absent if absent else self.identity_keys
        else:
            pool = self.identity_keys
        identity = pool[int(np.random.randint(len(pool)))]
        donors = self.identity_to_donors[identity]
        return int(donors[int(np.random.randint(len(donors)))])

    def sample_target_anchor_scene_bundle(self, present_semantics, requested):
        """Sample companions from one Train scene conditioned on a target semantic.

        The selected source anchor is excluded because the target scene already
        supplies its counterpart.  Returned companions therefore have genuine
        donor-to-donor co-occurrence and each co-occurred with the anchor
        semantic in the same source crop.
        """
        self._ensure_loaded()
        if self.sampling_mode != "target_anchor_scene_bundle":
            raise RuntimeError(
                "sample_target_anchor_scene_bundle requires "
                "sampling_mode=target_anchor_scene_bundle"
            )
        requested = max(1, int(requested))
        eligible_semantics = sorted(
            {
                int(semantic_id)
                for semantic_id in present_semantics
                if int(semantic_id) in self.semantic_to_bundle_scenes
            }
        )
        if not eligible_semantics:
            return None
        anchor_semantic = eligible_semantics[
            int(np.random.randint(len(eligible_semantics)))
        ]
        source_scenes = self.semantic_to_bundle_scenes[anchor_semantic]
        source_scene = source_scenes[int(np.random.randint(len(source_scenes)))]
        scene_donors = np.asarray(self.scene_to_donors[source_scene], dtype=np.int64)
        anchor_candidates = scene_donors[
            self.index["semantic_id"][scene_donors] == anchor_semantic
        ]
        source_anchor = int(
            anchor_candidates[int(np.random.randint(len(anchor_candidates)))]
        )
        companions = scene_donors[scene_donors != source_anchor]
        if len(companions) == 0:
            return None
        # Provide same-scene fallbacks when a large companion cannot satisfy
        # the target scene's bbox/grid constraints.  The dataset still accepts
        # at most ``requested`` instances; this only broadens the candidate
        # queue without breaking bundle provenance.
        take = min(max(requested, requested * 4), len(companions))
        selected = np.random.choice(companions, size=take, replace=False)
        return {
            "anchor_semantic": int(anchor_semantic),
            "source_scene": source_scene,
            "source_anchor": source_anchor,
            "donor_indices": [int(index) for index in selected.tolist()],
        }

    def donor(self, donor_index):
        self._ensure_loaded()
        donor_index = int(donor_index)
        line_start, line_end = self.index["line_offsets"][donor_index]
        primitive_start, primitive_end = self.index["primitive_offsets"][donor_index]
        return {
            "coords_local": np.asarray(self.coords[line_start:line_end]).copy(),
            "colors": np.asarray(self.colors[line_start:line_end]).copy(),
            "widths": np.asarray(self.widths[line_start:line_end]).copy(),
            "point_primitive_ids": np.asarray(
                self.point_primitive_ids[line_start:line_end]
            ).copy(),
            "layer_ids": np.asarray(self.layer_ids[line_start:line_end]).copy(),
            "primitive_semantic_ids": np.asarray(
                self.primitive_semantic_ids[primitive_start:primitive_end]
            ).copy(),
            "primitive_lengths": np.asarray(
                self.primitive_lengths[primitive_start:primitive_end]
            ).copy(),
            "semantic_id": int(self.index["semantic_id"][donor_index]),
            "primitive_id": int(self.index["primitive_id"][donor_index]),
            "rotation": float(self.index["rotation"][donor_index]),
            "scale": np.asarray(self.index["scale"][donor_index]).copy(),
            "flip": int(self.index["flip"][donor_index]),
            "size": np.asarray(self.index["size"][donor_index]).copy(),
            "local_bbox": np.asarray(self.index["local_bbox"][donor_index]).copy(),
            "estimated_postgrid_lines": int(
                self.index["estimated_postgrid_lines"][donor_index]
            ),
        }


@TRANSFORMS.register_module()
class CutMixInstance(object):
    """
    适配 InstanceData 的 CutMix 增强

    从预定义的 group 文件中随机选择一组 primitives 并粘贴到场景中

    Args:
        p: 执行增强的概率
        group_dir: group 文件目录
        scene_bound: 场景边界
        scale_inv: 是否忽略缩放
    """

    def __init__(self, p=0.5, group_dir=None, scene_bound=30000.0, scale_inv=False):
        self.p = p
        self.group_dir = group_dir
        self.scene_bound = scene_bound
        self.scale_inv = scale_inv

        if self.group_dir is not None and os.path.exists(self.group_dir):
            self.valid_group_lst = [
                os.path.join(self.group_dir, x)
                for x in os.listdir(self.group_dir)
                if x.endswith(".json")
            ]
        else:
            self.valid_group_lst = []

        self.primitive_lib = None
        self.primitive_ids = None
        self.primitive_weights = None
        self.primitive_to_semantic = None

    def import_primitive_lib(
        self, primitive_lib, primitive_ids, primitive_weights, primitive_to_semantic
    ):
        """导入 primitive 库"""
        self.primitive_lib = primitive_lib
        self.primitive_ids = primitive_ids
        self.primitive_weights = primitive_weights
        self.primitive_to_semantic = primitive_to_semantic

    def _apply_transform(self, coords, rotation, scale, flip, translation):
        """
        对 primitive 坐标应用旋转、缩放、翻转和平移变换
        """
        # 计算中心
        center = (np.min(coords[:, :2], axis=0) + np.max(coords[:, :2], axis=0)) / 2

        # 旋转变换
        angle_rad = np.radians(rotation)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])

        # 对坐标进行变换
        for i in [0, 2]:  # x1,y1 和 x2,y2
            xy = coords[:, i : i + 2] - center
            # 缩放
            if not self.scale_inv:
                xy = xy * scale
            # 翻转
            if flip:
                xy[:, 0] = -xy[:, 0]
            # 旋转
            xy = xy @ rotation_matrix.T
            coords[:, i : i + 2] = xy + center

        # 平移
        coords[:, 0] += translation[0]
        coords[:, 1] += translation[1]
        coords[:, 2] += translation[0]
        coords[:, 3] += translation[1]

        return coords

    def __call__(self, instance_data: InstanceData) -> InstanceData:
        """
        对 InstanceData 执行 CutMix 增强

        注意: 正确处理两层数据结构:
        - Point-level: coords, colors, widths, prim_ids, layer_ids
        - Primitive-level: semantic_ids (每个 instance 只有一个 primitive)
        """
        if self.primitive_lib is None or len(self.valid_group_lst) == 0:
            return instance_data

        num_instances = instance_data.num_instances()

        if np.random.rand() < self.p or num_instances == 0:
            # 随机选择一个 group 文件
            group_file = np.random.choice(self.valid_group_lst)
            with open(group_file, "r") as f:
                group_data = json.load(f)

            if len(group_data) == 0:
                return instance_data

            # 随机选择一个 group
            select_group_idx = np.random.choice(len(group_data))
            select_group = group_data[select_group_idx]

            # 计算 group 中心
            select_group_bbox = select_group["group_bbox"]
            select_group_center = np.array(
                [
                    (select_group_bbox[2] + select_group_bbox[0]) / 2,
                    (select_group_bbox[3] + select_group_bbox[1]) / 2,
                ],
                dtype=np.float32,
            )

            # select_group_center += (self.scene_bound / 2)

            # 获取 group 中的 primitive 信息
            select_group_primitive_ids = select_group["primitive_ids"]
            select_group_semantic_ids = select_group["semantic_ids"]
            select_group_translations = select_group["translations"]
            select_group_rotations = select_group["rotations"]
            select_group_scales = select_group["scales"]
            select_group_flips = select_group["flips"]

            # 随机放置位置
            select_center = np.random.uniform(
                0.4 * self.scene_bound, 0.6 * self.scene_bound, 2
            )
            relative_translation = select_center - select_group_center

            # 计算当前最大的 layer_id，为本次新增的所有 instance 分配一个新的共享 layer_id
            max_layer_id = 0
            for layer_arr in instance_data.instance_layer_ids:
                if len(layer_arr) > 0:
                    max_layer_id = max(max_layer_id, np.max(layer_arr))
            if (
                instance_data.bg_layer_ids is not None
                and len(instance_data.bg_layer_ids) > 0
            ):
                max_layer_id = max(max_layer_id, np.max(instance_data.bg_layer_ids))
            new_shared_layer_id = max_layer_id + 1

            # 遍历 group 中的每个 primitive
            for primitive_id, semantic_id, translation, rotation, scale, flip in zip(
                select_group_primitive_ids,
                select_group_semantic_ids,
                select_group_translations,
                select_group_rotations,
                select_group_scales,
                select_group_flips,
            ):
                if primitive_id not in self.primitive_lib:
                    continue

                # # 限制缩放范围
                # scale = np.clip(scale, 0.5, 2.0)

                prim_data = self.primitive_lib[primitive_id]

                # 复制 primitive 数据 (Point-level)
                raw_coords = prim_data["raw_coords"].copy()
                colors = prim_data["colors"].copy()
                widths = prim_data["widths"].copy()
                # 使用本次新增 instance 共享的新 layer_id
                n_points = len(raw_coords)
                layer_ids = np.full(n_points, new_shared_layer_id, dtype=np.int32)

                # 对于单个 primitive，所有点的 prim_ids 都应该是 0
                n_points = len(raw_coords)
                prim_ids = np.zeros(n_points, dtype=np.int32)

                # 计算更新后的平移
                update_translation = (
                    np.array(translation, dtype=np.float32) + relative_translation
                )

                # 应用变换
                raw_coords = self._apply_transform(
                    raw_coords, rotation, scale, flip, update_translation
                )
                raw_coords += self.scene_bound / 2

                # Append Point-level 数据到 instance_data
                instance_data.instance_coords.append(raw_coords)
                instance_data.instance_colors.append(colors)
                instance_data.instance_widths.append(widths)
                instance_data.instance_prim_ids.append(prim_ids)
                instance_data.instance_layer_ids.append(layer_ids)

                # Append Primitive-level 数据 (直接使用 primitive_lib 中保存的数据)
                # 注意：primitive_lib 中每个 key 本质上是一个 instance 级别的元素
                prim_sem_ids = prim_data[
                    "sem_ids"
                ].copy()  # 该 instance 内各子 primitive 的 semantic_id
                prim_inst_ids = prim_data[
                    "inst_ids"
                ].copy()  # 该 instance 内各子 primitive 的 instance_id
                prim_lengths = prim_data[
                    "primitive_lengths"
                ].copy()  # 该 instance 内各子 primitive 的长度

                instance_data.instance_semantic_ids.append(
                    prim_sem_ids.astype(np.int32)
                )
                # instance_id 使用新的索引（当前 instance 数量）
                new_instance_idx = (
                    instance_data.num_instances() - 1
                )  # 因为已经 append 了 coords
                instance_data.instance_instance_ids.append(
                    np.full(len(prim_sem_ids), new_instance_idx, dtype=np.int32)
                )
                instance_data.instance_primitive_lengths.append(
                    prim_lengths.astype(np.float32)
                )

                # 更新 GT 的 instance-level 属性
                instance_data.primitive_ids = np.concatenate(
                    [
                        instance_data.primitive_ids,
                        np.array([primitive_id], dtype=np.int32),
                    ]
                )
                instance_data.semantic_ids = np.concatenate(
                    [
                        instance_data.semantic_ids,
                        np.array([semantic_id], dtype=np.int32),
                    ]
                )
                # instance_data.translations = np.concatenate([
                #     instance_data.translations,
                #     (update_translation / (self.scene_bound / 2)).reshape(1, 2)
                # ], axis=0)
                instance_data.translations = np.concatenate(
                    [instance_data.translations, update_translation.reshape(1, 2)],
                    axis=0,
                )
                instance_data.rotations = np.concatenate(
                    [instance_data.rotations, np.array([rotation], dtype=np.float32)]
                )
                # instance_data.rotations = np.concatenate([
                #     instance_data.rotations,
                #     np.array([(360 - rotation) % 360], dtype=np.float32)
                # ])
                # scale 可能是标量或数组，需要处理
                if np.isscalar(scale):
                    scale_arr = np.array([[scale, scale]], dtype=np.float32)
                else:
                    scale_arr = np.array(scale, dtype=np.float32).reshape(1, 2)
                instance_data.scales = np.concatenate(
                    [instance_data.scales, scale_arr], axis=0
                )
                instance_data.flip_modes = np.concatenate(
                    [instance_data.flip_modes, np.array([flip], dtype=np.int32)]
                )
                if primitive_id in flip_invariant_primitive_id:
                    flip_invariant = True
                else:
                    flip_invariant = False
                instance_data.flip_invariant_primitives = np.concatenate(
                    [
                        instance_data.flip_invariant_primitives,
                        np.array([flip_invariant], dtype=np.bool),
                    ]
                )
                if instance_data.sizes is not None:
                    min_x = min(raw_coords[:, 0].min(), raw_coords[:, 2].min())
                    max_x = max(raw_coords[:, 0].max(), raw_coords[:, 2].max())
                    min_y = min(raw_coords[:, 1].min(), raw_coords[:, 3].min())
                    max_y = max(raw_coords[:, 1].max(), raw_coords[:, 3].max())
                    size = np.array([max_x - min_x, max_y - min_y], dtype=np.float32)
                    instance_data.sizes = np.concatenate(
                        [instance_data.sizes, size.reshape(1, 2)], axis=0
                    )

        return instance_data


@TRANSFORMS.register_module()
class InstanceJitterTranslate(object):
    """
    Instance-level 的 Translation 扰动增强

    对部分 instance 做小幅平移扰动；同步更新 instance_coords 和 translations

    Args:
        p: 触发概率
        sample_ratio: 被选中的实例比例
        max_delta: 每轴平移幅度（归一化坐标，eg. 归一化到[-1,1]则给0.02之类）
        axis_bias: None / 'x' / 'y' （若想沿某轴更常见，可加权）
        scene_bound: 场景边界大小（用于坐标归一化转换）
        trans_range: translation 的有效范围，超出该范围的扰动将被跳过
    """

    def __init__(
        self,
        p=0.5,
        sample_ratio=0.3,
        max_delta=0.1,
        axis_bias=None,
        scene_bound=20000.0,
        trans_range=[0.0, 1.0],
        random_translate=False,
    ):
        self.p = p
        self.sample_ratio = sample_ratio
        self.max_delta = max_delta
        self.axis_bias = axis_bias
        self.scene_bound = scene_bound
        self.trans_range = trans_range
        self.random_translate = random_translate

    def __call__(self, instance_data: InstanceData) -> InstanceData:
        """
        对 InstanceData 执行 Translation 扰动增强

        Args:
            instance_data: 输入的 InstanceData

        Returns:
            扰动后的 InstanceData
        """
        # 检查触发概率
        if np.random.rand() > self.p:
            return instance_data

        # 检查必要数据是否存在
        if instance_data.translations is None:
            return instance_data

        num_instances = instance_data.num_instances()
        if num_instances == 0:
            return instance_data

        # 选择要扰动的 instance 数量
        k = max(1, int(self.sample_ratio * num_instances))
        idxs = np.random.choice(num_instances, k, replace=False)

        for i in idxs:
            # 生成随机扰动
            if self.random_translate:
                target = np.asarray(
                    [
                        np.random.uniform(self.trans_range[0], self.trans_range[1]),
                        np.random.uniform(self.trans_range[0], self.trans_range[1]),
                    ],
                    dtype=np.float32,
                )
                old_translation = np.asarray(
                    instance_data.translations[i, :2], dtype=np.float32
                ).copy()
                offset = target - old_translation

                # 更新 instance_coords (线模式: x1, y1, x2, y2) [0, 30000]
                coords = instance_data.instance_coords[i]
                coords[:, 0] += offset[0]
                coords[:, 1] += offset[1]
                coords[:, 2] += offset[0]
                coords[:, 3] += offset[1]
                instance_data.instance_coords[i] = coords

                instance_data.translations[i, :2] = target
            else:
                dx = np.random.uniform(-self.max_delta, self.max_delta)
                dy = np.random.uniform(-self.max_delta, self.max_delta)

                # 轴向偏置：如果指定某轴偏置，则减小另一轴的扰动幅度
                if self.axis_bias == "x":
                    dy *= 0.5
                if self.axis_bias == "y":
                    dx *= 0.5

                delta = np.array([dx, dy], dtype=np.float32)

                # 检查更新后 translation 是否在有效范围内
                shifted_trans = instance_data.translations[i, :2] + delta * (
                    self.scene_bound / 2
                )
                trans_valid = (
                    (shifted_trans[0] >= self.trans_range[0])
                    & (shifted_trans[0] <= self.trans_range[1])
                    & (shifted_trans[1] >= self.trans_range[0])
                    & (shifted_trans[1] <= self.trans_range[1])
                )

                if not trans_valid:
                    # 更新后会超出范围，跳过该 instance
                    continue

                # 更新 instance_coords (线模式: x1, y1, x2, y2)
                coords = instance_data.instance_coords[i]
                # 计算坐标空间的实际偏移量
                coord_delta = delta * (self.scene_bound / 2)
                # 更新 x1, y1 (前两列)
                coords[:, 0] += coord_delta[0]
                coords[:, 1] += coord_delta[1]
                # 更新 x2, y2 (后两列)
                coords[:, 2] += coord_delta[0]
                coords[:, 3] += coord_delta[1]
                instance_data.instance_coords[i] = coords

                # 更新 translations
                instance_data.translations[i, :2] += delta

        return instance_data


class InstanceCompose(object):
    """
    Instance-level 增强的 Compose 类

    用于组合多个 instance-level 增强操作
    """

    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.transforms = []
        for t_cfg in self.cfg:
            self.transforms.append(TRANSFORMS.build(t_cfg))

    def import_primitive_lib(
        self, primitive_lib, primitive_ids, primitive_weights, primitive_to_semantic
    ):
        """将 primitive_lib 导入到所有子增强中"""
        for t in self.transforms:
            if hasattr(t, "import_primitive_lib"):
                t.import_primitive_lib(
                    primitive_lib,
                    primitive_ids,
                    primitive_weights,
                    primitive_to_semantic,
                )

    def __call__(self, instance_data: InstanceData) -> InstanceData:
        """依次应用所有 instance-level 增强"""
        for t in self.transforms:
            instance_data = t(instance_data)
        return instance_data


@TRANSFORMS.register_module()
class VecClipping(object):
    """
    对向量数据进行坐标裁剪，去除超出指定范围的线段及相关属性。

    裁剪流程:
    1. 移除坐标超出 clip_range 的线段
    2. 计算每个 instance 的线段保留比例
    3. 移除保留比例低于 min_instance_ratio 的 instance 及其所有线段
    4. 同步更新所有 point-level, primitive-level 和 instance-level 属性

    Args:
        clip_range: 裁剪范围 [min, max]，默认 [-1, 1]
        min_instance_ratio: instance 保留的最小线段比例，默认 0.3
        update_size: 是否根据实际保留部分重新计算被裁剪 instance 的 size，默认 False
    """

    def __init__(
        self,
        clip_range=[-1, 1],
        min_instance_ratio=0.3,
        update_size=False,
        instance_filter_metric="count",
        low_visibility_mode="drop",
        empty_input_fallback="drop_sample",
    ):
        self.clip_min = clip_range[0]
        self.clip_max = clip_range[1]
        self.min_instance_ratio = min_instance_ratio
        self.update_size = update_size
        if instance_filter_metric not in {"count", "length", "bbox"}:
            raise ValueError(
                "instance_filter_metric must be 'count', 'length', or 'bbox', got "
                f"{instance_filter_metric!r}"
            )
        if low_visibility_mode not in {"drop", "keep_as_noise"}:
            raise ValueError(
                "low_visibility_mode must be 'drop' or 'keep_as_noise', got "
                f"{low_visibility_mode!r}"
            )
        if empty_input_fallback not in {"drop_sample", "keep_as_noise"}:
            raise ValueError(
                "empty_input_fallback must be 'drop_sample' or "
                f"'keep_as_noise', got {empty_input_fallback!r}"
            )
        self.instance_filter_metric = instance_filter_metric
        self.low_visibility_mode = low_visibility_mode
        self.empty_input_fallback = empty_input_fallback

    def __call__(self, data_dict):
        assert "svg_data" in data_dict.keys()
        svg_data = data_dict["svg_data"]

        coord = np.asarray(svg_data.coords)
        n_points = coord.shape[0]
        audit_enabled = bool(data_dict.get("_recon_audit_enabled", False))
        audit = data_dict.setdefault("_recon_audit", {}) if audit_enabled else None
        if n_points == 0:
            return data_dict

        orig_prim_ids = np.asarray(svg_data.primitive_ids, dtype=np.int32)
        orig_instance_ids = np.asarray(svg_data.instance_ids, dtype=np.int32)
        orig_line_to_instance = orig_instance_ids[orig_prim_ids]
        num_instances_before = len(data_dict.get("translation", []))
        segment_coord = coord.reshape(n_points, -1, 2)
        segment_lengths = np.linalg.norm(
            segment_coord[:, -1] - segment_coord[:, 0], axis=1
        )
        foreground = orig_line_to_instance >= 0
        total_counts = np.bincount(
            orig_line_to_instance[foreground], minlength=num_instances_before
        ).astype(np.int64)
        total_lengths = np.bincount(
            orig_line_to_instance[foreground],
            weights=segment_lengths[foreground],
            minlength=num_instances_before,
        ).astype(np.float64)
        if audit_enabled:
            audit["full_translation_pre_clip"] = np.asarray(
                data_dict.get("translation", np.empty((0, 2))), dtype=np.float32
            ).copy()
            audit["clip_input_segment_counts"] = total_counts
            audit["clip_input_segment_lengths"] = total_lengths
            # Keep an immutable pre-filter copy for protocol comparisons.
            # ``clip_input_segment_counts`` is later remapped to the retained
            # target IDs, while this array must continue to describe every
            # original foreground instance.
            audit["visibility_filter_input_segment_counts_all"] = total_counts.copy()
            if bool(data_dict.get("_recon_capture_preclip_source", False)):
                # Snapshot after every upstream augmentation but before the
                # crop.  Only foreground membership is retained; these
                # segments remain a loss target and are never put into vec_data.
                audit["preclip_full_coords"] = segment_coord[foreground].astype(
                    np.float32, copy=True
                )
                audit["preclip_full_line_instance_ids"] = orig_line_to_instance[
                    foreground
                ].astype(np.int32, copy=True)

        # ====== Step 1: 计算坐标裁剪 mask ======
        in_range = (coord >= self.clip_min) & (coord <= self.clip_max)
        coord_keep_mask = in_range.reshape(n_points, -1).all(axis=1)

        if coord_keep_mask.all():
            if audit_enabled:
                audit["clip_count_ratio"] = np.ones(
                    num_instances_before, dtype=np.float32
                )
                audit["clip_length_ratio"] = np.ones(
                    num_instances_before, dtype=np.float32
                )
                audit["clip_bbox_ratio"] = np.ones(
                    num_instances_before, dtype=np.float32
                )
                audit["visibility_filter_count_ratio_all"] = np.ones(
                    num_instances_before, dtype=np.float32
                )
                audit["visibility_filter_length_ratio_all"] = np.ones(
                    num_instances_before, dtype=np.float32
                )
                audit["visibility_filter_bbox_ratio_all"] = np.ones(
                    num_instances_before, dtype=np.float32
                )
                audit["visibility_filter_semantic_id_all"] = np.asarray(
                    data_dict.get("semantic_id", np.full(num_instances_before, -1)),
                    dtype=np.int64,
                ).copy()
                audit["clip_removed_instances"] = 0
                audit["clip_removed_segments"] = 0
                audit["visibility_filter_ignored_instances"] = 0
                audit["visibility_filter_noise_segments"] = 0
                audit["visibility_filter_noise_length"] = 0.0
            return data_dict

        # ====== Step 2: 用 bincount 向量化计算 instance 保留比例 ======
        fg_line_mask = orig_line_to_instance >= 0
        bbox_ratio_per_inst = np.zeros(num_instances_before, dtype=np.float64)
        if fg_line_mask.any():
            max_inst = max(
                num_instances_before,
                int(orig_line_to_instance[fg_line_mask].max()) + 1,
            )
            total_per_inst = np.bincount(
                orig_line_to_instance[fg_line_mask], minlength=max_inst
            )
            surviving_per_inst = np.bincount(
                orig_line_to_instance[fg_line_mask & coord_keep_mask],
                minlength=max_inst,
            )
            surviving_length_per_inst = np.bincount(
                orig_line_to_instance[fg_line_mask & coord_keep_mask],
                weights=segment_lengths[fg_line_mask & coord_keep_mask],
                minlength=max_inst,
            )
            count_ratio_per_inst = np.divide(
                surviving_per_inst,
                total_per_inst,
                out=np.zeros_like(surviving_per_inst, dtype=np.float64),
                where=total_per_inst > 0,
            )
            length_ratio_per_inst = np.divide(
                surviving_length_per_inst,
                total_lengths[:max_inst],
                out=np.zeros_like(surviving_length_per_inst, dtype=np.float64),
                where=total_lengths[:max_inst] > 0,
            )

            # Axis-aligned bbox retention.  This is well matched to the
            # Industry data's mostly horizontal/vertical placement: for a
            # partially visible horizontal object, for example, the area
            # ratio reduces to its retained x-span ratio when y-span is
            # unchanged.  Degenerate line-like bboxes fall back to the major
            # axis span ratio instead of producing 0/0.
            def _bbox_extents(line_mask):
                x_min = np.full(max_inst, np.inf, dtype=np.float64)
                x_max = np.full(max_inst, -np.inf, dtype=np.float64)
                y_min = np.full(max_inst, np.inf, dtype=np.float64)
                y_max = np.full(max_inst, -np.inf, dtype=np.float64)
                selected = line_mask & fg_line_mask
                if selected.any():
                    selected_points = segment_coord[selected].reshape(-1, 2)
                    selected_ids = np.repeat(
                        orig_line_to_instance[selected], segment_coord.shape[1]
                    )
                    np.minimum.at(x_min, selected_ids, selected_points[:, 0])
                    np.maximum.at(x_max, selected_ids, selected_points[:, 0])
                    np.minimum.at(y_min, selected_ids, selected_points[:, 1])
                    np.maximum.at(y_max, selected_ids, selected_points[:, 1])
                width = np.where(
                    np.isfinite(x_min), np.maximum(x_max - x_min, 0.0), 0.0
                )
                height = np.where(
                    np.isfinite(y_min), np.maximum(y_max - y_min, 0.0), 0.0
                )
                return np.stack([width, height], axis=1)

            full_bbox_extent = _bbox_extents(np.ones(n_points, dtype=bool))
            visible_bbox_extent = _bbox_extents(coord_keep_mask)
            full_bbox_area = np.prod(full_bbox_extent, axis=1)
            visible_bbox_area = np.prod(visible_bbox_extent, axis=1)
            bbox_ratio_per_inst = np.zeros(max_inst, dtype=np.float64)
            nondegenerate = full_bbox_area > 1e-12
            bbox_ratio_per_inst[nondegenerate] = (
                visible_bbox_area[nondegenerate] / full_bbox_area[nondegenerate]
            )
            degenerate = ~nondegenerate & (total_per_inst > 0)
            full_major_span = full_bbox_extent.max(axis=1)
            visible_major_span = visible_bbox_extent.max(axis=1)
            line_like = degenerate & (full_major_span > 1e-12)
            bbox_ratio_per_inst[line_like] = (
                visible_major_span[line_like] / full_major_span[line_like]
            )
            point_like = degenerate & ~line_like
            bbox_ratio_per_inst[point_like] = (
                surviving_per_inst[point_like] > 0
            ).astype(np.float64)
            bbox_ratio_per_inst = np.clip(bbox_ratio_per_inst, 0.0, 1.0)

            metric_ratios = {
                "count": count_ratio_per_inst,
                "length": length_ratio_per_inst,
                "bbox": bbox_ratio_per_inst,
            }
            ratio_per_inst = metric_ratios[self.instance_filter_metric]
            low_ratio_insts = np.where(
                (ratio_per_inst < self.min_instance_ratio) & (total_per_inst > 0)
            )[0]

            if audit_enabled:
                audit["visibility_filter_count_ratio_all"] = (
                    count_ratio_per_inst.astype(np.float32)
                )
                audit["visibility_filter_length_ratio_all"] = (
                    length_ratio_per_inst.astype(np.float32)
                )
                audit["visibility_filter_bbox_ratio_all"] = bbox_ratio_per_inst.astype(
                    np.float32
                )
                audit["visibility_filter_bbox_full_extent_all"] = (
                    full_bbox_extent.astype(np.float32)
                )
                audit["visibility_filter_bbox_visible_extent_all"] = (
                    visible_bbox_extent.astype(np.float32)
                )
                audit["visibility_filter_semantic_id_all"] = np.asarray(
                    data_dict.get("semantic_id", np.full(num_instances_before, -1)),
                    dtype=np.int64,
                )[:max_inst].copy()
                audit["visibility_filter_metric"] = self.instance_filter_metric

            force_empty_input_noise = False
            if len(low_ratio_insts) > 0 and self.low_visibility_mode == "drop":
                discard_inst_mask = np.isin(orig_line_to_instance, low_ratio_insts)
                keep_mask = coord_keep_mask & ~discard_inst_mask
                # A strict visibility filter can occasionally remove every
                # visible line in a random crop.  For scratch experiments we
                # still want this crop to teach EOS/background handling.  In
                # that explicitly requested mode, retain the clipped geometry
                # as void/noise while keeping every low-visibility instance
                # out of the autoregressive target sequence.
                if (
                    not keep_mask.any()
                    and coord_keep_mask.any()
                    and self.empty_input_fallback == "keep_as_noise"
                ):
                    keep_mask = coord_keep_mask
                    force_empty_input_noise = True
            else:
                keep_mask = coord_keep_mask
                force_empty_input_noise = False
        else:
            low_ratio_insts = np.empty(0, dtype=np.int64)
            keep_mask = coord_keep_mask

        # In keep_as_noise mode the visible geometry remains part of the
        # conditioner input, but the low-visibility object is no longer a GT
        # instance.  Its primitive-level instance association is changed to
        # the existing void sentinel (-1); coordinates/features are untouched.
        effective_orig_instance_ids = orig_instance_ids.copy()
        keep_low_as_noise = (
            self.low_visibility_mode == "keep_as_noise" or force_empty_input_noise
        )
        if len(low_ratio_insts) and keep_low_as_noise:
            low_primitive_mask = np.isin(effective_orig_instance_ids, low_ratio_insts)
            effective_orig_instance_ids[low_primitive_mask] = -1

        if audit_enabled:
            kept_foreground = (orig_line_to_instance >= 0) & keep_mask
            kept_counts = np.bincount(
                orig_line_to_instance[kept_foreground], minlength=num_instances_before
            ).astype(np.int64)
            kept_lengths = np.bincount(
                orig_line_to_instance[kept_foreground],
                weights=segment_lengths[kept_foreground],
                minlength=num_instances_before,
            ).astype(np.float64)
            count_ratio_all = np.divide(
                kept_counts,
                total_counts,
                out=np.zeros(num_instances_before, dtype=np.float64),
                where=total_counts > 0,
            )
            length_ratio_all = np.divide(
                kept_lengths,
                total_lengths,
                out=np.zeros(num_instances_before, dtype=np.float64),
                where=total_lengths > 0,
            )
            bbox_ratio_all = np.zeros(num_instances_before, dtype=np.float64)
            bbox_ratio_all[: len(bbox_ratio_per_inst)] = bbox_ratio_per_inst
            visible_low_lines = coord_keep_mask & np.isin(
                orig_line_to_instance, low_ratio_insts
            )
            audit["visibility_filter_ignored_instances"] = int(len(low_ratio_insts))
            audit["visibility_filter_noise_segments"] = int(
                np.count_nonzero(visible_low_lines) if keep_low_as_noise else 0
            )
            audit["visibility_filter_noise_length"] = float(
                segment_lengths[visible_low_lines].sum() if keep_low_as_noise else 0.0
            )
            audit["visibility_filter_empty_input_fallback"] = bool(
                force_empty_input_noise
            )

        if not keep_mask.any():
            # 本样本在 clip_range 下无任何可保留线段：
            # - 若继续返回原 data_dict，会导致 instance-level 属性（translation/size）不受约束，
            #   下游统计/训练出现超界值。
            # - 交由 Dataset 在 __getitem__ 中跳过当前样本，读取下一个样本。
            data_dict["_drop_sample"] = True
            data_dict["_drop_reason"] = "vec_clipping_keep_mask_empty"
            return data_dict

        # ====== Step 3: 更新 Point-level 属性 ======
        colors = np.asarray(svg_data.colors)
        widths = np.asarray(svg_data.widths)
        layer_ids = np.asarray(svg_data.layer_ids)

        svg_data.coords = coord[keep_mask]
        svg_data.colors = colors[keep_mask]
        svg_data.widths = widths[keep_mask]
        svg_data.layer_ids = layer_ids[keep_mask]

        # ====== Step 4: 更新 Primitive-level 属性（LUT 重映射） ======
        surviving_prim_ids = orig_prim_ids[keep_mask]
        surviving_prims = np.unique(surviving_prim_ids)

        orig_semantic_ids = np.asarray(svg_data.semantic_ids)
        orig_prim_lengths = np.asarray(svg_data.primitive_lengths)

        if len(surviving_prims) < len(orig_semantic_ids):
            svg_data.semantic_ids = orig_semantic_ids[surviving_prims]
            svg_data.instance_ids = effective_orig_instance_ids[surviving_prims]
            svg_data.primitive_lengths = orig_prim_lengths[surviving_prims]
        else:
            # Primitive membership can still change when low-visibility
            # instances are retained as void/noise even if no primitive was
            # removed geometrically.
            svg_data.instance_ids = effective_orig_instance_ids

        max_prim_id = int(surviving_prims.max()) + 1
        prim_lut = np.empty(max_prim_id, dtype=np.int32)
        prim_lut[surviving_prims] = np.arange(len(surviving_prims), dtype=np.int32)
        svg_data.primitive_ids = prim_lut[surviving_prim_ids]

        # ====== Step 5: 更新 Instance-level 属性（LUT 重映射） ======
        updated_instance_ids = np.asarray(svg_data.instance_ids, dtype=np.int32)
        fg_inst_mask = updated_instance_ids >= 0
        present_instances = np.empty(0, dtype=np.int64)

        if fg_inst_mask.any():
            present_instances = np.unique(updated_instance_ids[fg_inst_mask])
            max_inst_id = int(present_instances.max()) + 1
            inst_lut = np.empty(max_inst_id, dtype=np.int32)
            inst_lut[present_instances] = np.arange(
                len(present_instances), dtype=np.int32
            )
            new_inst = updated_instance_ids.copy()
            new_inst[fg_inst_mask] = inst_lut[updated_instance_ids[fg_inst_mask]]
            svg_data.instance_ids = new_inst

        num_orig_instances = len(data_dict.get("translation", []))
        if num_orig_instances > 0 and len(present_instances) < num_orig_instances:
            keep_inst_mask = np.zeros(num_orig_instances, dtype=bool)
            valid_ids = present_instances[present_instances < num_orig_instances]
            keep_inst_mask[valid_ids] = True

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
                if key_name in data_dict and data_dict[key_name] is not None:
                    arr = data_dict[key_name]
                    if hasattr(arr, "__len__") and len(arr) == num_orig_instances:
                        data_dict[key_name] = arr[keep_inst_mask]

        if audit_enabled:
            for key_name in ("instance_origin_kind", "instance_jittered"):
                if key_name in audit:
                    audit[key_name] = np.asarray(audit[key_name])[present_instances]
            audit["clip_count_ratio"] = count_ratio_all[present_instances].astype(
                np.float32
            )
            audit["clip_length_ratio"] = length_ratio_all[present_instances].astype(
                np.float32
            )
            audit["clip_bbox_ratio"] = bbox_ratio_all[present_instances].astype(
                np.float32
            )
            audit["clip_input_segment_counts"] = total_counts[present_instances]
            audit["clip_input_segment_lengths"] = total_lengths[present_instances]
            audit["full_translation_pre_clip"] = audit["full_translation_pre_clip"][
                present_instances
            ]
            if "preclip_full_line_instance_ids" in audit:
                preclip_ids = np.asarray(
                    audit["preclip_full_line_instance_ids"],
                    dtype=np.int32,
                )
                preclip_keep = np.isin(preclip_ids, present_instances)
                audit["preclip_full_coords"] = np.asarray(
                    audit["preclip_full_coords"], dtype=np.float32
                )[preclip_keep]
                kept_preclip_ids = preclip_ids[preclip_keep]
                audit["preclip_full_line_instance_ids"] = np.searchsorted(
                    present_instances, kept_preclip_ids
                ).astype(np.int32)
            audit["clip_removed_instances"] = int(
                num_instances_before - len(present_instances)
            )
            audit["clip_removed_segments"] = int(n_points - np.count_nonzero(keep_mask))

        # ====== Step 6: 根据实际保留部分重新计算 partial instance 的 size 和 translation ======
        if self.update_size and "size" in data_dict and data_dict["size"] is not None:
            final_coords = np.asarray(svg_data.coords)
            final_prim_ids = np.asarray(svg_data.primitive_ids, dtype=np.int32)
            final_inst_ids = np.asarray(svg_data.instance_ids, dtype=np.int32)
            point_inst = final_inst_ids[final_prim_ids]

            fg = point_inst >= 0
            if fg.any():
                num_instances = len(data_dict["size"])
                fg_inst = point_inst[fg]
                fg_coords = final_coords[fg]

                if fg_coords.ndim == 3:
                    all_x = fg_coords[:, :, 0].ravel()
                    all_y = fg_coords[:, :, 1].ravel()
                    all_inst = np.repeat(fg_inst, 2)
                else:
                    all_x = np.concatenate([fg_coords[:, 0], fg_coords[:, 2]])
                    all_y = np.concatenate([fg_coords[:, 1], fg_coords[:, 3]])
                    all_inst = np.concatenate([fg_inst, fg_inst])

                x_min = np.full(num_instances, np.inf, dtype=np.float32)
                x_max = np.full(num_instances, -np.inf, dtype=np.float32)
                y_min = np.full(num_instances, np.inf, dtype=np.float32)
                y_max = np.full(num_instances, -np.inf, dtype=np.float32)

                np.minimum.at(x_min, all_inst, all_x.astype(np.float32))
                np.maximum.at(x_max, all_inst, all_x.astype(np.float32))
                np.minimum.at(y_min, all_inst, all_y.astype(np.float32))
                np.maximum.at(y_max, all_inst, all_y.astype(np.float32))

                data_dict["size"] = np.stack([x_max - x_min, y_max - y_min], axis=1)
                if "translation" in data_dict and data_dict["translation"] is not None:
                    center_x = (x_min + x_max) * 0.5
                    center_y = (y_min + y_max) * 0.5
                    data_dict["translation"][:, 0] = center_x
                    data_dict["translation"][:, 1] = center_y

        data_dict["svg_data"] = svg_data
        return data_dict


@TRANSFORMS.register_module()
class VecGridSample(object):
    """
    针对向量数据（2D 线段/点）的网格采样，确保输入网络的点密度保持一致。

    使用线段中点（或点坐标）进行 2D 网格划分，每个网格单元只保留一条
    代表线段/点（train 模式随机选取，test 模式选取第一条）。

    同时同步更新 point_primitive_ids，并可选地过滤失去所有采样点的
    instance 的 GT 属性。

    注意: VecGridSample 会改变点的数量。为了保持 svg_data 中 colors,
    widths, layer_ids 等 per-point 属性的一致性，该 transform 会将采样
    索引存入 data_dict['_vec_grid_sample_idx']，由 Dataset 的 _transform
    方法在 transform chain 结束后使用该索引同步更新 svg_data。

    数据要求:
        - coord: 点坐标 [N, 2, 2]（线段模式）或 [N, 2]（点模式）
        - point_primitive_ids: (可选) 点级别的 primitive_id [N,]
        - translation, rotation, scale 等 instance 级别属性 (可选)

    Args:
        grid_size: 网格大小（归一化坐标空间下，默认 0.05）
        hash_type: 哈希函数类型，"fnv" 或 "ravel"
        mode: "train" 每个 cell 随机选取, "test" 选取第一个
        return_grid_coord: 是否返回 grid 坐标
        filter_instance: 是否过滤失去所有点的 instance（同步更新 instance 级别属性）
    """

    def __init__(
        self,
        grid_size=0.05,
        hash_type="fnv",
        mode="train",
        return_grid_coord=False,
        filter_instance=True,
    ):
        self.grid_size = grid_size
        self.hash = self.fnv_hash_vec if hash_type == "fnv" else self.ravel_hash_vec
        assert mode in ["train", "test"]
        self.mode = mode
        self.return_grid_coord = return_grid_coord
        self.filter_instance = filter_instance

    def __call__(self, data_dict):
        assert "svg_data" in data_dict.keys()
        svg_data = data_dict["svg_data"]
        coord = np.asarray(svg_data.coords)
        colors = np.asarray(svg_data.colors)
        widths = np.asarray(svg_data.widths)
        primitive_ids = np.asarray(svg_data.primitive_ids, dtype=np.int32)
        layer_ids = np.asarray(svg_data.layer_ids)
        semantic_ids = np.asarray(svg_data.semantic_ids, dtype=np.int32)
        instance_ids = np.asarray(svg_data.instance_ids, dtype=np.int32)
        primitive_lengths = np.asarray(svg_data.primitive_lengths)

        n_points = coord.shape[0]
        audit_enabled = bool(data_dict.get("_recon_audit_enabled", False))
        audit = data_dict.setdefault("_recon_audit", {}) if audit_enabled else None
        if n_points == 0:
            return data_dict
        if audit_enabled:
            audit["grid_input_segments"] = int(n_points)
            audit["grid_input_instances"] = int(len(data_dict.get("translation", [])))
            pre_grid_line_instance = instance_ids[primitive_ids]
            pre_grid_foreground = pre_grid_line_instance >= 0
            audit["pre_grid_instance_coords"] = coord[pre_grid_foreground].copy()
            audit["pre_grid_line_instance_ids"] = pre_grid_line_instance[
                pre_grid_foreground
            ].astype(np.int32, copy=True)

        # ====== Step 1: 计算每条线段/点的代表坐标（用于网格划分） ======
        if len(coord.shape) == 3:
            representative_coord = coord.mean(axis=1)
        else:
            representative_coord = coord

        # ====== Step 2: 网格量化与哈希 ======
        scaled_coord = representative_coord / np.array(self.grid_size)
        grid_coord = np.floor(scaled_coord).astype(int)
        min_coord = grid_coord.min(0)
        grid_coord -= min_coord

        key = self.hash(grid_coord)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)

        # ====== Step 3: 每个 grid cell 选取代表线段 ======
        if self.mode == "train":
            idx_select = (
                np.cumsum(np.insert(count, 0, 0)[0:-1])
                + np.random.randint(0, count.max(), count.size) % count
            )
            idx_unique = idx_sort[idx_select]
        else:
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1])
            idx_unique = idx_sort[idx_select]

        # ====== Step 4: 更新 Point-level 属性 ======
        svg_data.coords = coord[idx_unique]
        svg_data.colors = colors[idx_unique]
        svg_data.widths = widths[idx_unique]
        svg_data.layer_ids = layer_ids[idx_unique]

        # ====== Step 5: 更新 Primitive-level 属性（LUT 重映射 primitive_ids） ======
        sampled_prim_ids = primitive_ids[idx_unique]
        surviving_prim_indices = np.unique(sampled_prim_ids)

        num_total_prims = len(semantic_ids)
        if len(surviving_prim_indices) < num_total_prims:
            svg_data.semantic_ids = semantic_ids[surviving_prim_indices]
            svg_data.instance_ids = instance_ids[surviving_prim_indices]
            svg_data.primitive_lengths = primitive_lengths[surviving_prim_indices]

            max_old_prim = int(surviving_prim_indices.max()) + 1
            prim_lut = np.empty(max_old_prim, dtype=np.int32)
            prim_lut[surviving_prim_indices] = np.arange(
                len(surviving_prim_indices), dtype=np.int32
            )
            svg_data.primitive_ids = prim_lut[sampled_prim_ids]
        else:
            svg_data.primitive_ids = sampled_prim_ids

        # ====== Step 6: 更新 Instance-level 属性（与 VecClipping Step 5 一致） ======
        # 网格采样可能导致某个 instance 的所有 primitive 都丢失，
        # 必须同步过滤 GT 侧的 instance-level 属性，否则 scene/gt 数量不一致。
        present_instances = np.arange(
            len(data_dict.get("translation", [])), dtype=np.int64
        )
        if self.filter_instance:
            updated_instance_ids = np.asarray(svg_data.instance_ids, dtype=np.int32)
            fg_inst_mask = updated_instance_ids >= 0
            present_instances = np.empty(0, dtype=np.int64)

            if fg_inst_mask.any():
                present_instances = np.unique(updated_instance_ids[fg_inst_mask])

                num_orig_instances = len(data_dict.get("translation", []))
                if (
                    num_orig_instances > 0
                    and len(present_instances) < num_orig_instances
                ):
                    # 过滤 GT instance-level 属性
                    keep_inst_mask = np.zeros(num_orig_instances, dtype=bool)
                    valid_ids = present_instances[
                        present_instances < num_orig_instances
                    ]
                    keep_inst_mask[valid_ids] = True

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
                        if key_name in data_dict and data_dict[key_name] is not None:
                            arr = data_dict[key_name]
                            if (
                                hasattr(arr, "__len__")
                                and len(arr) == num_orig_instances
                            ):
                                data_dict[key_name] = arr[keep_inst_mask]

                # instance_ids 重新连续编号（-1 保持不变）
                max_inst_id = int(present_instances.max()) + 1
                inst_lut = np.empty(max_inst_id, dtype=np.int32)
                inst_lut[present_instances] = np.arange(
                    len(present_instances), dtype=np.int32
                )
                new_inst = updated_instance_ids.copy()
                new_inst[fg_inst_mask] = inst_lut[updated_instance_ids[fg_inst_mask]]
                svg_data.instance_ids = new_inst

                if audit_enabled:
                    for key_name in (
                        "clip_count_ratio",
                        "clip_length_ratio",
                        "clip_bbox_ratio",
                        "clip_input_segment_counts",
                        "clip_input_segment_lengths",
                        "full_translation_pre_clip",
                        "instance_origin_kind",
                        "instance_jittered",
                    ):
                        if key_name in audit:
                            value = np.asarray(audit[key_name])
                            audit[key_name] = value[present_instances]

                    if "preclip_full_line_instance_ids" in audit:
                        preclip_ids = np.asarray(
                            audit["preclip_full_line_instance_ids"],
                            dtype=np.int32,
                        )
                        preclip_keep = np.isin(preclip_ids, present_instances)
                        audit["preclip_full_coords"] = np.asarray(
                            audit["preclip_full_coords"], dtype=np.float32
                        )[preclip_keep]
                        kept_preclip_ids = preclip_ids[preclip_keep]
                        audit["preclip_full_line_instance_ids"] = np.searchsorted(
                            present_instances, kept_preclip_ids
                        ).astype(np.int32)

        data_dict["svg_data"] = svg_data
        data_dict["_vec_grid_sample_idx"] = idx_unique

        if audit_enabled:
            pre_grid_coords = np.asarray(audit["pre_grid_instance_coords"])
            pre_grid_instance_ids = np.asarray(
                audit["pre_grid_line_instance_ids"], dtype=np.int32
            )
            if len(present_instances) < audit["grid_input_instances"]:
                pre_grid_keep = np.isin(pre_grid_instance_ids, present_instances)
                pre_grid_coords = pre_grid_coords[pre_grid_keep]
                pre_grid_instance_ids = pre_grid_instance_ids[pre_grid_keep]
                if len(present_instances):
                    pre_grid_instance_ids = np.searchsorted(
                        present_instances, pre_grid_instance_ids
                    ).astype(np.int32)
            audit["pre_grid_instance_coords"] = pre_grid_coords
            audit["pre_grid_line_instance_ids"] = pre_grid_instance_ids
            audit["grid_output_segments"] = int(len(idx_unique))
            audit["grid_output_instances"] = int(len(data_dict.get("translation", [])))
            audit["grid_removed_segments"] = int(n_points - len(idx_unique))
            audit["grid_removed_instances"] = int(
                audit["grid_input_instances"] - audit["grid_output_instances"]
            )

        if self.return_grid_coord:
            data_dict["grid_coord"] = grid_coord[idx_unique]

        return data_dict

    @staticmethod
    def ravel_hash_vec(arr):
        """
        Ravel the coordinates after subtracting the min coordinates.
        """
        assert arr.ndim == 2
        arr = arr.copy()
        arr -= arr.min(0)
        arr = arr.astype(np.uint64, copy=False)
        arr_max = arr.max(0).astype(np.uint64) + 1

        keys = np.zeros(arr.shape[0], dtype=np.uint64)
        # Fortran style indexing
        for j in range(arr.shape[1] - 1):
            keys += arr[:, j]
            keys *= arr_max[j + 1]
        keys += arr[:, -1]
        return keys

    @staticmethod
    def fnv_hash_vec(arr):
        """
        FNV64-1A
        """
        assert arr.ndim == 2
        # Floor first for negative coordinates
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(
            arr.shape[0], dtype=np.uint64
        )
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr
