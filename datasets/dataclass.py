from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch


@dataclass
class ProcessArgs:
    file_path: str
    input_dir: str
    output_dir: str
    save_type: str
    connect_lines: bool
    line_t_values: list[float]
    curve_t_values: list[float]
    dynamic_sampling: bool
    dynamic_sampling_ratio: float


@dataclass
class SVGData:
    viewBox: list[float]
    coords: list[list[float]]
    colors: list[list[int]]
    widths: list[float]
    primitive_ids: list[int]
    layer_ids: list[int]
    semantic_ids: list[int]
    instance_ids: list[int]
    primitive_lengths: list[float]


@dataclass
class SVGDataTensor:
    # N: number of sampled points or lines
    viewBox: torch.Tensor  # shape is (4,), [minx, miny, width, height]
    coords: (
        torch.Tensor
    )  # point mode: shape is (N, 2), [N * [x, y]]; line mode: shape is (N, 4), [N * [x1, y1, x2, y2]]
    colors: torch.Tensor  # shape is (N, 3), [N * [r, g, b]]
    widths: torch.Tensor  # shape is (N,), [N * [width]]
    primitive_ids: torch.Tensor  # shape is (N,), [N * [primitive_id]]
    layer_ids: torch.Tensor  # shape is (N,), [N * [layer_id]]
    semantic_ids: torch.Tensor  # shape is (N,), [N * [semantic_id]]
    instance_ids: torch.Tensor  # shape is (N,), [N * [instance_id]]
    primitive_lengths: torch.Tensor  # shape is (N,), [N * [length]]


@dataclass
class VecData:
    # metadata
    data_path: str  # file path for debug
    # --------------------- features --------------------- #
    # N: number of points or lines
    coords: torch.Tensor
    # coordinates of points or lines
    # point mode: shape is (N, 2), [N * [x, y]]
    # line mode: shape is (N, 4), [N * [x1, y1, x2, y2]]
    feats: torch.Tensor
    # features of points or lines
    # point mode: shape is (N, 7), [N * [cx, cy, pcx, pcy, color_r, color_g, color_b]]
    # line mode: shape is (N, 10), [N * [length, |dx|, |dy|, cx, cy, pcx, pcy, color_r, color_g, color_b]]
    prim_ids: torch.Tensor  # shape is (N,), [N * [primitive_id]]
    layer_ids: torch.Tensor  # shape is (N,), [N * [layer_id]]
    # ---------------------- labels ---------------------- #
    sem_ids: torch.Tensor  # shape is (N,), [N * [semantic_id]]
    inst_ids: torch.Tensor  # shape is (N,), [N * [instance_id]]
    prim_lengths: torch.Tensor  # shape is (N,), [N * [length]]


@dataclass
class VecDataTransformArgs:
    norm_range: tuple[float, float] = (-0.5, 0.5)  # range of normalized coordinates
    random_vertical_flip: float = 0.5  # probability of vertical flip
    random_horizontal_flip: float = 0.5  # probability of horizontal flip
    random_rotate: bool = True  # whether to rotate
    random_scale: tuple[float, float] = (
        0.8,
        1.2,
    )  # scale range: (min_scale_ratio, max_scale_ratio)
    random_translation: tuple[float, float] = (
        0.1,
        0.1,
    )  # translation range: (x_translation_ratio, y_translation_ratio)


@dataclass
class InstanceData:
    """
    Instance级别的数据表示，每个instance独立存储
    用于 CopyPaste 和 CutMix 等 instance-level 数据增强

    数据层次说明:
    - SVGData 有两层结构:
        - Point-level (采样点级别): coords, colors, widths, primitive_ids, layer_ids
        - Primitive-level (元素级别): semantic_ids, instance_ids, primitive_lengths
    - 映射关系: point 通过 primitive_ids[point_idx] 指向 primitive 索引

    InstanceData 按 instance_id 分组这些数据
    """

    viewBox: np.ndarray

    # ============ Per-instance 的点级数据 (List of arrays) ============
    # 每个列表元素对应一个 instance 内的所有采样点
    instance_coords: List[np.ndarray] = field(
        default_factory=list
    )  # List[(M_i, 4)] 每个instance的点坐标
    instance_colors: List[np.ndarray] = field(
        default_factory=list
    )  # List[(M_i, 3)] 每个instance点的颜色
    instance_widths: List[np.ndarray] = field(
        default_factory=list
    )  # List[(M_i,)] 每个instance点的宽度
    instance_prim_ids: List[np.ndarray] = field(
        default_factory=list
    )  # List[(M_i,)] 点属于instance内哪个primitive (从0开始重新编号)
    instance_layer_ids: List[np.ndarray] = field(
        default_factory=list
    )  # List[(M_i,)] 点的layer_id

    # ============ Per-instance 的 primitive-level 数据 (List of arrays) ============
    # 每个列表元素是该 instance 内各 primitive 的属性
    instance_semantic_ids: List[np.ndarray] = field(
        default_factory=list
    )  # List[(P_i,)] 每个instance内各primitive的semantic_id
    instance_instance_ids: List[np.ndarray] = field(
        default_factory=list
    )  # List[(P_i,)] 每个instance内各primitive的instance_id
    instance_primitive_lengths: List[np.ndarray] = field(
        default_factory=list
    )  # List[(P_i,)] 每个instance内各primitive的长度

    # ============ GT 的 instance-level 属性 (shape = num_instances) ============
    # 这些属性来自 GT 文件，每个 instance 一个值
    primitive_ids: Optional[np.ndarray] = (
        None  # (N,) 每个 instance 对应的 GT primitive 类型
    )
    semantic_ids: Optional[np.ndarray] = None  # (N,) 每个 instance 对应的 GT 语义类别
    translations: Optional[np.ndarray] = None  # (N, 2) 每个 instance 的平移
    rotations: Optional[np.ndarray] = None  # (N,) 每个 instance 的旋转角度
    scales: Optional[np.ndarray] = (
        None  # (N, 2) 每个 instance 的缩放 (x_scale, y_scale)
    )
    flip_modes: Optional[np.ndarray] = None  # (N,) 每个 instance 的翻转模式
    flip_invariant_primitives: Optional[np.ndarray] = (
        None  # (N,) 每个 instance 的翻转不变性
    )
    sizes: Optional[np.ndarray] = None  # (N, 2) 每个 instance 的大小 (width, height)
    # ============ 背景数据 (instance_id == -1 的 primitives) ============
    # Point-level 背景数据
    bg_coords: Optional[np.ndarray] = None  # (M_bg, 4) 背景点坐标
    bg_colors: Optional[np.ndarray] = None  # (M_bg, 3) 背景点颜色
    bg_widths: Optional[np.ndarray] = None  # (M_bg,) 背景点宽度
    bg_prim_ids: Optional[np.ndarray] = None  # (M_bg,) 背景点的primitive_id
    bg_layer_ids: Optional[np.ndarray] = None  # (M_bg,) 背景点的layer_id
    # Primitive-level 背景数据
    bg_semantic_ids: Optional[np.ndarray] = None  # (P_bg,) 背景各primitive的semantic_id
    bg_instance_ids: Optional[np.ndarray] = None  # (P_bg,) 背景各primitive的instance_id
    bg_primitive_lengths: Optional[np.ndarray] = None  # (P_bg,) 背景各primitive的长度

    def num_instances(self) -> int:
        """返回 instance 数量"""
        return len(self.instance_coords)

    def is_empty(self) -> bool:
        """判断是否没有任何 instance"""
        return len(self.instance_coords) == 0


"""
for group in groups:
    # Skip empty layer
    if len(group) == 0:
        continue
    for primitive in group:
        # ------------- sample primitives ------------ #
        sampled_coords, prim_width, prim_color, prim_length = parse_primitive(
            primitive, line_t_values, curve_t_values, max_length, bbox, connect_lines)
        if len(sampled_coords) == 0:
            continue
        # ------------------ get ids ----------------- #
        # In FloorPlanCAD dataset, semanticId starts from 1 to 35, and instanceId starts from 1,
        # instanceId=-1 means uncountable semantic.
        # we define semanticId=36 for background and instanceId=-1 for background/uncountable
        # semantic_id = int(primitive.attrib.get("semanticId", 36))
        semantic_id = int(primitive.attrib.get("semanticId", 36))
        if semantic_id == 36 or semantic_id == -1: # background semantic
            instance_id = -1
        else: # valid semantic
            instance_id = int(primitive.attrib.get("instanceId", -1))
        semantic_id -= 1  # shift id from [1, 36] to [0, 35] for better compatibility
        # ---------------- append data --------------- #
        for coord in sampled_coords:
            svg_data.coords.append(coord)
            svg_data.colors.append(prim_color)
            svg_data.widths.append(prim_width)
            svg_data.primitive_ids.append(primitive_id)
            svg_data.layer_ids.append(layer_id)
        svg_data.semantic_ids.append(semantic_id)
        svg_data.instance_ids.append(instance_id)
        svg_data.primitive_lengths.append(prim_length)
        # -------------------------------------------- #
        primitive_id += 1
    layer_id += 1
"""
