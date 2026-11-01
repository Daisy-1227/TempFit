import torch
import numpy as np
from torch_scatter import scatter
from datasets.augment_utils import (
    random_flip,
    random_rotate,
    random_scale,
    random_translation,
)

from datasets.dataclass import (
    SVGData,
    SVGDataTensor,
    VecData,
    VecDataTransformArgs,
    InstanceData,
)

# from .vec_data.floorplancad.dataclass_define import SVGData
from typing import Dict, Any, Tuple

# from .augment_utils import (
#     random_flip,
#     random_rotate,
#     random_scale,
#     random_translation
# )


def to_tensor(data: SVGData) -> SVGDataTensor:
    """
    Convert SVGData to SVGDataTensor

    Args:

        `data`: SVGData

    Returns:

        `SVGDataTensor`
    """
    try:
        return SVGDataTensor(
            viewBox=torch.tensor(data.viewBox, dtype=torch.float32),
            coords=torch.tensor(data.coords, dtype=torch.float32),
            colors=torch.tensor(data.colors, dtype=torch.int32),
            widths=torch.tensor(data.widths, dtype=torch.float32),
            primitive_ids=torch.tensor(data.primitive_ids, dtype=torch.int32),
            layer_ids=torch.tensor(data.layer_ids, dtype=torch.int32),
            semantic_ids=torch.tensor(data.semantic_ids, dtype=torch.int32),
            instance_ids=torch.tensor(data.instance_ids, dtype=torch.int32),
            primitive_lengths=torch.tensor(data.primitive_lengths, dtype=torch.float32),
        )
    except Exception as e:
        raise ValueError(f"Failed to convert data to tensor: {e}")


def merge_prims_coords(
    coords: list[list[list[float]]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Merge the coordinates of primitives

    Args:

        `coords` (`list[list[list[float]]]`): N * P * [x, y], N is the number of primitives, P is the number of points of each primitive,
            but the number of points of each primitive is not the same, so we need to merge them, and get
            the primitive ids of each point

    Returns:

        tuple of tensors (coords, prim_ids, num_points_per_prim),
            P is the total number of points, N is the number of primitives

            `coords` (`torch.Tensor`): shape is (P, 2), P * [x, y]

            `prim_ids` (`torch.Tensor`): shape is (P,), primitive_id

            `num_points_per_prim` (`torch.Tensor`): shape is (N,), number of points of each primitive
    """
    list_coords_tensor = [torch.tensor(coord, dtype=torch.float32) for coord in coords]
    num_points_per_prim = torch.tensor(
        [len(coord) for coord in list_coords_tensor], dtype=torch.int32
    )
    prim_ids = torch.repeat_interleave(
        torch.arange(len(num_points_per_prim)), num_points_per_prim
    )
    coords_tensor = torch.cat(list_coords_tensor, dim=0)
    return coords_tensor, prim_ids, num_points_per_prim


def norm_coords(
    coords: torch.Tensor, bbox: torch.Tensor, min_val: float, max_val: float
) -> torch.Tensor:
    """
    Normalize coordinates to range [min_val, max_val] based on bbox.

    Args:

        `coords`: Tensor of shape (..., 2), ... * [x, y]

        `bbox`: Tensor of shape (4,), [minx, miny, width, height]

        `max_val`: float, maximum value of the normalized coordinates

        `min_val`: float, minimum value of the normalized coordinates

    Returns:

        `torch.Tensor`: Normalized coordinates tensor in range [min_val, max_val]
    """
    # Calculate minx, maxx, miny, maxy if not provided
    minx, miny, width, height = bbox

    # Handle degenerate cases
    if width == 0 or height == 0:
        raise ValueError(
            f"Cannot scale: width or height is zero, bbox: {bbox.tolist()}"
        )
    if min_val >= max_val:
        raise ValueError(
            f"min_val must be less than max_val, min_val: {min_val}, max_val: {max_val}"
        )

    # Normalize coordinates in-place
    coords[..., 0] = (coords[..., 0] - minx) / width * (max_val - min_val) + min_val
    coords[..., 1] = (coords[..., 1] - miny) / height * (max_val - min_val) + min_val

    return coords


def augment_line_args(
    coords: torch.Tensor,
    gt_data: Dict[str, Any],
    min_val: float,
    max_val: float,
    transform_args: VecDataTransformArgs,
) -> torch.Tensor:
    """
    Augment line arguments

    Args:

        `coords`: Tensor of shape (..., 2), ... * [x, y]

        `transform_args`: VecDataTransformArgs

    Returns:

        `torch.Tensor`: Augmented coordinates tensor
    """
    # Random vertical flip

    coords = random_flip(
        coords=coords,
        gt_data=gt_data,
        min_val=min_val,
        max_val=max_val,
        flip_type="vertical",
        flip_prob=transform_args.random_vertical_flip,
    )
    # Random horizontal flip
    coords = random_flip(
        coords=coords,
        gt_data=gt_data,
        min_val=min_val,
        max_val=max_val,
        flip_type="horizontal",
        flip_prob=transform_args.random_horizontal_flip,
    )
    # Random rotate
    coords = random_rotate(
        coords=coords,
        gt_data=gt_data,
        min_val=min_val,
        max_val=max_val,
        if_rotate=transform_args.random_rotate,
    )
    # Random scale
    coords = random_scale(
        coords=coords,
        gt_data=gt_data,
        min_val=min_val,
        max_val=max_val,
        min_scale_ratio=transform_args.random_scale[0],
        max_scale_ratio=transform_args.random_scale[1],
    )
    # Random translation
    coords = random_translation(
        coords=coords,
        gt_data=gt_data,
        x_translation_ratio=transform_args.random_translation[0],
        y_translation_ratio=transform_args.random_translation[1],
    )

    return coords


def to_vec_data(svg_data_tensor: SVGDataTensor) -> VecData:
    """
    Convert SVGDataTensor to VecData
    """
    coords = get_coords(svg_data_tensor.coords)
    feats = get_feats(
        svg_data_tensor.coords, svg_data_tensor.primitive_ids, svg_data_tensor.colors
    )
    if coords.shape[-1] == 4:
        coords, feats = feats[..., [3, 4]], feats[..., :7]
    else:
        coords, feats = coords[..., :], feats[..., :4]
    return VecData(
        data_path="",
        coords=coords,
        feats=feats,
        prim_ids=svg_data_tensor.primitive_ids,
        layer_ids=svg_data_tensor.layer_ids,
        sem_ids=svg_data_tensor.semantic_ids,
        inst_ids=svg_data_tensor.instance_ids,
        prim_lengths=svg_data_tensor.primitive_lengths,
    )


def get_coords(coords: torch.Tensor) -> torch.Tensor:
    """
    This func helps to compute the coordinates of points or lines.

    Args:

        `coords` (`torch.Tensor`, shape is (N, 2) or (N, 4), N * [x, y] or N * [x1, y1, x2, y2])
            N is the number of points or lines

    Returns:

        `coords` (`torch.Tensor`, shape is (N, 2) or (N, 4)):
            N is the number of points or lines
        if `connect_lines` is True:
            coords:  N * [x_1, y_1, x_2, y_2] if (x_1 < x_2) or (x_1==x_2 and y_1 <= y_2)
                else N * [x_2, y_2, x_1, y_1]
        else:
            coords: N * [x, y]
    """
    if coords.shape[-1] == 4:  # if is line coords, standardize the order of points
        mask = (coords[..., 0] < coords[..., 2]) | (
            (coords[..., 0] == coords[..., 2]) & (coords[..., 1] <= coords[..., 3])
        )
        coords = torch.where(
            torch.stack([mask] * 4, dim=-1), coords, coords[..., [2, 3, 0, 1]]
        )
    return coords


def get_feats(
    coords: torch.Tensor, primitive_id: torch.Tensor, line_color: torch.Tensor
) -> torch.Tensor:
    """
    This func helps to compute the features of points or lines.

    Args:

        `coords`: Tensor of shape (N, 2) or (N, 4), N * [x, y] or N * [x1, y1, x2, y2]
            N is the number of points or lines

        `primitive_id`: Tensor of shape (N,), primitive_id

        `line_color`: Tensor of shape (N, 3), N * [r, g, b]

    Returns:

        `feats`: line mode: Tensor of shape (N, 10), N * [length, |dx|, |dy|, cx, cy, pcx, pcy, color_r, color_g, color_b]
            point mode: Tensor of shape (N, 7), N * [cx, cy, pcx, pcy, color_r, color_g, color_b]

            \\- `length`: sqrt((x_2 - x_1)^2 + (y_2 - y_1)^2), length of line segment

            \\- `|dx|`: abs(x_2 - x_1), absolute difference of x coordinates

            \\- `|dy|`: abs(y_2 - y_1), absolute difference of y coordinates

            \\- `cx`: if is line segment, (x_1 + x_2) / 2, center x coordinate of line segment
                if is point, x coordinate of point

            \\- `cy`: if is line segment, (y_1 + y_2) / 2, center y coordinate of line segment
                if is point, y coordinate of point

            \\- `pcx`: primitive center x coordinates

            \\- `pcy`: primitive center y coordinates

            \\- `color_r`: red channel of line color

            \\- `color_g`: green channel of line color

            \\- `color_b`: blue channel of line color
    """
    cx, cy, pcx, pcy = get_centers(coords, primitive_id)
    color_r, color_g, color_b = get_colors(line_color)
    if coords.shape[-1] == 4:
        length = torch.sqrt(
            torch.sum(torch.square(coords[..., :2] - coords[..., 2:4]), dim=-1)
        )
        vec = coords[..., 2:4] - coords[..., 0:2]  # (..., 2)
        length_ = torch.norm(vec, dim=-1, keepdim=True) + 1e-8
        unit_vec = vec / length_  # (..., 2)
        dx, dy = unit_vec[..., 0], unit_vec[..., 1]
        feats = torch.stack(
            [length, dx, dy, cx, cy, pcx, pcy, color_r, color_g, color_b], dim=-1
        )
    else:
        feats = torch.stack([cx, cy, pcx, pcy, color_r, color_g, color_b], dim=-1)
    return feats


def get_centers(
    coords: torch.Tensor, primitive_id: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    A primitive will be sampled as multiple points or line segments.
    This func helps to compute the center of the primitive
    from which each point or line segment is sampled.

    Args:

        `coords`: Tensor of shape (N, 2) or (N, 4), N * [x, y] or N * [x1, y1, x2, y2]

        `primitive_id`: Tensor of shape (N,) containing primitive IDs

    Returns:

        tuple of tensors (cx, cy, pcx, pcy), each of shape (N,):

            `cx`: if is line segment, (x_1 + x_2) / 2, center x coordinate of line segment
                if is point, x coordinate of point

            `cy`: if is line segment, (y_1 + y_2) / 2, center y coordinate of line segment
                if is point, y coordinate of point

            `pcx`: primitive center x coordinates

            `pcy`: primitive center y coordinates

    Raises:
        ValueError: If input tensors have invalid shapes or types
    """
    # Calculate centers of points or line segments
    if coords.shape[-1] == 4:
        cx = torch.mean(coords[..., [0, 2]], dim=-1)
        cy = torch.mean(coords[..., [1, 3]], dim=-1)
    else:
        cx, cy = coords[..., 0], coords[..., 1]

    # Calculate primitive centers
    pcx = scatter(cx, primitive_id.long(), dim=0, reduce="mean")
    pcy = scatter(cy, primitive_id.long(), dim=0, reduce="mean")

    # broadcast pcx and pcy to the same shape as cx and cy
    pcx = pcx[primitive_id]
    pcy = pcy[primitive_id]

    return cx, cy, pcx, pcy


def get_colors(
    line_color: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    This func helps to compute the colors of line segments.
    """
    # normalize line_color to [-0.5, 0.5]
    line_color = line_color / 255.0 - 0.5
    color_r, color_g, color_b = (
        line_color[..., 0],
        line_color[..., 1],
        line_color[..., 2],
    )
    return color_r, color_g, color_b


# ==================== Instance-level 数据转换函数 ==================== #


def svg_to_instance(
    svg_data: SVGData,
    model_data: Dict[str, Any],
) -> InstanceData:
    """
    将 SVGData + model_data 解耦为 InstanceData

    数据结构说明：
    - Point-level (采样点级别): coords, colors, widths, primitive_ids, layer_ids
    - Primitive-level (元素级别): semantic_ids, instance_ids, primitive_lengths
    - 映射关系: primitive_ids[point_idx] -> primitive 索引

    Args:
        svg_data: 包含 point-level 和 primitive-level 数据的 SVGData 对象
        model_data: 包含 instance-level 属性的字典 (来自 GT 文件)

    Returns:
        InstanceData: 解耦后的 instance 级别数据
    """
    # Point-level 数据转换为 numpy 数组
    coords = np.array(svg_data.coords, dtype=np.float32)  # (N_points, 4)
    colors = np.array(svg_data.colors, dtype=np.int32)  # (N_points, 3)
    widths = np.array(svg_data.widths, dtype=np.float32)  # (N_points,)
    point_prim_ids = np.array(
        svg_data.primitive_ids, dtype=np.int32
    )  # (N_points,) 每个点属于哪个primitive
    layer_ids = np.array(svg_data.layer_ids, dtype=np.int32)  # (N_points,)

    # Primitive-level 数据转换为 numpy 数组
    prim_semantic_ids = np.array(
        svg_data.semantic_ids, dtype=np.int32
    )  # (N_primitives,)
    prim_instance_ids = np.array(
        svg_data.instance_ids, dtype=np.int32
    )  # (N_primitives,)
    prim_lengths = np.array(
        svg_data.primitive_lengths, dtype=np.float32
    )  # (N_primitives,)

    num_primitives = len(prim_instance_ids)

    # 分离背景 (instance_id == -1) 和前景 primitives
    bg_prim_mask = prim_instance_ids == -1
    fg_prim_mask = ~bg_prim_mask

    # 获取背景 primitives 的索引
    bg_prim_indices = np.where(bg_prim_mask)[0]
    # 获取背景点: 点的 primitive_id 在背景 primitive 索引中
    if len(bg_prim_indices) > 0:
        bg_point_mask = np.isin(point_prim_ids, bg_prim_indices)
    else:
        bg_point_mask = np.zeros(len(coords), dtype=bool)

    # 背景点数据
    if bg_point_mask.any():
        bg_coords = coords[bg_point_mask]
        bg_colors = colors[bg_point_mask]
        bg_widths = widths[bg_point_mask]
        bg_point_prim_ids = point_prim_ids[bg_point_mask]
        bg_layer_ids = layer_ids[bg_point_mask]
        # 背景 primitives 的 semantic_ids
        bg_prim_semantic_ids = prim_semantic_ids[bg_prim_mask]
        bg_prim_instance_ids = prim_instance_ids[bg_prim_mask]
        bg_prim_lengths = prim_lengths[bg_prim_mask]
    else:
        bg_coords = None
        bg_colors = None
        bg_widths = None
        bg_point_prim_ids = None
        bg_layer_ids = None
        bg_prim_semantic_ids = None
        bg_prim_instance_ids = None
        bg_prim_lengths = None

    # 前景数据：按 instance_id 分组
    instance_coords_list = []
    instance_colors_list = []
    instance_widths_list = []
    instance_prim_ids_list = []
    instance_layer_ids_list = []
    # Primitive-level per instance
    instance_prim_semantic_ids_list = []
    instance_prim_instance_ids_list = []
    instance_prim_lengths_list = []

    if fg_prim_mask.any():
        fg_prim_instance_ids = prim_instance_ids[fg_prim_mask]
        unique_instances = np.unique(fg_prim_instance_ids)
        unique_instances = np.sort(unique_instances)  # 确保顺序一致

        for inst_id in unique_instances:
            # 找到属于该 instance 的所有 primitives
            inst_prim_mask = prim_instance_ids == inst_id
            inst_prim_indices = np.where(inst_prim_mask)[0]

            # 找到属于这些 primitives 的所有点
            inst_point_mask = np.isin(point_prim_ids, inst_prim_indices)

            if inst_point_mask.any():
                # 重新映射 primitive_ids: 使其从 0 开始连续编号
                inst_coords = coords[inst_point_mask]
                inst_colors = colors[inst_point_mask]
                inst_widths = widths[inst_point_mask]
                inst_point_prim_ids = point_prim_ids[inst_point_mask]
                inst_layer_ids = layer_ids[inst_point_mask]

                # 创建从原始 primitive index 到新 index 的映射
                old_to_new = {
                    old_idx: new_idx
                    for new_idx, old_idx in enumerate(inst_prim_indices)
                }
                inst_point_prim_ids_remapped = np.array(
                    [old_to_new[pid] for pid in inst_point_prim_ids], dtype=np.int32
                )

                instance_coords_list.append(inst_coords)
                instance_colors_list.append(inst_colors)
                instance_widths_list.append(inst_widths)
                instance_prim_ids_list.append(inst_point_prim_ids_remapped)
                instance_layer_ids_list.append(inst_layer_ids)

                # Primitive-level 数据
                instance_prim_semantic_ids_list.append(
                    prim_semantic_ids[inst_prim_mask]
                )
                instance_prim_instance_ids_list.append(
                    prim_instance_ids[inst_prim_mask]
                )
                instance_prim_lengths_list.append(prim_lengths[inst_prim_mask])

    # 从 model_data 获取 instance-level 属性 (来自 GT)
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.numpy()
        elif isinstance(x, np.ndarray):
            return x
        else:
            return np.array(x)

    gt_primitive_ids = to_numpy(
        model_data.get("primitive_id", np.array([], dtype=np.int32))
    )
    gt_semantic_ids = to_numpy(
        model_data.get("semantic_id", np.array([], dtype=np.int32))
    )
    gt_translations = to_numpy(
        model_data.get("translation", np.empty((0, 2), dtype=np.float32))
    )
    gt_rotations = to_numpy(model_data.get("rotation", np.array([], dtype=np.float32)))
    gt_scales = to_numpy(model_data.get("scale", np.empty((0, 2), dtype=np.float32)))
    gt_flip_modes = to_numpy(model_data.get("flip_mode", np.array([], dtype=np.int32)))
    gt_flip_invariant_primitives = to_numpy(
        model_data.get("flip_invariant_primitive", np.array([], dtype=np.bool))
    )
    gt_sizes = to_numpy(model_data.get("size", np.empty((0, 2), dtype=np.float32)))
    return InstanceData(
        viewBox=np.array(svg_data.viewBox, dtype=np.float32),
        # Per-instance 的点级数据
        instance_coords=instance_coords_list,
        instance_colors=instance_colors_list,
        instance_widths=instance_widths_list,
        instance_prim_ids=instance_prim_ids_list,
        instance_layer_ids=instance_layer_ids_list,
        # Per-instance 的 primitive-level 数据
        instance_semantic_ids=instance_prim_semantic_ids_list,
        instance_instance_ids=instance_prim_instance_ids_list,
        instance_primitive_lengths=instance_prim_lengths_list,
        # GT 的 instance-level 属性
        primitive_ids=gt_primitive_ids,
        semantic_ids=gt_semantic_ids,
        translations=gt_translations,
        rotations=gt_rotations,
        scales=gt_scales,
        flip_modes=gt_flip_modes,
        flip_invariant_primitives=gt_flip_invariant_primitives,
        sizes=gt_sizes,
        # 背景数据
        bg_coords=bg_coords,
        bg_colors=bg_colors,
        bg_widths=bg_widths,
        bg_prim_ids=bg_point_prim_ids,
        bg_layer_ids=bg_layer_ids,
        bg_semantic_ids=bg_prim_semantic_ids,
        bg_instance_ids=bg_prim_instance_ids,
        bg_primitive_lengths=bg_prim_lengths,
    )


def instance_to_svg(instance_data: InstanceData) -> Tuple[SVGData, Dict[str, Any]]:
    """
    将 InstanceData 合并回 SVGData 格式

    注意 SVGData 的两层结构:
    - Point-level: coords, colors, widths, primitive_ids, layer_ids
    - Primitive-level: semantic_ids, instance_ids, primitive_lengths

    Args:
        instance_data: instance 级别的数据

    Returns:
        Tuple[SVGData, Dict]: 合并后的 SVGData 和更新后的 model_data
    """
    # Point-level 数据
    all_coords = []
    all_colors = []
    all_widths = []
    all_prim_ids = []  # 需要重新编号
    all_layer_ids = []

    # Primitive-level 数据
    all_prim_semantic_ids = []
    all_prim_instance_ids = []
    all_prim_lengths = []

    current_prim_offset = 0  # 用于重新编号 primitive_ids

    # 合并前景 instance 数据
    num_instances = len(instance_data.instance_coords)
    for i in range(num_instances):
        inst_coords = instance_data.instance_coords[i]
        inst_colors = instance_data.instance_colors[i]
        inst_widths = instance_data.instance_widths[i]
        inst_prim_ids = instance_data.instance_prim_ids[i]
        inst_layer_ids = instance_data.instance_layer_ids[i]

        # 计算该 instance 内有多少个 primitives
        if len(inst_prim_ids) > 0:
            num_prims_in_inst = inst_prim_ids.max() + 1
        else:
            num_prims_in_inst = 0

        # 重新编号 primitive_ids
        remapped_prim_ids = inst_prim_ids + current_prim_offset

        all_coords.append(inst_coords)
        all_colors.append(inst_colors)
        all_widths.append(inst_widths)
        all_prim_ids.append(remapped_prim_ids)
        all_layer_ids.append(inst_layer_ids)

        # Primitive-level: 该 instance 的所有 primitive 都属于同一个 instance
        # instance_semantic_ids 存储的是该 instance 内各 primitive 的 semantic_id
        if (
            len(instance_data.instance_semantic_ids) > i
            and len(instance_data.instance_semantic_ids[i]) > 0
        ):
            inst_prim_semantic_ids = instance_data.instance_semantic_ids[i]
        else:
            # 如果没有，使用 GT 的 semantic_id
            inst_prim_semantic_ids = np.full(
                num_prims_in_inst,
                (
                    instance_data.semantic_ids[i]
                    if i < len(instance_data.semantic_ids)
                    else 0
                ),
                dtype=np.int32,
            )

        all_prim_semantic_ids.append(inst_prim_semantic_ids)
        all_prim_instance_ids.append(np.full(num_prims_in_inst, i, dtype=np.int32))

        # Primitive-level: 使用存储的原始 primitive_lengths
        if (
            len(instance_data.instance_primitive_lengths) > i
            and len(instance_data.instance_primitive_lengths[i]) > 0
        ):
            all_prim_lengths.append(
                instance_data.instance_primitive_lengths[i].astype(np.float32)
            )
        else:
            # fallback: 用点数作为 primitive_lengths
            prim_lengths_in_inst = []
            for prim_idx in range(num_prims_in_inst):
                prim_lengths_in_inst.append(np.sum(inst_prim_ids == prim_idx))
            all_prim_lengths.append(np.array(prim_lengths_in_inst, dtype=np.float32))

        current_prim_offset += num_prims_in_inst

    # 合并背景数据 (instance_id = -1)
    if instance_data.bg_coords is not None and len(instance_data.bg_coords) > 0:
        bg_prim_ids = instance_data.bg_prim_ids

        # 计算背景 primitives 数量
        if len(bg_prim_ids) > 0:
            # 重新编号背景的 primitive_ids
            unique_bg_prims = np.unique(bg_prim_ids)
            bg_old_to_new = {
                old_idx: new_idx + current_prim_offset
                for new_idx, old_idx in enumerate(unique_bg_prims)
            }
            remapped_bg_prim_ids = np.array(
                [bg_old_to_new[pid] for pid in bg_prim_ids], dtype=np.int32
            )
            num_bg_prims = len(unique_bg_prims)
        else:
            remapped_bg_prim_ids = bg_prim_ids
            num_bg_prims = 0

        all_coords.append(instance_data.bg_coords)
        all_colors.append(instance_data.bg_colors)
        all_widths.append(instance_data.bg_widths)
        all_prim_ids.append(remapped_bg_prim_ids)
        all_layer_ids.append(instance_data.bg_layer_ids)

        # 背景 primitives 的 semantic_ids 和 instance_ids
        if instance_data.bg_semantic_ids is not None:
            all_prim_semantic_ids.append(instance_data.bg_semantic_ids)
        else:
            all_prim_semantic_ids.append(np.full(num_bg_prims, -1, dtype=np.int32))
        all_prim_instance_ids.append(np.full(num_bg_prims, -1, dtype=np.int32))

        # 背景 primitive_lengths: 使用存储的原始值
        if (
            instance_data.bg_primitive_lengths is not None
            and len(instance_data.bg_primitive_lengths) > 0
        ):
            all_prim_lengths.append(
                instance_data.bg_primitive_lengths.astype(np.float32)
            )
        else:
            # fallback: 用点数作为 primitive_lengths
            bg_prim_lengths = []
            for prim_idx in unique_bg_prims:
                bg_prim_lengths.append(np.sum(bg_prim_ids == prim_idx))
            all_prim_lengths.append(np.array(bg_prim_lengths, dtype=np.float32))

    # Concatenate Point-level 数据
    if len(all_coords) > 0:
        all_coords = np.concatenate(all_coords, axis=0)
        all_colors = np.concatenate(all_colors, axis=0)
        all_widths = np.concatenate(all_widths, axis=0)
        all_prim_ids = np.concatenate(all_prim_ids, axis=0)
        all_layer_ids = np.concatenate(all_layer_ids, axis=0)
    else:
        all_coords = np.empty((0, 4), dtype=np.float32)
        all_colors = np.empty((0, 3), dtype=np.int32)
        all_widths = np.empty((0,), dtype=np.float32)
        all_prim_ids = np.empty((0,), dtype=np.int32)
        all_layer_ids = np.empty((0,), dtype=np.int32)

    # Concatenate Primitive-level 数据
    if len(all_prim_semantic_ids) > 0:
        all_prim_semantic_ids = np.concatenate(all_prim_semantic_ids, axis=0)
        all_prim_instance_ids = np.concatenate(all_prim_instance_ids, axis=0)
        all_prim_lengths = np.concatenate(all_prim_lengths, axis=0)
    else:
        all_prim_semantic_ids = np.empty((0,), dtype=np.int32)
        all_prim_instance_ids = np.empty((0,), dtype=np.int32)
        all_prim_lengths = np.empty((0,), dtype=np.float32)

    svg_data = SVGData(
        viewBox=instance_data.viewBox.tolist(),
        # Point-level
        coords=all_coords.tolist(),
        colors=all_colors.tolist(),
        widths=all_widths.tolist(),
        primitive_ids=all_prim_ids.tolist(),
        layer_ids=all_layer_ids.tolist(),
        # Primitive-level
        semantic_ids=all_prim_semantic_ids.tolist(),
        instance_ids=all_prim_instance_ids.tolist(),
        primitive_lengths=all_prim_lengths.tolist(),
    )

    # 构建更新后的 model_data
    model_data = {
        "translation": instance_data.translations,
        "rotation": instance_data.rotations,
        "scale": instance_data.scales,
        "flip_mode": instance_data.flip_modes,
        "primitive_id": instance_data.primitive_ids,
        "semantic_id": instance_data.semantic_ids,
        "flip_invariant_primitive": instance_data.flip_invariant_primitives,
        "size": (
            instance_data.sizes
            if instance_data.sizes is not None
            else np.empty((0, 2), dtype=np.float32)
        ),
    }

    return svg_data, model_data
