import torch
from torch_scatter import scatter

from datasets.vec_data.floorplancad.dataclass_define import (
    SVGData,
    SVGDataTensor,
    VecData,
    VecDataTransformArgs,
)
from datasets.vec_data.floorplancad.augment_utils import (
    random_flip,
    random_rotate,
    random_scale,
    random_translation,
)


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
        min_val=min_val,
        max_val=max_val,
        flip_type="vertical",
        flip_prob=transform_args.random_vertical_flip,
    )
    # Random horizontal flip
    coords = random_flip(
        coords=coords,
        min_val=min_val,
        max_val=max_val,
        flip_type="horizontal",
        flip_prob=transform_args.random_horizontal_flip,
    )
    # Random rotate
    coords = random_rotate(
        coords=coords,
        min_val=min_val,
        max_val=max_val,
        if_rotate=transform_args.random_rotate,
    )
    # Random scale
    coords = random_scale(
        coords=coords,
        min_val=min_val,
        max_val=max_val,
        min_scale_ratio=transform_args.random_scale[0],
        max_scale_ratio=transform_args.random_scale[1],
    )
    # Random translation
    coords = random_translation(
        coords=coords,
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
