import torch
from typing import Dict, Any


def random_flip(
    coords: torch.Tensor,
    gt_data: Dict[str, Any],
    min_val: float,
    max_val: float,
    flip_type: str,
    flip_prob: float,
) -> torch.Tensor:
    """
    Randomly flip the coordinates

    Args:

        `coords`: Tensor of coordinates to flip (..., 2), ... * [x, y]

        `max_val`: float, maximum value of the coordinate

        `min_val`: float, minimum value of the coordinate

        `flip_type`: str, "vertical" or "horizontal"

        `flip_prob`: float, probability of flipping

    Returns:

        `torch.Tensor`: Flipped coordinates tensor
    """
    if flip_prob < 0 or flip_prob > 1:
        raise ValueError(
            f"Invalid flip probability: {flip_prob}, must be in range [0, 1]"
        )
    if flip_type not in ["vertical", "horizontal"]:
        raise ValueError(
            f"Invalid flip type: {flip_type}, must be in ['vertical', 'horizontal']"
        )

    # if flip_prob is 0, do not flip
    if flip_prob == 0.0:
        return coords

    if flip_type == "vertical":
        if torch.rand(1) < flip_prob:
            coords[..., 1] = max_val + min_val - coords[..., 1]

    elif flip_type == "horizontal":
        if torch.rand(1) < flip_prob:
            coords[..., 0] = max_val + min_val - coords[..., 0]

    return coords


def random_rotate(
    coords: torch.Tensor, min_val: float, max_val: float, if_rotate: bool
) -> torch.Tensor:
    """
    Randomly rotate the coordinates

    Args:
        `coords`: Tensor of coordinates to rotate (..., 2), ... * [x, y]

        `min_val`: float, minimum value of the coordinate

        `max_val`: float, maximum value of the coordinate

        `if_rotate`: bool, whether to rotate

    Returns:

        `torch.Tensor`: Rotated coordinates tensor
    """
    # if if_rotate is False, do not rotate
    if not if_rotate:
        return coords

    # Generate random angle in radians
    theta = torch.rand(1, device=coords.device) * 2 * torch.pi

    # Create rotation matrix
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    rotation_matrix = torch.tensor([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

    # Move coords to start from (0, 0)
    coords = coords - (max_val + min_val) / 2

    # Apply rotation: (N, 2) = (N, 2) @ (2, 2)
    rotated_points = torch.matmul(coords, rotation_matrix.T)

    # Move back to original position
    coords = rotated_points + (max_val + min_val) / 2

    return coords


def random_scale(
    coords: torch.Tensor,
    min_val: float,
    max_val: float,
    min_scale_ratio: float,
    max_scale_ratio: float,
) -> torch.Tensor:
    """
    Randomly scale the coordinates

    Args:

        `coords`: Tensor of coordinates to scale (..., 2), ... * [x, y]

        `min_val`: float, minimum value of the coordinate

        `max_val`: float, maximum value of the coordinate

        `min_scale_ratio`: float, minimum scale ratio

        `max_scale_ratio`: float, maximum scale ratio

    Returns:

        `torch.Tensor`: Scaled coordinates tensor
    """
    # Validate scale ratios
    if min_scale_ratio < 0 or max_scale_ratio < 0:
        raise ValueError(
            f"Scale ratios must be positive, "
            f"min_scale_ratio: {min_scale_ratio}, max_scale_ratio: {max_scale_ratio}"
        )
    if min_scale_ratio > max_scale_ratio:
        raise ValueError(
            f"min_scale_ratio must be less than or equal to max_scale_ratio, "
            f"min_scale_ratio: {min_scale_ratio}, max_scale_ratio: {max_scale_ratio}"
        )

    # if min_scale_ratio and max_scale_ratio are both 1.0, do not scale
    if min_scale_ratio == 1.0 and max_scale_ratio == 1.0:
        return coords

    if min_scale_ratio == max_scale_ratio:
        scale_factor = min_scale_ratio
    else:
        # Generate random scale factor between min and max
        scale_factor = (
            torch.rand(1, device=coords.device) * (max_scale_ratio - min_scale_ratio)
            + min_scale_ratio
        )

    # Apply scaling
    coords = coords - (max_val + min_val) / 2
    coords = coords * scale_factor
    coords = coords + (max_val + min_val) / 2

    return coords


def random_translation(
    coords: torch.Tensor, x_translation_ratio: float, y_translation_ratio: float
) -> torch.Tensor:
    """
    Randomly translate the coordinates

    Args:
        `coords`: Tensor of coordinates to translate (..., 2), ... * [x, y]

        `x_translation_ratio`: float, x translation ratio, randomly translate from [-x_translation_ratio, x_translation_ratio]

        `y_translation_ratio`: float, y translation ratio, randomly translate from [-y_translation_ratio, y_translation_ratio]

    Returns:

        `torch.Tensor`: Translated coordinates tensor
    """
    # Validate translation ratios
    if x_translation_ratio < 0 or y_translation_ratio < 0:
        raise ValueError("Translation ratios must be non-negative")

    # if x_translation_ratio and y_translation_ratio are both 0, do not translate
    if x_translation_ratio == 0.0 and y_translation_ratio == 0.0:
        return coords

    # Generate random translation offsets
    x_offset = (
        torch.rand(1, device=coords.device) * 2 - 1
    ) * x_translation_ratio  # Random value in [-x_ratio, x_ratio]
    y_offset = (
        torch.rand(1, device=coords.device) * 2 - 1
    ) * y_translation_ratio  # Random value in [-y_ratio, y_ratio]

    # Create translation offsets tensor
    translation = torch.tensor([x_offset, y_offset], device=coords.device)

    # Apply translation
    return coords + translation
