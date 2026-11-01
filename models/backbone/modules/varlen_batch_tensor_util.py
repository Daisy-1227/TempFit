from typing import List, Tuple

import torch


def list_tensor_to_cat_tensor(
    list_tensor: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a list of tensors to a concatenated tensor and a cumulative sequence length tensor

    Args:
        `list_tensor` (`List[torch.Tensor]`): A list of tensors

    Returns:

        `torch.Tensor`: A concatenated tensor

        `torch.Tensor`: A cumulative sequence length tensor
    """
    seq_lens = torch.tensor(
        [len(tensor) for tensor in list_tensor],
        dtype=torch.int32,
        device=list_tensor[0].device,
    )
    cu_seqlens = torch.cat(
        [
            torch.tensor([0], dtype=torch.int32, device=list_tensor[0].device),
            torch.cumsum(seq_lens, dim=0, dtype=torch.int32),
        ]
    )
    return torch.cat(list_tensor, dim=0), cu_seqlens


def cat_tensor_to_list_tensor(
    cat_tensor: torch.Tensor, cu_seqlens: torch.Tensor
) -> List[torch.Tensor]:
    """
    Convert a concatenated tensor and a cumulative sequence length tensor to a list of tensors

    Args:
        `cat_tensor` (`torch.Tensor`): A concatenated tensor

        `cu_seqlens` (`torch.Tensor`): A cumulative sequence length tensor

    Returns:
        A list of tensors
    """
    return list(torch.split(cat_tensor, (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()))
