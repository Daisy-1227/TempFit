import torch
import torch.nn as nn
from torch_scatter import scatter

from models.backbone.modules.addnorm import AddNorm


class GroupFeatFusion(nn.Module):

    def __init__(self, embed_dim: int, dropout: float):
        super().__init__()
        self.add_norm = AddNorm(embed_dim, dropout)

    def forward(self, feats, group_id_map, cu_seqlens):
        """
        Args:
            `feats` (`torch.Tensor`, shape is (N1+N2+..., embed_dim)): Features of sequence

            `group_id_map` (`torch.Tensor`, shape is (N1+N2+...,)): Group id map, ensure id is non-negative

            `cu_seqlens` (`torch.Tensor`, shape is (batch_size + 1,)): Cumulative sequence lengths
        """
        fused_feats = torch.zeros_like(feats)
        for batch_idx in range(len(cu_seqlens) - 1):
            idx_start = cu_seqlens[batch_idx]
            idx_end = cu_seqlens[batch_idx + 1]
            group_id_map_batch = group_id_map[idx_start:idx_end]
            feats_batch = feats[idx_start:idx_end]
            pooled_feats = self._group_pooling(feats_batch, group_id_map_batch)
            broadcasted_pooled_feats = self._broadcast_pooled_feats(
                pooled_feats, group_id_map_batch
            )
            feats_batch = self.add_norm(broadcasted_pooled_feats, feats_batch)
            fused_feats[idx_start:idx_end] = feats_batch
        return fused_feats

    @staticmethod
    def _group_pooling(feats, group_id_map):
        """
        Args:
            `feats` (`torch.Tensor`, shape is (N, embed_dim))

            `group_id_map` (`torch.Tensor`, shape is (N,))
        """
        max_prim_feats = scatter(feats, group_id_map.long(), dim=0, reduce="max")
        mean_prim_feats = scatter(feats, group_id_map.long(), dim=0, reduce="mean")
        return max_prim_feats + mean_prim_feats

    @staticmethod
    def _broadcast_pooled_feats(pooled_feats, group_id_map):
        """
        Args:
            `pooled_feats` (`torch.Tensor`, shape is (Grouped_N, embed_dim))

            `group_id_map` (`torch.Tensor`, shape is (N,))
        """
        N = group_id_map.shape[0]
        E = pooled_feats.shape[1]
        broadcasted = torch.zeros(
            size=(N, E), dtype=pooled_feats.dtype, device=pooled_feats.device
        )
        idxs = torch.arange(N, device=pooled_feats.device)
        broadcasted[idxs] = pooled_feats[group_id_map[idxs]]
        return broadcasted
