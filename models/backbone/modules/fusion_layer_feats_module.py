import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter


class FusionLayerFeatsModule(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int):
        super(FusionLayerFeatsModule, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.fc1 = nn.Linear(self.input_dim * 3, self.embed_dim)
        self.fc2 = nn.Linear(self.embed_dim, self.input_dim)
        self.attention = nn.Linear(self.input_dim, 1)
        self.fc_concat = nn.Linear(self.input_dim * 2, self.input_dim)
        self.gelu = nn.GELU()

    def forward(
        self, feats: torch.Tensor, cu_seqlens: torch.Tensor, layer_ids: torch.Tensor
    ) -> torch.Tensor:
        new_feats = torch.zeros_like(feats)
        assert feats.shape[0] == layer_ids.shape[0]

        for batch_idx in range(len(cu_seqlens) - 1):
            idx_start, idx_end = cu_seqlens[batch_idx], cu_seqlens[batch_idx + 1]
            layer_ids_batch = layer_ids[idx_start:idx_end]
            feats_batch = feats[idx_start:idx_end]

            avg_pool = scatter(
                feats_batch, layer_ids_batch.long(), dim=0, reduce="mean"
            )  # shape (num_layers, input_dim)
            max_pool = scatter(
                feats_batch, layer_ids_batch.long(), dim=0, reduce="max"
            )  # shape (num_layers, input_dim)
            attention_weights = F.softmax(
                self.attention(feats_batch), dim=0
            )  # shape (N, )
            weighted_feats = torch.mul(
                feats_batch, attention_weights.expand_as(feats_batch)
            )  # shape (N, input_dim)
            attention_pool = scatter(
                weighted_feats, layer_ids_batch.long(), dim=0, reduce="sum"
            )  # shape (num_layers, input_dim)

            feats_concat = torch.cat(
                (avg_pool, max_pool, attention_pool), dim=1
            )  # shape (num_layers, input_dim * 3)
            new_layer_feats = self.fc1(feats_concat)  # shape (num_layers, embed_dim)
            new_layer_feats = self.gelu(
                new_layer_feats
            )  # shape (num_layers, embed_dim)
            new_layer_feats = self.fc2(new_layer_feats)  # shape (num_layers, input_dim)
            # Expand new_layer_feats to the shape of feats_batch, following the order of layer_ids_batch
            new_layer_feats = new_layer_feats.index_select(
                0, layer_ids_batch.long()
            )  # shape (N, input_dim)
            feats_concat = torch.cat(
                (new_layer_feats, feats_batch), dim=1
            )  # shape (N, 2 * input_dim)
            new_feats[idx_start:idx_end] = self.fc_concat(
                feats_concat
            )  # shape (N, input_dim)
        return new_feats
