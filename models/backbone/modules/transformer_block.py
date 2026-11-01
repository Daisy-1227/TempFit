import torch.nn as nn

from models.backbone.modules.feedforward import FFN
from models.backbone.modules.addnorm import AddNorm
from models.backbone.modules.attention import VarlenSelfAttentionWithRoPE


class TransformerBlock(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        attn_drop: float,
        activation: str,
        dropout: float,
        rope_dim: int = 4,
        rope_theta: float = 10000.0,
        rope_learnable: bool = True,
    ):
        super().__init__()
        self.attn = VarlenSelfAttentionWithRoPE(
            embed_dim, n_heads, attn_drop, dropout, rope_dim, rope_theta, rope_learnable
        )
        self.attn_norm = AddNorm(embed_dim, dropout)
        self.ffn = FFN(embed_dim, activation, dropout)
        self.ffn_norm = AddNorm(embed_dim, dropout)

    def forward(self, coords, feats, cu_seqlens):
        feats = self.attn_norm(self.attn(coords, feats, cu_seqlens), feats)
        feats = self.ffn_norm(self.ffn(feats), feats)
        return feats
