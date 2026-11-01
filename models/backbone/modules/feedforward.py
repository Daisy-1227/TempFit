import torch.nn as nn


class FFN(nn.Module):

    def __init__(self, embed_dim: int, activation: str, dropout: float):
        super().__init__()
        self.up_proj = nn.Linear(embed_dim, embed_dim * 4)
        self.activation = getattr(nn, activation)()
        self.dropout = nn.Dropout(dropout)
        self.down_proj = nn.Linear(embed_dim * 4, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.up_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        x = self.norm(x)
        return x
