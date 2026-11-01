import torch.nn as nn


class AddNorm(nn.Module):

    def __init__(self, embed_dim: int, dropout: float):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, residual):
        return self.ln(input + self.dropout(residual))
