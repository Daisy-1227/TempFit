import torch
import torch.nn as nn


class Projection(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: str,
        dropout: float,
        use_norm: bool = True,
        use_sigmoid: bool = False,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Linear(output_dim, output_dim * 4),
            getattr(nn, activation)(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 4, output_dim),
        )

        if use_norm:
            self.norm = nn.LayerNorm(output_dim)
        else:
            self.norm = lambda x: x

        if use_sigmoid:
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = lambda x: x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.norm(self.proj(x)))
