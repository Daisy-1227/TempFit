import math
import torch
from torch import nn


class TargetSelector(nn.Module):
    def __init__(self, dim=768, hidden=256, match_dim=128):
        super().__init__()

        def projection():
            return nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, match_dim),
            )

        self.query = projection()
        self.key = projection()
        self.position = nn.Sequential(
            nn.Linear(2, 64), nn.GELU(), nn.Linear(64, match_dim)
        )
        self.offset = nn.Parameter(torch.zeros(()))
        self.match_dim = match_dim

    def keys(self, features, coords):
        with torch.autocast(features.device.type, enabled=False):
            return self.key(features.float()) + self.position(coords.float())

    def forward(self, queries, features=None, coords=None, keys=None):
        with torch.autocast(queries.device.type, enabled=False):
            keys = self.keys(features, coords) if keys is None else keys
            return (
                torch.einsum("bid,bkd->bik", self.query(queries.float()), keys)
                / math.sqrt(self.match_dim)
                + self.offset
            )
