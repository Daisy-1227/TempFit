import torch
import torch.nn as nn


class AbsolutePosEmbedding(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        coords_dim: int = 4,
        theta: float = 10000.0,
        learnable: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.coords_dim = coords_dim
        self.learnable = learnable
        self.theta = theta

        mag = self._get_sinusoidal_mag(theta)

        if learnable:
            self.mag = nn.Parameter(mag)
        else:
            self.register_buffer("mag", mag)

    @torch.no_grad()
    def _get_sinusoidal_mag(self, theta: float):
        """
        Get the sinusoidal magnitude

        Returns:

            `torch.Tensor`, shape is (embed_dim // coords_dim // 2): The sinusoidal magnitudes
        """
        embed_dim_each_coord = self.embed_dim // self.coords_dim
        mag = 1 / (
            theta
            ** (
                torch.arange(0, embed_dim_each_coord, 2)[
                    : (embed_dim_each_coord // 2)
                ].float()
                / self.embed_dim
            )
        )
        return mag

    def forward(self, feats: torch.Tensor, coords: torch.Tensor):
        """
        Add learnable absolute position embedding to features.

        Args:

            `feats` (`torch.Tensor`): Input features with shape (total_seq_len, embed_dim)

            `coords` (`torch.Tensor`): Coordinates with shape (total_seq_len, coords_dim)

        Returns:

            `torch.Tensor`, shape is (total_seq_len, embed_dim): The features with absolute position embedding
        """
        N, D = coords.shape
        freqs_cis = torch.cat(coords.chunk(D, dim=-1), dim=0) * self.mag
        freqs_cis = torch.cat(freqs_cis.chunk(D, dim=0), dim=-1)
        pos_embed = torch.zeros_like(feats)
        pos_embed[..., 0::2] = torch.sin(freqs_cis)
        pos_embed[..., 1::2] = torch.cos(freqs_cis)
        return feats + pos_embed
