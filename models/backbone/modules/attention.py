import torch
import torch.nn as nn
import flash_attn


class VarlenSelfAttention(nn.Module):
    """
    Forward Args:

        `inputs` (`torch.Tensor`, shape is (total_seq_len=N1+N2+..., embed_dim))

        `cu_seqlens` (`torch.Tensor`, shape is (B+1,))
    """

    def __init__(self, embed_dim: int, n_heads: int, attn_drop: float, dropout: float):
        super().__init__()

        self.embed_dim: int = embed_dim
        self.n_heads: int = n_heads
        self.head_dim: int = self.embed_dim // self.n_heads
        self.attn_drop: float = attn_drop
        self.dropout: nn.Dropout = nn.Dropout(dropout)

        assert self.head_dim * self.n_heads == self.embed_dim

        self.qkv_proj = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, inputs, cu_seqlens):
        total_seq_len, E = inputs.shape
        H = self.n_heads
        D = self.head_dim

        qkv_packed = self.qkv_proj(inputs).reshape(
            total_seq_len, 3, H, D
        )  # (total_seq_len, 3, H, D)
        max_seqlen = torch.max(cu_seqlens[1:] - cu_seqlens[:-1]).item()

        attn_output = flash_attn.flash_attn_varlen_qkvpacked_func(
            qkv_packed.half(),
            cu_seqlens,
            max_seqlen,
            dropout_p=self.attn_drop if self.training else 0.0,
        )  # (total_seq_len, H, D) # type: ignore
        attn_output = attn_output.to(inputs.dtype)  # type: ignore
        attn_output = attn_output.reshape(total_seq_len, E)  # (total_seq_len, E)

        output = self.dropout(self.out_proj(attn_output))

        return output


class VarlenSelfAttentionWithRoPE(VarlenSelfAttention):
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        attn_drop: float,
        dropout: float,
        rope_dim: int = 4,
        rope_theta: float = 10000.0,
        rope_learnable: bool = True,
    ):
        super().__init__(embed_dim, n_heads, attn_drop, dropout)
        self.rope_dim = rope_dim
        self.rope_theta = rope_theta
        mag = 1 / (
            rope_theta
            ** (
                torch.arange(0, self.head_dim, 2 * rope_dim)[
                    : (self.head_dim // 2 * rope_dim)
                ].float()
                / self.head_dim
            )
        )
        if rope_learnable:
            self.mag = nn.Parameter(mag)
        else:
            self.register_buffer("mag", mag)

    @torch.no_grad()
    def _compute_cis(self, coords: torch.Tensor):
        """
        Compute the frequencies for the RoPE.

        Args:

            `coords` (`torch.Tensor`, shape is (total_seq_len, coords_dim)): The coordinates of the tokens.

        Returns:

            `torch.Tensor`, shape is (total_seq_len, head_dim // 2): The frequencies of the RoPE.
        """
        N, D = coords.shape  # N = total_seq_len, D = coords_dim
        freqs_cis = torch.cat(coords.chunk(D, dim=-1), dim=0) * self.mag
        freqs_cis = torch.polar(torch.ones_like(freqs_cis), freqs_cis)
        freqs_cis = torch.cat(freqs_cis.chunk(D, dim=0), dim=-1)
        return freqs_cis

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor):
        """
        Apply the RoPE to the queries and keys.

        Args:

            `q` (`torch.Tensor`, shape is (total_seq_len, n_heads, head_dim)): The queries.

            `k` (`torch.Tensor`, shape is (total_seq_len, n_heads, head_dim)): The keys.

            `freqs_cis` (`torch.Tensor`, shape is (total_seq_len, head_dim // 2)): The frequencies of the RoPE.

        Returns:

            `torch.Tensor`, shape is (total_seq_len, n_heads, head_dim): The queries and keys with the RoPE applied.
        """
        raw_q_dtype, raw_k_dtype = q.dtype, k.dtype
        q_ = torch.view_as_complex(
            q.float().reshape(*q.shape[:-1], -1, 2)
        )  # (total_seq_len, n_heads, head_dim // 2)
        k_ = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(1)  # (total_seq_len, 1, head_dim // 2)
        q_out = torch.view_as_real(q_ * freqs_cis).flatten(
            2
        )  # (total_seq_len, n_heads, head_dim)
        k_out = torch.view_as_real(k_ * freqs_cis).flatten(2)
        return q_out.to(dtype=raw_q_dtype), k_out.to(dtype=raw_k_dtype)

    def forward(self, coords, inputs, cu_seqlens):
        total_seq_len, E = inputs.shape
        H = self.n_heads
        D = self.head_dim

        qkv_packed = self.qkv_proj(inputs).reshape(
            total_seq_len, 3, H, D
        )  # (total_seq_len, 3, H, D)
        max_seqlen = torch.max(cu_seqlens[1:] - cu_seqlens[:-1]).item()

        # ------------------ Apply RoPE ------------------ #
        freqs_cis = self._compute_cis(coords)
        qkv_packed[:, 0], qkv_packed[:, 1] = self._apply_rope(
            qkv_packed[:, 0], qkv_packed[:, 1], freqs_cis
        )
        # ------------------------------------------------ #

        attn_output = flash_attn.flash_attn_varlen_qkvpacked_func(
            qkv_packed.half(),
            cu_seqlens,
            max_seqlen,
            dropout_p=self.attn_drop if self.training else 0.0,
        )  # (total_seq_len, H, D) # type: ignore
        attn_output = attn_output.to(inputs.dtype)  # type: ignore
        attn_output = attn_output.reshape(total_seq_len, E)  # (total_seq_len, E)

        output = self.dropout(self.out_proj(attn_output))

        return output


class VarlenCrossAttention(nn.Module):
    """
    Forward Args:

        `inputs` (`torch.Tensor`, shape is (total_seq_len=N1+N2+..., embed_dim)): N1, N2, ... are the sequence lengths of the inputs

        `cu_seqlens_inputs` (`torch.Tensor`, shape is (B+1,)): cu_seqlens_inputs[i] is the cumulative sequence length of the i-th batch of inputs

        `queries` (`torch.Tensor`, shape is (total_seq_len=Q1+Q2+..., embed_dim)): Q1, Q2, ... are the sequence lengths of the queries

        `cu_seqlens_queries` (`torch.Tensor`, shape is (B+1,)): cu_seqlens_queries[i] is the cumulative sequence length of the i-th batch of queries
    """

    def __init__(self, embed_dim: int, n_heads: int, attn_drop: float, dropout: float):
        super().__init__()

        self.embed_dim: int = embed_dim
        self.n_heads: int = n_heads
        self.head_dim: int = self.embed_dim // self.n_heads
        self.attn_drop: float = attn_drop
        self.dropout: nn.Dropout = nn.Dropout(dropout)

        assert self.head_dim * self.n_heads == self.embed_dim

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.kv_proj = nn.Linear(self.embed_dim, 2 * self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, inputs, cu_seqlens_inputs, queries, cu_seqlens_queries):
        total_seqlen_inputs, E = inputs.shape
        total_seqlen_queries, E = queries.shape
        H = self.n_heads
        D = self.head_dim

        q = self.q_proj(queries).reshape(total_seqlen_queries, H, D)
        max_seqlen_inputs = torch.max(
            cu_seqlens_inputs[1:] - cu_seqlens_inputs[:-1]
        ).item()

        kv = self.kv_proj(inputs).reshape(total_seqlen_inputs, 2, H, D)
        max_seqlen_queries = torch.max(
            cu_seqlens_queries[1:] - cu_seqlens_queries[:-1]
        ).item()

        attn_output = flash_attn.flash_attn_varlen_func(
            q.half(),
            kv[:, 0].half(),
            kv[:, 1].half(),
            cu_seqlens_queries,
            cu_seqlens_inputs,
            max_seqlen_queries,
            max_seqlen_inputs,
            dropout_p=self.attn_drop if self.training else 0.0,
        )  # (total_seqlen_queries, H, D) # type: ignore
        attn_output = attn_output.to(queries.dtype)  # type: ignore
        attn_output = attn_output.reshape(
            total_seqlen_queries, E
        )  # (total_seqlen_queries, E)

        output = self.dropout(self.out_proj(attn_output))

        return output


class VarlenCrossAttentionWithMask(nn.Module):

    def __init__(self, embed_dim: int, n_heads: int, dropout: float):
        super().__init__()

        self.embed_dim: int = embed_dim
        self.n_heads: int = n_heads
        self.head_dim: int = self.embed_dim // self.n_heads

        assert self.head_dim * self.n_heads == self.embed_dim

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=n_heads, dropout=dropout
        )

    def forward(
        self, inputs, cu_seqlens_inputs, queries, cu_seqlens_queries, attn_masks=None
    ):
        outputs = []
        for batch_idx in range(len(cu_seqlens_inputs) - 1):
            idx_start_input, idx_end_input = (
                cu_seqlens_inputs[batch_idx],
                cu_seqlens_inputs[batch_idx + 1],
            )
            idx_start_query, idx_end_query = (
                cu_seqlens_queries[batch_idx],
                cu_seqlens_queries[batch_idx + 1],
            )
            output, _ = self.attn(
                query=queries[idx_start_query:idx_end_query],
                key=inputs[idx_start_input:idx_end_input],
                value=inputs[idx_start_input:idx_end_input],
                attn_mask=attn_masks[batch_idx] if attn_masks is not None else None,
            )
            outputs.append(output)

        return torch.cat(outputs, dim=0)
