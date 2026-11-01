import torch
import torch.nn.functional as F
from typing import Optional, List
import matplotlib.pyplot as plt


@torch.no_grad()
def extract_last_cross_attn_to_pc(
    attended_primitive_codes: torch.Tensor,  # (B, N_dec, D) —— 即你的 decoder 输出
    layer_cache,  # LayerIntermediates —— 即你称的 coarse_cache
    pc_embed_key_padding_mask: Optional[
        torch.Tensor
    ],  # (B, N_ctx)  True=有效  False=pad
    q_index: Optional[int] = None,  # 关注哪个“输出token”，默认最后一个
    q_indices: Optional[List[int]] = None,  # 关注哪些“输出token”，默认最后一个
    drop_mem_kv: int = 0,  # 若 cross_attn_num_mem_kv>0，则设为对应的数以剔除前若干列
    average_heads: bool = True,  # 聚合多头
):
    """
    返回:
      attn_vec:   (B, N_ctx_eff)  —— 指定输出token对每个上下文位置的注意力（已考虑mask / 可选去mem_kv / 可选按头平均）
      attn_mat:   (B, H, N_dec, N_ctx_tot) —— 最后一个 cross-attn 的完整注意力矩阵（未裁剪）
    """
    assert hasattr(layer_cache, "attn_intermediates") and (
        layer_cache.attn_intermediates is not None
    ), "未拿到 attn_intermediates，请确保 decoder(..., return_hiddens=True)"

    # 仅保留 cross-attn 的 intermediates，并取最后一个
    cross_inters = [
        it
        for it in layer_cache.attn_intermediates
        if getattr(it, "layer_type", None) == "c"
    ]
    assert (
        len(cross_inters) > 0
    ), "本模型结构下似乎没有 cross-attn 层，或未开启 cross_attend"
    last_cross = cross_inters[-1]

    # 注意力: (B, H, N_dec, N_ctx_tot)
    attn_mat = last_cross.post_softmax_attn
    B, H, N_dec, N_ctx_tot = attn_mat.shape

    if q_indices is not None:
        q_indices = [int(q_index) for q_index in q_indices]
        assert all(
            0 <= q_index < N_dec for q_index in q_indices
        ), f"q_indices 超界: 0..{N_dec-1}"
        attn_q = attn_mat[:, :, q_indices, :]
    else:
        # 默认取“最后一个输出 token”（也可自行传入q_index）
        if q_index is None:
            q_index = attended_primitive_codes.shape[1] - 1
        q_index = int(q_index)
        assert 0 <= q_index < N_dec, f"q_index 超界: 0..{N_dec-1}"

        # 取该输出token对应的 cross-attn 列向量 (B, H, N_ctx_tot)
        attn_q = attn_mat[:, :, q_index, :]

    # 如有 mem_kv，剔除最左侧的前 drop_mem_kv 列，只看真实 pc_context
    if drop_mem_kv > 0:
        assert drop_mem_kv <= N_ctx_tot
        attn_q = attn_q[..., drop_mem_kv:]
        N_ctx_eff = N_ctx_tot - drop_mem_kv
    else:
        N_ctx_eff = N_ctx_tot

    # 应用 context padding mask（True=有效）
    if pc_embed_key_padding_mask is not None:
        assert pc_embed_key_padding_mask.shape == (
            B,
            N_ctx_eff,
        ), f"context mask 维度应为 {(B, N_ctx_eff)}，实际 {tuple(pc_embed_key_padding_mask.shape)}"
        attn_q = attn_q * pc_embed_key_padding_mask[:, None, :].to(attn_q.dtype)

    # 可选：按头平均 -> (B, N_ctx_eff)
    attn_vec = attn_q.mean(dim=1) if average_heads else attn_q

    # 重新归一化，避免mask后总和<1
    denom = attn_vec.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    attn_vec = attn_vec / denom

    return attn_vec, attn_mat


def plot_attn_vector(
    attn_vec_1d, title="attn(last output token → pc_context)", save_path=None
):
    # attn_vec_1d: (N_ctx_eff,)
    plt.figure()
    plt.imshow(attn_vec_1d[None, :], aspect="auto")
    plt.xlabel("pc_context index")
    plt.yticks([])
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path)
    # plt.show()
    plt.close()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def plot_attn_on_points(
    pc_coord,
    attn,
    *,
    cmap: str = "magma",  # 颜色映射：'viridis'/'plasma'/'magma' 等
    figsize=(6, 6),
    equal_axis: bool = True,  # 是否等比坐标
    invert_y: bool = False,  # 若你的坐标是图像像素系（左上为原点），可置 True
    axis_off: bool = False,  # 隐藏坐标轴
    title: str | None = None,
    save_path: str | None = None,  # 若提供路径则保存
    scale_size_by_attn: bool = True,
    size_range=(8.0, 48.0),  # 点大小范围（按注意力插值）
    alpha_range=(1.00, 1.00),  # 透明度范围（按注意力插值）
    background: (
        np.ndarray | None
    ) = None,  # 可选背景图 (H,W) 或 (H,W,3/4)，先 imshow 再散点
    valid_mask: np.ndarray | None = None,  # 可选：有效点掩码 (N,)；无效点会被忽略
):
    """
    可视化二维点上的注意力热度。

    参数
    ----
    pc_coord: (N,2)  - 二维点坐标 (x, y)
    attn:     (N,)   - 已归一化的注意力权重，建议在 [0,1]，用于着色/大小/透明度
    cmap:     str    - matplotlib colormap 名称
    background:      - 若给定，将先绘制背景（如原图），再绘制散点
    valid_mask:      - True=保留；False=忽略（适合处理padding）
    """
    # --- to numpy & shape checks ---
    if hasattr(pc_coord, "detach"):  # torch.Tensor
        pc_coord = pc_coord.detach().cpu().numpy()
    else:
        pc_coord = np.asarray(pc_coord)

    if hasattr(attn, "detach"):
        attn = attn.detach().cpu().numpy()
    else:
        attn = np.asarray(attn)

    assert (
        pc_coord.ndim == 2 and pc_coord.shape[1] == 2
    ), f"pc_coord 形状应为 (N,2)，得到 {pc_coord.shape}"
    assert (
        attn.ndim == 1 and attn.shape[0] == pc_coord.shape[0]
    ), f"attn 形状应为 (N,)，且 N 与坐标一致，得到 {attn.shape}"

    # --- filter invalids / NaN ---
    nan_mask = np.isfinite(pc_coord).all(axis=1) & np.isfinite(attn)
    if valid_mask is not None:
        nan_mask &= valid_mask.astype(bool)
    pc = pc_coord[nan_mask]
    a = attn[nan_mask]
    if pc.shape[0] == 0:
        raise ValueError("有效点数为 0，请检查 mask / NaN / 形状。")

    # --- (可选) 轻度归一以防输入略越界 ---
    a = np.clip(a, 0.0, 1.0)
    norm = Normalize(vmin=0.0, vmax=1.0)
    cmap_obj = plt.get_cmap(cmap)
    cmap_obj = cmap_obj.reversed()

    # 构造 RGBA 颜色，按注意力映射，并给 alpha 设置梯度
    colors = cmap_obj(norm(a))
    # if alpha_range is not None:
    #     amin, amax = alpha_range
    #     alpha = amin + (amax - amin) * a
    #     colors[:, 3] = np.clip(alpha, 0.0, 1.0)

    # 点大小（可随注意力变化）
    if scale_size_by_attn:
        smin, smax = size_range
        sizes = smin + (smax - smin) * a
    else:
        sizes = size_range[1]  # 固定使用较大的一个

    # --- 绘图 ---
    fig, ax = plt.subplots(figsize=figsize)

    # 背景（可选）
    if background is not None:
        ax.imshow(background, origin="upper" if not invert_y else "lower")

    sc = ax.scatter(pc[:, 0], pc[:, 1], s=sizes, c=colors, marker="o", linewidths=0)

    # colorbar 依据 colormap（与散点颜色保持一致）
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Attention", rotation=90)

    if equal_axis:
        ax.set_aspect("equal")
    if invert_y:
        ax.invert_yaxis()
    if axis_off:
        ax.axis("off")
    if title is not None:
        ax.set_title(title)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200)
    plt.close()
    return fig, ax


import numpy as np


def enhance_attn(
    attn, method="percentile", *, p_low=0.80, p_high=0.995, gamma=3.0, softmax_T=0.02
):
    """
    将很小的注意力分数做“显示增强”，让高值更突出（只用于可视化）。
    输入:
      attn: (N,) 建议非负；可为 sum=1 的分布，也可为任意范围
    返回:
      attn_disp: (N,) ∈ [0,1] 仅用于上色/大小/透明度
    """
    a = np.asarray(attn).astype(np.float64)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    if a.max() <= 0:
        return np.zeros_like(a)

    if method == "percentile":
        # 1) 百分位对比拉伸 + 2) 幂指数锐化
        lo = np.quantile(a, p_low)
        hi = np.quantile(a, p_high)
        if not np.isfinite(lo):
            lo = 0.0
        if not np.isfinite(hi) or hi <= lo:
            hi = a.max()
        w = np.clip((a - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
        # gamma > 1 让高值更亮；可试 2~6
        w = w**gamma
        # 最后归一到 [0,1]
        w = (w - w.min()) / max(w.max() - w.min(), 1e-12)
        return w

    elif method == "softmax":
        # 温度 softmax（T 越小越“尖锐”）
        z = (a / max(a.std(), 1e-12)) if a.std() > 0 else a
        w = np.exp(z / softmax_T)
        w = w / max(w.sum(), 1e-12)
        # 拉到 [0,1] 便于 colormap
        w = (w - w.min()) / max(w.max() - w.min(), 1e-12)
        return w

    elif method == "rank":
        # 仅按排序分级着色（对极端值更鲁棒）
        r = np.argsort(np.argsort(a))  # 0..N-1
        w = r.astype(np.float64) / max(len(a) - 1, 1)
        return w

    elif method == "log":
        w = np.log1p(a / a.max() * 100.0)  # 100 可调
        w = (w - w.min()) / max(w.max() - w.min(), 1e-12)
        return w

    else:
        # 默认回退：线性归一
        w = a / a.max()
        return w


import numpy as np
import matplotlib.pyplot as plt


def highlight_topk(
    ax,
    pc_coord,
    attn,
    k=20,
    *,
    marker="o",
    size_mul=2.5,
    edgecolor="white",
    linewidth=1.6,
    annotate=False,
    fmt="{:.3f}",
    color=None,
):
    """
    在已有坐标轴 ax 上高亮注意力最高的 k 个点。
    """
    if hasattr(pc_coord, "detach"):
        pc = pc_coord.detach().cpu().numpy()
    else:
        pc = np.asarray(pc_coord)
    a = attn.detach().cpu().numpy() if hasattr(attn, "detach") else np.asarray(attn)

    k = int(np.clip(k, 1, len(a)))
    idx = np.argpartition(a, -k)[-k:]
    idx = idx[np.argsort(-a[idx])]  # 从大到小排序

    # 自动估一个基准尺寸（如果你之前用的是按注意力缩放，可以直接给一个固定较大的值）
    base_s = 36.0
    s = np.full(k, base_s * size_mul)

    # 前景层：亮边 + 可选统一颜色（否则沿用默认 colormap）
    ax.scatter(
        pc[idx, 0],
        pc[idx, 1],
        s=s,
        facecolors="none" if color is None else color,
        edgecolors=edgecolor,
        linewidths=linewidth,
        marker=marker,
        zorder=3,
    )

    if annotate:
        for i in idx:
            ax.text(
                pc[i, 0],
                pc[i, 1],
                fmt.format(a[i]),
                color=edgecolor,
                fontsize=9,
                weight="bold",
                ha="center",
                va="center",
                zorder=4,
                bbox=dict(facecolor="black", alpha=0.35, pad=1.5, linewidth=0),
            )


import torch
from typing import Optional, List


@torch.no_grad()
def extract_all_cross_attn_to_pc(
    attended_primitive_codes: torch.Tensor,  # (B, N_dec, D) —— 即你的 decoder 输出
    layer_cache,  # LayerIntermediates —— 即你称的 coarse_cache
    pc_embed_key_padding_mask: Optional[
        torch.Tensor
    ],  # (B, N_ctx)  True=有效  False=pad
    q_index: Optional[int] = None,  # 关注哪个"输出token"，默认最后一个
    q_indices: Optional[List[int]] = None,  # 关注哪些"输出token"，默认最后一个
    drop_mem_kv: int = 0,  # 若 cross_attn_num_mem_kv>0，则设为对应的数以剔除前若干列
    average_heads: bool = True,  # 聚合多头
    num_cross_per_block: int = 2,  # 每个block中cross-attn的数量（用于分组）
    return_per_block: bool = False,  # 是否按block分组返回
):
    """
    提取所有 cross-attention 层的注意力图，支持按 block 分组。

    返回:
      如果 return_per_block=False:
        attn_vecs:  List[(B, N_ctx_eff)] —— 每个 cross-attn 层的注意力向量列表，长度为总cross-attn层数
        attn_mats:  List[(B, H, N_dec, N_ctx_tot)] —— 每个 cross-attn 层的完整注意力矩阵列表

      如果 return_per_block=True:
        attn_vecs:  List[List[(B, N_ctx_eff)]] —— 按block分组的注意力向量，外层list长度为block数，内层为每个block的cross-attn层
        attn_mats:  List[List[(B, H, N_dec, N_ctx_tot)]] —— 按block分组的注意力矩阵

    例如：custom_layers = ('a', 'c', 'c', 'f') * attn_depth
         每个block有2个cross-attn，如果attn_depth=4，共8个cross-attn层
         return_per_block=False: 返回长度为8的列表
         return_per_block=True: 返回长度为4的列表，每个元素是长度为2的列表
    """
    assert hasattr(layer_cache, "attn_intermediates") and (
        layer_cache.attn_intermediates is not None
    ), "未拿到 attn_intermediates，请确保 decoder(..., return_hiddens=True)"

    # 仅保留 cross-attn 的 intermediates
    cross_inters = [
        it
        for it in layer_cache.attn_intermediates
        if getattr(it, "layer_type", None) == "c"
    ]
    assert (
        len(cross_inters) > 0
    ), "本模型结构下似乎没有 cross-attn 层，或未开启 cross_attend"

    attn_vecs = []
    attn_mats = []

    for cross_layer in cross_inters:
        # 注意力: (B, H, N_dec, N_ctx_tot)
        attn_mat = cross_layer.post_softmax_attn
        B, H, N_dec, N_ctx_tot = attn_mat.shape

        if q_indices is not None:
            q_indices_int = [int(qi) for qi in q_indices]
            attn_q = attn_mat[:, :, q_indices_int, :]
        else:
            # 默认取"最后一个输出 token"
            if q_index is None:
                q_idx = attended_primitive_codes.shape[1] - 1
            else:
                q_idx = q_index
            q_idx = int(q_idx)
            assert 0 <= q_idx < N_dec, f"q_index 超界: 0..{N_dec-1}"
            attn_q = attn_mat[:, :, q_idx, :]

        # 如有 mem_kv，剔除最左侧的前 drop_mem_kv 列
        if drop_mem_kv > 0:
            assert drop_mem_kv <= N_ctx_tot
            attn_q = attn_q[..., drop_mem_kv:]
            N_ctx_eff = N_ctx_tot - drop_mem_kv
        else:
            N_ctx_eff = N_ctx_tot

        # 应用 context padding mask
        if pc_embed_key_padding_mask is not None:
            assert pc_embed_key_padding_mask.shape == (
                B,
                N_ctx_eff,
            ), f"context mask 维度应为 {(B, N_ctx_eff)}，实际 {tuple(pc_embed_key_padding_mask.shape)}"
            attn_q = attn_q * pc_embed_key_padding_mask[:, None, :].to(attn_q.dtype)

        # 按头平均
        attn_vec = attn_q.mean(dim=1) if average_heads else attn_q

        # 重新归一化
        denom = attn_vec.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        attn_vec = attn_vec / denom

        attn_vecs.append(attn_vec)
        attn_mats.append(attn_mat)

    # 如果需要按block分组
    if return_per_block and num_cross_per_block > 1:
        num_total_cross = len(attn_vecs)
        num_blocks = num_total_cross // num_cross_per_block

        # 重新组织为 [block][cross_idx_in_block]
        attn_vecs_grouped = []
        attn_mats_grouped = []
        for block_idx in range(num_blocks):
            start_idx = block_idx * num_cross_per_block
            end_idx = start_idx + num_cross_per_block
            block_attn_vecs = attn_vecs[start_idx:end_idx]
            attn_vecs_grouped.append(block_attn_vecs)
            attn_mats_grouped.append(attn_mats[start_idx:end_idx])

        return attn_vecs_grouped, attn_mats_grouped

    return attn_vecs, attn_mats


@torch.no_grad()
def extract_last_cross_attn_to_pc(
    attended_primitive_codes: torch.Tensor,  # (B, N_dec, D) —— 即你的 decoder 输出
    layer_cache,  # LayerIntermediates —— 即你称的 coarse_cache
    pc_embed_key_padding_mask: Optional[
        torch.Tensor
    ],  # (B, N_ctx)  True=有效  False=pad
    q_index: Optional[int] = None,  # 关注哪个“输出token”，默认最后一个
    q_indices: Optional[List[int]] = None,  # 关注哪些“输出token”，默认最后一个
    drop_mem_kv: int = 0,  # 若 cross_attn_num_mem_kv>0，则设为对应的数以剔除前若干列
    average_heads: bool = True,  # 聚合多头
):
    """
    返回:
      attn_vec:   (B, N_ctx_eff)  —— 指定输出token对每个上下文位置的注意力（已考虑mask / 可选去mem_kv / 可选按头平均）
      attn_mat:   (B, H, N_dec, N_ctx_tot) —— 最后一个 cross-attn 的完整注意力矩阵（未裁剪）
    """
    assert hasattr(layer_cache, "attn_intermediates") and (
        layer_cache.attn_intermediates is not None
    ), "未拿到 attn_intermediates，请确保 decoder(..., return_hiddens=True)"

    # 仅保留 cross-attn 的 intermediates，并取最后一个
    cross_inters = [
        it
        for it in layer_cache.attn_intermediates
        if getattr(it, "layer_type", None) == "c"
    ]
    assert (
        len(cross_inters) > 0
    ), "本模型结构下似乎没有 cross-attn 层，或未开启 cross_attend"
    last_cross = cross_inters[-1]

    # 注意力: (B, H, N_dec, N_ctx_tot)
    attn_mat = last_cross.post_softmax_attn
    B, H, N_dec, N_ctx_tot = attn_mat.shape

    if q_indices is not None:
        q_indices = [int(q_index) for q_index in q_indices]
        # assert all(0 <= q_index < N_dec for q_index in q_indices), f"q_indices 超界: 0..{N_dec-1}"
        attn_q = attn_mat[:, :, q_indices, :]
    else:
        # 默认取“最后一个输出 token”（也可自行传入q_index）
        if q_index is None:
            q_index = attended_primitive_codes.shape[1] - 1
        q_index = int(q_index)
        assert 0 <= q_index < N_dec, f"q_index 超界: 0..{N_dec-1}"

        # 取该输出token对应的 cross-attn 列向量 (B, H, N_ctx_tot)
        attn_q = attn_mat[:, :, q_index, :]

    # 如有 mem_kv，剔除最左侧的前 drop_mem_kv 列，只看真实 pc_context
    if drop_mem_kv > 0:
        assert drop_mem_kv <= N_ctx_tot
        attn_q = attn_q[..., drop_mem_kv:]
        N_ctx_eff = N_ctx_tot - drop_mem_kv
    else:
        N_ctx_eff = N_ctx_tot

    # 应用 context padding mask（True=有效）
    if pc_embed_key_padding_mask is not None:
        assert pc_embed_key_padding_mask.shape == (
            B,
            N_ctx_eff,
        ), f"context mask 维度应为 {(B, N_ctx_eff)}，实际 {tuple(pc_embed_key_padding_mask.shape)}"
        attn_q = attn_q * pc_embed_key_padding_mask[:, None, :].to(attn_q.dtype)

    # 可选：按头平均 -> (B, N_ctx_eff)
    attn_vec = attn_q.mean(dim=1) if average_heads else attn_q

    # 重新归一化，避免mask后总和<1
    denom = attn_vec.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    attn_vec = attn_vec / denom

    return attn_vec, attn_mat


@torch.no_grad()
def _build_q_indices_abs(
    dec_len: int, prim_codes_len: int, primitive_mask: torch.Tensor
):
    """
    计算“实例 token 在 Decoder 全序列中的绝对行位置”。
    primitive_mask: (B, N_inst_all) True=有效
    返回 list[Tensor]，每个元素 shape=(N_inst_valid_b,)
    """
    start = dec_len - (prim_codes_len + 1)  # +1 for EOS
    batch = primitive_mask.size(0)
    out = []
    for b in range(batch):
        rel = torch.nonzero(primitive_mask[b], as_tuple=False).squeeze(1)
        out.append(start + rel)
    return out


def extract_last_local_cross_attn(cache, drop_mem_kv: int, loss_mask_type: str = None):
    """
    从 decoder cache 的 attn_intermediates 里取“最后一次 local cross”的注意力:
    返回 (B, H, N_dec, N_ctx_eff)；已去掉 mem_kv 列
    """
    # cache.attn_intermediates 是一个 list，含所有 self/cross 的 inter
    # 我们只保留 layer_type == 'c' 的，并取倒数第一个；因为我们把 local 设为每个 block 的第二个 'c'，
    # 最后一个 'c' 就是“最后一层 local cross”
    inters = [
        it
        for it in getattr(cache, "attn_intermediates", [])
        if getattr(it, "layer_type", None) == "c"
    ]
    assert len(inters) > 0 and hasattr(
        inters[-1], "post_softmax_attn"
    ), "没有拿到 cross-attn 注意力，请确认 local cross 已禁用 flash 并 return_intermediates=True"
    if loss_mask_type == None:
        attn = inters[-1].post_softmax_attn  # (B,H,N_dec,N_memkv+N_ctx)
    elif loss_mask_type == "cross_entropy":
        attn = inters[-1].pre_softmax_attn  # (B,H,N_dec,N_memkv+N_ctx)
    if drop_mem_kv > 0:
        attn = attn[..., drop_mem_kv:]  # 去掉 memory 槽
    return attn


def extract_local_cross_attn(cache, drop_mem_kv: int, loss_mask_type: str = None):
    inters = [
        it
        for it in getattr(cache, "attn_intermediates", [])
        if getattr(it, "layer_type", None) == "c"
    ]
    assert len(inters) > 0 and hasattr(
        inters[-1], "post_softmax_attn"
    ), "没有拿到 cross-attn 注意力，请确认 local cross 已禁用 flash 并 return_intermediates=True"
    local_attn_indices = [2 * i + 1 for i in range(len(inters) // 2)]
    if loss_mask_type == None:
        local_attn = [inters[i].post_softmax_attn for i in local_attn_indices]
    elif loss_mask_type == "cross_entropy":
        local_attn = [inters[i].pre_softmax_attn for i in local_attn_indices]
    if drop_mem_kv > 0:
        local_attn = [attn[..., drop_mem_kv:] for attn in local_attn]
    return local_attn


def attention_alignment_loss_from_cache(
    attended_primitive_codes: torch.Tensor,  # decoder 输出序列 (B, N_dec, C)
    cache,  # decoder 返回的 cache，里含 attn_intermediates
    gt_instance_masks: list,  # 长度 B，每个元素 Tensor (N_inst_valid_b, N_ctx_valid_b) 0/1 或 bool
    pc_embed_key_padding_mask: torch.Tensor,  # (B, N_ctx) True=有效
    primitive_mask: torch.Tensor,  # (B, N_inst_all) True=有效实例
    prim_codes_len: int,  # 当前 step 的 instance token 数 (不含 SOS/EOS)
    num_mem_kv: int = 0,  # cross-attn 的 mem_kv 数（左侧列数）
    mode: str = "kl",  # 'kl' 或 'cov'
    eps: float = 1e-8,
    loss_mask_type: str = None,
    aux_mask_supervision: bool = False,
    attn_constraint: bool = False,
    lambda_attn_col_ce: float = 0.1,
    attn_constraint_last_layer: bool = False,
):
    """
    计算“最后一层 local cross”对齐到 GT mask 的小权重损失。
    返回: (loss_scalar, alpha_list) 其中 alpha_list 用于可视化，每个元素 (N_inst_valid_b, N_ctx_valid_b)
    """
    loss_all = []
    loss_attn_constraint = []
    alphas_out = []

    if aux_mask_supervision:
        local_attn = extract_local_cross_attn(
            cache, drop_mem_kv=num_mem_kv, loss_mask_type=loss_mask_type
        )
        n_layers = len(local_attn)
        for layer_idx, attn in enumerate(local_attn):
            B, H, N_dec, N_ctx = attn.shape
            q_indices_abs_list = _build_q_indices_abs(
                dec_len=N_dec,
                prim_codes_len=prim_codes_len,
                primitive_mask=primitive_mask,
            )
            for b in range(B):
                # 2) 取该样本的 q 行与有效列
                q_idx = q_indices_abs_list[b]  # (N_inst_valid_b,)
                if q_idx.numel() == 0:
                    continue
                valid_ctx = pc_embed_key_padding_mask[b]  # (N_ctx,)
                attn_b = attn[b, :, q_idx, :][
                    :, :, valid_ctx
                ]  # (H, N_inst_valid_b, N_ctx_valid_b)
                if loss_mask_type == None:
                    alpha = attn_b.mean(dim=0)  # (N_inst_valid_b, N_ctx_valid_b)
                    # 数值安全
                    alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(eps)

                    # 3) GT 对齐（保证与 alpha 同尺寸）
                    t = gt_instance_masks[b].to(
                        alpha.dtype
                    )  # (N_inst_valid_b, N_ctx_valid_b) 0/1
                    # 允许用户传原始全长 mask，这里兼容裁切：
                    if t.size(-1) != alpha.size(-1):
                        # 假设用户传的是 (N_inst_valid_b, N_ctx) 未裁列：做列裁切
                        t = t[..., valid_ctx]

                    pos_sum = t.sum(dim=-1, keepdim=True)  # (N_inst_valid_b,1)
                    has_pos = pos_sum.squeeze(-1) > 0
                    if has_pos.any():
                        t_norm = torch.zeros_like(t)
                        t_norm[has_pos] = t[has_pos] / pos_sum[has_pos].clamp_min(eps)

                        if mode == "kl":
                            # KL(t || alpha)，仅对有正样本的行求均值
                            kl = (
                                t_norm * (t_norm.add(eps).log() - alpha.add(eps).log())
                            ).sum(dim=-1)
                            loss_all.append(kl[has_pos].mean())
                        else:
                            # 覆盖-泄漏（更鲁棒）
                            coverage = (alpha * t).sum(dim=-1)  # 越接近 1 越好
                            leak = (alpha * (1 - t)).sum(dim=-1)
                            loss_cov = (1.0 - coverage).pow(2)
                            loss_leak = leak
                            loss_all.append((loss_cov + loss_leak)[has_pos].mean())
                    alphas_out.append(alpha.detach())
                elif loss_mask_type == "cross_entropy":
                    L = attn_b.mean(dim=0)
                    t = gt_instance_masks[b].to(L.dtype)

                    # 类不平衡（可选）：pos_weight = (#neg / #pos)
                    pos = t.sum()
                    neg = t.numel() - pos
                    pos_weight = (neg / pos).clamp(min=1.0) if pos > 0 else None
                    # pos_weight = None

                    # BCE-with-logits（逐点）
                    bce = F.binary_cross_entropy_with_logits(
                        L,
                        t,
                        reduction="mean",
                        pos_weight=(
                            pos_weight if isinstance(pos_weight, torch.Tensor) else None
                        ),
                    )

                    # Dice（对 sigmoid(L)）
                    p = torch.sigmoid(L)
                    num = 2 * (p * t).sum(dim=-1)
                    den = (p.pow(2) + t.pow(2)).sum(dim=-1).clamp_min(1e-6)
                    dice = (1 - (num + 1) / (den + 1)).mean()
                    loss_all.append(bce + dice)
                    # sub_loss = bce + dice
                    if attn_constraint:
                        # # L: (I, P) 预softmax logits，t: (I, P) 浮点；你已有
                        # t_bin = (t > 0.5)              # 这里每列恰好一个 True
                        # # 1) 列向 CE（主力项）
                        # owner_idx = t_bin.long().argmax(dim=0)   # (P,)
                        # logits_col = L.transpose(0, 1)           # (P, I)
                        # loss_col_ce = F.cross_entropy(logits_col, owner_idx, reduction='mean')
                        # loss_attn_constraint.append(lambda_attn_col_ce * loss_col_ce)
                        eps = 1e-8
                        owner_idx = t.long().argmax(dim=0)  # (P,)
                        alpha_row = torch.softmax(
                            L, dim=-1
                        )  # (I, P)  行内对 key 维正规化
                        q = alpha_row / alpha_row.sum(dim=0, keepdim=True).clamp_min(
                            eps
                        )  # (I, P)  列归一 → 责任分布
                        log_q_col = (q + eps).log().transpose(0, 1)  # (P, I)  类维=实例
                        loss_col_ce = F.nll_loss(log_q_col, owner_idx, reduction="mean")
                        loss_attn_constraint.append(lambda_attn_col_ce * loss_col_ce)

                    # loss_all.append(sub_loss)
    else:
        attn = extract_last_local_cross_attn(
            cache, drop_mem_kv=num_mem_kv, loss_mask_type=loss_mask_type
        )  # (B,H,N_dec,N_ctx)
        B, H, N_dec, N_ctx = attn.shape

        # 1) 计算每个 batch 的实例绝对 q 行索引
        q_indices_abs_list = _build_q_indices_abs(
            dec_len=N_dec, prim_codes_len=prim_codes_len, primitive_mask=primitive_mask
        )

        for b in range(B):
            # 2) 取该样本的 q 行与有效列
            q_idx = q_indices_abs_list[b]  # (N_inst_valid_b,)
            if q_idx.numel() == 0:
                continue
            valid_ctx = pc_embed_key_padding_mask[b]  # (N_ctx,)
            attn_b = attn[b, :, q_idx, :][
                :, :, valid_ctx
            ]  # (H, N_inst_valid_b, N_ctx_valid_b)
            if loss_mask_type == None:
                alpha = attn_b.mean(dim=0)  # (N_inst_valid_b, N_ctx_valid_b)
                # 数值安全
                alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(eps)

                # 3) GT 对齐（保证与 alpha 同尺寸）
                t = gt_instance_masks[b].to(
                    alpha.dtype
                )  # (N_inst_valid_b, N_ctx_valid_b) 0/1
                # 允许用户传原始全长 mask，这里兼容裁切：
                if t.size(-1) != alpha.size(-1):
                    # 假设用户传的是 (N_inst_valid_b, N_ctx) 未裁列：做列裁切
                    t = t[..., valid_ctx]

                pos_sum = t.sum(dim=-1, keepdim=True)  # (N_inst_valid_b,1)
                has_pos = pos_sum.squeeze(-1) > 0
                if has_pos.any():
                    t_norm = torch.zeros_like(t)
                    t_norm[has_pos] = t[has_pos] / pos_sum[has_pos].clamp_min(eps)

                    if mode == "kl":
                        # KL(t || alpha)，仅对有正样本的行求均值
                        kl = (
                            t_norm * (t_norm.add(eps).log() - alpha.add(eps).log())
                        ).sum(dim=-1)
                        loss_all.append(kl[has_pos].mean())
                    else:
                        # 覆盖-泄漏（更鲁棒）
                        coverage = (alpha * t).sum(dim=-1)  # 越接近 1 越好
                        leak = (alpha * (1 - t)).sum(dim=-1)
                        loss_cov = (1.0 - coverage).pow(2)
                        loss_leak = leak
                        loss_all.append((loss_cov + loss_leak)[has_pos].mean())
                alphas_out.append(alpha.detach())
            elif loss_mask_type == "cross_entropy":
                L = attn_b.mean(dim=0)
                t = gt_instance_masks[b].to(L.dtype)

                # 类不平衡（可选）：pos_weight = (#neg / #pos)
                pos = t.sum()
                neg = t.numel() - pos
                pos_weight = (neg / pos).clamp(min=1.0) if pos > 0 else None
                # pos_weight = None

                # BCE-with-logits（逐点）
                bce = F.binary_cross_entropy_with_logits(
                    L,
                    t,
                    reduction="mean",
                    pos_weight=(
                        pos_weight if isinstance(pos_weight, torch.Tensor) else None
                    ),
                )

                # Dice（对 sigmoid(L)）
                p = torch.sigmoid(L)
                num = 2 * (p * t).sum(dim=-1)
                den = (p.pow(2) + t.pow(2)).sum(dim=-1).clamp_min(1e-6)
                dice = (1 - (num + 1) / (den + 1)).mean()

                loss_all.append(bce + dice)

    if len(loss_all) == 0:
        loss = torch.zeros(
            [],
            device=attended_primitive_codes.device,
            dtype=attended_primitive_codes.dtype,
        )
    else:
        loss_all = torch.stack(loss_all).mean()
        if len(loss_attn_constraint) > 0:
            if attn_constraint_last_layer:
                loss_attn_constraint = loss_attn_constraint[-1]
            else:
                loss_attn_constraint = torch.stack(loss_attn_constraint).mean()
            loss = loss_all + loss_attn_constraint
        else:
            loss = loss_all
            loss_attn_constraint = 0.0

    return loss, alphas_out, loss_attn_constraint
