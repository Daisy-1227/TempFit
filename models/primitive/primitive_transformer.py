from __future__ import annotations

from functools import partial
from math import ceil
import os
import math
from accelerate.utils import DistributedDataParallelKwargs
from beartype.typing import Tuple, Callable, List
import json
from einops import rearrange, repeat, reduce, pack
from gateloop_transformer import SimpleGateLoopLayer
from huggingface_hub import PyTorchModelHubMixin
import numpy as np

# open3d is only referenced in commented-out debug code; make it optional so
# packaged builds can exclude it (~770 MB savings).
try:
    import open3d as o3d
except ImportError:
    o3d = None
from tqdm import tqdm
import torch
from torch import nn, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance
from pytorch3d.transforms import euler_angles_to_matrix
from contextlib import contextmanager
from x_transformers import Decoder, Encoder, PrefixDecoder
from x_transformers.x_transformers import LayerIntermediates
from x_transformers.autoregressive_wrapper import eval_decorator
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # 使用非交互式后端，适合服务器环境

# michelangelo + img_encoder are only used when shape_condition_model_type
# selects them or use_img_conditioner=True. Guard them so the heavy GUI deps
# they pull (cv2 / VTK / matplotlib backends) can be excluded from packaged
# builds. Module load fails silently → caller code paths that try to use
# them will hit a clear NameError, but our default config (PTv3) doesn't.
try:
    from models.primitive.michelangelo import ShapeConditioner as ShapeConditioner_miche
except Exception:
    ShapeConditioner_miche = None
from models.backbone import create_point_extractor, PointTransformerV3

try:
    from models.primitive.img_encoder import ImgConditioner
except Exception:
    ImgConditioner = None
from models.primitive.slot_attn import SlotAttentionEncoder

from models.primitive.utils import (
    discretize,
    undiscretize,
    set_module_requires_grad_,
    default,
    exists,
    safe_cat,
    identity,
    is_tensor_empty,
)
from models.primitive.utils.typing import Float, Int, Bool, typecheck
from models.primitive.utils.attn_utils import (
    extract_last_cross_attn_to_pc,
    extract_all_cross_attn_to_pc,
    attention_alignment_loss_from_cache,
)

### Add for positional encoding
from models.primitive.positional_embedding import PositionEmbeddingCoordsSine

from datasets.transform_utils import (
    to_tensor,
    norm_coords,
    augment_line_args,
    to_vec_data,
)
from models.primitive.rotation_symmetry import (
    canonicalize_raw_angle,
    phase_to_canonical_angle,
    raw_angle_to_phase,
    symmetry_order_table,
)

# from .matcher_ref import HungarianMatcher
# constants

DEFAULT_DDP_KWARGS = DistributedDataParallelKwargs(find_unused_parameters=True)

# 可视化用的颜色列表
COLORS = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#bcf60c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#9a6324",
    "#fffac8",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd8b1",
    "#000075",
    "#808080",
    "#C56932",
    "#b7a58c",
    "#3a627d",
    "#9abc15",
    "#54810c",
    "#a7389c",
    "#687253",
    "#61f584",
    "#9a17d4",
    "#52b0c1",
    "#21f5b4",
    "#a2856c",
    "#9b1c34",
    "#4b1062",
    "#7cf406",
    "#0b1f63",
]


def discretize_log_scale(value, *, num_discrete, continuous_range):
    """Discretize positive relative scale uniformly in log space."""
    lower, upper = map(float, continuous_range)
    if lower <= 0 or upper <= lower:
        raise ValueError(f"invalid positive scale range: {continuous_range}")
    log_range = [math.log(lower), math.log(upper)]
    return discretize(
        torch.log(value.clamp_min(1e-8)),
        num_discrete=num_discrete,
        continuous_range=log_range,
    )


def undiscretize_log_scale(value, *, num_discrete, continuous_range):
    """Map log-uniform scale bins back to positive linear scale."""
    lower, upper = map(float, continuous_range)
    if lower <= 0 or upper <= lower:
        raise ValueError(f"invalid positive scale range: {continuous_range}")
    log_range = [math.log(lower), math.log(upper)]
    return torch.exp(
        undiscretize(
            value,
            num_discrete=num_discrete,
            continuous_range=log_range,
        )
    )


def visualize_downsampled_instance_mask(
    pc_coord,
    downsampled_instance_masks,
    save_path="./debug_vis/downsampled_mask.png",
    batch_idx=0,
):
    """
    可视化降采样后的点云坐标和instance masks (只显示XY平面)

    Args:
        pc_coord: [b, n_coords, 3] 降采样后的点云坐标
        downsampled_instance_masks: [N_inst, n_coords] 降采样后的instance masks (float类型，0或1)
        save_path: 保存图像的路径
        batch_idx: 要可视化的batch索引
    """
    import os

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 获取指定batch的数据
    if pc_coord.dim() == 3:
        coord = pc_coord[batch_idx].cpu().numpy()  # [n_coords, 3]
    else:
        coord = pc_coord.cpu().numpy()  # [n_coords, 3]

    if downsampled_instance_masks is None:
        print("Warning: downsampled_instance_masks is None, cannot visualize.")
        return

    masks = downsampled_instance_masks.cpu().numpy()  # [N_inst, n_coords]

    # 提取XY坐标
    x = coord[:, 0]
    y = coord[:, 1]

    num_instances = masks.shape[0]
    n_points = coord.shape[0]

    # 为每个点分配instance ID（一个点可能属于多个instance，取第一个）
    point_instance_ids = np.ones(n_points, dtype=np.int32) * -1

    for inst_idx in range(num_instances):
        inst_mask = masks[inst_idx] > 0.5  # 二值化
        point_instance_ids[inst_mask] = inst_idx

    # 创建可视化
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # 左图：所有实例叠加显示
    ax1 = axes[0]

    # 绘制背景点（不属于任何instance）
    background_mask = point_instance_ids == -1
    if background_mask.sum() > 0:
        ax1.scatter(
            x[background_mask],
            y[background_mask],
            c="lightgray",
            s=30,
            alpha=0.5,
            label="Background",
            edgecolors="black",
            linewidths=0.5,
        )

    # 绘制每个实例
    for inst_idx in range(num_instances):
        inst_mask = point_instance_ids == inst_idx
        if inst_mask.sum() > 0:
            color = COLORS[inst_idx % len(COLORS)]
            ax1.scatter(
                x[inst_mask],
                y[inst_mask],
                c=color,
                s=50,
                alpha=0.8,
                label=f"Instance {inst_idx}",
                edgecolors="black",
                linewidths=0.5,
            )

    ax1.set_xlabel("X Coordinate", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Y Coordinate", fontsize=14, fontweight="bold")
    ax1.set_title(
        f"Downsampled Point Cloud - All Instances (Batch {batch_idx})",
        fontsize=16,
        fontweight="bold",
    )
    ax1.legend(loc="best", fontsize=10, ncol=2)
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.set_aspect("equal", adjustable="box")

    # 右图：分instance显示
    ax2 = axes[1]

    # 先画所有点作为背景
    ax2.scatter(x, y, c="lightgray", s=20, alpha=0.3, edgecolors="none")

    # 叠加显示每个instance的mask
    for inst_idx in range(num_instances):
        inst_mask = masks[inst_idx] > 0.5
        if inst_mask.sum() > 0:
            color = COLORS[inst_idx % len(COLORS)]
            ax2.scatter(
                x[inst_mask],
                y[inst_mask],
                c=color,
                s=50,
                alpha=0.7,
                label=f"Inst {inst_idx}",
                edgecolors="white",
                linewidths=1,
            )

    ax2.set_xlabel("X Coordinate", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Y Coordinate", fontsize=14, fontweight="bold")
    ax2.set_title(
        f"Instance Masks Overlay (Batch {batch_idx})", fontsize=16, fontweight="bold"
    )
    ax2.legend(loc="best", fontsize=10, ncol=2)
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Visualization saved to: {save_path}")
    print(f"  - Total points: {n_points}")
    print(f"  - Num instances: {num_instances}")
    print(f"  - Background points: {background_mask.sum()}")
    for inst_idx in range(num_instances):
        inst_count = (point_instance_ids == inst_idx).sum()
        print(f"  - Instance {inst_idx}: {inst_count} points")


def downsample_point_cloud_data(
    coord,
    grid_coord,
    offset,
    feat,
    instance_masks=None,
    target_max_points=10000,
    method="random",
):
    """
    对点云数据进行降采样

    Args:
        coord: [N_total, 3] 点云坐标
        grid_coord: [N_total, 3] 网格坐标
        offset: [batch_size] 累积偏移量
        feat: [N_total, C] 点云特征
        instance_masks: [n_inst, N_total] 或 None，instance masks（如果存在）
        target_max_points: int，目标最大点数
        method: str，降采样方法，'random'或'fps'

    Returns:
        降采样后的 coord, grid_coord, offset, feat, instance_masks
    """
    N_total = coord.shape[0]

    if N_total <= target_max_points:
        # 不需要降采样
        return coord, grid_coord, offset, feat, instance_masks

    device = coord.device
    batch_size = offset.shape[0]

    # 计算每个样本的起始和结束位置
    lengths = offset.clone()
    if batch_size > 1:
        lengths[1:] = offset[1:] - offset[:-1]

    # 计算降采样比例
    downsample_ratio = target_max_points / N_total

    # 对每个batch样本分别降采样
    downsampled_coords = []
    downsampled_grid_coords = []
    downsampled_feats = []
    downsampled_masks_list = [] if instance_masks is not None else None
    new_lengths = []

    start_idx = 0
    for i in range(batch_size):
        end_idx = start_idx + lengths[i].item()
        sample_length = lengths[i].item()

        # 计算该样本的目标点数
        target_points = max(1, int(sample_length * downsample_ratio))

        # 提取当前样本的数据
        sample_coord = coord[start_idx:end_idx]
        sample_grid_coord = grid_coord[start_idx:end_idx]
        sample_feat = feat[start_idx:end_idx]

        # 降采样
        if method == "random":
            # 随机采样
            indices = torch.randperm(sample_length, device=device)[:target_points]
            indices = torch.sort(indices)[0]  # 排序以保持顺序
        elif method == "fps":
            # FPS（这里简化为均匀采样，真正的FPS计算开销较大）
            indices = torch.linspace(
                0, sample_length - 1, target_points, device=device
            ).long()
        else:
            raise ValueError(f"Unknown downsampling method: {method}")

        # 应用降采样
        downsampled_coords.append(sample_coord[indices])
        downsampled_grid_coords.append(sample_grid_coord[indices])
        downsampled_feats.append(sample_feat[indices])
        new_lengths.append(target_points)

        # 如果有instance_masks，也进行降采样
        if instance_masks is not None:
            sample_masks = instance_masks[
                :, start_idx:end_idx
            ]  # [n_inst, sample_length]
            downsampled_masks_list.append(
                sample_masks[:, indices]
            )  # [n_inst, target_points]

        start_idx = end_idx

    # 拼接所有batch的数据
    coord_downsampled = torch.cat(downsampled_coords, dim=0)
    grid_coord_downsampled = torch.cat(downsampled_grid_coords, dim=0)
    feat_downsampled = torch.cat(downsampled_feats, dim=0)

    # 重新计算offset
    new_lengths_tensor = torch.tensor(new_lengths, dtype=offset.dtype, device=device)
    offset_downsampled = torch.cumsum(new_lengths_tensor, dim=0)

    # 拼接instance_masks（如果存在）
    instance_masks_downsampled = None
    if instance_masks is not None:
        instance_masks_downsampled = torch.cat(downsampled_masks_list, dim=1)

    # print(f"Downsampled point cloud from {N_total} to {coord_downsampled.shape[0]} points")

    return (
        coord_downsampled,
        grid_coord_downsampled,
        offset_downsampled,
        feat_downsampled,
        instance_masks_downsampled,
    )


def visualize_instance_mask(
    pc_coord, pc_embed_key_padding_mask, gt_instance_mask, save_dir="./debug_vis"
):
    """
    可视化点云和实例掩码，用于检查降采样后的索引是否正确

    Args:
        pc_coord: [bs, N, 3] 点云坐标
        pc_embed_key_padding_mask: [bs, N] bool矩阵，True表示有效点
        gt_instance_mask: 长度为bs的list，每个元素是 [N_inst, N_p] 的bool矩阵
        save_dir: 保存图像的目录
    """
    import os

    os.makedirs(save_dir, exist_ok=True)

    bs = pc_coord.shape[0]

    for batch_idx in range(bs):
        # 获取当前样本的数据
        coord = pc_coord[batch_idx].cpu().numpy()  # [N, 3]
        valid_mask = pc_embed_key_padding_mask[batch_idx].cpu().numpy()  # [N]
        instance_masks = gt_instance_mask[batch_idx]  # [N_inst, N_p]

        # 只取有效的点
        valid_coord = coord[valid_mask]  # [N_valid, 3]

        if valid_coord.shape[0] == 0:
            print(
                f"Warning: Batch {batch_idx} has no valid points, skipping visualization."
            )
            continue

        # 只取前两维进行2D可视化
        x = valid_coord[:, 0]
        y = valid_coord[:, 1]

        # 为每个点分配实例ID（颜色）
        # 初始化为-1（背景/无实例）
        point_instance_ids = np.ones(valid_coord.shape[0], dtype=np.int32) * -1

        if isinstance(instance_masks, torch.Tensor):
            instance_masks = instance_masks.cpu().numpy()

        num_instances = instance_masks.shape[0]

        # 为每个点分配所属的实例ID
        for inst_idx in range(num_instances):
            inst_mask = instance_masks[inst_idx]  # [N_p]
            # 需要考虑到instance_mask可能对应原始点云，需要根据valid_mask筛选
            if len(inst_mask) == len(valid_mask):
                # instance_mask对应完整的padding后的点云
                inst_mask_valid = inst_mask[valid_mask]
            else:
                # instance_mask已经是有效点的mask
                inst_mask_valid = inst_mask[: len(point_instance_ids)]

            # 为属于当前实例的点分配ID
            point_instance_ids[inst_mask_valid] = inst_idx

        # 创建可视化
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # 左图：所有点（不同实例用不同颜色）
        ax1 = axes[0]

        # 绘制背景点（没有实例的点）
        background_mask = point_instance_ids == -1
        if background_mask.sum() > 0:
            ax1.scatter(
                x[background_mask],
                y[background_mask],
                c="lightgray",
                s=10,
                alpha=0.3,
                label="Background",
            )

        # 绘制每个实例
        for inst_idx in range(num_instances):
            inst_mask = point_instance_ids == inst_idx
            if inst_mask.sum() > 0:
                color = COLORS[inst_idx % len(COLORS)]
                ax1.scatter(
                    x[inst_mask],
                    y[inst_mask],
                    c=color,
                    s=20,
                    alpha=0.7,
                    label=f"Instance {inst_idx}",
                )

        ax1.set_xlabel("X Coordinate", fontsize=12)
        ax1.set_ylabel("Y Coordinate", fontsize=12)
        ax1.set_title(
            f"Batch {batch_idx} - All Instances", fontsize=14, fontweight="bold"
        )
        ax1.legend(loc="best", fontsize=8, ncol=2)
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect("equal", adjustable="box")

        # 右图：每个实例单独显示
        ax2 = axes[1]

        # 绘制所有点作为背景
        ax2.scatter(x, y, c="lightgray", s=5, alpha=0.2)

        # 高亮显示每个实例（分别用不同的标记）
        markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h", "H", "+", "x"]
        for inst_idx in range(min(num_instances, len(markers))):
            inst_mask = point_instance_ids == inst_idx
            if inst_mask.sum() > 0:
                color = COLORS[inst_idx % len(COLORS)]
                marker = markers[inst_idx % len(markers)]
                ax2.scatter(
                    x[inst_mask],
                    y[inst_mask],
                    c=color,
                    s=30,
                    alpha=0.8,
                    marker=marker,
                    label=f"Inst {inst_idx}",
                )

        ax2.set_xlabel("X Coordinate", fontsize=12)
        ax2.set_ylabel("Y Coordinate", fontsize=12)
        ax2.set_title(
            f"Batch {batch_idx} - Instance Details", fontsize=14, fontweight="bold"
        )
        ax2.legend(loc="best", fontsize=8, ncol=2)
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect("equal", adjustable="box")

        # 添加统计信息
        stats_text = f"Total Points: {len(x)}\nInstances: {num_instances}\n"
        for inst_idx in range(num_instances):
            inst_count = (point_instance_ids == inst_idx).sum()
            stats_text += f"Inst{inst_idx}: {inst_count} pts\n"

        fig.text(
            0.02,
            0.98,
            stats_text,
            transform=fig.transFigure,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()

        # 保存图像
        save_path = os.path.join(save_dir, f"instance_mask_batch_{batch_idx}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Visualization saved to: {save_path}")
        print(f"  - Valid points: {len(x)}")
        print(f"  - Instances: {num_instances}")
        print(
            f"  - Points per instance: {[int((point_instance_ids == i).sum()) for i in range(num_instances)]}"
        )

    print(f"\n✓ All visualizations saved to: {save_dir}")


# SHAPE_CODE = {
#     'CubeBevel': 0,
#     'SphereSharp': 1,
#     'CylinderSharp': 2,
# }
# BS_NAME = {
#     0: 'CubeBevel',
#     1: 'SphereSharp',
#     2: 'CylinderSharp',
# }


def run_decoder(decoder, codes, context=None, context_mask=None, cache=None):
    out = decoder(
        codes,
        cache=cache,
        return_hiddens=True,
        context=context,
        context_mask=context_mask,
    )
    return out[0] if isinstance(out, tuple) else out  # [B, L, D]


@torch.no_grad()
def sanity_check_prefix_mask(
    decoder,
    primitive_codes,  # [B, L, D]
    pc_prefix_len,  # int
    primitive_lib_embed,  # [B, Lc, D]
    primitive_lib_mask=None,  # [B, Lc]
    verbose=True,
):
    was_training = decoder.training
    decoder.eval()
    try:
        B, L, _ = primitive_codes.shape
        device = primitive_codes.device

        # ① 基线：传 context，但把整段查询都禁用 cross-attn（全 False）
        qmask_none = torch.ones((B, L), dtype=torch.bool, device=device)
        with mask_cross_attn_on_prefix(decoder, qmask_none):
            out_nocontext = run_decoder(
                decoder,
                primitive_codes,
                context=primitive_lib_embed,
                context_mask=primitive_lib_mask,
                cache=None,
            )

        # ② 带先验：仅前缀禁用 cross-attn，其余启用
        qmask_prefix = torch.ones((B, L), dtype=torch.bool, device=device)
        if pc_prefix_len > 0:
            qmask_prefix[:, :pc_prefix_len] = False

        with mask_cross_attn_on_prefix(decoder, qmask_prefix):
            out_withcontext = run_decoder(
                decoder,
                primitive_codes,
                context=primitive_lib_embed,
                context_mask=primitive_lib_mask,
                cache=None,
            )

        # —— 比较 —— #
        pref_a = out_nocontext[:, :pc_prefix_len, :]
        pref_b = out_withcontext[:, :pc_prefix_len, :]
        suf_a = out_nocontext[:, pc_prefix_len:, :]
        suf_b = out_withcontext[:, pc_prefix_len:, :]

        prefix_equal = torch.equal(pref_a.cpu(), pref_b.cpu())
        prefix_max_abs_diff = (
            (pref_a - pref_b).abs().max().item() if pref_a.numel() else 0.0
        )
        suffix_equal = torch.equal(suf_a.cpu(), suf_b.cpu())
        suffix_max_abs_diff = (
            (suf_a - suf_b).abs().max().item() if suf_a.numel() else 0.0
        )

        if verbose:
            print(f"[Prefix len = {pc_prefix_len}]")
            print(f"  Prefix equal (byte-level): {prefix_equal}")
            print(f"  Prefix max |Δ|: {prefix_max_abs_diff:.3e}")
            print(f"  Suffix equal (byte-level): {suffix_equal}")
            print(f"  Suffix max |Δ|: {suffix_max_abs_diff:.3e}")

        return {
            "prefix_equal": prefix_equal,
            "prefix_max_abs_diff": prefix_max_abs_diff,
            "suffix_equal": suffix_equal,
            "suffix_max_abs_diff": suffix_max_abs_diff,
            "out_nocontext": out_nocontext,
            "out_withcontext": out_withcontext,
        }
    finally:
        if was_training:
            decoder.train()


def gaussian_kernel_1d(
    C: int, sigma: float, circular: bool, device=None, dtype=torch.float32
):
    i = torch.arange(C, device=device, dtype=dtype)
    d = (i[None, :] - i[:, None]).abs()  # (C, C)
    if circular:
        d = torch.minimum(d, C - d)  # 环形距离
    denom = 2.0 * (sigma**2 + 1e-12)
    K = torch.exp(-(d**2) / denom)  # (C, C)
    K = K / K.sum(dim=1, keepdim=True)
    return K  # 每一行是某真类的软目标分布


# ori_semantic_lst = [1, 2, 11, 12, 13, 15, 16, 18, 19, 23, 25, 26, 27]
# primitive_folder = "<external-path-required>"

# semantic_lst = [x for x in range(len(ori_semantic_lst))]
# primitive_lst = [len(os.listdir(f"{primitive_folder}/semantic_{x}")) for x in ori_semantic_lst]

# num_semantic = len(primitive_lst)
# num_primitive = sum(primitive_lst)

# offsets = [0]
# for c in primitive_lst[:-1]:
#     offsets.append(offsets[-1] + c)      # [0, 8, 16, 25, 35]

# # 全局 primitive_id -> semantic_id
# primitive2semantic = torch.zeros(num_primitive, dtype=torch.long)
# for sid, (off, cnt) in enumerate(zip(offsets, primitive_lst)):
#     primitive2semantic[off:off + cnt] = sid

# semantic2primitive = torch.zeros(num_semantic, num_primitive, dtype=torch.bool)
# for sid, (off, cnt) in enumerate(zip(offsets, primitive_lst)):
#     semantic2primitive[sid, off:off + cnt] = True


@contextmanager
def mask_cross_attn_on_prefix(decoder, query_mask_bool: torch.Tensor):
    """
    仅在 cross-attention（即 forward(context=...)）时，把前缀查询位置的跨注意力残差置零。
    query_mask_bool: [B, L]，True=允许跨注意力；False=禁用（pc 前缀位置）
    """
    qmask_base = query_mask_bool.unsqueeze(-1)  # [B, L, 1] 用于广播
    handles = []

    def hook(module, args, kwargs, output):
        # 自注意力（没传 context）直接跳过
        if kwargs.get("context", None) is None:
            return output

        # 取出 out，用它的形状对齐 mask
        if isinstance(output, tuple):
            out = output[0]
        else:
            out = output

        B, L, _ = out.shape

        # 广播 / 裁剪到当前步长度（自回归 + KV cache 时很重要）
        qmask = qmask_base
        if qmask.shape[0] == 1 and B > 1:
            qmask = qmask.expand(B, -1, -1)  # 允许 batch=1 的广播
        if qmask.shape[1] != L:
            qmask = qmask[:, :L]  # 只取本步长度
        if qmask.dtype != out.dtype:
            qmask = qmask.to(out.dtype)

        out_masked = out * qmask  # [B, L, D]

        # 如果还有 intermediates，就原样带回去
        if isinstance(output, tuple):
            # （可选）如果你还想把注意力权重也同步置零，解除注释以下几行
            # inter = output[1]
            # if hasattr(inter, 'attn') and inter.attn is not None:
            #     # inter.attn: [B, H, Lq, Lk]，按查询维度做 mask
            #     attn_qmask = qmask.squeeze(-1).unsqueeze(1).unsqueeze(-1)  # [B,1,L,1]
            #     inter.attn = inter.attn * attn_qmask
            return (out_masked, *output[1:])
        else:
            return out_masked

    # 给所有 Attention 模块挂上 hook（只在带 context 的调用时生效）
    for m in decoder.modules():
        if m.__class__.__name__ == "Attention":
            h = m.register_forward_hook(hook, with_kwargs=True)
            handles.append(h)

    try:
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def eval_no_grad(module):
    was_training = module.training
    try:
        module.eval()
        with torch.no_grad():
            yield
    finally:
        # 仅在进入前是 train 的情况下恢复
        if was_training:
            module.train()


@contextmanager
def bn_eval_only(module):
    # Hybrid mode for the lib forward path: BN goes into eval (no running-stats
    # update, no train-mode batch-stat normalization), but the autograd graph
    # remains live so gradients can flow back through learnable conv/linear
    # weights. Used when cond_grad_on_lib=True so we get the old version's
    # gradient signal without inheriting its BN-pollution bug.
    bn_types = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
    )
    saved = []
    for m in module.modules():
        if isinstance(m, bn_types):
            saved.append((m, m.training))
            m.eval()
    try:
        yield
    finally:
        for m, t in saved:
            m.train(t)


def masked_cross_entropy_with_logits(
    primitive_logits,
    gt_primitive,
    gt_semantic,
    primitive2semantic,
    semantic2primitive,
    ignore_index=-1,
    reduction="mean",
):
    primitive2semantic = primitive2semantic.to(primitive_logits.device)
    semantic2primitive = semantic2primitive.to(primitive_logits.device)

    ## get primitive mask corresponding to gt_semantic
    mask = semantic2primitive[gt_semantic]

    ## masked log softmax
    masked_logits = primitive_logits.masked_fill(~mask, float("-inf"))

    # 去掉被 ignore 的 pad 位置；把 (B, N, P) 展平成 (B*N, P)
    flat_logits = masked_logits.view(-1, masked_logits.size(-1))
    flat_targets = gt_primitive.view(-1)

    if ignore_index is not None:
        loss = F.cross_entropy(
            flat_logits, flat_targets, ignore_index=ignore_index, reduction=reduction
        )
    else:
        loss = F.cross_entropy(flat_logits, flat_targets, reduction=reduction)
    return loss


# Perceiver Resampler: compress variable-length encoder features into fixed-length prefix tokens


class PerceiverResampler(Module):
    """
    Perceiver-style resampler that compresses variable-length point cloud features
    into a fixed number of latent tokens via learnable queries + cross-attention.
    Inspired by MeshAnything / Flamingo / BLIP-2 Q-Former.
    """

    def __init__(self, input_dim, hidden_dim, num_latents, depth=2, heads=8):
        super().__init__()
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.randn(num_latents, hidden_dim) * 0.02)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.cross_attn_norm_q = nn.LayerNorm(hidden_dim)
        self.cross_attn_norm_kv = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.self_attn_layers = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, key_padding_mask=None):
        """
        Args:
            x: [B, N, input_dim] variable-length encoder features
            key_padding_mask: [B, N] bool, True = valid position
        Returns:
            [B, num_latents, hidden_dim]
        """
        B = x.shape[0]
        x = self.input_proj(x)  # [B, N, hidden_dim]
        q = repeat(self.latents, "n d -> b n d", b=B)
        q_norm = self.cross_attn_norm_q(q)
        x_norm = self.cross_attn_norm_kv(x)
        # PyTorch MHA key_padding_mask: True = IGNORE; our mask has True = valid → invert
        attn_key_mask = ~key_padding_mask if key_padding_mask is not None else None
        latents, _ = self.cross_attn(
            q_norm, x_norm, x_norm, key_padding_mask=attn_key_mask
        )
        latents = latents + q  # residual
        # latents = self.self_attn_layers(latents)                     # [B, K, hidden_dim]
        return self.output_norm(latents)


# FiLM block


class FiLM(Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = default(dim_out, dim)

        self.to_gamma = nn.Linear(dim, dim_out, bias=False)
        self.to_beta = nn.Linear(dim, dim_out)

        self.gamma_mult = nn.Parameter(
            torch.zeros(
                1,
            )
        )
        self.beta_mult = nn.Parameter(
            torch.zeros(
                1,
            )
        )

    def forward(self, x, cond):
        gamma, beta = self.to_gamma(cond), self.to_beta(cond)
        # gamma, beta = tuple(rearrange(t, 'b d -> b 1 d') for t in (gamma, beta))

        # for initializing to identity

        gamma = 1 + self.gamma_mult * gamma.tanh()
        beta = beta.tanh() * self.beta_mult

        # classic film

        return x * gamma + beta


# gateloop layers


class GateLoopBlock(Module):
    def __init__(self, dim, *, depth, use_heinsen=True):
        super().__init__()
        self.gateloops = ModuleList([])

        for _ in range(depth):
            gateloop = SimpleGateLoopLayer(dim=dim, use_heinsen=use_heinsen)
            self.gateloops.append(gateloop)

    def forward(self, x, cache=None):
        received_cache = exists(cache)

        if is_tensor_empty(x):
            return x, None

        if received_cache:
            prev, x = x[:, :-1], x[:, -1:]

        cache = default(cache, [])
        cache = iter(cache)

        new_caches = []
        for gateloop in self.gateloops:
            layer_cache = next(cache, None)
            out, new_cache = gateloop(x, cache=layer_cache, return_cache=True)
            new_caches.append(new_cache)
            x = x + out

        if received_cache:
            x = torch.cat((prev, x), dim=-2)

        return x, new_caches


def top_k_2(logits, frac_num_tokens=0.1, k=None):
    num_tokens = logits.shape[-1]

    k = default(k, ceil(frac_num_tokens * num_tokens))
    k = min(k, num_tokens)

    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(2, ind, val)
    return probs


def soft_argmax(labels):
    indices = torch.arange(labels.size(-1), dtype=labels.dtype, device=labels.device)
    soft_argmax = torch.sum(labels * indices, dim=-1)
    return soft_argmax


def _init_seed_param(shape, scale=0.02):
    p = nn.Parameter(torch.empty(*shape))
    nn.init.trunc_normal_(p, std=scale, a=-0.08, b=0.08)
    return p


class AttentionPooling(nn.Module):
    """
    Perceiver-style cross-attention pooling. Aggregates {x_i} to a single token
    using a learnable query.

    Standard pre-norm transformer block:
        attn_out = MHA(LN(q), LN(x), x, key_padding_mask=mask)
        pooled   = q + attn_out                       # residual on query
        pooled   = pooled + FFN(LN(pooled))           # residual on FFN

    Inputs:
        x:    (B, N, D)
        mask: (B, N)  True = padding (i.e. the "key_padding_mask" convention used
              by torch.nn.MultiheadAttention)
    Returns:
        pooled: (B, 1, D)
        attn:   (B, 1, N)
    """

    def __init__(self, d_model, n_heads=4, q_learnable=True, dropout=0.0, ffn_mult=4):
        super().__init__()
        self.d_model = d_model
        self.q_learnable = q_learnable
        if q_learnable:
            self.query = _init_seed_param((1, 1, d_model))
        else:
            self.register_parameter("query", None)
        # 3 LayerNorms (pre-norm style): one on q, one on kv-input, one on FFN-input.
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Linear(ffn_mult * d_model, d_model),
        )

    def forward(self, x, mask=None, query=None):
        B, N, D = x.shape
        assert D == self.d_model
        if query is None:
            if self.q_learnable:
                q = self.query.expand(B, -1, -1)  # (B, 1, D)
            else:
                # Neutral query = mean over keys (detached so no extra gradient path).
                q = x.mean(dim=1, keepdim=True).detach()
        else:
            q = query

        # Pre-norm cross-attention with residual on q.
        q_n = self.norm_q(q)
        kv_n = self.norm_kv(x)
        attn_out, attn = self.mha(q_n, kv_n, x, key_padding_mask=mask)
        pooled = q + attn_out

        # Pre-norm FFN with residual.
        pooled = pooled + self.ffn(self.norm_ffn(pooled))
        return pooled, attn  # (B, 1, D), (B, 1, N)


class PMA(nn.Module):
    """
    Set Transformer 风格的 PMA：用 S 个可学习 seed 将 {x_i} 聚合为 S 个 token。
    x: (B, N, D)
    mask: (B, N)  True=padding/无效
    返回 tokens: (B, S, D), attn: (B, S, N)
    """

    def __init__(
        self,
        d_model,
        n_seeds=4,
        n_heads=4,
        dropout=0.0,
        individual_query=False,
        num_primitive=None,
    ):
        super().__init__()
        self.individual_query = individual_query
        self.d_model = d_model
        if self.individual_query:
            self.num_primitive = num_primitive
            self.seeds = _init_seed_param((self.num_primitive, n_seeds, d_model))
        else:
            self.seeds = _init_seed_param((1, n_seeds, d_model))
        self.mha = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x, mask=None, seeds=None):
        """
        seeds: (B, S, D) 或 None；若 None 则使用可学习 seeds
        """
        B, N, D = x.shape
        assert D == self.d_model
        if self.individual_query:
            q = self.seeds
        else:
            if seeds is None:
                q = self.seeds.expand(B, -1, -1)  # (B,S,D)
            else:
                q = seeds  # 外部给定的多查询

        tokens, attn = self.mha(q, x, x, key_padding_mask=mask)  # (B,S,D), (B,S,N)
        y = self.norm(tokens + self.ffn(tokens))
        return y, attn


def get_sin_cos_positional_embedding(num_positions, embedding_dim):
    """
    Generate sinusoidal positional embeddings.

    :param num_positions: Number of positions (sequence length)
    :param embedding_dim: Dimensionality of the embeddings
    :return: A matrix of shape (num_positions, embedding_dim) with positional embeddings
    """
    # Initialize the position indices and dimension indices
    position = np.arange(num_positions)[:, np.newaxis]  # Shape: (num_positions, 1)
    dim = np.arange(embedding_dim)[np.newaxis, :]  # Shape: (1, embedding_dim)

    # Compute the term for scaling position and dimension
    angle_rates = 1 / np.power(10000, (2 * (dim // 2)) / np.float32(embedding_dim))

    # Apply sin to even indices and cos to odd indices
    positional_embedding = np.zeros((num_positions, embedding_dim))
    positional_embedding[:, 0::2] = np.sin(
        position * angle_rates[:, 0::2]
    )  # Sin on even indices
    positional_embedding[:, 1::2] = np.cos(
        position * angle_rates[:, 1::2]
    )  # Cos on odd indices

    return positional_embedding


def get_fourier_angle_embedding(num_discrete_angles, embedding_dim):
    assert embedding_dim % 2 == 0, "embedding_dim must be even for sin/cos pairs"
    num_freq_pairs = embedding_dim // 2
    K = num_discrete_angles
    n_freq = num_freq_pairs

    idx = torch.arange(K).float()  # 0..K-1
    theta = 2 * math.pi * idx / K  # 离散角
    feats = []
    for m in range(1, n_freq + 1):
        feats.append(torch.sin(m * theta))
        feats.append(torch.cos(m * theta))
    codebook = torch.stack(feats, dim=-1)  # [K, 2*n_freq]
    codebook = F.normalize(codebook, dim=-1)  # 归一化，数值更稳
    return codebook


#### add for geom_embed
import torch, torch.nn as nn, torch.nn.functional as F
from math import sqrt


class LibAttnFuse(nn.Module):
    def __init__(self, dim, n_heads=4, init_gate=0.0, bias_to_id=2.0):
        super().__init__()
        self.h = n_heads
        self.dk = dim // n_heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
        self.ln_x = nn.LayerNorm(dim)
        self.ln_lib = nn.LayerNorm(dim)
        self.gate = nn.Parameter(torch.tensor(init_gate))  # 零起点门控
        self.bias_to_id = bias_to_id  # 给正确 id 的加性偏置(可调)

    def forward(self, x_inst, lib_embed, primitive_ids, allow_ids=None):
        """
        x_inst:      [B, N, D]   实例 tokens（当前层输入或子层输出）
        lib_embed:   [B, L, D]   primitive_lib_embed（作为条件库）
        primitive_ids: [B, N]    每个实例的 primitive_id (long)
        allow_ids:   [B, N, L] 可选布尔掩码，True=允许；不传则仅允许对应 id
        """
        B, N, D = x_inst.shape
        L = lib_embed.size(1)

        # 归一 + 投影为多头
        q = (
            self.q(self.ln_x(x_inst)).view(B, N, self.h, self.dk).transpose(1, 2)
        )  # [B,h,N,dk]
        k = (
            self.k(self.ln_lib(lib_embed)).view(B, L, self.h, self.dk).transpose(1, 2)
        )  # [B,h,L,dk]
        v = self.v(lib_embed).view(B, L, self.h, self.dk).transpose(1, 2)  # [B,h,L,dk]

        # 注意力 logits
        logits = torch.matmul(q, k.transpose(-1, -2)) / sqrt(self.dk)  # [B,h,N,L]

        # 只看指定 id（或其集合）
        if allow_ids is None:
            mask = torch.zeros(B, N, L, dtype=torch.bool, device=x_inst.device)
            bidx = torch.arange(B, device=x_inst.device)[:, None]
            nidx = torch.arange(N, device=x_inst.device)[None, :]
            mask[bidx, nidx, primitive_ids] = True  # 只放开对应 id
        else:
            mask = allow_ids  # True=允许

        # 给正确 id 额外正偏置，其他被屏蔽为 -inf
        bias = torch.zeros_like(logits)
        bias = bias + self.bias_to_id * mask[:, None, :, :]  # [B,1,N,L]广播到[h]
        logits = logits + bias
        logits = logits.masked_fill(~mask[:, None, :, :], float("-inf"))

        attn = logits.softmax(dim=-1)  # [B,h,N,L]
        z = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, D)  # [B,N,D]
        z = self.o(z)

        # 零起点门控残差（不破坏分布）
        # return x_inst + torch.tanh(self.gate) * z
        return z


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks
    # return loss.sum()


dice_loss_jit = torch.jit.script(dice_loss)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks
    # return loss.mean(1).sum()


sigmoid_ce_loss_jit = torch.jit.script(sigmoid_ce_loss)  # type: torch.jit.ScriptModule

# import torch
# import torch.nn.functional as F


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    alpha: float = 0.25,
    gamma: float = 2.0,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape. (The raw logits)
        targets: A float tensor with the same shape as inputs. (0 or 1)
        num_masks: Normalization factor
        alpha: Weighting factor in range (0,1) to balance positive vs negative classes.
               0.25 means positive class gets lower base weight, BUT combined with gamma
               it usually balances out large negative counts.
               For high imbalance (few positives), try alpha=0.75 or 0.8 to boost positives.
        gamma: Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.
    """
    prob = inputs.sigmoid()

    # 计算 BCE Term
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    # 计算 Focal Term
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    # Apply alpha weighting
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_masks


# JIT 编译
sigmoid_focal_loss_jit = torch.jit.script(sigmoid_focal_loss)


from datasets.dataclass import SVGData, VecData


class HierarchicalPrimitiveLoss(nn.Module):
    def __init__(self, primitive_id2semantic, coarse_weight=1.0):
        """
        Args:
            primitive_id2semantic: (num_primitive,) 的 Tensor。
                                   表示每个 primitive 对应的 semantic 类别 ID。
                                   例如: [0, 0, 1, 2, 0 ...]
            coarse_weight:         粗粒度（语义类别）损失的权重，默认 1.0。
        """
        super().__init__()
        self.coarse_weight = coarse_weight

        # 将映射表注册为 buffer，这样它会自动随模型移动到 GPU，但不是可训练参数
        self.register_buffer("primitive_id2semantic", primitive_id2semantic)

        # 计算语义类别的总数
        self.num_semantics = primitive_id2semantic.max().item() + 1

    def forward(self, primitive_logits, gt_primitive, gt_semantic_pad=None):
        """
        Args:
            primitive_logits: (B, num_inst, num_primitive) - 模型输出的未归一化分数
            gt_primitive:     (B, num_inst) - Ground Truth 的 Primitive ID。
                              无效/Padding 部分必须为 -1。

        Returns:
            total_loss: scalar
            loss_dict:  dict containing 'fine_loss' and 'coarse_loss' for logging
        """
        # 1. 数据扁平化 (Flatten)
        # 展平成 (Total_Inst, Num_Primitive) 以便统一处理
        logits_flat = primitive_logits.flatten(0, 1)  # (B*N, P)
        gt_flat = gt_primitive.flatten()  # (B*N,)

        # 2. 过滤无效样本 (Masking)
        # 只需要计算 gt != -1 的部分
        valid_mask = gt_flat != -1

        # 如果当前 batch 全是 padding (虽然罕见)，直接返回 0
        if not valid_mask.any():
            return (
                torch.tensor(0.0, device=primitive_logits.device, requires_grad=True),
                {},
            )

        valid_logits = logits_flat[valid_mask]  # (M, P)
        valid_gt = gt_flat[valid_mask]  # (M,)

        # --- Part 1: 细粒度损失 (Fine-grained Loss) ---
        # 强制模型预测出具体的那个 primitive (例如: 必须是"椅子A")
        loss_fine = F.cross_entropy(valid_logits, valid_gt)

        # --- Part 2: 粗粒度损失 (Coarse-grained / Hierarchical Loss) ---
        # 强制模型预测出的概率分布，在聚合后能对应正确的 semantic 类别 (例如: 必须是"椅子"类)

        # A. 计算 Softmax 概率
        probs = F.softmax(valid_logits, dim=-1)  # (M, P)

        # B. 概率聚合 (Probability Aggregation)
        # 初始化粗粒度概率矩阵 (M, Num_Semantics)
        semantic_probs = torch.zeros(
            probs.shape[0], self.num_semantics, device=probs.device, dtype=probs.dtype
        )

        # 核心操作: 将属于同一语义类别的 primitive 概率相加
        # index_add_(dim, index, source): 将 source 加到 self 的 index 位置
        # 这里的 self.primitive_id2semantic 会自动广播应用到每一行
        semantic_probs.index_add_(1, self.primitive_id2semantic, probs)

        # C. 加上极小值防止 log(0)
        semantic_probs = semantic_probs + 1e-8

        # D. 获取粗粒度的 Ground Truth
        # 利用映射表，将细粒度 GT 转换为粗粒度 GT
        gt_semantic = self.primitive_id2semantic[valid_gt]  # (M,)

        # E. 计算 NLL Loss (因为我们已经有了概率，取 log 后即为 LogSoftmax)
        # NLLLoss(log(p), target) 等价于 CrossEntropy
        loss_coarse = F.nll_loss(torch.log(semantic_probs), gt_semantic)

        # 3. 总损失
        total_loss = loss_fine + self.coarse_weight * loss_coarse

        return total_loss, {
            "loss_fine": loss_fine.detach(),
            "loss_coarse": loss_coarse.detach(),
            "loss_total": total_loss.detach(),
        }


def _ensure_float(x):
    return x.float() if not x.is_floating_point() else x


@torch.no_grad()
def _make_onehot_from_bins(gt_bins: torch.Tensor, K: int) -> torch.Tensor:
    """
    gt_bins: (B, NP, 2) int64 in [0,K-1]
    return:  (B, NP, 2, K) one-hot
    """
    return F.one_hot(gt_bins.clamp(0, K - 1), num_classes=K).to(torch.float32)


def make_soft_target_two_neighbours(
    gt_cont: torch.Tensor,
    bin_centers: torch.Tensor,
) -> torch.Tensor:
    """
    连续值 -> 相邻两个bin的线性插值软标签（推荐，最贴合量化）

    gt_cont:      (B, NP, 2) float
    bin_centers:  (K,) 或 (1,1,1,K) float，要求单调递增
    return q:     (B, NP, 2, K)
    """
    # shapes
    if bin_centers.dim() == 1:
        centers = bin_centers.view(1, 1, 1, -1)  # (1,1,1,K)
    else:
        centers = bin_centers
    B, NP, D = gt_cont.shape
    K = centers.shape[-1]

    # 在 centers 上找插入位置：idx_hi 是第一个 centers[idx] > y 的位置
    # searchsorted 需要 1D，故 flatten 再还原
    y = gt_cont.reshape(-1)  # (B*NP*2,)
    c1d = centers.reshape(-1)  # (K,)
    idx_hi = torch.searchsorted(c1d, y, right=False)  # in [0..K]
    idx_hi = idx_hi.clamp(1, K - 1)  # 保证有左右邻居
    idx_lo = idx_hi - 1

    c_lo = c1d[idx_lo]  # (B*NP*2,)
    c_hi = c1d[idx_hi]
    # 线性插值权重
    t = (y - c_lo) / (c_hi - c_lo + 1e-12)
    w_lo = (1.0 - t).clamp(0.0, 1.0)
    w_hi = t.clamp(0.0, 1.0)

    q = torch.zeros((y.numel(), K), device=gt_cont.device, dtype=torch.float32)
    q.scatter_(1, idx_lo.unsqueeze(1), w_lo.unsqueeze(1))
    q.scatter_add_(1, idx_hi.unsqueeze(1), w_hi.unsqueeze(1))
    return q.view(B, NP, D, K)


def make_soft_target_gaussian(
    gt_cont: torch.Tensor,
    bin_centers: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """
    连续值 -> 高斯软标签（更平滑，sigma 用“中心间距”的倍数较直观）

    gt_cont:     (B, NP, 2)
    bin_centers: (K,) 或 (1,1,1,K)
    return q:    (B, NP, 2, K)
    """
    if bin_centers.dim() == 1:
        centers = bin_centers.view(1, 1, 1, -1)  # (1,1,1,K)
    else:
        centers = bin_centers
    y = gt_cont.unsqueeze(-1)  # (B,NP,2,1)
    dist2 = (y - centers) ** 2
    q = torch.exp(-0.5 * dist2 / (sigma**2 + 1e-12))
    q = q / (q.sum(dim=-1, keepdim=True) + 1e-12)
    return q


def cdf_distance_w1(
    p: torch.Tensor,
    q: torch.Tensor,
    bin_width: torch.Tensor | float | None = None,
    valid_mask: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    1D Wasserstein-1 / EMD via CDF L1:
      W1(p,q) = sum_k |CDF_p(k) - CDF_q(k)| * Δ
    p,q: (B, NP, 2, K) each sums to 1 on last dim
    bin_width: scalar or (K-1,) or (K,) or None (treat as 1)
    valid_mask: (B, NP, 2) bool/0-1, True means include
    """
    p = _ensure_float(p)
    q = _ensure_float(q)
    cdf_p = p.cumsum(dim=-1)
    cdf_q = q.cumsum(dim=-1)
    diff = (cdf_p - cdf_q).abs()  # (B,NP,2,K)

    if bin_width is None:
        w = 1.0
    else:
        # allow scalar or tensor; broadcast to (...,K)
        if not torch.is_tensor(bin_width):
            w = float(bin_width)
        else:
            bw = bin_width.to(p.device).to(p.dtype)
            if bw.dim() == 1:
                if bw.numel() == p.size(-1) - 1:
                    # widths between centers; pad last
                    bw = torch.cat([bw, bw[-1:]], dim=0)
                bw = bw.view(1, 1, 1, -1)
            w = bw

    per = diff * w  # (B,NP,2,K)
    per = per.sum(dim=-1)  # (B,NP,2)

    if valid_mask is not None:
        vm = valid_mask.to(per.dtype)
        per = per * vm
        denom = vm.sum().clamp_min(1.0)
    else:
        denom = per.numel()

    if reduction == "mean":
        return per.sum() / denom
    if reduction == "sum":
        return per.sum()
    return per  # no reduction


def cdf_distance_crps(
    p: torch.Tensor,
    q: torch.Tensor,
    bin_width: torch.Tensor | float | None = None,
    valid_mask: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    CRPS via CDF L2 (discrete approximation):
      CRPS(p,q) = sum_k (CDF_p(k) - CDF_q(k))^2 * Δ
    p,q: (B, NP, 2, K)
    """
    p = _ensure_float(p)
    q = _ensure_float(q)
    cdf_p = p.cumsum(dim=-1)
    cdf_q = q.cumsum(dim=-1)
    diff2 = (cdf_p - cdf_q) ** 2  # (B,NP,2,K)

    if bin_width is None:
        w = 1.0
    else:
        if not torch.is_tensor(bin_width):
            w = float(bin_width)
        else:
            bw = bin_width.to(p.device).to(p.dtype)
            if bw.dim() == 1:
                if bw.numel() == p.size(-1) - 1:
                    bw = torch.cat([bw, bw[-1:]], dim=0)
                bw = bw.view(1, 1, 1, -1)
            w = bw

    per = (diff2 * w).sum(dim=-1)  # (B,NP,2)

    if valid_mask is not None:
        vm = valid_mask.to(per.dtype)
        per = per * vm
        denom = vm.sum().clamp_min(1.0)
    else:
        denom = per.numel()

    if reduction == "mean":
        return per.sum() / denom
    if reduction == "sum":
        return per.sum()
    return per


# import torch
# import torch.nn.functional as F

# ── Straight-Through helpers ──────────────────────────────────────────────────
RECON_DECODE_MODES = frozenset({"soft_expectation", "hard_st"})
RECON_TR_MODES = frozenset({"joint_bidir", "factorized"})
RECON_ROTATION_DIRECTIONS = frozenset(
    {
        "bidirectional",
        "source_to_template",
        "template_to_source",
    }
)


def validate_recon_decode_mode(mode: str) -> str:
    """Validate and return the reconstruction-logit decoding policy."""
    if mode not in RECON_DECODE_MODES:
        choices = "/".join(sorted(RECON_DECODE_MODES))
        raise ValueError(f"recon_decode_mode must be {choices}, got {mode!r}")
    return mode


def validate_recon_tr_mode(mode: str) -> str:
    """Validate and return the reconstruction T/R coupling policy."""
    if mode not in RECON_TR_MODES:
        choices = "/".join(sorted(RECON_TR_MODES))
        raise ValueError(f"recon_tr_mode must be {choices}, got {mode!r}")
    return mode


def validate_recon_rotation_direction(direction: str) -> str:
    """Validate the factorized rotation geometry direction."""
    if direction not in RECON_ROTATION_DIRECTIONS:
        choices = "/".join(sorted(RECON_ROTATION_DIRECTIONS))
        raise ValueError(
            f"recon_rotation_direction must be {choices}, got {direction!r}"
        )
    return direction


def deterministic_st_argmax_onehot(logits: torch.Tensor) -> torch.Tensor:
    """Return hard argmax values with the ordinary softmax backward pass.

    This estimator is deterministic: its forward value is exactly a one-hot
    encoding of ``argmax(logits)`` and it consumes no random numbers.
    """
    probabilities = logits.softmax(dim=-1)
    hard = F.one_hot(logits.argmax(dim=-1), num_classes=logits.shape[-1]).to(
        dtype=logits.dtype
    )
    return (hard - probabilities).detach() + probabilities


def st_gumbel_onehot(
    logits: torch.Tensor,
    tau: float = 1.0,
    stochastic: bool = True,
) -> torch.Tensor:
    """
    Straight-through Gumbel estimator.
    Forward : hard one-hot  (argmax of Gumbel-perturbed logits)
    Backward: gradient flows through the soft Gumbel-softmax approximation,
              so upstream logits receive gradients from all downstream losses.
    """
    if not stochastic:
        return F.one_hot(logits.argmax(dim=-1), num_classes=logits.shape[-1]).to(
            dtype=logits.dtype
        )

    y_soft = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
    y_hard = torch.zeros_like(y_soft).scatter_(
        -1, y_soft.argmax(dim=-1, keepdim=True), 1.0
    )
    return (y_hard - y_soft).detach() + y_soft  # forward=hard, backward=soft


def onehot_to_embed(onehot: torch.Tensor, embedding: nn.Embedding) -> torch.Tensor:
    """Matrix-multiply one-hot (or soft) with embedding table, preserving gradients."""
    return onehot @ embedding.weight


def _ss_mix(
    pred_embed: torch.Tensor, gt_embed: torch.Tensor, ss_eps: float
) -> torch.Tensor:
    """Scheduled-sampling mix: each primitive slot independently uses pred (prob=ss_eps)
    or GT (prob=1-ss_eps). Gradient only flows through the pred branch."""
    if ss_eps <= 0.0:
        return gt_embed.detach()
    if ss_eps >= 1.0:
        return pred_embed
    B, np_ = pred_embed.shape[:2]
    mask = (torch.rand(B, np_, 1, device=pred_embed.device) < ss_eps).float()
    return mask * pred_embed + (1.0 - mask) * gt_embed.detach()


# ─────────────────────────────────────────────────────────────────────────────


def dist_loss_cdf(
    logits: torch.Tensor,
    target: torch.Tensor,
    pad_id: int,
    kind: str = "crps",  # "crps" or "w1"
    q_from_kernel: torch.Tensor | None = None,  # (C, C) or None -> one-hot
    bin_width: torch.Tensor | float | None = None,  # scalar or (C,) or (C-1,)
):
    """
    Distribution loss via CDF distance (vectorized).

    logits: (..., C)
    target: (...) int64
    q_from_kernel: (C, C) row i gives target distribution for class i (optional)
    kind:
      - "w1"   : sum_k |CDF_p - CDF_q| * Δ
      - "crps" : sum_k (CDF_p - CDF_q)^2 * Δ
    bin_width:
      - None -> treat as 1
      - scalar
      - tensor of shape (C-1,) (between bins) or (C,) (per-bin weight)
    """
    C = logits.size(-1)
    mask = target != pad_id
    if not mask.any():
        return logits.new_zeros(())

    # flatten & select valid
    logits_f = logits.reshape(-1, C)
    tgt_f = target.reshape(-1).long()
    mask_f = mask.reshape(-1)

    logits_v = logits_f[mask_f]  # (M, C)
    tgt_v = tgt_f[mask_f]  # (M,)

    # predicted distribution
    p = F.softmax(logits_v, dim=-1)  # (M, C)

    # target distribution q
    if q_from_kernel is None:
        q = F.one_hot(tgt_v.clamp(0, C - 1), num_classes=C).to(p.dtype)  # (M, C)
    else:
        K = q_from_kernel.to(device=p.device, dtype=p.dtype)  # (C, C)
        q = K.index_select(0, tgt_v.clamp(0, C - 1))  # (M, C)
        # safety normalize (in case K rows don't sum to 1 perfectly)
        q = q / (q.sum(dim=-1, keepdim=True) + 1e-12)

    # CDFs
    cdf_p = p.cumsum(dim=-1)  # (M, C)
    cdf_q = q.cumsum(dim=-1)  # (M, C)
    diff = cdf_p - cdf_q

    # weights (Δ)
    if bin_width is None:
        w = None
    else:
        if not torch.is_tensor(bin_width):
            w = torch.tensor(float(bin_width), device=p.device, dtype=p.dtype).view(
                1, 1
            )  # scalar
        else:
            bw = bin_width.to(device=p.device, dtype=p.dtype)
            if bw.dim() != 1:
                raise ValueError("bin_width must be scalar or 1D tensor")
            if bw.numel() == C - 1:
                bw = torch.cat([bw, bw[-1:]], dim=0)  # pad to (C,)
            if bw.numel() != C:
                raise ValueError(
                    f"bin_width must have length C or C-1 (got {bw.numel()}), C={C}"
                )
            w = bw.view(1, C)  # broadcast to (M,C)

    # per-sample dist
    if kind.lower() == "w1":
        per = diff.abs()
    elif kind.lower() == "crps":
        per = diff.pow(2)
    else:
        raise ValueError("kind must be 'w1' or 'crps'")

    if w is not None:
        per = per * w

    # per = per.sum(dim=-1)  # (M,)
    per = per.mean(dim=-1)
    return per.mean()


class PrimitiveTransformerDiscrete(Module, PyTorchModelHubMixin):
    @typecheck
    def __init__(
        self,
        *,
        num_discrete_scale=128,
        continuous_range_scale: List[float, float] = [0, 1],
        dim_scale_embed=64,
        num_discrete_rotation=180,
        continuous_range_rotation: List[float, float] = [-180, 180],
        dim_rotation_embed=64,
        num_discrete_translation=128,
        continuous_range_translation: List[float, float] = [-1, 1],
        dim_translation_embed=64,
        # num_type = 3,
        num_semantic=13,
        num_primitive=121,
        dim_type_embed=64,
        embed_order="ctrs",
        bin_smooth_blur_sigma=0.5,
        dim: int | Tuple[int, int] = 512,
        flash_attn=True,
        attn_depth=12,
        attn_dim_head=64,
        attn_heads=16,
        attn_kwargs: dict = dict(ff_glu=True, attn_num_mem_kv=4),
        max_primitive_len=144,
        dropout=0.0,
        coarse_pre_gateloop_depth=2,
        coarse_post_gateloop_depth=0,
        coarse_adaptive_rmsnorm=False,
        gateloop_use_heinsen=False,
        pad_id=-1,
        num_sos_tokens=None,
        condition_on_shape=True,
        shape_cond_with_cross_attn=False,
        shape_cond_with_film=False,
        shape_cond_with_cat=False,
        shape_cond_with_perceiver_prefix=False,
        perceiver_num_latents=256,
        perceiver_depth=2,
        perceiver_heads=8,
        shape_condition_model_type="michelangelo",
        shape_condition_len=1,
        shape_condition_dim=None,
        cross_attn_num_mem_kv=4,  # needed for preventing nan when dropping out shape condition
        loss_weight: dict = dict(
            # eos = 1.0,
            # type = 1.0,
            # scale = 1.0,
            # rotation = 1.0,
            # translation = 1.0,
            # reconstruction = 1.0,
            # scale_huber = 1.0,
            # rotation_huber = 1.0,
            # translation_huber = 1.0,
            eos=1.0,
            reconstruction=1.0,
            rotation=1.0,
            scale=1.0,
            translation=1.0,
            # type = 1.0
            semantic_id=1.0,
            primitive_id=1.0,
            huber_translation=1.0,
            huber_rotation=1.0,
            huber_scale=1.0,
        ),
        bs_pc_dir=None,
        ### Added by Honghao
        cad_pc_conditioner_path=None,
        freeze_backbone=True,
        use_img_conditioner=False,
        img_conditioner_model_name="hrnet_w18",
        img_conditioner_feature_index=-1,
        primitive_dir=None,
        # ── Lib-pooling strategy ──
        # Pass `pooling_type` directly (one of 'mean', 'max', 'attn',
        # 'mean_max', 'pma') — single source of truth.
        # The old per-strategy bools (mean_pooling/attn_pooling/pma_pooling)
        # remain accepted for backward compatibility and are mapped onto
        # `pooling_type` if it is left None.
        pooling_type: str | None = None,
        mean_pooling=False,
        attn_pooling=False,
        pma_pooling=False,
        primitive_cond_with_cat=False,
        filter_primitive_lib=False,
        merge_attn=False,
        mask_cross=False,
        normalize_coord=False,
        sin_cos_embedding=False,
        pred_flip=False,
        dim_flip_embed=16,
        sin_cos_embedding_rotation=False,
        fourier_embedding_rotation=False,
        smooth_loss=False,
        add_pe=False,
        label_padding=False,
        direct_recon=False,
        use_geom_embed=False,
        circular_rotate=False,
        two_stage_decoder=False,
        pretrain_weight=None,
        use_prefix_decoder=False,
        bg_pred=False,
        scene_bound=30000.0,
        keep_ratio=0.1,
        min_keep=50,
        use_slot_attn=False,
        mask_guide=False,
        use_dense_feat=False,
        attn_hard_constraint=False,
        global_local_ca=False,
        rm_mask_pred=False,
        init_attr_from_mask=False,
        loss_mask_type=None,
        return_point_mapping=False,
        aux_mask_supervision=False,
        supervise_attn_bias=False,
        individual_query=False,
        use_loftr=False,
        down_sampled_dense_feat=False,
        attn_constraint=False,
        attn_constraint_last_layer=False,
        sem_geom_decouple=False,
        geom_sem_decouple=False,
        attr_decouple=False,
        # Controlled semantic-token ablation for the attr-decoupled decoder.
        # The baseline serializes six tokens and feeds GT semantic to the
        # primitive head.  Setting attr_emit_semantic_token=False removes only
        # the serialized token; attr_primitive_semantic_condition controls the
        # independent direct head conditioning.
        attr_emit_semantic_token: bool = True,
        attr_primitive_semantic_condition: str = "gt",
        attr_semantic_supervision: bool = True,
        # Attr-decouple ablation: replace the learned primitive-ID embedding
        # with the selected template's encoded geometry.  Default False keeps
        # existing checkpoints/configs behaviorally unchanged.
        attr_replace_primitive_id_with_geom=False,
        pretrain=False,
        learnable_anchor_embedding=False,
        inst_wise_learnable_anchor_embedding=False,
        sem_prim_pretrain=False,
        lm_loss=False,
        backbone_config: dict = None,
        bg_fitting=False,
        bg_matching=None,
        fg_fitting=False,
        primitive_contrastive_learning=False,
        # ───────── Fit-test ablation flags (V2/V3 + cond_grad + cache) ─────────
        # All default OFF → behaviour identical to previous main.
        cond_grad_on_lib: bool = False,  # A: hybrid mode (BN eval + grad on) for lib forward
        cache_lib_feats: bool = False,  # B: cache conditioner output for lib (only valid when cond_grad_on_lib=False)
        use_lib_in_sem: bool = False,  # C: V2 — residual lib[primitive_id] into sem branch
        use_lib_in_geom: bool = False,  # D: V3 — residual lib[primitive_id] into geom branch
        # ──────────────────────────────────────────────────────────────────────
        primitive_mapping_path=None,
        primitive_coarse_loss_weight: float = 1.0,
        soft_ce=False,
        dist_loss=False,
        ## add for size pred ##
        size_pred=False,
        num_discrete_size=128,
        continuous_range_size: List[float, float] = [0, 1],
        dim_size_embed=16,
        gumbel_loss=False,
        # Predict relative scale with log-uniform bins.  Disabled by default so
        # existing checkpoints/configurations retain their exact semantics.
        scale_log=False,
        ## factorized AR for 2-channel attrs (translation, size): P(y|h, x) instead of P(y|h)
        factorized_xy=False,
        ## When True, zero out auxiliary "metadata" channels that the conditioner
        ## currently uses:
        ##   - layer_ids: normalized + concatenated as the 3rd coord channel in
        ##                _get_data_dict (line mode). Zeroed → coord 3rd channel stays 0.
        ##   - color (color_r/g/b): present at indices 7/8/9 of feats, but the
        ##                          to_vec_data slice `feats[..., :7]` currently drops
        ##                          them before they reach the model — so zeroing
        ##                          here is defensive (no-op under the current 7-ch
        ##                          slice; takes effect if a future change keeps them).
        ## Default False = baseline. Use to ablate the model's reliance on these
        ## auxiliary modalities.
        no_color_layer=False,
        # PTv3 serialization grid size in normalized scene coordinates.  The
        # same value is used for scene and primitive-library conditioning.
        conditioner_grid_size: float = 0.01,
        # Scale-label/reconstruction representation.  Defaults retain exact
        # historical behavior.  Conditioner normalization is independent so
        # N1 (label-only) and N2 (label + normalized conditioner) are separable.
        scale_parameterization: str = "relative",
        normalize_primitive_conditioner: bool = False,
        # Some primitive preprocessing uses torch.randperm to build fixed
        # library caches before all trainable heads have been constructed.
        # Isolating that RNG use makes random parameter initialization exactly
        # comparable when an ablation changes normalized library point counts.
        isolate_primitive_library_rng: bool = False,
        # R0 full-instance reconstruction.  Defaults preserve all existing
        # configurations; the loss is evaluated only when its weight is > 0.
        recon_dense_template_limit: int = 256,
        recon_source_limit: int = 96,
        recon_support_threshold: float = 0.0025,
        recon_support_cap: int = 32,
        recon_min_support: int = 8,
        recon_gt_geometry_threshold: float = 0.001,
        recon_warmup_epochs: int = 5,
        recon_visibility_aware: bool = False,
        recon_normalize_by_gt_diagonal: bool = False,
        recon_clip_range: List[float, float] = [-1.0, 1.0],
        recon_partial_weight_power: float = 0.5,
        recon_partial_min_support: int = 1,
        recon_relative_beta: float = 0.05,
        recon_use_gt_scale: bool = True,
        # Tri-state scale ablation for reconstruction.  None preserves the
        # historical recon_use_gt_scale boolean; explicit modes isolate the
        # geometry-forward effect of predicted scale from its gradient effect.
        recon_scale_mode: str | None = None,
        # Optional partial-only override.  None preserves the historical
        # behavior in which full and partial instances share scale_mode.
        # pred_detach is the strict control for predicted partial geometry
        # without a partial-scale reconstruction gradient.
        recon_partial_scale_mode: str | None = None,
        # visible: historical GT-clipped partial objective.
        # preclip_full: use the complete source captured immediately before
        # VecClipping and apply the same full-template objective as a full
        # instance.  The extra source is a loss-only target.
        recon_partial_source_mode: str = "preclip_full",
        recon_partial_mode: str = "exclude",
        # Relative partial reconstruction weight: under instance_mean it is a
        # per-instance partial weight; under the
        # legacy partitioned reduction it multiplies the partial-group mean.
        recon_partial_weight: float = 1.0,
        # instance_mean: one weighted mean over every valid instance.  With
        # preclip_full and partial_weight=1, every full/partial instance has
        # exactly the same weight.  partitioned preserves the old
        # mean(full)+lambda*mean(partial) experimental reduction.
        recon_instance_reduction: str = "instance_mean",
        recon_rotation_anchor_weight: float = 0.05,
        # Recon-only translation/rotation decoding.  soft_expectation keeps
        # historical behavior; hard_st evaluates the forward geometry at the
        # deterministic argmax bins used by formal AR inference.
        recon_decode_mode: str = "soft_expectation",
        # Opt-in shared deterministic continuous decoder; legacy preserves old runs.
        continuous_decode_mode: str = "legacy",
        # Strictly opt-in T/R factorization.  joint_bidir retains the exact
        # historical path: one template transformed by pred T + pred R and a
        # 0.5/0.5 bidirectional geometry loss.
        recon_tr_mode: str = "joint_bidir",
        # Used only by recon_tr_mode='factorized'.  Direction names are from
        # the anchor/source perspective and intentionally unambiguous.
        recon_rotation_direction: str = "bidirectional",
        recon_rotation_weight: float = 1.0,
        # Detached circular pred-vs-GT gate for the factorized R component.
        # Any negative value disables it.
        recon_rotation_gate_deg: float = -1.0,
        # Opt-in rotation representation used by the symmetry experiment.
        # off: historical endpoint-bin angle target;
        # force_one: shared circular phase bins with N=1 for every template;
        # audited: shared phase bins with the audited per-template N.
        rotation_symmetry_mode: str = "off",
        rotation_circular_label_sigma: float = 0.0,
    ):
        super().__init__()
        from models.primitive.scale_parameterization import (
            validate_scale_parameterization,
        )

        self.scale_parameterization = validate_scale_parameterization(
            scale_parameterization
        )
        self.normalize_primitive_conditioner = bool(normalize_primitive_conditioner)
        self.isolate_primitive_library_rng = bool(isolate_primitive_library_rng)
        if (
            self.normalize_primitive_conditioner
            and self.scale_parameterization != "absolute_local_extent"
        ):
            raise ValueError(
                "normalize_primitive_conditioner=true requires "
                "scale_parameterization=absolute_local_extent"
            )
        if conditioner_grid_size <= 0:
            raise ValueError(
                "conditioner_grid_size must be positive, got "
                f"{conditioner_grid_size}"
            )
        self.conditioner_grid_size = float(conditioner_grid_size)
        self.recon_dense_template_limit = int(recon_dense_template_limit)
        self.recon_source_limit = int(recon_source_limit)
        self.recon_support_threshold = float(recon_support_threshold)
        if (
            not math.isfinite(self.recon_support_threshold)
            or self.recon_support_threshold <= 0
        ):
            raise ValueError(
                "recon_support_threshold must be finite and positive, got "
                f"{recon_support_threshold}"
            )
        self.recon_support_cap = int(recon_support_cap)
        self.recon_min_support = int(recon_min_support)
        self.recon_gt_geometry_threshold = float(recon_gt_geometry_threshold)
        self.recon_warmup_epochs = int(recon_warmup_epochs)
        self.recon_visibility_aware = bool(recon_visibility_aware)
        self.recon_normalize_by_gt_diagonal = bool(recon_normalize_by_gt_diagonal)
        if len(recon_clip_range) != 2 or recon_clip_range[1] <= recon_clip_range[0]:
            raise ValueError(f"invalid recon_clip_range: {recon_clip_range}")
        self.recon_clip_range = tuple(map(float, recon_clip_range))
        self.recon_partial_weight_power = float(recon_partial_weight_power)
        self.recon_partial_min_support = int(recon_partial_min_support)
        self.recon_relative_beta = float(recon_relative_beta)
        if recon_scale_mode is None:
            recon_scale_mode = "gt" if recon_use_gt_scale else "pred"
        if recon_scale_mode not in {"gt", "pred_detach", "pred"}:
            raise ValueError(
                "recon_scale_mode must be one of gt/pred_detach/pred, got "
                f"{recon_scale_mode!r}"
            )
        self.recon_scale_mode = str(recon_scale_mode)
        if recon_partial_scale_mode is None:
            recon_partial_scale_mode = self.recon_scale_mode
        if recon_partial_scale_mode not in {"gt", "pred_detach", "pred"}:
            raise ValueError(
                "recon_partial_scale_mode must be one of "
                "gt/pred_detach/pred, got "
                f"{recon_partial_scale_mode!r}"
            )
        self.recon_partial_scale_mode = str(recon_partial_scale_mode)
        if recon_partial_source_mode not in {"visible", "preclip_full"}:
            raise ValueError(
                "recon_partial_source_mode must be visible/preclip_full, got "
                f"{recon_partial_source_mode!r}"
            )
        self.recon_partial_source_mode = str(recon_partial_source_mode)
        # Retain this public attribute for old diagnostics/checkpoints.  New
        # code must use recon_scale_mode because pred_detach has no boolean
        # representation.
        self.recon_use_gt_scale = self.recon_scale_mode == "gt"
        self.recon_partial_weight = float(recon_partial_weight)
        if (
            not math.isfinite(self.recon_partial_weight)
            or self.recon_partial_weight < 0
        ):
            raise ValueError(
                "recon_partial_weight must be finite and nonnegative, got "
                f"{recon_partial_weight}"
            )
        if recon_instance_reduction not in {"instance_mean", "partitioned"}:
            raise ValueError(
                "recon_instance_reduction must be instance_mean/partitioned, "
                f"got {recon_instance_reduction!r}"
            )
        self.recon_instance_reduction = str(recon_instance_reduction)
        if not 0.0 < float(recon_rotation_anchor_weight) < 1.0:
            raise ValueError(
                "recon_rotation_anchor_weight must be in (0, 1), got "
                f"{recon_rotation_anchor_weight}"
            )
        self.recon_rotation_anchor_weight = float(recon_rotation_anchor_weight)
        self.recon_decode_mode = validate_recon_decode_mode(recon_decode_mode)
        if continuous_decode_mode not in ("legacy", "soft_expectation"):
            raise ValueError(
                "continuous_decode_mode must be legacy or soft_expectation"
            )
        if continuous_decode_mode == "soft_expectation" and (
            gumbel_loss or recon_decode_mode != "soft_expectation"
        ):
            raise ValueError(
                "shared soft decoding requires gumbel_loss=False and recon_decode_mode=soft_expectation"
            )
        self.continuous_decode_mode = continuous_decode_mode

        self.recon_tr_mode = validate_recon_tr_mode(recon_tr_mode)
        self.recon_rotation_direction = validate_recon_rotation_direction(
            recon_rotation_direction
        )
        self.recon_rotation_weight = float(recon_rotation_weight)
        if (
            not math.isfinite(self.recon_rotation_weight)
            or self.recon_rotation_weight < 0
        ):
            raise ValueError(
                "recon_rotation_weight must be finite and nonnegative, got "
                f"{recon_rotation_weight}"
            )
        self.recon_rotation_gate_deg = float(recon_rotation_gate_deg)
        if not math.isfinite(self.recon_rotation_gate_deg):
            raise ValueError(
                "recon_rotation_gate_deg must be finite, got "
                f"{recon_rotation_gate_deg}"
            )
        if rotation_symmetry_mode not in {"off", "force_one", "audited"}:
            raise ValueError(
                "rotation_symmetry_mode must be off/force_one/audited, got "
                f"{rotation_symmetry_mode!r}"
            )
        self.rotation_symmetry_mode = rotation_symmetry_mode
        self.rotation_circular_label_sigma = float(rotation_circular_label_sigma)
        if (
            not math.isfinite(self.rotation_circular_label_sigma)
            or self.rotation_circular_label_sigma < 0
        ):
            raise ValueError(
                "rotation_circular_label_sigma must be finite and nonnegative"
            )
        if self.rotation_symmetry_mode != "off" and not attr_decouple:
            raise ValueError(
                "rotation_symmetry_mode is currently supported only for "
                "attr_decouple=True"
            )
        if recon_partial_mode not in {"exclude", "source_only", "visible_bidir"}:
            raise ValueError(
                "recon_partial_mode must be exclude/source_only/visible_bidir, "
                f"got {recon_partial_mode!r}"
            )
        self.recon_partial_mode = recon_partial_mode
        self._recon_epoch = 0
        if (
            min(
                self.recon_dense_template_limit,
                self.recon_source_limit,
                self.recon_support_cap,
                self.recon_min_support,
            )
            <= 0
        ):
            raise ValueError("all reconstruction bounds must be positive")
        if self.recon_min_support > self.recon_support_cap:
            raise ValueError("recon_min_support cannot exceed recon_support_cap")
        if not 0 <= self.recon_partial_min_support <= self.recon_support_cap:
            raise ValueError("recon_partial_min_support must be in [0, support_cap]")
        if self.recon_partial_weight_power < 0 or self.recon_relative_beta <= 0:
            raise ValueError("invalid partial reconstruction weighting/beta")
        self.size_pred = size_pred
        self.gumbel_loss = gumbel_loss
        self.scale_log = bool(scale_log)
        self.factorized_xy = factorized_xy
        self.no_color_layer = no_color_layer
        self.sem_geom_decouple = sem_geom_decouple
        self.geom_sem_decouple = geom_sem_decouple
        self.attr_decouple = attr_decouple
        if attr_primitive_semantic_condition not in {"gt", "none"}:
            raise ValueError(
                "attr_primitive_semantic_condition must be gt/none, got "
                f"{attr_primitive_semantic_condition!r}"
            )
        self.attr_emit_semantic_token = bool(attr_emit_semantic_token)
        self.attr_primitive_semantic_condition = str(attr_primitive_semantic_condition)
        self.attr_semantic_supervision = bool(attr_semantic_supervision)
        if self.attr_emit_semantic_token and not self.attr_semantic_supervision:
            raise ValueError(
                "A serialized semantic token requires semantic supervision"
            )
        self.attr_replace_primitive_id_with_geom = bool(
            attr_replace_primitive_id_with_geom
        )
        if self.attr_replace_primitive_id_with_geom and not self.attr_decouple:
            raise ValueError(
                "attr_replace_primitive_id_with_geom requires attr_decouple=True"
            )
        _decode_modes = (
            int(self.sem_geom_decouple)
            + int(self.geom_sem_decouple)
            + int(self.attr_decouple)
        )
        if _decode_modes > 1:
            raise ValueError(
                "sem_geom_decouple, geom_sem_decouple and attr_decouple are mutually exclusive"
            )
        self.num_attr_tokens = 6 if self.attr_emit_semantic_token else 5
        self.pretrain = pretrain
        self.learnable_anchor_embedding = learnable_anchor_embedding
        self.inst_wise_learnable_anchor_embedding = inst_wise_learnable_anchor_embedding
        self.sem_prim_pretrain = sem_prim_pretrain
        self.lm_loss = lm_loss
        self.bg_fitting = bg_fitting
        self.bg_matching = bg_matching
        self.fg_fitting = fg_fitting
        self.soft_ce = soft_ce
        self.dist_loss = dist_loss
        if self.soft_ce:
            kernel_translation = gaussian_kernel_1d(
                num_discrete_translation, bin_smooth_blur_sigma, False
            )
            kernel_rotation = gaussian_kernel_1d(
                num_discrete_rotation, bin_smooth_blur_sigma, True
            )
            kernel_scale = gaussian_kernel_1d(
                num_discrete_scale, bin_smooth_blur_sigma, False
            )
            self.register_buffer("kernel_translation", kernel_translation)
            self.register_buffer("kernel_rotation", kernel_rotation)
            self.register_buffer("kernel_scale", kernel_scale)
        self.primitive_contrastive_learning = primitive_contrastive_learning
        if self.primitive_contrastive_learning:
            self.primitive_mapping_path = primitive_mapping_path
            with open(self.primitive_mapping_path, "r", encoding="utf-8") as f:
                self.primitive_mapping = json.load(f)
            self.primitive_id2semantic_id = self.primitive_mapping[
                "primitive_id2semantic_id"
            ]
            self.primitive_id2semantic_id = torch.tensor(
                [v for v in self.primitive_id2semantic_id.values()], dtype=torch.long
            )
            primitive_head_dim = dim + (
                dim_type_embed if self.attr_primitive_semantic_condition == "gt" else 0
            )
            self.pred_proj = nn.Sequential(
                nn.Linear(primitive_head_dim, dim), nn.GELU(), nn.Linear(dim, dim)
            )
            self.lib_proj = nn.Sequential(
                nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)
            )
            init_temperature = np.log(1 / 0.07)
            self.logit_scale = nn.Parameter(torch.ones([]) * init_temperature)
            self.max_temperature = 100.0
            self.Hierarchical_primitive_criterion = HierarchicalPrimitiveLoss(
                self.primitive_id2semantic_id,
                coarse_weight=float(primitive_coarse_loss_weight),
            )
            self.primitive_coarse_loss_weight = float(primitive_coarse_loss_weight)
        #### 定义vocab mapping ####
        #### 定义size pred ####
        if self.size_pred:
            self.num_discrete_size = num_discrete_size
            self.continuous_range_size = continuous_range_size
            self.discretize_size = partial(
                discretize,
                num_discrete=num_discrete_size,
                continuous_range=continuous_range_size,
            )
            self.undiscretize_size = partial(
                undiscretize,
                num_discrete=num_discrete_size,
                continuous_range=continuous_range_size,
            )
            self.size_embed = nn.Embedding(num_discrete_size, dim_size_embed)

        if self.lm_loss:
            if self.sem_prim_pretrain:
                C_SEM = num_semantic
                C_PRIM = num_primitive
                vocab_size = C_SEM + C_PRIM
                self.vocab_size = vocab_size
                self.lm_head = nn.Linear(dim, vocab_size)
            elif self.attr_decouple:
                #### adjust translation prediction with two tokens ####
                C_SEM = num_semantic
                C_PRIM = num_primitive
                C_TRANS = 2 * num_discrete_translation
                C_FLIP = 1
                C_ROT = num_discrete_rotation
                C_SCALE = num_discrete_scale
                vocab_size = C_SEM + C_PRIM + C_TRANS + C_FLIP + C_ROT + C_SCALE
                self.vocab_size = vocab_size
                self.lm_head = nn.Linear(dim, vocab_size)

        if self.learnable_anchor_embedding:
            #### 所有instance共享一个anchor embedding，也可以不同instance用不同的anchor embedding，先用共享的测试
            self.anchor_embedding = nn.Embedding(1, dim)
            if self.inst_wise_learnable_anchor_embedding:
                self.inst_wise_anchor_embedding = nn.Embedding(max_primitive_len, dim)

        self.global_local_ca = global_local_ca
        self.attn_hard_constraint = attn_hard_constraint
        self.use_dense_feat = use_dense_feat
        self.init_attr_from_mask = init_attr_from_mask
        self.loss_mask_type = loss_mask_type
        self.return_point_mapping = return_point_mapping
        self.aux_mask_supervision = aux_mask_supervision
        self.rm_mask_pred = rm_mask_pred
        self.supervise_attn_bias = supervise_attn_bias
        self.individual_query = individual_query
        self.use_loftr = use_loftr
        self.down_sampled_dense_feat = down_sampled_dense_feat
        self.attn_constraint = attn_constraint
        self.attn_constraint_last_layer = attn_constraint_last_layer
        if self.use_loftr:
            self.loftr_q_proj = nn.Linear(dim, dim)
            self.loftr_k_proj = nn.Linear(dim, dim)

        if self.supervise_attn_bias:
            self.cross_bias_sigma = 3.0
            self.cross_bias_scale = 0.5
            ### TODO: 调整一下参数？
        if self.use_dense_feat:
            self.mask_features_head = nn.Sequential(
                # nn.Linear(self.conditioner.dec_channels, dim),
                nn.Linear(64, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )
            self.inst_proj = nn.Sequential(
                # nn.Linear(self.conditioner.dec_channels, dim),
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )
            self.pc_proj = nn.Sequential(
                # nn.Linear(self.conditioner.dec_channels, dim),
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )

        #### Add for slot attention
        self.use_slot_attn = use_slot_attn
        self.mask_guide = mask_guide
        if self.use_slot_attn:
            num_iterations = 3
            num_slots = 128  ####### 假设当前场景最多有128个instance
            slot_size = dim
            mlp_hidden_size = 1024
            truncate = "none"
            init_method = "shared_gaussian"
            num_heads = 16
            self.slot_attention_encoder = SlotAttentionEncoder(
                num_iterations=num_iterations,
                num_slots=num_slots,
                input_channels=dim,
                slot_size=slot_size,
                mlp_hidden_size=mlp_hidden_size,
                truncate=truncate,
                init_method=init_method,
                num_heads=num_heads,
            )

        self.scene_bound = scene_bound
        self.keep_ratio = keep_ratio
        self.min_keep = min_keep
        # feature embedding for scale, rotation, translation, and type
        self.label_padding = label_padding
        self.direct_recon = direct_recon
        self.use_geom_embed = use_geom_embed
        self.two_stage_decoder = two_stage_decoder
        self.use_prefix_decoder = use_prefix_decoder
        self.bg_pred = bg_pred
        self.num_bg_classes = 5  # 有效背景类: semantic 30-34
        if self.bg_pred or self.bg_fitting:
            self.bg_implicit_primitive = nn.Embedding(self.num_bg_classes, dim)
            #### add for mask_embed
            self.mask_features_head = nn.Sequential(
                # nn.Linear(self.conditioner.dec_channels, dim),
                nn.Linear(64, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )
            # primitive-level 背景分类: 将 pc_embed 映射到背景分类子空间
            self.pc_embed_to_bg_space = nn.Linear(dim, dim)

            if self.bg_matching is not None and self.bg_matching == "hungarian":
                """
                如果背景部分用匈牙利的话，那么需要两个改动，
                1. 需要背景类别预测
                2. 根据背景类别和背景mask来做匈牙利匹配
                3. 基于匹配结果来计算loss
                """
                ### 预测类别：num_bg_classes + 1, +1的部分表示对应query无效
                self.bg_class_pred = nn.Linear(dim, self.num_bg_classes + 1)
                ### Define the matcher for Hungarian matching
                self.bg_matcher = HungarianMatcher(
                    cost_class=1.0, cost_mask=2.0, cost_dice=2.0
                )

        if self.use_geom_embed:
            self.instance_retrieve_head = nn.Linear(dim, dim)
            self.primitive_lib_retrieve_head = nn.Linear(dim, dim)

        self.circular_rotate = circular_rotate

        self.num_discrete_scale = num_discrete_scale
        self.continuous_range_scale = continuous_range_scale
        scale_discretizer = discretize_log_scale if self.scale_log else discretize
        scale_undiscretizer = undiscretize_log_scale if self.scale_log else undiscretize
        self.discretize_scale = partial(
            scale_discretizer,
            num_discrete=num_discrete_scale,
            continuous_range=continuous_range_scale,
        )
        self.undiscretize_scale = partial(
            scale_undiscretizer,
            num_discrete=num_discrete_scale,
            continuous_range=continuous_range_scale,
        )
        if self.label_padding:
            self.scale_embed = nn.Embedding(num_discrete_scale + 1, dim_scale_embed)
        else:
            self.scale_embed = nn.Embedding(num_discrete_scale, dim_scale_embed)
        ## partial - 这里的作用是冻结后面的参数，后续调用时只需要传入一个参数即可；discrete: 连续->离散, undiscretize: 离散 -> 连续

        self.num_discrete_rotation = num_discrete_rotation
        self.continuous_range_rotation = continuous_range_rotation
        if self.rotation_symmetry_mode == "off":
            self.discretize_rotation = partial(
                discretize,
                num_discrete=num_discrete_rotation,
                continuous_range=continuous_range_rotation,
            )
            self.undiscretize_rotation = partial(
                undiscretize,
                num_discrete=num_discrete_rotation,
                continuous_range=continuous_range_rotation,
            )
        else:
            if tuple(map(float, continuous_range_rotation)) != (0.0, 360.0):
                raise ValueError(
                    "rotation symmetry phase mode requires "
                    "continuous_range_rotation=[0, 360]"
                )
            self.discretize_rotation = self._discretize_rotation_phase
            self.undiscretize_rotation = self._undiscretize_rotation_phase
        self.register_buffer(
            "rotation_symmetry_orders",
            symmetry_order_table(
                num_primitive,
                audited=self.rotation_symmetry_mode == "audited",
            ),
        )
        if self.label_padding:
            self.rotation_embed = nn.Embedding(
                num_discrete_rotation + 1, dim_rotation_embed
            )
        else:
            self.rotation_embed = nn.Embedding(
                num_discrete_rotation, dim_rotation_embed
            )
        self.sin_cos_embedding_rotation = sin_cos_embedding_rotation
        if sin_cos_embedding_rotation:
            sin_cos_values_rotation = get_sin_cos_positional_embedding(
                num_discrete_rotation, dim_rotation_embed
            )
            self.static_sin_cos_embedding_rotation = nn.Embedding(
                num_discrete_rotation, dim_rotation_embed
            )
            self.static_sin_cos_embedding_rotation.weight.data = torch.tensor(
                sin_cos_values_rotation, dtype=torch.float32
            )
            self.static_sin_cos_embedding_rotation.weight.requires_grad = False
        self.fourier_embedding_rotation = fourier_embedding_rotation
        if fourier_embedding_rotation:
            fourier_values_rotation = get_fourier_angle_embedding(
                num_discrete_rotation, dim_rotation_embed
            )
            self.static_fourier_embedding_rotation = nn.Embedding(
                num_discrete_rotation, dim_rotation_embed
            )
            self.static_fourier_embedding_rotation.weight.data = torch.tensor(
                fourier_values_rotation, dtype=torch.float32
            )
            self.static_fourier_embedding_rotation.weight.requires_grad = False
        self.num_discrete_translation = num_discrete_translation
        self.continuous_range_translation = continuous_range_translation
        self.discretize_translation = partial(
            discretize,
            num_discrete=num_discrete_translation,
            continuous_range=continuous_range_translation,
        )
        self.undiscretize_translation = partial(
            undiscretize,
            num_discrete=num_discrete_translation,
            continuous_range=continuous_range_translation,
        )
        if self.label_padding:
            self.translation_embed = nn.Embedding(
                num_discrete_translation + 1, dim_translation_embed
            )
        else:
            self.translation_embed = nn.Embedding(
                num_discrete_translation, dim_translation_embed
            )
        ### add sin_cos_embedding for translation
        self.sin_cos_embedding = sin_cos_embedding
        if self.sin_cos_embedding:
            sin_cos_values = get_sin_cos_positional_embedding(
                num_discrete_translation, dim_translation_embed
            )
            self.static_sin_cos_embedding = nn.Embedding(
                num_discrete_translation, dim_translation_embed
            )
            self.static_sin_cos_embedding.weight.data = torch.tensor(
                sin_cos_values, dtype=torch.float32
            )
            self.static_sin_cos_embedding.weight.requires_grad = False

        # ### Modified num_type
        # self.num_type = num_type
        # self.type_embed = nn.Embedding(num_type, dim_type_embed)        self.num_semantic = num_semantic
        self.num_primitive = num_primitive
        if self.label_padding:
            self.semantic_embed = nn.Embedding(num_semantic + 1, dim_type_embed)
            self.primitive_embed = nn.Embedding(num_primitive + 1, dim_type_embed)
        else:
            self.semantic_embed = nn.Embedding(num_semantic, dim_type_embed)
            self.primitive_embed = nn.Embedding(num_primitive, dim_type_embed)

        ### What does embed_order mean ---- the order of concatenated features when encoding
        self.embed_order = embed_order
        self.bin_smooth_blur_sigma = bin_smooth_blur_sigma

        # initial dimension

        self.dim = dim
        # init_dim = 3 * (dim_scale_embed + dim_rotation_embed + dim_translation_embed) + dim_type_embed
        #### Modified into cad setting
        self.pred_flip = pred_flip
        if pred_flip:
            init_dim = (
                2 * (dim_translation_embed)
                + dim_scale_embed
                + dim_rotation_embed
                + dim_type_embed
                + dim_type_embed
                + dim_flip_embed
            )
        else:
            init_dim = (
                2 * (dim_translation_embed)
                + dim_scale_embed
                + dim_rotation_embed
                + dim_type_embed
                + dim_type_embed
            )
        ### 每个primitive: (scale_x, scale_y, scale_z) + (rotation_x, rotation_y, rotation_z) + (translation_x, translation_y, translation_z) + type_code
        # project into model dimension

        self.project_in = nn.Linear(init_dim, dim)

        if self.sem_geom_decouple or self.geom_sem_decouple:
            self.sem_proj_in = nn.Linear(dim_type_embed * 2, dim)
            if self.pretrain:
                init_geom_dim = 2 * dim_translation_embed
            else:
                init_geom_dim = (
                    2 * (dim_translation_embed)
                    + dim_flip_embed
                    + dim_rotation_embed
                    + dim_scale_embed
                )
            init_geom_dim += dim_scale_embed
            self.geom_proj_in = nn.Linear(init_geom_dim, dim)
            self.attr_type_embed = nn.Embedding(2, dim)
            self.attr_instance_embed = nn.Embedding(max_primitive_len, dim)

        if self.attr_decouple:
            # One causal token per attribute:
            # [semantic ->] primitive -> translation -> flip -> rotation -> size/scale.
            self.sem_proj_in = nn.Linear(dim_type_embed, dim)
            self.prim_proj_in = nn.Linear(dim_type_embed, dim)
            self.trans_proj_in = nn.Linear(2 * dim_translation_embed, dim)
            self.flip_proj_in = nn.Linear(dim_flip_embed, dim)
            self.rot_proj_in = nn.Linear(dim_rotation_embed, dim)
            if self.size_pred:
                self.shape_proj_in = nn.Linear(2 * dim_size_embed, dim)
            else:
                self.shape_proj_in = nn.Linear(2 * dim_scale_embed, dim)

            self.attr_type_embed = nn.Embedding(self.num_attr_tokens, dim)
            self.attr_instance_embed = nn.Embedding(max_primitive_len, dim)

        nn.init.zeros_(self.attr_type_embed.weight)

        # Define sos token \in (1, dim) and eos token \in (1, dim)
        num_sos_tokens = default(
            num_sos_tokens,
            1 if not condition_on_shape or not shape_cond_with_film else 4,
        )
        assert num_sos_tokens > 0

        self.num_sos_tokens = num_sos_tokens
        self.sos_token = nn.Parameter(torch.randn(num_sos_tokens, dim))

        # the transformer eos token
        self.eos_token = nn.Parameter(torch.randn(1, dim))

        self.emb_layernorm = nn.LayerNorm(dim)
        self.max_seq_len = max_primitive_len

        # shape condition
        self.condition_on_shape = condition_on_shape
        self.shape_cond_with_cross_attn = False
        self.shape_cond_with_cat = False
        self.shape_condition_model_type = ""
        self.conditioner = None
        dim_shape = None

        #### Modified into Condition on PC encoder
        if condition_on_shape:
            assert (
                shape_cond_with_cross_attn
                or shape_cond_with_film
                or shape_cond_with_cat
                or shape_cond_with_perceiver_prefix
            )
            self.shape_cond_with_cross_attn = shape_cond_with_cross_attn
            self.shape_cond_with_cat = shape_cond_with_cat
            self.shape_cond_with_perceiver_prefix = shape_cond_with_perceiver_prefix
            self.shape_condition_model_type = shape_condition_model_type
            if "michelangelo" in shape_condition_model_type:
                self.conditioner = ShapeConditioner_miche(
                    dim_latent=shape_condition_dim
                )
                self.to_cond_dim = nn.Linear(
                    self.conditioner.dim_model_out * 2, self.conditioner.dim_latent
                )
                self.to_cond_dim_head = nn.Linear(
                    self.conditioner.dim_model_out, self.conditioner.dim_latent
                )
                dim_shape = self.conditioner.dim_latent
                set_module_requires_grad_(self.conditioner, False)
                ## 根据coarse shape来调整特征分布, 但是似乎这里不需要用到；
                self.shape_coarse_film_cond = (
                    FiLM(dim_shape, dim) if shape_cond_with_film else identity
                )
            elif "point_transformer_v3" in shape_condition_model_type:
                self.conditioner = PointTransformerV3(**backbone_config)
                dim_shape = dim
                self.shape_coarse_film_cond = (
                    FiLM(dim_shape, dim) if shape_cond_with_film else identity
                )
                self.to_cond_dim = nn.Linear(64, dim_shape)
                if shape_cond_with_perceiver_prefix:
                    self.perceiver_resampler = PerceiverResampler(
                        input_dim=64,
                        hidden_dim=dim_shape,
                        num_latents=perceiver_num_latents,
                        depth=perceiver_depth,
                        heads=perceiver_heads,
                    )
                    self.perceiver_num_latents = perceiver_num_latents
                    self.cond_prefix_pe = nn.Embedding(perceiver_num_latents, dim_shape)
            else:
                raise ValueError(
                    f"unknown shape_condition_model_type {self.shape_condition_model_type}"
                )

        self.condition_on_img = use_img_conditioner
        if self.condition_on_img:
            self.img_conditioner = ImgConditioner(
                model_name=img_conditioner_model_name,
                feature_index=img_conditioner_feature_index,
            )
            self.to_cond_dim_img = nn.Linear(self.img_conditioner.out_channels, dim)
            set_module_requires_grad_(self.img_conditioner, True)

        #### What is the use of coarse_gateloop_block
        self.coarse_gateloop_block = (
            GateLoopBlock(
                dim, depth=coarse_pre_gateloop_depth, use_heinsen=gateloop_use_heinsen
            )
            if coarse_pre_gateloop_depth > 0
            else None
        )
        self.coarse_post_gateloop_block = (
            GateLoopBlock(
                dim, depth=coarse_post_gateloop_depth, use_heinsen=gateloop_use_heinsen
            )
            if coarse_post_gateloop_depth > 0
            else None
        )
        self.coarse_adaptive_rmsnorm = coarse_adaptive_rmsnorm
        #### Define decoder-only transformer from x-transformer
        if self.use_prefix_decoder:
            if self.global_local_ca:
                custom_layers = ("a", "c", "c", "f") * attn_depth
                self.decoder = PrefixDecoder(
                    dim=dim,
                    depth=attn_depth,
                    heads=attn_heads,
                    attn_dim_head=attn_dim_head,
                    attn_flash=flash_attn,
                    attn_dropout=dropout,
                    ff_dropout=dropout,
                    use_adaptive_rmsnorm=coarse_adaptive_rmsnorm,
                    dim_condition=dim_shape,
                    cross_attend=self.shape_cond_with_cross_attn,
                    cross_attn_dim_context=dim_shape,
                    cross_attn_num_mem_kv=cross_attn_num_mem_kv,
                    ### Add for global-local cross-attention ####
                    custom_layers=custom_layers,
                    global_local_ca=self.global_local_ca,
                    cross_flash_pattern=(
                        True,
                        False,
                    ),  # ← 关键：global=True, local=False，按 cross 层顺序循环
                    cross_return_intermediates_pattern=(
                        False,
                        True,
                    ),  # 只在 local 返回注意力中间量
                    **attn_kwargs,
                )
            else:
                self.decoder = PrefixDecoder(
                    dim=dim,
                    depth=attn_depth,
                    heads=attn_heads,
                    attn_dim_head=attn_dim_head,
                    attn_flash=flash_attn,
                    attn_dropout=dropout,
                    ff_dropout=dropout,
                    use_adaptive_rmsnorm=coarse_adaptive_rmsnorm,
                    dim_condition=dim_shape,
                    cross_attend=self.shape_cond_with_cross_attn,
                    cross_attn_dim_context=dim_shape,
                    cross_attn_num_mem_kv=cross_attn_num_mem_kv,
                    **attn_kwargs,
                )
        else:
            self.decoder = Decoder(
                dim=dim,
                depth=attn_depth,
                heads=attn_heads,
                attn_dim_head=attn_dim_head,
                attn_flash=flash_attn,
                attn_dropout=dropout,
                ff_dropout=dropout,
                use_adaptive_rmsnorm=coarse_adaptive_rmsnorm,
                dim_condition=dim_shape,
                cross_attend=self.shape_cond_with_cross_attn,
                cross_attn_dim_context=dim_shape,
                cross_attn_num_mem_kv=cross_attn_num_mem_kv,
                **attn_kwargs,
            )

        if pred_flip:
            self.to_flip_logits = nn.Sequential(
                nn.Linear(dim + dim_type_embed * 2 + 2 * dim_translation_embed, dim),
                nn.ReLU(),
                nn.Linear(dim, 1),
            )
            self.flip_embed = nn.Embedding(2, dim_flip_embed)

        # to logits
        self.to_eos_logits = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1)
        )
        # self.to_type_logits = nn.Sequential(
        #     nn.Linear(dim, dim),
        #     nn.ReLU(),
        #     nn.Linear(dim, num_type)
        # )
        self.num_semantic = num_semantic
        self.num_primitive = num_primitive
        self.to_semantic_logits = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, num_semantic)
        )
        self.to_primitive_logits = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, num_primitive)
        )

        # self.to_translation_logits = nn.Sequential(
        #     nn.Linear(dim + dim_type_embed, dim),
        #     nn.ReLU(),
        #     nn.Linear(dim, 3 * num_discrete_translation)
        # )
        # self.to_rotation_logits = nn.Sequential(
        #     nn.Linear(dim + dim_type_embed + 3 * dim_translation_embed, dim),
        #     nn.ReLU(),
        #     nn.Linear(dim, 3 * num_discrete_rotation)
        # )
        # self.to_scale_logits = nn.Sequential(
        #     nn.Linear(dim + dim_type_embed + 3 * (dim_translation_embed + dim_rotation_embed), dim),
        #     nn.ReLU(),
        #     nn.Linear(dim, 3 * num_discrete_scale)
        # )
        ###### Modified into cad setting

        ###### 直接不用primitive_embed
        self.to_translation_logits = nn.Sequential(
            nn.Linear(dim + dim_type_embed * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 2 * num_discrete_translation),
        )
        if self.pred_flip:
            self.to_rotation_logits = nn.Sequential(
                nn.Linear(
                    dim
                    + dim_type_embed * 2
                    + 2 * dim_translation_embed
                    + dim_flip_embed,
                    dim,
                ),
                nn.ReLU(),
                nn.Linear(dim, num_discrete_rotation),
            )
            self.to_scale_logits = nn.Sequential(
                nn.Linear(
                    dim
                    + dim_type_embed * 2
                    + 2 * dim_translation_embed
                    + dim_flip_embed
                    + dim_rotation_embed,
                    dim,
                ),
                nn.ReLU(),
                nn.Linear(dim, num_discrete_scale),
            )
        else:
            self.to_rotation_logits = nn.Sequential(
                nn.Linear(dim + dim_type_embed * 2 + 2 * dim_translation_embed, dim),
                nn.ReLU(),
                nn.Linear(dim, num_discrete_rotation),
            )
            self.to_scale_logits = nn.Sequential(
                nn.Linear(
                    dim
                    + dim_type_embed * 2
                    + 2 * dim_translation_embed
                    + dim_rotation_embed,
                    dim,
                ),
                nn.ReLU(),
                nn.Linear(dim, num_discrete_scale),
            )
        # else:
        #     self.to_translation_logits = nn.Sequential(
        #         nn.Linear(dim + dim_type_embed, dim),
        #         nn.ReLU(),
        #         nn.Linear(dim, 2 * num_discrete_translation)
        #     )
        #     if self.pred_flip:
        #         self.to_rotation_logits = nn.Sequential(
        #             nn.Linear(dim + dim_type_embed + 2 * dim_translation_embed + dim_flip_embed, dim),
        #             nn.ReLU(),
        #             nn.Linear(dim, num_discrete_rotation)
        #         )
        #         self.to_scale_logits = nn.Sequential(
        #             nn.Linear(dim + dim_type_embed + 2 * dim_translation_embed + dim_flip_embed + dim_rotation_embed, dim),
        #             nn.ReLU(),
        #             nn.Linear(dim, num_discrete_scale)
        #         )
        #     else:
        #         self.to_rotation_logits = nn.Sequential(
        #             nn.Linear(dim + dim_type_embed + 2 * dim_translation_embed, dim),
        #             nn.ReLU(),
        #             nn.Linear(dim, num_discrete_rotation)
        #         )
        #         self.to_scale_logits = nn.Sequential(
        #             nn.Linear(dim + dim_type_embed + 2 * dim_translation_embed + dim_rotation_embed, dim),
        #             nn.ReLU(),
        #             nn.Linear(dim, num_discrete_scale)
        #         )

        self.pad_id = pad_id

        # save the primitive info
        bs_pc_map = {}
        # for bs_name, type_code in SHAPE_CODE.items():
        #     pc = o3d.io.read_point_cloud(os.path.join(bs_pc_dir, f'SM_GR_BS_{bs_name}_001.ply'))
        #     bs_pc_map[type_code] = torch.from_numpy(np.asarray(pc.points)).float()
        bs_pc_list = []
        # for i in range(len(bs_pc_map)):
        #     bs_pc_list.append(bs_pc_map[i])
        # self.bs_pc = torch.stack(bs_pc_list, dim=0)

        self.loss_weight = loss_weight
        # self.register_buffer('semantic2primitive', semantic2primitive)
        # self.register_buffer('primitive2semantic', primitive2semantic)

        self.rotation_matrix_align_coord = (
            euler_angles_to_matrix(torch.Tensor([np.pi / 2, 0, 0]), "XYZ")
            .unsqueeze(0)
            .unsqueeze(0)
        )
        ### Load primitive lib
        # Resolve pooling_type: prefer the explicit string flag; if None, fall
        # back to the legacy bools so existing YAML configs keep working.
        if pooling_type is None:
            if attn_pooling:
                pooling_type = "attn"
            elif pma_pooling:
                pooling_type = "pma"
            elif mean_pooling:
                pooling_type = "mean"
            else:
                pooling_type = "mean"  # historical default
        valid_pool_types = {"mean", "max", "attn", "mean_max", "pma"}
        assert (
            pooling_type in valid_pool_types
        ), f"pooling_type={pooling_type!r} not in {valid_pool_types}"
        self.pooling_type = pooling_type
        # Keep the legacy bools in sync so that any other code that reads them
        # (e.g. the AR generate path) sees a consistent state.
        self.mean_pooling = pooling_type == "mean"
        self.attn_pooling = pooling_type == "attn"
        self.use_pma_pooling = pooling_type == "pma"
        self.max_pooling = pooling_type == "max"
        self.mean_max_pooling = pooling_type == "mean_max"

        self.primitive_cond_with_cat = primitive_cond_with_cat
        self.primitive_dir = primitive_dir
        self.to_cond_dim_primitive_lib = None

        self.normalize_coord = normalize_coord
        if self.primitive_dir is not None:
            self.grid_size = 0.002
            if self.isolate_primitive_library_rng:
                with torch.random.fork_rng(devices=[]):
                    self.load_primitive_lib()
            else:
                self.load_primitive_lib()
            # if self.bg_pred:
            #     self.to_cond_dim_primitive_lib = nn.Linear(self.conditioner.dec_channels, dim_shape)
            # else:
            #     self.to_cond_dim_primitive_lib = nn.Linear(self.conditioner.enc_channels, dim_shape)
            # self.to_cond_dim_primitive_lib = nn.Linear(backbone_config['dec_channels'][-1], dim_shape)
            self.to_cond_dim_primitive_lib = nn.Linear(64, dim_shape)
            self.to_cond_dim_primitive_embed = nn.Linear(dim_shape, dim_type_embed)

            # ── Fit-test ablation: V2/V3 modules + lib_feats cache ──
            # alpha/beta are scalar gates initialised at 0 so that turning the
            # flag ON without training is exactly equivalent to baseline. The
            # projection layers map lib dim (==`dim`) → sem/geom packed dim,
            # which after `sem_proj_in`/`geom_proj_in` is also `dim`. So the
            # projection is `Linear(dim, dim)`.
            self.cond_grad_on_lib = cond_grad_on_lib
            self.cache_lib_feats = cache_lib_feats
            self.use_lib_in_sem = use_lib_in_sem
            self.use_lib_in_geom = use_lib_in_geom
            if self.cache_lib_feats and self.cond_grad_on_lib:
                raise ValueError(
                    "cache_lib_feats and cond_grad_on_lib are mutually exclusive: "
                    "caching the conditioner output severs the gradient path."
                )
            if self.use_lib_in_sem:
                self.proj_lib_to_sem = nn.Linear(dim_shape, dim)
                # Zero-init the projection so residual ≡ 0 at step 0 (baseline-equivalent),
                # but dL/dW_proj is still non-zero on the first backward pass — so the
                # projection trains from step 1. This avoids the chicken-and-egg problem
                # we'd get from alpha=0 gating: with α=0 the projection never receives
                # gradient because dL/dW_proj = α · (dL/dsem · inputᵀ) collapses to 0.
                nn.init.zeros_(self.proj_lib_to_sem.weight)
                nn.init.zeros_(self.proj_lib_to_sem.bias)
                # alpha kept as a learnable scalar but initialised to 1.0; serves mainly
                # as observability (its trajectory tells us whether the residual term
                # is being suppressed or amplified during training). It also offers an
                # extra parameter to absorb scale, which can speed up adaptation.
                self.alpha_lib_sem = nn.Parameter(torch.ones(1))
            if self.use_lib_in_geom:
                self.proj_lib_to_geom = nn.Linear(dim_shape, dim)
                nn.init.zeros_(self.proj_lib_to_geom.weight)
                nn.init.zeros_(self.proj_lib_to_geom.bias)
                self.beta_lib_geom = nn.Parameter(torch.ones(1))
            # cache buffer; not persistent → won't be saved to ckpt.
            self.register_buffer("_lib_feats_cache", torch.empty(0), persistent=False)
            self._lib_feats_cache_valid = False
        else:
            # Even when primitive_dir is None we accept the flags (no-op).
            self.cond_grad_on_lib = False
            self.cache_lib_feats = False
            self.use_lib_in_sem = False
            self.use_lib_in_geom = False
            # self.to_cond_dim_primitive_lib = nn.Linear(self.conditioner.enc_channels, dim_type_embed)
            # self.to_cond_dim_primitive_lib = nn.Linear(self.conditioner.dec_channels, dim_shape)

        self.src_type_embed = nn.Embedding(3, dim)
        # 初始化为 0，确保一开始行为不变，后续由训练自动学习区分
        nn.init.zeros_(self.src_type_embed.weight)

        ### Add for reconstruction loss
        bin_centers_translation = self.undiscretize_translation(
            torch.arange(self.num_discrete_translation)
        )  # (K,)
        bin_centers_rotation = self.undiscretize_rotation(
            torch.arange(self.num_discrete_rotation)
        )  # (K,)
        bin_centers_scale = self.undiscretize_scale(
            torch.arange(self.num_discrete_scale)
        )  # (K,)
        self.register_buffer("bin_centers_translation", bin_centers_translation)
        self.register_buffer("bin_centers_rotation", bin_centers_rotation)
        self.register_buffer("bin_centers_scale", bin_centers_scale)

        if self.size_pred:
            bin_centers_size = self.undiscretize_size(
                torch.arange(self.num_discrete_size)
            )  # (K,)
            self.register_buffer("bin_centers_size", bin_centers_size)

        if self.attn_pooling:
            # Standardised perceiver-style cross-attention pooling
            # (pre-norm, 3 LayerNorm + 2 residual; see AttentionPooling above).
            self.attention_pooling = AttentionPooling(dim)
        if self.use_pma_pooling:
            self.n_seeds = 1
            self.pma_pooling = PMA(
                dim,
                n_seeds=self.n_seeds,
                individual_query=self.individual_query,
                num_primitive=self.num_primitive,
            )
            self.slot_embed = nn.Embedding(self.n_seeds, dim)
        # mean / max / mean_max are all parameter-free → no module construction needed.

        self.filter_primitive_lib = filter_primitive_lib
        self.merge_attn = merge_attn
        self.mask_cross = mask_cross

        ### Add for gaussian kernel
        self.smooth_loss = smooth_loss
        kernel_translation = gaussian_kernel_1d(
            num_discrete_translation, bin_smooth_blur_sigma, False
        )
        kernel_rotation = gaussian_kernel_1d(
            num_discrete_rotation, bin_smooth_blur_sigma, True
        )
        kernel_scale = gaussian_kernel_1d(
            num_discrete_scale, bin_smooth_blur_sigma, False
        )
        self.register_buffer("kernel_translation", kernel_translation)
        self.register_buffer("kernel_rotation", kernel_rotation)
        self.register_buffer("kernel_scale", kernel_scale)
        symmetry_sigma = max(self.rotation_circular_label_sigma, 1e-6)
        self.register_buffer(
            "kernel_rotation_symmetry",
            gaussian_kernel_1d(num_discrete_rotation, symmetry_sigma, True),
        )
        ### Add for fourier positional encoding
        self.add_pe = add_pe
        self.pc_pe = PositionEmbeddingCoordsSine(pos_type="fourier", d_pos=dim, d_in=2)
        self.inst_pe = PositionEmbeddingCoordsSine(
            pos_type="fourier", d_pos=dim, d_in=2
        )
        if self.isolate_primitive_library_rng:
            with torch.random.fork_rng(devices=[]):
                self.preload_all_primitives([_ for _ in range(self.num_primitive)])
        else:
            self.preload_all_primitives([_ for _ in range(self.num_primitive)])
        self._initialize_attr_recon_templates()

        self.invalid_geom_embed = nn.Embedding(1, dim)
        #### 冻结参数，不随训练过程而改变
        self.invalid_geom_embed.weight.requires_grad = False
        # self.invalid_prim_embed = nn.Embedding(1, dim)
        # self.invalid_prim_embed.weight.requires_grad = False
        if self.sem_geom_decouple or self.geom_sem_decouple:
            self.to_semantic_logits = nn.Sequential(
                nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, num_semantic)
            )
            self.to_primitive_logits = nn.Sequential(
                nn.Linear(dim + dim_type_embed, dim),
                nn.ReLU(),
                nn.Linear(dim, num_primitive),
            )
            self.to_translation_logits = nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, 2 * num_discrete_translation),
            )
            self.to_flip_logits = nn.Sequential(
                nn.Linear(dim + 2 * dim_translation_embed, dim),
                nn.ReLU(),
                nn.Linear(dim, 1),
            )
            self.to_rotation_logits = nn.Sequential(
                nn.Linear(dim + 2 * dim_translation_embed + dim_flip_embed, dim),
                nn.ReLU(),
                nn.Linear(dim, num_discrete_rotation),
            )
            self.to_scale_logits = nn.Sequential(
                nn.Linear(
                    dim
                    + 2 * dim_translation_embed
                    + dim_flip_embed
                    + dim_rotation_embed,
                    dim,
                ),
                nn.ReLU(),
                nn.Linear(dim, num_discrete_scale * 2),
            )
            if self.size_pred:
                self.to_size_logits = nn.Sequential(
                    nn.Linear(
                        dim
                        + 2 * dim_translation_embed
                        + dim_flip_embed
                        + dim_rotation_embed,
                        dim,
                    ),
                    nn.ReLU(),
                    nn.Linear(dim, num_discrete_size * 2),
                )
            if self.factorized_xy:
                # Factorized 2-channel heads: P(x|h), P(y|h, x_embed)
                # ported from _131 — replaces the joint `to_translation_logits`
                # (and `to_size_logits` if size_pred) when factorized_xy=True.
                self.to_trans_x_logits = nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.ReLU(),
                    nn.Linear(dim, num_discrete_translation),
                )
                self.to_trans_y_logits = nn.Sequential(
                    nn.Linear(dim + dim_translation_embed, dim),
                    nn.ReLU(),
                    nn.Linear(dim, num_discrete_translation),
                )
                if self.size_pred:
                    _size_h_in = (
                        dim
                        + 2 * dim_translation_embed
                        + dim_flip_embed
                        + dim_rotation_embed
                    )
                    self.to_size_w_logits = nn.Sequential(
                        nn.Linear(_size_h_in, dim),
                        nn.ReLU(),
                        nn.Linear(dim, num_discrete_size),
                    )
                    self.to_size_h_logits = nn.Sequential(
                        nn.Linear(_size_h_in + dim_size_embed, dim),
                        nn.ReLU(),
                        nn.Linear(dim, num_discrete_size),
                    )
        if self.attr_decouple:
            if (
                self.pretrain
                or self.sem_prim_pretrain
                or self.lm_loss
                or self.learnable_anchor_embedding
            ):
                raise ValueError(
                    "The production attr_decouple path requires pretrain=False, "
                    "sem_prim_pretrain=False, lm_loss=False and "
                    "learnable_anchor_embedding=False"
                )
            if not self.pred_flip:
                raise ValueError(
                    "The production attr_decouple path currently requires pred_flip=True"
                )
            if self.soft_ce or self.dist_loss:
                raise ValueError(
                    "The production attr_decouple path currently requires soft_ce=False "
                    "and dist_loss=False"
                )

            self.to_semantic_logits = nn.Sequential(
                nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, num_semantic)
            )
            primitive_head_dim = dim + (
                dim_type_embed if self.attr_primitive_semantic_condition == "gt" else 0
            )
            self.to_primitive_logits = nn.Sequential(
                nn.Linear(primitive_head_dim, dim),
                nn.ReLU(),
                nn.Linear(dim, num_primitive),
            )
            if self.factorized_xy:
                self.to_trans_x_logits = nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.ReLU(),
                    nn.Linear(dim, num_discrete_translation),
                )
                self.to_trans_y_logits = nn.Sequential(
                    nn.Linear(dim + dim_translation_embed, dim),
                    nn.ReLU(),
                    nn.Linear(dim, num_discrete_translation),
                )
            else:
                self.to_translation_logits = nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.ReLU(),
                    nn.Linear(dim, 2 * num_discrete_translation),
                )
            self.to_flip_logits = nn.Sequential(
                nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1)
            )
            self.to_rotation_logits = nn.Sequential(
                nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, num_discrete_rotation)
            )
            if self.size_pred:
                if self.factorized_xy:
                    self.to_size_w_logits = nn.Sequential(
                        nn.Linear(dim, dim),
                        nn.ReLU(),
                        nn.Linear(dim, num_discrete_size),
                    )
                    self.to_size_h_logits = nn.Sequential(
                        nn.Linear(dim + dim_size_embed, dim),
                        nn.ReLU(),
                        nn.Linear(dim, num_discrete_size),
                    )
                else:
                    self.to_size_logits = nn.Sequential(
                        nn.Linear(dim, dim),
                        nn.ReLU(),
                        nn.Linear(dim, 2 * num_discrete_size),
                    )
            else:
                self.to_scale_logits = nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.ReLU(),
                    nn.Linear(dim, 2 * num_discrete_scale),
                )

    def _factorized_2ch_logits(self, h, x_head, y_head, embed_table, cond_x_idx=None):
        """Factorized 2-channel head: returns logits [B, np, 2, K] with y conditioned on x.

        - At training time, pass cond_x_idx = GT x bin (teacher forcing for the
          inner factorization).
        - At AR inference, pass cond_x_idx=None and the function uses the
          argmax of x_logits to feed y_head.
        """
        x_logits = x_head(h)
        if cond_x_idx is None:
            cond_x_idx = x_logits.argmax(dim=-1)
        cond_x_idx = cond_x_idx.clamp(min=0, max=embed_table.num_embeddings - 1)
        x_embed = embed_table(cond_x_idx)
        y_logits = y_head(torch.cat([h, x_embed], dim=-1))
        return torch.stack([x_logits, y_logits], dim=-2)  # [B, np, 2, K]

    def soft_ce_with_kernel(
        self, logits: torch.Tensor, target: torch.Tensor, K: torch.Tensor, pad_id: int
    ):
        # logits: (..., C), target: (...) int64, K: (C, C)
        C = logits.size(-1)
        mask = target != pad_id
        if not mask.any():
            return logits.new_zeros(())
        logits_f = logits.reshape(-1, C)
        tgt_f = target.reshape(-1)
        mask_f = mask.reshape(-1)
        logits_v = logits_f[mask_f]  # (M, C)
        idx_v = tgt_f[mask_f].long()  # (M,)
        soft_t = K.index_select(0, idx_v)  # (M, C)
        logp = F.log_softmax(logits_v, dim=-1)
        loss = -(soft_t * logp).sum(dim=-1).sum() / mask_f.sum().clamp_min(1)
        return loss

    def _transform(
        self,
        data: SVGData,
        scene_bound: float,
        normalize_local_extent: bool = False,
    ) -> VecData:
        if normalize_local_extent:
            from models.primitive.scale_parameterization import (
                normalize_coords_to_local_extent,
            )

            data.coords = normalize_coords_to_local_extent(data.coords).astype(
                np.float32
            )
        svg_data_tensor = to_tensor(data)
        # transform all line coords from [N * [x1, y1, x2, y2]] to [N * 2 * [x, y]] for easier processing
        if svg_data_tensor.coords.shape[-1] == 4:
            svg_data_tensor.coords = svg_data_tensor.coords.reshape(-1, 2, 2)
        # # ---------------- normalize lines --------------- #
        # svg_data_tensor.coords = norm_coords(
        #     coords=svg_data_tensor.coords,
        #     bbox=svg_data_tensor.viewBox,
        #     min_val=transform_args.norm_range[0],
        #     max_val=transform_args.norm_range[1])
        # 归一化思路还是按照之前的逻辑 #
        # svg_data_tensor.coords = svg_data_tensor.coords / (scene_bound / 2)
        # svg_data_tensor.coords = svg_data_tensor.coords - (scene_bound / 2)
        if not normalize_local_extent:
            svg_data_tensor.coords = svg_data_tensor.coords / (scene_bound / 2)
        # # ----------------- augment lines ---------------- #
        # svg_data_tensor.coords = augment_line_args(
        #     coords=svg_data_tensor.coords,
        #     min_val=transform_args.norm_range[0],
        #     max_val=transform_args.norm_range[1],
        #     transform_args=transform_args)
        # ------------ transform to line data ------------ #
        # transform all line coords back to [N * [x1, y1, x2, y2]]
        if len(svg_data_tensor.coords.shape) == 3:
            svg_data_tensor.coords = svg_data_tensor.coords.reshape(-1, 4)
        vec_data = to_vec_data(svg_data_tensor)
        return vec_data

    def load_primitive_lib(self):
        self.primitive_lib = {}
        ### modify into the vecformer-style primitive lib ###
        ##### assert the bound of the scenes is [0, 20000]
        # scene_bound = 20000.0
        primitive_files_lst = sorted(
            [x for x in os.listdir(self.primitive_dir) if x.endswith(".json")]
        )
        for file in primitive_files_lst:
            json_data = json.load(open(os.path.join(self.primitive_dir, file)))
            svg_data = SVGData(**json_data)
            vec_data = self._transform(
                svg_data,
                self.scene_bound,
                normalize_local_extent=self.normalize_primitive_conditioner,
            )
            primitive_id = int(file.split(".json")[0])
            self.primitive_lib[primitive_id] = {
                "coord": vec_data.coords,
                "feat": vec_data.feats,
                "prim_ids": vec_data.prim_ids,
                "layer_ids": vec_data.layer_ids,
            }

        ### 将primitive_lib编码成Point对象
        primitive_lib_coords = []
        primitive_lib_feats = []
        primitive_lib_prim_ids = []
        primitive_lib_layer_ids = []
        primitive_lib_cu_seqlens = [0]

        for primitive_id in sorted(self.primitive_lib.keys()):
            primitive_coords = self.primitive_lib[primitive_id]["coord"]
            primitive_feats = self.primitive_lib[primitive_id]["feat"]
            primitive_prim_ids = self.primitive_lib[primitive_id]["prim_ids"]
            primitive_layer_ids = self.primitive_lib[primitive_id]["layer_ids"]
            assert primitive_coords.shape[0] == primitive_feats.shape[0]
            assert primitive_coords.shape[0] == primitive_prim_ids.shape[0]
            assert primitive_coords.shape[0] == primitive_layer_ids.shape[0]
            primitive_lib_coords.append(primitive_coords)
            primitive_lib_feats.append(primitive_feats)
            primitive_lib_prim_ids.append(primitive_prim_ids)
            primitive_lib_layer_ids.append(primitive_layer_ids)
            primitive_lib_cu_seqlens.append(
                primitive_lib_cu_seqlens[-1] + primitive_coords.shape[0]
            )

        primitive_lib_coords = torch.cat(primitive_lib_coords, dim=0)
        primitive_lib_feats = torch.cat(primitive_lib_feats, dim=0)
        primitive_lib_prim_ids = torch.cat(primitive_lib_prim_ids, dim=0)
        primitive_lib_layer_ids = torch.cat(primitive_lib_layer_ids, dim=0)
        primitive_lib_cu_seqlens = torch.tensor(primitive_lib_cu_seqlens)

        self.register_buffer("primitive_lib_coords", primitive_lib_coords)
        self.register_buffer("primitive_lib_feats", primitive_lib_feats)
        self.register_buffer("primitive_lib_prim_ids", primitive_lib_prim_ids)
        self.register_buffer("primitive_lib_layer_ids", primitive_lib_layer_ids)
        self.register_buffer("primitive_lib_cu_seqlens", primitive_lib_cu_seqlens)

    def load_primitive_coords(self, primitive_id: int):
        """加载primitive坐标，支持批量优化"""
        # if primitive_id == 0:
        #     start_idx = 0
        # else:
        #     start_idx = self.primitive_lib_offset[primitive_id - 1].item()
        start_idx = self.primitive_lib_cu_seqlens[primitive_id].item()
        end_idx = self.primitive_lib_cu_seqlens[primitive_id + 1].item()
        primitive_coords = self.primitive_lib_coords[start_idx:end_idx]
        return primitive_coords

    def load_primitive_coords_batch(self, primitive_ids):
        """批量加载primitive坐标，进一步优化内存访问"""
        coords_list = []
        for primitive_id in primitive_ids:
            coords = self.load_primitive_coords(primitive_id)
            coords_list.append(coords[:, :2].contiguous())  # 只取x,y坐标
        return coords_list

    @property
    def device(self):
        return next(self.parameters()).device

    @typecheck
    @torch.no_grad()
    def embed_pc(self, pc: Tensor):
        if "michelangelo" in self.shape_condition_model_type:
            pc_head, pc_embed = self.conditioner(shape=pc)
            pc_embed = torch.cat(
                [self.to_cond_dim_head(pc_head), self.to_cond_dim(pc_embed)], dim=-2
            ).detach()
        else:
            raise ValueError(
                f"unknown shape_condition_model_type {self.shape_condition_model_type}"
            )

        return pc_embed

    @typecheck
    def get_discretize_gt(
        self,
        *,
        scale: Float["b np 3"],
        rotation: Float["b np 3"],
        translation: Float["b np 3"],
        semantic_id: Int["b np"],
        primitive_id: Int["b np"],
        # primitive_mask: Bool['b np'],
    ):
        discretize_scale = self.discretize_scale(scale)
        rotation_supervision = self._rotation_gt_phase(rotation, primitive_id)
        discretize_rotation = self.discretize_rotation(rotation_supervision)
        discretize_translation = self.discretize_translation(translation)
        return (
            discretize_scale,
            discretize_rotation,
            discretize_translation,
            semantic_id,
            primitive_id,
        )

    @typecheck
    def _attr_primitive_identity_embed(
        self,
        primitive_id: Tensor,
        primitive_mask: Tensor,
        primitive_geom_embed: Tensor | None,
    ) -> Tensor:
        """Return the 48-D primitive identity used by attr-decouple tokens.

        The control path uses the learned ID table.  The experimental path
        gathers the selected template's pooled geometry and maps it to the
        same 48-D interface, matching Floorplan's ``use_geom_embed`` design.
        """
        safe_primitive = primitive_id.masked_fill(~primitive_mask, 0).long()
        if not self.attr_replace_primitive_id_with_geom:
            return self.primitive_embed(safe_primitive)

        if primitive_geom_embed is None:
            raise RuntimeError(
                "attr_replace_primitive_id_with_geom=True requires encoded "
                "primitive template geometry"
            )
        if primitive_geom_embed.ndim == 2:
            primitive_geom = primitive_geom_embed[safe_primitive]
        elif primitive_geom_embed.ndim == 3:
            batch = safe_primitive.shape[0]
            if primitive_geom_embed.shape[0] == 1 and batch != 1:
                primitive_geom_embed = primitive_geom_embed.expand(batch, -1, -1)
            if primitive_geom_embed.shape[0] != batch:
                raise ValueError(
                    "Template batch does not match primitive IDs: "
                    f"{primitive_geom_embed.shape[0]} != {batch}"
                )
            gather_index = safe_primitive.unsqueeze(-1).expand(
                -1, -1, primitive_geom_embed.shape[-1]
            )
            primitive_geom = torch.gather(
                primitive_geom_embed, dim=1, index=gather_index
            )
        else:
            raise ValueError(
                "Expected primitive template geometry [P,D] or [B,P,D], got "
                f"{tuple(primitive_geom_embed.shape)}"
            )
        return self.to_cond_dim_primitive_embed(primitive_geom).masked_fill(
            ~primitive_mask.unsqueeze(-1), 0.0
        )

    @typecheck
    def encode_attr(
        self,
        *,
        scale: Float["b np 3"],
        rotation: Float["b np 3"],
        translation: Float["b np 3"],
        # type_code: Int['b np'],
        semantic_id: Int["b np"],
        primitive_id: Int["b np"],
        flip_mode: Int["b np"],
        primitive_mask: Bool["b np"],
        return_primitives=False,
        primitive_geom_embed=None,
        size: Float["b np 2"] | None = None,
    ):
        # The current Industry data represents translation, scale and size with
        # two axes. Older attr-decouple code assumed scalar scale; normalize the
        # legacy scalar form here so the rest of the path has one stable schema.
        if scale.ndim == 2:
            scale = scale.unsqueeze(-1).expand(-1, -1, 2)

        discretize_scale = self.discretize_scale(scale)
        discretize_scale_safe = discretize_scale.masked_fill(
            ~primitive_mask.unsqueeze(-1), 0
        )
        scale_embed_raw = rearrange(
            self.scale_embed(discretize_scale_safe), "b np c d -> b np (c d)"
        )

        rotation_supervision = self._rotation_gt_phase(rotation, primitive_id)
        discretize_rotation = self.discretize_rotation(rotation_supervision)
        discretize_rotation_safe = discretize_rotation.masked_fill(~primitive_mask, 0)
        rotation_embed_raw = self.rotation_embed(discretize_rotation_safe)

        flip_embed_raw = self.flip_embed(flip_mode.masked_fill(~primitive_mask, 0))

        discretize_translation = self.discretize_translation(translation)
        discretize_translation_safe = discretize_translation.masked_fill(
            ~primitive_mask.unsqueeze(-1), 0
        )
        translation_embed_raw = rearrange(
            self.translation_embed(discretize_translation_safe),
            "b np c d -> b np (c d)",
        )

        discretize_size = None
        if self.size_pred:
            if size is None:
                raise ValueError(
                    "size must be provided when attr_decouple and size_pred are enabled"
                )
            discretize_size = self.discretize_size(size)
            discretize_size_safe = discretize_size.masked_fill(
                ~primitive_mask.unsqueeze(-1), 0
            )
            shape_embed_raw = rearrange(
                self.size_embed(discretize_size_safe), "b np c d -> b np (c d)"
            )
        else:
            shape_embed_raw = scale_embed_raw

        safe_semantic = semantic_id.masked_fill(~primitive_mask, 0)
        safe_primitive = primitive_id.masked_fill(~primitive_mask, 0)
        semantic_embed_raw = self.semantic_embed(safe_semantic)
        primitive_embed_raw = self._attr_primitive_identity_embed(
            primitive_id, primitive_mask, primitive_geom_embed
        )

        semantic_type_embed = self.sem_proj_in(semantic_embed_raw)
        primitive_type_embed = self.prim_proj_in(primitive_embed_raw)
        translation_embed = self.trans_proj_in(translation_embed_raw)
        flip_embed = self.flip_proj_in(flip_embed_raw)
        rotation_embed = self.rot_proj_in(rotation_embed_raw)
        shape_embed = self.shape_proj_in(shape_embed_raw)

        # Preserve the baseline primitive-library residual. In the decoupled
        # sequence it is injected into the primitive token and is therefore
        # available to all later geometry tokens through causal attention.
        if (
            getattr(self, "use_lib_in_sem", False)
            or getattr(self, "use_lib_in_geom", False)
        ) and primitive_geom_embed is not None:
            lib_unbatched = (
                primitive_geom_embed
                if primitive_geom_embed.dim() == 2
                else primitive_geom_embed[0]
            )
            lib_per_inst = lib_unbatched[safe_primitive]
            lib_per_inst = lib_per_inst.masked_fill(~primitive_mask.unsqueeze(-1), 0.0)
            if getattr(self, "use_lib_in_sem", False):
                semantic_type_embed = (
                    semantic_type_embed
                    + self.alpha_lib_sem * self.proj_lib_to_sem(lib_per_inst)
                )
            if (
                getattr(self, "use_lib_in_geom", False)
                and not self.attr_replace_primitive_id_with_geom
            ):
                primitive_type_embed = (
                    primitive_type_embed
                    + self.beta_lib_geom * self.proj_lib_to_geom(lib_per_inst)
                )

        projected = (
            semantic_type_embed,
            primitive_type_embed,
            translation_embed,
            flip_embed,
            rotation_embed,
            shape_embed,
        )
        projected = tuple(
            value.masked_fill(~primitive_mask.unsqueeze(-1), 0.0) for value in projected
        )
        (
            semantic_type_embed,
            primitive_type_embed,
            translation_embed,
            flip_embed,
            rotation_embed,
            shape_embed,
        ) = projected

        if not return_primitives:
            return projected

        primitive_embed_unpacked = {
            "scale": scale_embed_raw,
            "size": shape_embed_raw if self.size_pred else None,
            "rotation": rotation_embed_raw,
            "translation": translation_embed_raw,
            "semantic_id": semantic_embed_raw,
            "primitive_id": primitive_embed_raw,
            "flip_mode": flip_embed_raw,
        }

        primitives_gt = {
            "scale": discretize_scale.masked_fill(
                ~primitive_mask.unsqueeze(-1), self.pad_id
            ),
            "size": (
                discretize_size.masked_fill(~primitive_mask.unsqueeze(-1), self.pad_id)
                if self.size_pred
                else None
            ),
            "rotation": discretize_rotation.masked_fill(~primitive_mask, self.pad_id),
            "translation": discretize_translation.masked_fill(
                ~primitive_mask.unsqueeze(-1), self.pad_id
            ),
            "semantic_id": semantic_id,
            "primitive_id": primitive_id,
            "flip_mode": flip_mode,
        }

        return (*projected, primitive_embed_unpacked, primitives_gt)

    @typecheck
    def encode_sem_geom(
        self,
        *,
        scale: Float["b np 3"],
        rotation: Float["b np 3"],
        translation: Float["b np 3"],
        # type_code: Int['b np'],
        semantic_id: Int["b np"],
        primitive_id: Int["b np"],
        flip_mode: Int["b np"],
        primitive_mask: Bool["b np"],
        return_primitives=False,
        primitive_geom_embed=None,
        size: Float["b np 2"] | None = None,
    ):
        #### 首先获取每个instance's attribute对应的embedding表示，随后统一放到一个token中
        #### Geometry Embedding ####
        # discretize_scale = self.discretize_scale(scale)
        # discretize_scale = discretize_scale.masked_fill(~primitive_mask, 0)
        # scale_embed = self.scale_embed(discretize_scale)
        # scale_embed = rearrange(scale_embed, 'b np d -> b np d')
        discretize_scale = self.discretize_scale(scale)
        discretize_scale = discretize_scale.masked_fill(
            ~primitive_mask.unsqueeze(-1), 0
        )
        scale_embed = self.scale_embed(discretize_scale)
        scale_embed = rearrange(scale_embed, "b np c d -> b np (c d)")

        discretize_rotation = self.discretize_rotation(rotation)
        discretize_rotation = discretize_rotation.masked_fill(~primitive_mask, 0)
        rotation_embed = self.rotation_embed(discretize_rotation)
        rotation_embed = rearrange(rotation_embed, "b np d -> b np d")

        flip_embed = self.flip_embed(flip_mode.masked_fill(~primitive_mask, 0))
        flip_embed = rearrange(flip_embed, "b np d -> b np d")

        discretize_translation = self.discretize_translation(translation)
        discretize_translation = discretize_translation.masked_fill(
            ~primitive_mask.unsqueeze(-1), 0
        )
        translation_embed = self.translation_embed(discretize_translation)
        translation_embed = rearrange(translation_embed, "b np c d -> b np (c d)")

        if self.size_pred:
            discretize_size = self.discretize_size(size)
            discretize_size = discretize_size.masked_fill(
                ~primitive_mask.unsqueeze(-1), 0
            )
            size_embed = self.size_embed(discretize_size)
            size_embed = rearrange(size_embed, "b np c d -> b np (c d)")

        if self.pretrain:
            geom_embed = self.geom_proj_in(translation_embed)
        else:
            if self.size_pred:
                geom_embed, _ = pack(
                    [translation_embed, flip_embed, rotation_embed, size_embed],
                    "b np *",
                )
            else:
                geom_embed, _ = pack(
                    [translation_embed, flip_embed, rotation_embed, scale_embed],
                    "b np *",
                )
            geom_embed = self.geom_proj_in(geom_embed)
        #### Geometry Embedding ####

        #### Semantic Embedding ####
        semantic_type_embed = self.semantic_embed(
            semantic_id.masked_fill(~primitive_mask, 0)
        )
        primitive_type_embed = self.primitive_embed(
            primitive_id.masked_fill(~primitive_mask, 0)
        )
        ###### TODO: 考虑Primitive Geom信息 #####
        sem_embed, _ = pack([semantic_type_embed, primitive_type_embed], "b np *")
        sem_embed = self.sem_proj_in(sem_embed)
        #### Semantic Embedding ####

        # ─────────────────────────────────────────────────────────────────────
        # Fit-test ablation: V2 (sem residual) and V3 (geom residual).
        # Index-lookup the primitive lib by GT primitive_id, project to model
        # dim, and add as a gated residual to sem_embed / geom_embed.
        # primitive_geom_embed is the *unbatched* lib tensor [num_prims, dim]
        # under mean_pooling (the active config). For attn/pma/else cases that
        # already broadcast to [B, ..., dim], we fall back to slicing batch-0.
        # ─────────────────────────────────────────────────────────────────────
        if (
            getattr(self, "use_lib_in_sem", False)
            or getattr(self, "use_lib_in_geom", False)
        ) and primitive_geom_embed is not None:
            if primitive_geom_embed.dim() == 2:
                lib_unbatched = primitive_geom_embed  # [P, D]
            elif primitive_geom_embed.dim() == 3:
                lib_unbatched = primitive_geom_embed[
                    0
                ]  # [P, D]; rows are identical across batch
            else:
                lib_unbatched = primitive_geom_embed.reshape(
                    -1, primitive_geom_embed.shape[-1]
                )
            safe_pid = primitive_id.masked_fill(~primitive_mask, 0)  # [B, np]
            lib_per_inst = lib_unbatched[safe_pid]  # [B, np, D]
            # Zero-out invalid positions so they don't add noise to padding tokens.
            lib_per_inst = lib_per_inst.masked_fill(~primitive_mask.unsqueeze(-1), 0.0)
            if getattr(self, "use_lib_in_sem", False):
                sem_embed = sem_embed + self.alpha_lib_sem * self.proj_lib_to_sem(
                    lib_per_inst
                )
            if getattr(self, "use_lib_in_geom", False):
                geom_embed = geom_embed + self.beta_lib_geom * self.proj_lib_to_geom(
                    lib_per_inst
                )

        # ### 位置编码，两个都加 ####
        # if self.add_pe:
        #     trans_pe = self.inst_pe(translation.float(), num_channels=self.dim)
        #     trans_pe = trans_pe.masked_fill(~primitive_mask.unsqueeze(-1), 0.)
        #     sem_embed = sem_embed + trans_pe
        #     geom_embed = geom_embed + trans_pe
        # ### 位置编码，两个都加 ####

        if not return_primitives:
            return sem_embed, geom_embed

        primitive_embed_unpacked = {
            "scale": scale_embed,
            "rotation": rotation_embed,
            "translation": translation_embed,
            "semantic_id": semantic_type_embed,
            "primitive_id": primitive_type_embed,
            "flip_mode": flip_embed,
            "size": size_embed if self.size_pred else None,
        }

        ## Add for pad_id
        # discretize_scale = discretize_scale.masked_fill(~primitive_mask, self.pad_id)
        discretize_scale = discretize_scale.masked_fill(
            ~primitive_mask.unsqueeze(-1), self.pad_id
        )
        discretize_rotation = discretize_rotation.masked_fill(
            ~primitive_mask, self.pad_id
        )
        discretize_translation = discretize_translation.masked_fill(
            ~primitive_mask.unsqueeze(-1), self.pad_id
        )
        if self.size_pred:
            discretize_size = discretize_size.masked_fill(
                ~primitive_mask.unsqueeze(-1), self.pad_id
            )
        primitives_gt = {
            "scale": discretize_scale,
            "rotation": discretize_rotation,
            "translation": discretize_translation,
            "semantic_id": semantic_id,
            "primitive_id": primitive_id,
            "flip_mode": flip_mode,
            "size": discretize_size if self.size_pred else None,
        }

        return sem_embed, geom_embed, primitive_embed_unpacked, primitives_gt

    @typecheck
    def encode(
        self,
        *,
        scale: Float["b np 3"],
        rotation: Float["b np 3"],
        translation: Float["b np 3"],
        # type_code: Int['b np'],
        semantic_id: Int["b np"],
        primitive_id: Int["b np"],
        flip_mode: Int["b np"],
        primitive_mask: Bool["b np"],
        return_primitives=False,
        primitive_geom_embed=None,
    ):
        """
        einops:
        b - batch
        np - number of primitives
        c - coordinates (3)
        d - embed dim
        """

        # compute feature embedding
        discretize_scale = self.discretize_scale(scale)
        if self.label_padding:
            discretize_scale = discretize_scale.masked_fill(
                ~primitive_mask, self.num_discrete_scale
            )
        else:
            discretize_scale = discretize_scale.masked_fill(~primitive_mask, 0)
        scale_embed = self.scale_embed(discretize_scale)
        scale_embed = rearrange(scale_embed, "b np d -> b np d")

        discretize_rotation = self.discretize_rotation(rotation)
        if self.label_padding:
            discretize_rotation = discretize_rotation.masked_fill(
                ~primitive_mask, self.num_discrete_rotation
            )
        else:
            discretize_rotation = discretize_rotation.masked_fill(~primitive_mask, 0)

        rotation_embed = self.rotation_embed(discretize_rotation)
        if self.sin_cos_embedding_rotation:
            rotation_embed = rotation_embed + self.static_sin_cos_embedding_rotation(
                discretize_rotation
            )
        elif self.fourier_embedding_rotation:
            rotation_embed = rotation_embed + self.static_fourier_embedding_rotation(
                discretize_rotation
            )
        # rotation_embed = rearrange(rotation_embed, 'b np c d -> b np (c d)')
        rotation_embed = rearrange(rotation_embed, "b np d -> b np d")

        flip_embed = None
        if self.pred_flip:
            flip_embed = self.flip_embed(flip_mode.masked_fill(~primitive_mask, 0))
            flip_embed = rearrange(flip_embed, "b np d -> b np d")

        discretize_translation = self.discretize_translation(translation)
        if self.label_padding:
            discretize_translation = discretize_translation.masked_fill(
                ~primitive_mask.unsqueeze(-1), self.num_discrete_translation
            )
        else:
            discretize_translation = discretize_translation.masked_fill(
                ~primitive_mask.unsqueeze(-1), 0
            )
        translation_embed = self.translation_embed(discretize_translation)
        if self.sin_cos_embedding:
            translation_embed = translation_embed + self.static_sin_cos_embedding(
                discretize_translation
            )
        translation_embed = rearrange(translation_embed, "b np c d -> b np (c d)")

        # type_embed = self.type_embed(type_code.masked_fill(~primitive_mask, 0))
        if self.label_padding:
            semantic_type_embed = self.semantic_embed(
                semantic_id.masked_fill(~primitive_mask, self.num_semantic)
            )
            primitive_type_embed = self.primitive_embed(
                primitive_id.masked_fill(~primitive_mask, self.num_primitive)
            )
        else:
            semantic_type_embed = self.semantic_embed(
                semantic_id.masked_fill(~primitive_mask, 0)
            )
            primitive_type_embed = self.primitive_embed(
                primitive_id.masked_fill(~primitive_mask, 0)
            )

        if self.use_geom_embed:
            primitive_type_embed = primitive_geom_embed[
                primitive_id.masked_fill(~primitive_mask, 0)
            ]
            primitive_type_embed = self.to_cond_dim_primitive_embed(
                primitive_type_embed
            )

        # combine all features and project into model dimension
        if self.embed_order == "srtc":
            if self.pred_flip:
                primitive_embed, _ = pack(
                    [
                        scale_embed,
                        rotation_embed,
                        flip_embed,
                        translation_embed,
                        semantic_type_embed,
                        primitive_type_embed,
                    ],
                    "b np *",
                )
            else:
                primitive_embed, _ = pack(
                    [
                        scale_embed,
                        rotation_embed,
                        translation_embed,
                        semantic_type_embed,
                        primitive_type_embed,
                    ],
                    "b np *",
                )
        else:
            if self.pred_flip:
                primitive_embed, _ = pack(
                    [
                        semantic_type_embed,
                        primitive_type_embed,
                        translation_embed,
                        flip_embed,
                        rotation_embed,
                        scale_embed,
                    ],
                    "b np *",
                )
            else:
                primitive_embed, _ = pack(
                    [
                        semantic_type_embed,
                        primitive_type_embed,
                        translation_embed,
                        rotation_embed,
                        scale_embed,
                    ],
                    "b np *",
                )

        primitive_embed = self.project_in(primitive_embed)
        primitive_embed = primitive_embed.masked_fill(
            ~primitive_mask.unsqueeze(-1), 0.0
        )

        if self.add_pe:
            # trans_pe = self.inst_pe(discretize_translation.float(), num_channels=self.dim)
            ##### refine for translation
            trans_pe = self.inst_pe(translation.float(), num_channels=self.dim)
            trans_pe = trans_pe.masked_fill(~primitive_mask.unsqueeze(-1), 0.0)
            primitive_embed = primitive_embed + trans_pe

        # ###### add for geom_embed
        # if self.use_geom_embed:
        #     # primitive_embed = self.geom_fusion(primitive_embed, primitive_geom_embed, primitive_id.masked_fill(~primitive_mask, 0))
        #     geom_embed = primitive_geom_embed[primitive_id.masked_fill(~primitive_mask, 0)]
        #     primitive_embed = primitive_embed + geom_embed
        #     # geom_embed[~primitive_mask] = self.invalid_geom_embed.weight
        #     ##### 尝试各种融合方式
        #     ##### 1. 直接加
        #     # primitive_embed = primitive_embed + geom_embed
        #     ##### 2. film加
        #     # primitive_embed[~primitive_mask] = self.invalid_prim_embed.weight
        #     # primitive_embed = self.geom_film(primitive_embed, geom_embed)

        if not return_primitives:
            return primitive_embed

        primitive_embed_unpacked = {
            "scale": scale_embed,
            "rotation": rotation_embed,
            "translation": translation_embed,
            "semantic_id": semantic_type_embed,
            "primitive_id": primitive_type_embed,
            "flip_mode": flip_embed,
        }

        ## Add for pad_id
        discretize_scale = discretize_scale.masked_fill(~primitive_mask, self.pad_id)
        discretize_rotation = discretize_rotation.masked_fill(
            ~primitive_mask, self.pad_id
        )
        discretize_translation = discretize_translation.masked_fill(
            ~primitive_mask.unsqueeze(-1), self.pad_id
        )

        if self.label_padding:
            semantic_id = semantic_id.masked_fill(~primitive_mask, self.num_semantic)
            primitive_id = primitive_id.masked_fill(~primitive_mask, self.num_primitive)

        primitives_gt = {
            "scale": discretize_scale,
            "rotation": discretize_rotation,
            "translation": discretize_translation,
            "semantic_id": semantic_id,
            "primitive_id": primitive_id,
            "flip_mode": flip_mode,
        }

        return primitive_embed, primitive_embed_unpacked, primitives_gt

    @typecheck
    def transform_primitive_points(
        self,
        primitive_coords: Float["n 2"],
        scale: Float["1 3"],
        rotation: Float["1 3"],
        translation: Float["1 3"],
        flip: int = 0,
    ):
        device = primitive_coords.device
        dtype = primitive_coords.dtype

        # 确保输入是正确的形状
        pts = primitive_coords.clone()  # [n, 2]
        scale = scale.squeeze()  # [2]
        rotation = rotation.squeeze()  # []
        translation = translation.squeeze()  # [2]

        center_vec = pts.mean(dim=0)

        # 1) 旋转（绕中心点）
        theta = torch.deg2rad(rotation)  # 转换为弧度
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        # R = torch.tensor([[cos_theta, -sin_theta],
        #                 [sin_theta, cos_theta]], device=device, dtype=dtype)  # [2, 2]
        R = torch.stack(
            [
                torch.stack([cos_theta, -sin_theta], dim=-1),
                torch.stack([sin_theta, cos_theta], dim=-1),
            ],
            dim=-2,
        )
        pts_centered = pts - center_vec  # [n, 2]
        pts = torch.matmul(pts_centered, R.T)  # [n, 2]

        # 2) 缩放（绕中心点）
        pts = pts * scale.unsqueeze(0) + center_vec  # [n, 2]

        # 3) 翻转（暂时跳过，如需要可以添加）
        # if flip == 1:
        #     pts[:, 0] = 2.0 * cx - pts[:, 0]

        # 4) 平移
        pts = pts + translation.unsqueeze(0)  # [n, 2]

        return pts

    def transform_primitive_points_batch_padded(
        self,
        primitive_coords_list,  # List[Tensor[n_i, 2]]
        scale: Float["batch"],
        rotation: Float["batch"],
        translation: Float["batch 2"],
        flip: Float["batch"],
    ):
        """支持不同点数的批量变换，通过padding实现"""
        device = scale.device
        dtype = scale.dtype
        batch_size = len(primitive_coords_list)

        # 找到最大点数
        max_points = max(coords.size(0) for coords in primitive_coords_list)

        # 创建padding后的批量坐标张量
        batch_coords = torch.zeros(
            batch_size, max_points, 2, device=device, dtype=dtype
        )
        valid_lengths = []

        for i, coords in enumerate(primitive_coords_list):
            n_points = coords.size(0)
            batch_coords[i, :n_points] = coords
            valid_lengths.append(n_points)

        # 确保输入形状正确
        # if flip is not None:
        #     flip = flip.view(batch_size)         # [batch]
        # scale = scale.view(batch_size)         # [batch]
        scale = scale.view(batch_size, 2)
        rotation = rotation.view(batch_size)  # [batch]
        translation = translation.view(batch_size, 2)  # [batch, 2]

        # 🚀 完全向量化的变换实现

        # 1) 向量化计算所有中心点
        # 高效创建mask - 完全向量化
        valid_lengths_tensor = torch.tensor(valid_lengths, device=device)
        arange_max = torch.arange(max_points, device=device).unsqueeze(
            0
        )  # [1, max_points]
        valid_mask = arange_max < valid_lengths_tensor.unsqueeze(
            1
        )  # [batch, max_points]

        # 向量化计算中心点
        # 方法：使用masked操作计算平均值
        masked_coords = (
            batch_coords * valid_mask.unsqueeze(-1).float()
        )  # [batch, max_points, 2]
        center_vecs = (
            masked_coords.sum(dim=1) / valid_lengths_tensor.unsqueeze(-1).float()
        )  # [batch, 2]

        # 2) 向量化变换
        pts = batch_coords.clone()  # [batch, max_points, 2]

        # Step 1: 向量化缩放（绕中心点）
        # 广播中心点到所有点
        centers_expanded = center_vecs.unsqueeze(1)  # [batch, 1, 2]
        #### 先根据flip进行翻转 - 向量化实现，考虑valid_mask
        if flip is not None:
            # 🔄 向量化flip操作，只对有效点进行翻转
            flip_mask = flip.bool().unsqueeze(1).unsqueeze(2)  # [batch, 1, 1]
            x_flip_factor = torch.where(flip_mask, -1.0, 1.0)  # [batch, 1, 1]
            # x_coords = pts[:, :, 0:1]  # [batch, max_points, 1]
            # pts[:, :, 0:1] = x_coords * x_flip_factor
            pts_flip = pts.clone()
            pts_flip[:, :, 0:1] = pts_flip[:, :, 0:1] * x_flip_factor
        else:
            pts_flip = pts

        # pts_centered = pts - centers_expanded  # [batch, max_points, 2]
        pts_centered = pts_flip
        # 向量化缩放：scale是[batch]，需要扩展到[batch, 1, 1]以正确广播

        # scale_expanded = scale.view(batch_size, 1, 1)  # [batch, 1, 1]
        scale_expanded = scale.view(batch_size, 1, 2)
        # pts_scaled = pts_centered * scale_expanded + centers_expanded  # [batch, max_points, 2]
        pts_scaled = pts_centered * scale_expanded

        # Step 2: 向量化旋转（绕中心点）
        theta = torch.deg2rad(rotation)  # [batch]
        cos_theta = torch.cos(theta)  # [batch]
        sin_theta = torch.sin(theta)  # [batch]

        # 构建旋转矩阵 [batch, 2, 2] - 更高效的方式
        R = torch.stack(
            [
                torch.stack([cos_theta, -sin_theta], dim=-1),
                torch.stack([sin_theta, cos_theta], dim=-1),
            ],
            dim=-2,
        )  # [batch, 2, 2]

        # 重新中心化用于旋转
        # pts_centered_for_rot = pts_scaled - centers_expanded  # [batch, max_points, 2]
        pts_centered_for_rot = pts_scaled

        # 向量化旋转：批量矩阵乘法
        # pts_centered_for_rot: [batch, max_points, 2]
        # R: [batch, 2, 2]
        # 使用einsum进行高效的批量矩阵乘法
        # pts_rotated = torch.einsum('bni,bij->bnj', pts_centered_for_rot, R) + centers_expanded  # [batch, max_points, 2]
        # pts_rotated = torch.einsum('bni,bij->bnj', pts_centered_for_rot, R)
        pts_rotated = torch.einsum("bni,bji->bnj", pts_centered_for_rot, R)

        # Step 3: 向量化平移
        translation_expanded = translation.unsqueeze(1)  # [batch, 1, 2]
        pts_final = pts_rotated + translation_expanded  # [batch, max_points, 2]

        # 应用mask，确保padding部分不受影响（可选，因为padding部分不会用于loss计算）
        pts = pts_final * valid_mask.unsqueeze(-1).float()

        # 返回变换后的坐标和有效长度
        return pts, valid_lengths

    def compute_direct_recon_loss(
        self,
        pred_coords_padded: Float["batch max_points 2"],
        gt_coords_padded,  # List[Tensor[n_i, 2]]，每个元素是[n, 2]的数组
        pred_valid_lengths,  # List[int] or Tensor[batch]，每个元素表示pred_coords_padded中相应数组的有效元素个数
        gt_valid_lengths,  # List[int] or Tensor[batch]，每个元素表示gt_coords_padded中相应数组的有效元素个数
        visualize=False,  # 是否进行可视化
        vis_save_dir=None,  # 可视化结果保存目录
        vis_prefix="recon_comparison",  # 可视化文件名前缀
        vis_batch_idx=0,  # 可视化哪个batch的数据
    ):
        batch_size, max_points, _ = pred_coords_padded.shape
        pred_valid_lengths = torch.tensor(
            pred_valid_lengths, dtype=torch.long, device=pred_coords_padded.device
        ).unsqueeze(-1)
        valid_mask = (
            torch.arange(max_points, device=pred_coords_padded.device).repeat(
                batch_size, 1
            )
            < pred_valid_lengths
        )
        pred_coords = pred_coords_padded[valid_mask]
        gt_valid_lengths = torch.tensor(
            gt_valid_lengths, dtype=torch.long, device=gt_coords_padded.device
        ).unsqueeze(-1)
        gt_valid_mask = (
            torch.arange(max_points, device=gt_coords_padded.device).repeat(
                batch_size, 1
            )
            < gt_valid_lengths
        )
        gt_coords = gt_coords_padded[gt_valid_mask]
        # gt_coords = torch.concatenate(gt_coords_padded)
        # gt_coords = gt_coords_padded[:, :2]
        assert pred_coords.shape[0] == gt_coords.shape[0]
        dist = F.l1_loss(pred_coords, gt_coords, reduction="mean")

        # 可视化功能
        if visualize:
            self._visualize_coords_comparison(
                pred_coords,
                gt_coords,
                dist.item(),
                vis_save_dir,
                vis_prefix,
                vis_batch_idx,
            )

        return dist

    def _visualize_coords_comparison(
        self,
        pred_coords,
        gt_coords,
        l1_loss_value,
        vis_save_dir=None,
        vis_prefix="recon_comparison",
        vis_batch_idx=0,
    ):
        """
        可视化预测坐标和真实坐标的对比

        Args:
            pred_coords_padded: 预测的坐标 [batch, max_points, 2]
            gt_coords_padded: 真实的坐标 [batch, max_points, 2] (前两维)
            valid_lengths: 有效长度 [batch, 1]
            valid_mask: 有效mask [batch, max_points]
            l1_loss_value: L1损失值
            vis_save_dir: 保存目录
            vis_prefix: 文件名前缀
            vis_batch_idx: 要可视化的batch索引
        """
        # 设置保存目录
        if vis_save_dir is None:
            vis_save_dir = "./visualization_output"
        os.makedirs(vis_save_dir, exist_ok=True)

        # # 将数据移到CPU并转为numpy
        # pred_coords_np = pred_coords_padded[vis_batch_idx].detach().cpu().numpy()
        # gt_coords_np = gt_coords_padded[vis_batch_idx, :, :2].detach().cpu().numpy()
        # valid_len = valid_lengths[vis_batch_idx].item()

        # 只取有效的点
        pred_coords_valid = pred_coords.detach().cpu().numpy()
        gt_coords_valid = gt_coords.detach().cpu().numpy()

        valid_len = gt_coords_valid.shape[0]

        # 计算逐点的L1距离
        point_wise_dist = np.linalg.norm(pred_coords_valid - gt_coords_valid, axis=1)

        # 创建可视化
        fig, axes = plt.subplots(1, 2, figsize=(16, 16))

        # 子图1: 预测坐标和真实坐标的叠加显示
        ax1 = axes[0]
        ax1.scatter(
            gt_coords_valid[:, 0],
            gt_coords_valid[:, 1],
            c="blue",
            s=50,
            alpha=0.6,
            label="Ground Truth",
            marker="o",
        )
        ax1.scatter(
            pred_coords_valid[:, 0],
            pred_coords_valid[:, 1],
            c="red",
            s=50,
            alpha=0.6,
            label="Prediction",
            marker="x",
        )
        ax1.set_xlabel("X坐标", fontsize=12)
        ax1.set_ylabel("Y坐标", fontsize=12)
        ax1.set_title(
            f"预测vs真实坐标叠加图\nL1 Loss: {l1_loss_value:.6f}", fontsize=14
        )
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect("equal", "box")

        # 子图2: 差异向量图（用箭头连接对应点）
        ax2 = axes[1]
        ax2.scatter(
            gt_coords_valid[:, 0],
            gt_coords_valid[:, 1],
            c="blue",
            s=50,
            alpha=0.6,
            label="Ground Truth",
            marker="o",
        )
        # 绘制从GT到Pred的箭头
        for i in range(len(gt_coords_valid)):
            dx = pred_coords_valid[i, 0] - gt_coords_valid[i, 0]
            dy = pred_coords_valid[i, 1] - gt_coords_valid[i, 1]
            ax2.arrow(
                gt_coords_valid[i, 0],
                gt_coords_valid[i, 1],
                dx,
                dy,
                head_width=0.01,
                head_length=0.01,
                fc="red",
                ec="red",
                alpha=0.5,
                linewidth=0.5,
            )
        ax2.set_xlabel("X坐标", fontsize=12)
        ax2.set_ylabel("Y坐标", fontsize=12)
        ax2.set_title("差异向量图（箭头指向预测点）", fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect("equal", "box")

        # 添加统计信息
        stats_text = f"summary:\n"
        stats_text += f"n_points: {valid_len}\n"
        stats_text += f"mean: {point_wise_dist.mean():.6f}\n"
        stats_text += f"median: {np.median(point_wise_dist):.6f}\n"
        stats_text += f"max: {point_wise_dist.max():.6f}\n"
        stats_text += f"min: {point_wise_dist.min():.6f}\n"
        stats_text += f"std: {point_wise_dist.std():.6f}"
        fig.text(
            0.02,
            0.98,
            stats_text,
            transform=fig.transFigure,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # 保存图片
        save_path = os.path.join(vis_save_dir, f"{vis_batch_idx}_{vis_prefix}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"可视化结果已保存到: {save_path}")
        print(
            f"统计信息: 点数={valid_len}, 平均误差={point_wise_dist.mean():.6f}, "
            f"最大误差={point_wise_dist.max():.6f}"
        )

    def preload_all_primitives(self, primitive_ids_to_load=None):
        """
        预加载所有primitive坐标到内存缓存中，显著提升训练速度

        Args:
            primitive_ids_to_load: 可选，指定要预加载的primitive ID列表。
                                 如果为None，则从数据集中自动获取所有unique primitive IDs
        """
        print("开始预加载primitive坐标...")

        if not hasattr(self, "_primitive_coords_cache"):
            self._primitive_coords_cache = {}

        if primitive_ids_to_load is None:
            # 如果没有指定，可以从训练数据集中收集所有unique primitive IDs
            # 这需要数据集配合，或者可以扫描primitive目录
            print("Warning: 没有指定primitive_ids_to_load，请手动调用或从数据集获取")
            return

        # 过滤出未缓存的IDs
        uncached_ids = [
            pid
            for pid in primitive_ids_to_load
            if pid not in self._primitive_coords_cache
        ]

        if not uncached_ids:
            print("所有primitive已缓存，无需重复加载")
            return

        print(f"预加载 {len(uncached_ids)} 个primitive...")

        # 串行加载primitive坐标
        for primitive_id in uncached_ids:
            try:
                primitive_coords = self.load_primitive_coords(primitive_id)  # [n, 3]
                primitive_coords_2d = primitive_coords[:, :2].contiguous()  # [n, 2]
                self._primitive_coords_cache[primitive_id] = primitive_coords_2d
            except Exception as e:
                print(f"Warning: Failed to load primitive {primitive_id}: {e}")
                # 使用默认占位符避免中断训练
                self._primitive_coords_cache[primitive_id] = torch.zeros(
                    (10, 2), dtype=torch.float32
                )

        print(f"预加载完成！缓存中现有 {len(self._primitive_coords_cache)} 个primitive")

    def get_cache_stats(self):
        """获取缓存统计信息"""
        if hasattr(self, "_primitive_coords_cache"):
            total_size = sum(
                coords.numel() * coords.element_size()
                for coords in self._primitive_coords_cache.values()
            )
            return {
                "cached_primitives": len(self._primitive_coords_cache),
                "total_memory_mb": total_size / 1024 / 1024,
            }
        return {"cached_primitives": 0, "total_memory_mb": 0}

    def set_recon_epoch(self, epoch: int):
        """Set the zero-based epoch used by the R0 reconstruction warmup."""
        self._recon_epoch = max(int(epoch), 0)

    def _discretize_rotation_phase(self, phase_degrees):
        """Nearest circular phase bin; unlike ``discretize``, 0 is a center."""

        step = 360.0 / float(self.num_discrete_rotation)
        return (
            torch.round(torch.remainder(phase_degrees, 360.0) / step)
            .long()
            .remainder(self.num_discrete_rotation)
        )

    def _undiscretize_rotation_phase(self, phase_index):
        """Decode circular phase bins whose centers are j * 360 / M."""

        step = 360.0 / float(self.num_discrete_rotation)
        return torch.remainder(phase_index.float(), self.num_discrete_rotation) * step

    def _rotation_gt_phase(self, raw_angle, primitive_id):
        if self.rotation_symmetry_mode == "off":
            return raw_angle
        return raw_angle_to_phase(
            raw_angle, primitive_id, self.rotation_symmetry_orders
        )

    def _rotation_phase_to_physical(self, phase, primitive_id):
        if self.rotation_symmetry_mode == "off":
            return phase
        return phase_to_canonical_angle(
            phase, primitive_id, self.rotation_symmetry_orders
        )

    def _rotation_gt_to_physical(self, raw_angle, primitive_id):
        if self.rotation_symmetry_mode == "off":
            return raw_angle
        return canonicalize_raw_angle(
            raw_angle, primitive_id, self.rotation_symmetry_orders
        )

    def _stable_circular_rotation_expectation(self, probabilities):
        """Differentiable circular mean with a detached modal anchor.

        A plain atan2(E[sin], E[cos]) is singular for an approximately uniform
        rotation distribution, exactly the regime seen by a newly reset head.
        Mixing a small detached unit vector at the modal bin bounds that
        derivative while preserving circular behavior once the distribution
        becomes confident.
        """
        radians = torch.deg2rad(self.bin_centers_rotation).to(
            probabilities.device, probabilities.dtype
        )
        sine = (probabilities * torch.sin(radians)).sum(dim=-1)
        cosine = (probabilities * torch.cos(radians)).sum(dim=-1)
        anchor_index = probabilities.detach().argmax(dim=-1)
        anchor = radians[anchor_index]
        weight = self.recon_rotation_anchor_weight
        sine = (1.0 - weight) * sine + weight * torch.sin(anchor)
        cosine = (1.0 - weight) * cosine + weight * torch.cos(anchor)
        return torch.rad2deg(torch.atan2(sine, cosine)).remainder(360.0)

    def _attr_soft_decode_continuous(
        self,
        translation_logits,
        rotation_logits,
        shape_logits,
        shape_centers,
        primitive_id,
    ):
        """One deterministic T/phase/R/S graph shared by Huber and recon.

        Accumulate probabilities in FP32 under BF16 training. Rotation keeps
        the existing detached modal anchor and symmetry convention. Log scale
        is a geometric expectation, not an arithmetic mean of log-spaced bins.
        """
        translation_probs = translation_logits.float().softmax(-1)
        rotation_probs = rotation_logits.float().softmax(-1)
        shape_probs = shape_logits.float().softmax(-1)
        translation = (translation_probs * self.bin_centers_translation.float()).sum(-1)
        phase = self._stable_circular_rotation_expectation(rotation_probs)
        rotation = self._rotation_phase_to_physical(phase, primitive_id)
        if not self.size_pred and self.scale_log:
            shape = torch.exp(
                (
                    shape_probs * self.bin_centers_scale.float().clamp_min(1e-8).log()
                ).sum(-1)
            )
        else:
            shape = (shape_probs * shape_centers.float()).sum(-1)
        return translation, phase, rotation, shape

    def _attr_recon_decode_probabilities(self, logits):
        """Decode translation/rotation logits for the reconstruction path."""
        if self.recon_decode_mode == "soft_expectation":
            return logits.softmax(dim=-1)
        if self.recon_decode_mode == "hard_st":
            return deterministic_st_argmax_onehot(logits)
        # Constructor validation makes this unreachable in normal use.  Keep
        # it defensive for lightweight harnesses and mutated model state.
        validate_recon_decode_mode(self.recon_decode_mode)
        raise AssertionError("unreachable")

    def recon_weight_factor(self) -> float:
        if self.recon_warmup_epochs <= 1:
            return 1.0
        return min(
            max(self._recon_epoch / float(self.recon_warmup_epochs - 1), 0.0),
            1.0,
        )

    @staticmethod
    def _recon_stratified_segments(segments, limit):
        segments = np.asarray(segments, dtype=np.float32).reshape(-1, 2, 2)
        if len(segments) <= limit:
            return segments
        lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
        order = np.argsort(lengths, kind="stable")
        ranks = np.linspace(0, len(order) - 1, limit).round().astype(np.int64)
        return segments[order[ranks]]

    def _initialize_attr_recon_templates(self):
        """Cache bounded templates in the selected scale coordinate system."""
        template = torch.zeros(
            self.num_primitive,
            self.recon_dense_template_limit,
            2,
            2,
            dtype=torch.float32,
        )
        mask = torch.zeros(
            self.num_primitive,
            self.recon_dense_template_limit,
            dtype=torch.bool,
        )
        if self.primitive_dir is not None:
            normalizer = float(self.scene_bound) / 2.0
            for primitive_id in range(self.num_primitive):
                path = os.path.join(self.primitive_dir, f"{primitive_id}.json")
                if not os.path.isfile(path):
                    continue
                with open(path, "r", encoding="utf-8") as handle:
                    record = json.load(handle)
                raw_segments = record.get("raw_coords", record.get("coords"))
                if raw_segments is None:
                    continue
                raw_segments = np.asarray(raw_segments, dtype=np.float32).reshape(
                    -1, 2, 2
                )
                if self.scale_parameterization == "absolute_local_extent":
                    from models.primitive.scale_parameterization import (
                        normalize_coords_to_local_extent,
                    )

                    transformed_segments = normalize_coords_to_local_extent(
                        raw_segments
                    ).astype(np.float32)
                else:
                    transformed_segments = raw_segments / normalizer
                segments = self._recon_stratified_segments(
                    transformed_segments,
                    self.recon_dense_template_limit,
                )
                count = min(len(segments), self.recon_dense_template_limit)
                if count:
                    template[primitive_id, :count] = torch.from_numpy(segments[:count])
                    mask[primitive_id, :count] = True
        # These are deterministic derivatives of primitive_dir, not checkpoint
        # state.  Keeping them non-persistent preserves old checkpoint loading.
        self.register_buffer("attr_recon_template_segments", template, persistent=False)
        self.register_buffer("attr_recon_template_mask", mask, persistent=False)

    @staticmethod
    def _attr_recon_transform(template, scale, rotation_deg, translation, flip):
        points = template.clone()
        points[..., 0] = torch.where(
            flip[:, None, None], -points[..., 0], points[..., 0]
        )
        points = points * scale[:, None, None, :]
        rotation = torch.deg2rad(rotation_deg)
        cosine = torch.cos(rotation)[:, None, None]
        sine = torch.sin(rotation)[:, None, None]
        x, y = points[..., 0], points[..., 1]
        points = torch.stack((cosine * x - sine * y, sine * x + cosine * y), dim=-1)
        return points + translation[:, None, None, :]

    @staticmethod
    def _attr_recon_anchors(segments):
        return torch.cat(
            (segments[:, :, 0], segments[:, :, 1], segments.mean(dim=2)), dim=1
        )

    @staticmethod
    def _attr_recon_anchor_mask(segment_mask):
        return torch.cat((segment_mask, segment_mask, segment_mask), dim=1)

    @staticmethod
    def _attr_recon_point_segment_distance(points, segments, segment_mask):
        start = segments[:, None, :, 0]
        delta = segments[:, None, :, 1] - start
        relative = points[:, :, None] - start
        denominator = (delta * delta).sum(dim=-1).clamp_min(1e-12)
        projection = ((relative * delta).sum(dim=-1) / denominator).clamp(0.0, 1.0)
        closest = start + projection[..., None] * delta
        distance = torch.linalg.vector_norm(points[:, :, None] - closest, dim=-1)
        return distance.masked_fill(~segment_mask[:, None, :], float("inf"))

    def _attr_recon_directional_loss(
        self,
        source,
        source_mask,
        target,
        target_mask,
        beta=0.01,
        distance_scale=None,
    ):
        point_mask = self._attr_recon_anchor_mask(source_mask)
        distance = self._attr_recon_point_segment_distance(
            self._attr_recon_anchors(source), target, target_mask
        ).amin(dim=-1)
        if distance_scale is not None:
            distance = distance / distance_scale[:, None].clamp_min(1e-8)
        valid_distance = distance[point_mask]
        loss = F.smooth_l1_loss(
            valid_distance,
            torch.zeros_like(valid_distance),
            beta=beta,
            reduction="none",
        )
        owners = torch.arange(len(source), device=source.device)[:, None]
        owners = owners.expand_as(point_mask)[point_mask]
        totals = torch.zeros(len(source), device=source.device, dtype=loss.dtype)
        counts = torch.zeros_like(totals)
        totals.scatter_add_(0, owners, loss)
        counts.scatter_add_(0, owners, torch.ones_like(loss))
        return totals / counts.clamp_min(1.0)

    @staticmethod
    def _attr_recon_segment_diagonal(segments, segment_mask):
        point_mask = (
            segment_mask[:, :, None].expand(-1, -1, 2).reshape(len(segments), -1)
        )
        points = segments.reshape(len(segments), -1, 2)
        minimum = points.masked_fill(~point_mask[:, :, None], float("inf")).amin(1)
        maximum = points.masked_fill(~point_mask[:, :, None], -float("inf")).amax(1)
        diagonal = torch.linalg.vector_norm(maximum - minimum, dim=-1)
        diagonal = torch.where(
            torch.isfinite(diagonal), diagonal, torch.ones_like(diagonal)
        )
        return diagonal.clamp_min(1e-6)

    def _attr_recon_gt_clip_parameters(self, gt_segments, segment_mask):
        """Compute fixed line/box intersection intervals from GT geometry."""
        start = gt_segments[:, :, 0]
        delta = gt_segments[:, :, 1] - start
        enter = torch.zeros_like(delta[..., 0])
        leave = torch.ones_like(enter)
        valid = segment_mask.clone()
        lower, upper = self.recon_clip_range
        for axis in range(2):
            origin = start[..., axis]
            direction = delta[..., axis]
            parallel = direction.abs() <= 1e-10
            valid = valid & (~parallel | ((origin >= lower) & (origin <= upper)))
            safe_direction = torch.where(
                parallel, torch.ones_like(direction), direction
            )
            first = (lower - origin) / safe_direction
            second = (upper - origin) / safe_direction
            axis_enter = torch.minimum(first, second)
            axis_leave = torch.maximum(first, second)
            enter = torch.maximum(enter, torch.where(parallel, enter, axis_enter))
            leave = torch.minimum(leave, torch.where(parallel, leave, axis_leave))
        enter = enter.clamp(0.0, 1.0)
        leave = leave.clamp(0.0, 1.0)
        valid = valid & (leave >= enter)
        clipped_length = torch.linalg.vector_norm(
            delta * (leave - enter)[..., None], dim=-1
        )
        valid = valid & (clipped_length > 1e-8)
        return enter.detach(), leave.detach(), valid.detach()

    @staticmethod
    def _attr_recon_apply_clip_parameters(segments, enter, leave):
        start = segments[:, :, 0]
        delta = segments[:, :, 1] - start
        return torch.stack(
            (
                start + enter[..., None] * delta,
                start + leave[..., None] * delta,
            ),
            dim=2,
        )

    def _attr_recon_support_selection(
        self,
        gt_template,
        template_mask,
        source,
        source_mask,
        distance_scale=None,
        min_support=None,
    ):
        distance = self._attr_recon_point_segment_distance(
            self._attr_recon_anchors(gt_template), source, source_mask
        )
        distance = distance.amin(dim=-1).reshape(len(gt_template), 3, -1).mean(dim=1)
        if distance_scale is not None:
            distance = distance / distance_scale[:, None].clamp_min(1e-8)
        distance = distance.masked_fill(~template_mask, float("inf"))
        raw_supported = (
            (distance <= self.recon_support_threshold) & template_mask
        ).sum(dim=-1)
        if min_support is None:
            min_support = torch.full_like(raw_supported, self.recon_min_support)
        elif not torch.is_tensor(min_support):
            min_support = torch.full_like(raw_supported, int(min_support))
        else:
            min_support = min_support.to(raw_supported.device, raw_supported.dtype)
        desired = torch.maximum(raw_supported, min_support).clamp(
            max=self.recon_support_cap
        )
        desired = torch.minimum(desired, template_mask.sum(dim=-1))
        order = distance.argsort(dim=-1)
        selected_indices = order[:, : self.recon_support_cap]
        selected_mask = (
            torch.arange(self.recon_support_cap, device=order.device)[None, :]
            < desired[:, None]
        )
        return selected_indices, selected_mask, raw_supported

    @staticmethod
    def _attr_recon_partitioned_mean(
        per_instance,
        is_partial,
        partial_visibility_weight,
        partial_multiplier,
        reduction_mode="instance_mean",
    ):
        """Reduce per-instance losses while retaining partition diagnostics.

        instance_mean is the production default: full and partial numerators
        share one denominator.  partitioned retains the historical
        mean(full) + lambda * mean(partial) behavior for exact ablations.
        """
        if partial_multiplier < 0:
            raise ValueError("partial_multiplier must be nonnegative")
        if reduction_mode not in {"instance_mean", "partitioned"}:
            raise ValueError(
                "reduction_mode must be instance_mean/partitioned, got "
                f"{reduction_mode!r}"
            )
        is_partial = is_partial.bool()
        full_mask = ~is_partial
        zero = per_instance.sum() * 0.0

        if full_mask.any():
            full_values = per_instance[full_mask]
            # Deliberately retain the historical exclude reduction exactly:
            # sum(loss * 1) / sum(1), including its operation order.
            full_weights = torch.ones_like(full_values)
            full_effective_count = full_weights.sum()
            loss_full = (
                full_values * full_weights
            ).sum() / full_effective_count.clamp_min(1e-8)
            full_numerator = (full_values * full_weights).sum()
        else:
            loss_full = zero
            full_effective_count = zero.detach()
            full_numerator = zero

        if is_partial.any():
            partial_weights = partial_visibility_weight[is_partial]
            partial_effective_count = partial_weights.sum()
            loss_partial = (
                per_instance[is_partial] * partial_weights
            ).sum() / partial_effective_count.clamp_min(1e-8)
            partial_numerator = (per_instance[is_partial] * partial_weights).sum()
        else:
            loss_partial = zero
            partial_effective_count = zero.detach()
            partial_numerator = zero

        if reduction_mode == "instance_mean":
            partial_multiplier_tensor = per_instance.new_tensor(
                float(partial_multiplier)
            )
            total_effective_count = (
                full_effective_count
                + partial_multiplier_tensor * partial_effective_count
            )
            loss = (
                full_numerator + partial_multiplier_tensor * partial_numerator
            ) / total_effective_count.clamp_min(1e-8)
        else:
            loss = loss_full + float(partial_multiplier) * loss_partial
        return (
            loss,
            loss_full,
            loss_partial,
            full_effective_count.detach(),
            partial_effective_count.detach(),
        )

    @staticmethod
    def _attr_recon_partitioned_masked_mean(
        per_instance,
        is_partial,
        partial_visibility_weight,
        partial_multiplier,
        valid_mask,
        reduction_mode="instance_mean",
    ):
        """Partitioned reduction with an additional detached validity mask.

        This is used only by the opt-in factorized rotation branch.  The mask
        covers the optional circular-error gate and the requirement that a
        template->source term has at least one real GT-supported template
        segment.  Translation continues to use the historical unmasked
        reduction above.
        """
        if partial_multiplier < 0:
            raise ValueError("partial_multiplier must be nonnegative")
        if reduction_mode not in {"instance_mean", "partitioned"}:
            raise ValueError(
                "reduction_mode must be instance_mean/partitioned, got "
                f"{reduction_mode!r}"
            )
        is_partial = is_partial.bool()
        valid_mask = valid_mask.bool().detach()
        full_mask = (~is_partial) & valid_mask
        partial_mask = is_partial & valid_mask
        zero = per_instance.sum() * 0.0

        if full_mask.any():
            full_values = per_instance[full_mask]
            full_weights = torch.ones_like(full_values)
            full_effective_count = full_weights.sum()
            loss_full = (
                full_values * full_weights
            ).sum() / full_effective_count.clamp_min(1e-8)
            full_numerator = (full_values * full_weights).sum()
        else:
            loss_full = zero
            full_effective_count = zero.detach()
            full_numerator = zero

        if partial_mask.any():
            partial_weights = partial_visibility_weight[partial_mask]
            partial_effective_count = partial_weights.sum()
            loss_partial = (
                per_instance[partial_mask] * partial_weights
            ).sum() / partial_effective_count.clamp_min(1e-8)
            partial_numerator = (per_instance[partial_mask] * partial_weights).sum()
        else:
            loss_partial = zero
            partial_effective_count = zero.detach()
            partial_numerator = zero

        if reduction_mode == "instance_mean":
            partial_multiplier_tensor = per_instance.new_tensor(
                float(partial_multiplier)
            )
            total_effective_count = (
                full_effective_count
                + partial_multiplier_tensor * partial_effective_count
            )
            loss = (
                full_numerator + partial_multiplier_tensor * partial_numerator
            ) / total_effective_count.clamp_min(1e-8)
        else:
            loss = loss_full + float(partial_multiplier) * loss_partial
        return (
            loss,
            loss_full,
            loss_partial,
            full_effective_count.detach(),
            partial_effective_count.detach(),
        )

    def _compute_attr_full_recon_loss(
        self,
        *,
        pred_translation,
        pred_rotation,
        pred_scale,
        gt_primitive_id,
        gt_scale,
        gt_rotation,
        gt_flip,
        primitive_mask,
        recon_source_segments,
        recon_source_mask,
        recon_valid_mask,
        recon_full_translation,
        recon_clip_count_ratio=None,
        recon_clip_length_ratio=None,
        recon_is_partial=None,
        recon_preclip_source_segments=None,
        recon_preclip_source_mask=None,
    ):
        zero = pred_translation.sum() * 0.0 + pred_rotation.sum() * 0.0
        scale_mode = getattr(
            self,
            "recon_scale_mode",
            "gt" if getattr(self, "recon_use_gt_scale", True) else "pred",
        )
        partial_scale_mode = getattr(self, "recon_partial_scale_mode", scale_mode)
        if (scale_mode != "gt" or partial_scale_mode != "gt") and pred_scale is None:
            raise ValueError(
                "predicted reconstruction scale mode requires pred_scale: "
                f"full={scale_mode!r}, partial={partial_scale_mode!r}"
            )
        # Only pred creates an autograd edge. pred_detach intentionally uses
        # identical forward geometry while serving as a no-scale-gradient
        # control.
        if scale_mode == "pred" or partial_scale_mode == "pred":
            zero = zero + pred_scale.sum() * 0.0
        if any(
            value is None
            for value in (
                recon_source_segments,
                recon_source_mask,
                recon_valid_mask,
                recon_full_translation,
            )
        ):
            raise ValueError("reconstruction>0 requires dataset.return_recon_data=true")
        partial_multiplier = (
            0.0 if self.recon_partial_mode == "exclude" else self.recon_partial_weight
        )
        # A legacy/full-only configuration does not need the extra payload
        # even though preclip_full is now the model default.  Once partial
        # reconstruction is active, preclip_full is strict: silently falling
        # back to visible geometry would make the two objectives inconsistent.
        use_preclip_full_source = (
            getattr(self, "recon_partial_source_mode", "visible") == "preclip_full"
            and partial_multiplier > 0.0
        )
        if use_preclip_full_source and any(
            value is None
            for value in (
                recon_preclip_source_segments,
                recon_preclip_source_mask,
            )
        ):
            raise ValueError(
                "recon_partial_source_mode=preclip_full requires dataset "
                "recon_preclip_full_source=true"
            )
        selected_source_segments = (
            recon_preclip_source_segments
            if use_preclip_full_source
            else recon_source_segments
        )
        selected_source_mask = (
            recon_preclip_source_mask if use_preclip_full_source else recon_source_mask
        )
        valid = primitive_mask & recon_valid_mask.bool()
        # λ_partial=0 intentionally takes the same early full-only path as
        # recon_partial_mode='exclude'.  Besides avoiding wasted work, this
        # makes the loss value and gradients a strict ablation control.
        if partial_multiplier == 0.0:
            if recon_is_partial is not None:
                valid = valid & ~recon_is_partial.bool()
            elif recon_clip_count_ratio is not None:
                valid = valid & (recon_clip_count_ratio >= 1.0 - 1e-7)
        valid = valid & selected_source_mask.bool().any(dim=-1)
        if not valid.any():
            return zero, {
                "candidate": 0,
                "geometry_valid": 0,
                "partial_valid": 0,
                "full_valid": 0,
                "loss_full": zero,
                "loss_partial": zero,
                "full_effective_count": zero.detach(),
                "partial_effective_count": zero.detach(),
            }

        primitive_id = (
            gt_primitive_id[valid].long().clamp(min=0, max=self.num_primitive - 1)
        )
        template = self.attr_recon_template_segments[primitive_id].float()
        template_mask = self.attr_recon_template_mask[primitive_id]
        source = selected_source_segments[valid].float()
        source_mask = selected_source_mask[valid].bool()
        gt_scale_valid = gt_scale[valid].float().detach()
        rotation = gt_rotation[valid].float().detach().reshape(-1)
        translation = recon_full_translation[valid].float().detach()
        flip = gt_flip[valid].bool().detach()
        if recon_clip_count_ratio is None:
            clip_count_ratio = torch.ones_like(rotation)
        else:
            clip_count_ratio = recon_clip_count_ratio[valid].float().detach()
        if recon_clip_length_ratio is None:
            clip_length_ratio = clip_count_ratio
        else:
            clip_length_ratio = recon_clip_length_ratio[valid].float().detach()
        if recon_is_partial is None:
            is_partial = clip_count_ratio < 1.0 - 1e-7
        else:
            is_partial = recon_is_partial[valid].bool().detach()
        candidate_count = int(valid.sum().detach())

        with torch.no_grad():
            gt_template = self._attr_recon_transform(
                template, gt_scale_valid, rotation, translation, flip
            )
            gt_diagonal = self._attr_recon_segment_diagonal(gt_template, template_mask)
            if self.recon_visibility_aware and not use_preclip_full_source:
                clip_enter, clip_leave, visible_mask = (
                    self._attr_recon_gt_clip_parameters(gt_template, template_mask)
                )
                # Visibility clipping is a partial-instance operation.  A full
                # instance must retain the complete template even when small
                # source/template residuals put a template endpoint marginally
                # outside the crop window.
                clip_enter = torch.where(
                    is_partial[:, None], clip_enter, torch.zeros_like(clip_enter)
                )
                clip_leave = torch.where(
                    is_partial[:, None], clip_leave, torch.ones_like(clip_leave)
                )
                visible_mask = torch.where(
                    is_partial[:, None], visible_mask, template_mask
                )
                gt_visible_template = self._attr_recon_apply_clip_parameters(
                    gt_template, clip_enter, clip_leave
                )
            else:
                clip_enter = torch.zeros_like(template_mask, dtype=gt_template.dtype)
                clip_leave = torch.ones_like(clip_enter)
                visible_mask = template_mask
                gt_visible_template = gt_template
            distance_scale = (
                gt_diagonal if self.recon_normalize_by_gt_diagonal else None
            )
            beta = self.recon_relative_beta if distance_scale is not None else 0.01
            gt_geometry_loss = self._attr_recon_directional_loss(
                source,
                source_mask,
                gt_visible_template,
                visible_mask,
                beta=beta,
                distance_scale=distance_scale,
            )
            geometry_valid = visible_mask.any(dim=-1)
            if self.recon_gt_geometry_threshold >= 0:
                geometry_valid = geometry_valid & (
                    gt_geometry_loss <= self.recon_gt_geometry_threshold
                )
        if not geometry_valid.any():
            return zero, {
                "candidate": candidate_count,
                "geometry_valid": 0,
                "partial_valid": 0,
                "full_valid": 0,
                "loss_full": zero,
                "loss_partial": zero,
                "full_effective_count": zero.detach(),
                "partial_effective_count": zero.detach(),
            }

        template = template[geometry_valid]
        template_mask = template_mask[geometry_valid]
        source = source[geometry_valid]
        source_mask = source_mask[geometry_valid]
        gt_scale_valid = gt_scale_valid[geometry_valid]
        rotation = rotation[geometry_valid]
        translation = translation[geometry_valid]
        flip = flip[geometry_valid]
        gt_template = gt_template[geometry_valid]
        gt_visible_template = gt_visible_template[geometry_valid]
        visible_mask = visible_mask[geometry_valid]
        clip_enter = clip_enter[geometry_valid]
        clip_leave = clip_leave[geometry_valid]
        gt_diagonal = gt_diagonal[geometry_valid]
        clip_length_ratio = clip_length_ratio[geometry_valid]
        is_partial = is_partial[geometry_valid]
        pred_translation = pred_translation[valid].float()[geometry_valid]
        pred_rotation = pred_rotation[valid].float().reshape(-1)[geometry_valid]
        pred_scale_valid = None
        if scale_mode != "gt" or partial_scale_mode != "gt":
            pred_scale_valid = pred_scale[valid].float()[geometry_valid]

        def select_scale(mode):
            if mode == "gt":
                return gt_scale_valid
            if mode == "pred_detach":
                return pred_scale_valid.detach()
            return pred_scale_valid

        selected_full_scale = select_scale(scale_mode)
        selected_partial_scale = select_scale(partial_scale_mode)
        selected_pred_scale = torch.where(
            is_partial[:, None],
            selected_partial_scale,
            selected_full_scale,
        )

        with torch.no_grad():
            if use_preclip_full_source:
                support_minimum = torch.full_like(
                    is_partial, self.recon_min_support, dtype=torch.long
                )
            else:
                support_minimum = torch.where(
                    is_partial,
                    torch.full_like(
                        is_partial,
                        self.recon_partial_min_support,
                        dtype=torch.long,
                    ),
                    torch.full_like(
                        is_partial, self.recon_min_support, dtype=torch.long
                    ),
                )
            support_indices, support_mask, raw_support = (
                self._attr_recon_support_selection(
                    gt_visible_template,
                    visible_mask,
                    source,
                    source_mask,
                    distance_scale=(
                        gt_diagonal if self.recon_normalize_by_gt_diagonal else None
                    ),
                    min_support=support_minimum,
                )
            )
            # A partial crop with no threshold-supported template edge must
            # not invent a reverse correspondence by force-selecting its
            # nearest edge.  Keep the source->template term in that rare case.
            support_mask = support_mask & ((~is_partial | (raw_support > 0))[:, None])
        recon_tr_mode = getattr(self, "recon_tr_mode", "joint_bidir")
        distance_scale = gt_diagonal if self.recon_normalize_by_gt_diagonal else None
        beta = self.recon_relative_beta if distance_scale is not None else 0.01

        if recon_tr_mode == "joint_bidir":
            # Keep this block operation-for-operation identical to the
            # historical R2 implementation.  The new factorized behavior is
            # strictly opt-in and must not perturb legacy loss bits/gradients.
            pred_template = self._attr_recon_transform(
                template,
                selected_pred_scale,
                pred_rotation,
                pred_translation,
                flip,
            )
            pred_visible_template = self._attr_recon_apply_clip_parameters(
                pred_template, clip_enter, clip_leave
            )
            source_to_template = self._attr_recon_directional_loss(
                source,
                source_mask,
                pred_visible_template,
                visible_mask,
                beta=beta,
                distance_scale=distance_scale,
            )
            if self.recon_partial_mode == "source_only" and is_partial.any():
                source_to_full_template = self._attr_recon_directional_loss(
                    source,
                    source_mask,
                    pred_template,
                    template_mask,
                    beta=beta,
                    distance_scale=distance_scale,
                )
                source_to_template = torch.where(
                    is_partial, source_to_full_template, source_to_template
                )
            gather_indices = support_indices[:, :, None, None].expand(-1, -1, 2, 2)
            supported_template = pred_visible_template.gather(1, gather_indices)
            template_to_source = self._attr_recon_directional_loss(
                supported_template,
                support_mask,
                source,
                source_mask,
                beta=beta,
                distance_scale=distance_scale,
            )
            has_reverse_support = support_mask.any(dim=-1)
            per_instance = 0.5 * (source_to_template + template_to_source)
            per_instance = torch.where(
                is_partial & ~has_reverse_support,
                source_to_template,
                per_instance,
            )
            if self.recon_partial_mode == "source_only":
                per_instance = torch.where(is_partial, source_to_template, per_instance)
            if use_preclip_full_source:
                weights = torch.ones_like(clip_length_ratio)
            else:
                weights = clip_length_ratio.clamp(0.0, 1.0).pow(
                    self.recon_partial_weight_power
                )
                weights = torch.where(is_partial, weights, torch.ones_like(weights))
            (
                loss,
                loss_full,
                loss_partial,
                full_effective_count,
                partial_effective_count,
            ) = self._attr_recon_partitioned_mean(
                per_instance,
                is_partial,
                weights,
                partial_multiplier,
                getattr(self, "recon_instance_reduction", "instance_mean"),
            )
            return loss, {
                "candidate": candidate_count,
                "geometry_valid": int(geometry_valid.sum().detach()),
                "partial_valid": int(is_partial.sum().detach()),
                "full_valid": int((~is_partial).sum().detach()),
                "loss_full": loss_full,
                "loss_partial": loss_partial,
                "full_effective_count": full_effective_count,
                "partial_effective_count": partial_effective_count,
                "loss_translation": zero,
                "loss_rotation": zero,
                "rotation_valid": 0,
                "rotation_effective_count": zero.detach(),
            }

        # The constructor validates this in ordinary use; retain a defensive
        # check for lightweight harnesses that mutate attributes directly.
        validate_recon_tr_mode(recon_tr_mode)
        rotation_direction = validate_recon_rotation_direction(
            getattr(self, "recon_rotation_direction", "bidirectional")
        )
        rotation_weight = float(getattr(self, "recon_rotation_weight", 1.0))
        rotation_gate_deg = float(getattr(self, "recon_rotation_gate_deg", -1.0))

        # Factorized translation: pred T + detached GT R.
        translation_template = self._attr_recon_transform(
            template,
            selected_pred_scale,
            rotation.detach(),
            pred_translation,
            flip,
        )
        # Factorized rotation: detached GT T + pred R.
        rotation_template = self._attr_recon_transform(
            template,
            selected_pred_scale,
            pred_rotation,
            translation.detach(),
            flip,
        )

        def directional_terms(pred_template):
            pred_visible_template = self._attr_recon_apply_clip_parameters(
                pred_template, clip_enter, clip_leave
            )
            source_to_pred_template = self._attr_recon_directional_loss(
                source,
                source_mask,
                pred_visible_template,
                visible_mask,
                beta=beta,
                distance_scale=distance_scale,
            )
            if self.recon_partial_mode == "source_only" and is_partial.any():
                source_to_full_template = self._attr_recon_directional_loss(
                    source,
                    source_mask,
                    pred_template,
                    template_mask,
                    beta=beta,
                    distance_scale=distance_scale,
                )
                source_to_pred_template = torch.where(
                    is_partial,
                    source_to_full_template,
                    source_to_pred_template,
                )
            gather_indices = support_indices[:, :, None, None].expand(-1, -1, 2, 2)
            supported_pred_template = pred_visible_template.gather(1, gather_indices)
            pred_template_to_source = self._attr_recon_directional_loss(
                supported_pred_template,
                support_mask,
                source,
                source_mask,
                beta=beta,
                distance_scale=distance_scale,
            )
            return source_to_pred_template, pred_template_to_source

        (
            translation_source_to_template,
            translation_template_to_source,
        ) = directional_terms(translation_template)
        (
            rotation_source_to_template,
            rotation_template_to_source,
        ) = directional_terms(rotation_template)
        has_reverse_support = support_mask.any(dim=-1)

        per_translation = 0.5 * (
            translation_source_to_template + translation_template_to_source
        )
        per_translation = torch.where(
            is_partial & ~has_reverse_support,
            translation_source_to_template,
            per_translation,
        )
        if self.recon_partial_mode == "source_only":
            per_translation = torch.where(
                is_partial,
                translation_source_to_template,
                per_translation,
            )

        if rotation_direction == "source_to_template":
            per_rotation = rotation_source_to_template
        elif rotation_direction == "template_to_source":
            per_rotation = rotation_template_to_source
            # Preserve the existing robust partial behavior: a partial crop
            # with no true reverse support falls back to source->template.
            per_rotation = torch.where(
                is_partial & ~has_reverse_support,
                rotation_source_to_template,
                per_rotation,
            )
        else:
            per_rotation = 0.5 * (
                rotation_source_to_template + rotation_template_to_source
            )
            per_rotation = torch.where(
                is_partial & ~has_reverse_support,
                rotation_source_to_template,
                per_rotation,
            )
        if self.recon_partial_mode == "source_only":
            per_rotation = torch.where(
                is_partial,
                rotation_source_to_template,
                per_rotation,
            )

        if use_preclip_full_source:
            weights = torch.ones_like(clip_length_ratio)
        else:
            weights = clip_length_ratio.clamp(0.0, 1.0).pow(
                self.recon_partial_weight_power
            )
            weights = torch.where(is_partial, weights, torch.ones_like(weights))
        (
            loss_translation,
            loss_translation_full,
            loss_translation_partial,
            full_effective_count,
            partial_effective_count,
        ) = self._attr_recon_partitioned_mean(
            per_translation,
            is_partial,
            weights,
            partial_multiplier,
            getattr(self, "recon_instance_reduction", "instance_mean"),
        )

        rotation_valid = torch.ones_like(is_partial, dtype=torch.bool)
        if rotation_gate_deg >= 0.0:
            with torch.no_grad():
                circular_error = (
                    pred_rotation.detach() - rotation.detach() + 180.0
                ).remainder(360.0) - 180.0
                rotation_valid = circular_error.abs() <= rotation_gate_deg
        (
            loss_rotation,
            loss_rotation_full,
            loss_rotation_partial,
            rotation_full_effective_count,
            rotation_partial_effective_count,
        ) = self._attr_recon_partitioned_masked_mean(
            per_rotation,
            is_partial,
            weights,
            partial_multiplier,
            rotation_valid,
            getattr(self, "recon_instance_reduction", "instance_mean"),
        )
        loss = loss_translation + rotation_weight * loss_rotation
        loss_full = loss_translation_full + rotation_weight * loss_rotation_full
        loss_partial = (
            loss_translation_partial + rotation_weight * loss_rotation_partial
        )
        rotation_effective_count = (
            rotation_full_effective_count + rotation_partial_effective_count
        )
        return loss, {
            "candidate": candidate_count,
            "geometry_valid": int(geometry_valid.sum().detach()),
            "partial_valid": int(is_partial.sum().detach()),
            "full_valid": int((~is_partial).sum().detach()),
            "loss_full": loss_full,
            "loss_partial": loss_partial,
            "full_effective_count": full_effective_count,
            "partial_effective_count": partial_effective_count,
            "loss_translation": loss_translation,
            "loss_rotation": loss_rotation,
            "loss_translation_full": loss_translation_full,
            "loss_translation_partial": loss_translation_partial,
            "loss_rotation_full": loss_rotation_full,
            "loss_rotation_partial": loss_rotation_partial,
            "rotation_valid": int(rotation_valid.sum().detach()),
            "rotation_full_valid": int((rotation_valid & ~is_partial).sum().detach()),
            "rotation_partial_valid": int((rotation_valid & is_partial).sum().detach()),
            "rotation_effective_count": rotation_effective_count,
            "rotation_full_effective_count": (rotation_full_effective_count),
            "rotation_partial_effective_count": (rotation_partial_effective_count),
        }

    def compute_chamfer_distance_with_padding(
        self,
        pred_coords_padded: Float["batch max_points 2"],
        gt_coords_padded: Float["batch max_points 2"],
        valid_lengths,
        single_directional: bool = False,
    ):
        """计算支持padding的chamfer distance，进一步优化版本"""
        batch_size, max_points, _ = pred_coords_padded.shape
        device = pred_coords_padded.device

        # 检查是否可以完全批量处理（所有primitives点数相同）
        if len(set(valid_lengths)) == 1:
            # 所有primitives点数相同，可以直接批量计算
            length = valid_lengths[0]
            pred_coords = pred_coords_padded[:, :length, :]  # [batch, length, 2]
            gt_coords = gt_coords_padded[:, :length, :]  # [batch, length, 2]

            # 直接批量计算chamfer distance
            chamfer_loss, _ = chamfer_distance(
                gt_coords, pred_coords, single_directional=single_directional
            )
            return chamfer_loss  # 已经是[batch]形状

        else:
            # 不同点数，需要分别计算（保持原有逻辑）
            chamfer_losses = []

            for i in range(batch_size):
                length = valid_lengths[i]
                pred_coords = pred_coords_padded[i, :length].unsqueeze(
                    0
                )  # [1, length, 2]
                # gt_coords = gt_coords_padded[i, :length].unsqueeze(0)      # [1, length, 2]
                gt_coords = gt_coords_padded[i].unsqueeze(0)  # [1, length, 2]
                chamfer_loss, _ = chamfer_distance(
                    gt_coords, pred_coords, single_directional=single_directional
                )
                chamfer_losses.append(chamfer_loss)

            return torch.stack(chamfer_losses)

    @typecheck
    def compute_recon_loss(
        self,
        translation_logits: Float["b np nd"],
        rotation_logits: Float["b np nd"],
        scale_logits: Float["b np nd"],
        gt_primitive_id: Int["b np"],
        gt_translation: Float["b np nd"],
        gt_rotation: Float["b np nd"],
        gt_scale: Float["b np nd"],
        gt_flip: Int["b np"],
        primitive_mask: Bool["b np"],
        tau: float = 1.0,
        pc_coord: Tensor | None = None,
        pc_offset: Tensor | None = None,
        pc_primitive_index: Tensor | None = None,
        gt_primitive_coords: Tensor | None = None,
        pred_gumbel_cont_translation: Tensor | None = None,
        pred_gumbel_cont_rotation: Tensor | None = None,
        pred_gumbel_cont_scale: Tensor | None = None,
        target_cont_translation: Tensor | None = None,
        target_cont_rotation: Tensor | None = None,
        target_cont_scale: Tensor | None = None,
        pred_flip: Tensor | None = None,
    ):
        device = gt_primitive_id.device

        #### First get the input primitive points as GT
        # bs = pc_offset.shape[0]
        # pc_start_idx = 0
        # gt_primitive_coords = []
        # #### primitive_dataset -> 预先计算gt_primitive_coords
        # #### 这一步感觉也可以预先计算
        # for b_i in range(bs):
        #     pc_coord_i = pc_coord[pc_start_idx:pc_offset[b_i]]
        #     ##### get the primitive coord
        #     pc_primitive_index_i = pc_primitive_index[b_i]
        #     primitive_coords_per_batch = []
        #     primitive_num_i = len(pc_primitive_index_i)
        #     for p_i in range(primitive_num_i):
        #         pc_primitive_coord_i = pc_coord_i[pc_primitive_index_i[p_i, 0]:pc_primitive_index_i[p_i, 1]][:, :2]
        #         assert pc_primitive_coord_i.shape[0] >= 1
        #         primitive_coords_per_batch.append(pc_primitive_coord_i)
        #     gt_primitive_coords.extend(primitive_coords_per_batch)
        #     pc_start_idx = pc_offset[b_i]

        primitive_ids = gt_primitive_id[primitive_mask]
        if len(primitive_ids) == 0:
            return torch.tensor(0.0, device=device)

        # 超级优化：批量加载primitive坐标，最小化IO开销
        all_primitive_coords = []
        primitive_id_list = (
            primitive_ids.cpu().numpy()
            if primitive_ids.is_cuda
            else primitive_ids.numpy()
        )

        # 策略1：全局缓存（训练期间一次性加载所有primitive）
        if not hasattr(self, "_primitive_coords_cache"):
            self._primitive_coords_cache = {}

        # 策略2：智能批量加载 - 只加载当前batch需要且未缓存的
        unique_ids = list(set(int(pid) for pid in primitive_id_list))
        uncached_ids = [
            pid for pid in unique_ids if pid not in self._primitive_coords_cache
        ]

        # 策略3：加载未缓存的坐标
        if uncached_ids:
            # 批量加载未缓存的坐标
            for primitive_id in uncached_ids:
                try:
                    primitive_coords = self.load_primitive_coords(
                        primitive_id
                    )  # [n, 3]
                    primitive_coords_2d = primitive_coords[:, :2].contiguous()  # [n, 2]
                    self._primitive_coords_cache[primitive_id] = primitive_coords_2d
                except Exception as e:
                    print(f"Warning: Failed to load primitive {primitive_id}: {e}")
                    # 使用默认占位符避免中断训练
                    self._primitive_coords_cache[primitive_id] = torch.zeros(
                        (10, 2), dtype=torch.float32
                    )

        # 策略4：向量化提取，避免循环
        all_primitive_coords = [
            self._primitive_coords_cache[int(pid)] for pid in primitive_id_list
        ]

        # assert len(all_primitive_coords) == len(gt_primitive_coords)
        # for i in range(len(all_primitive_coords)):
        #     assert all_primitive_coords[i].shape[0] == gt_primitive_coords[i].shape[0]

        # 使用padding批量变换 - 预测值
        pred_transformed_coords_padded, pred_valid_lengths = (
            self.transform_primitive_points_batch_padded(
                all_primitive_coords,
                scale=pred_gumbel_cont_scale,  # [batch, 1]
                rotation=pred_gumbel_cont_rotation,  # [batch, 1]
                translation=pred_gumbel_cont_translation,  # [batch, 2]
                flip=pred_flip,
            )
        )  # [batch, max_points, 2], List[int]

        gt_transformed_coords_padded, gt_valid_lengths = (
            self.transform_primitive_points_batch_padded(
                all_primitive_coords,
                scale=target_cont_scale,  # [batch, 2]
                rotation=target_cont_rotation,  # [batch, 1]
                translation=target_cont_translation,  # [batch, 2]
                flip=gt_flip,
            )
        )  # [batch, max_points, 2], List[int]

        # # 使用padding批量变换 - GT值
        # with torch.no_grad():
        #     gt_transformed_coords_padded, _ = self.transform_primitive_points_batch_padded(
        #         all_primitive_coords,
        #         scale=gt_cont_scale,          # [batch, 2]
        #         rotation=gt_cont_rotation,    # [batch, 1]
        #         translation=gt_cont_translation  # [batch, 2]
        #     )  # [batch, max_points, 2], List[int]

        if self.direct_recon:
            ##### 直接计算pair-wise的distance
            ##### debug for recon loss ######
            # gt_transformed_coords_padded, valid_lengths = self.transform_primitive_points_batch_padded(
            #     all_primitive_coords,
            #     scale=target_cont_scale,     # [batch, 1]
            #     rotation=target_cont_rotation,   # [batch, 1]
            #     translation=target_cont_translation,  # [batch, 2]
            #     # flip=None,
            #     # flip=gt_flip[primitive_mask]
            # )  # [batch, max_points, 2], List[int]

            total_recon_loss = self.compute_direct_recon_loss(
                pred_transformed_coords_padded,
                gt_transformed_coords_padded,
                pred_valid_lengths,
                gt_valid_lengths,
                visualize=False,
            )

        else:
            # 批量计算chamfer distance（考虑padding）
            chamfer_losses = self.compute_chamfer_distance_with_padding(
                pred_transformed_coords_padded,
                gt_primitive_coords,
                valid_lengths,
                single_directional=False,
            )  # [batch]

            total_recon_loss = chamfer_losses.mean()

        # 可选：添加调试信息（训练时可以注释掉）
        # if self.training and hasattr(self, '_recon_stats_counter'):
        #     self._recon_stats_counter = getattr(self, '_recon_stats_counter', 0) + 1
        #     if self._recon_stats_counter % 100 == 0:  # 每100次打印一次
        #         unique_lengths = len(set(valid_lengths))
        #         max_length = max(valid_lengths)
        #         min_length = min(valid_lengths)
        #         avg_length = sum(valid_lengths) / len(valid_lengths)
        #         print(f"Recon loss stats - Batch size: {len(valid_lengths)}, "
        #               f"Unique lengths: {unique_lengths}, Range: [{min_length}, {max_length}], "
        #               f"Avg: {avg_length:.1f}, Padding efficiency: {avg_length/max_length:.2%}")

        return total_recon_loss

    def encode_pc(self, input_dict, return_mapping=False):
        with torch.autocast("cuda", enabled=False):
            (
                pc_embed,
                pc_coord,
                pc_embed_key_padding_mask,
                dense_feat,
                dense_coord,
                dense_feat_key_padding_mask,
                tgt_point_mapping,
                downsampled_instance_masks,
            ) = self.conditioner(input_dict, return_mapping=return_mapping)
        pc_embed = self.to_cond_dim(pc_embed)
        if dense_feat is not None:
            mask_features = self.mask_features_head(dense_feat)
        else:
            mask_features = None
        return (
            pc_embed,
            pc_coord,
            pc_embed_key_padding_mask,
            mask_features,
            dense_coord,
            dense_feat_key_padding_mask,
            tgt_point_mapping,
            downsampled_instance_masks,
        )

    def encode_primitive_lib(self, input_dict):
        #### primitive_lib可能会出现空
        with torch.inference_mode():
            with eval_no_grad(self.conditioner):
                with torch.autocast("cuda", enabled=False):
                    (
                        primitive_lib_embed,
                        primitive_lib_coord,
                        primitive_lib_embed_key_padding_mask,
                        dense_feat,
                        dense_coord,
                        dense_feat_key_padding_mask,
                        tgt_point_mapping,
                        _,
                    ) = self.conditioner(input_dict, is_primitive_lib=True)
        primitive_lib_embed = self.to_cond_dim_primitive_lib(primitive_lib_embed)
        return (
            primitive_lib_embed,
            primitive_lib_coord,
            primitive_lib_embed_key_padding_mask,
        )

    @torch.no_grad()
    def _vec_grid_sample_tensor(
        self, coords, feats, layer_ids, cu_seqlens, prim_ids=None, grid_size=0.01
    ):
        """
        对 primitive_lib 的 tensor 数据执行网格采样，逻辑与 VecGridSample 一致。
        针对每条 primitive（由 cu_seqlens 分段），以线段中点作为代表坐标进行
        2D 网格量化，每个格子保留第一个点（确定性 test 模式），从而减少点数。

        Args:
            coords     (Tensor): [N, 4]  每条线段 (x1,y1,x2,y2)，已归一化
            feats      (Tensor): [N, D]  点级特征
            layer_ids  (Tensor): [N]     层 ID
            cu_seqlens (Tensor): [M+1]   各 primitive 的累计偏移
            prim_ids   (Tensor): [N]     点级 primitive ID（可选，同步过滤）
            grid_size  (float):          网格大小（与输入数据归一化空间一致，默认 0.01）

        Returns:
            coords_out     (Tensor): [N', 4]
            feats_out      (Tensor): [N', D]
            layer_ids_out  (Tensor): [N']
            cu_seqlens_out (Tensor): [M+1]
            prim_ids_out   (Tensor): [N'] 或 None
        """
        device = coords.device
        M = cu_seqlens.shape[0] - 1  # number of primitives

        coords_list = []
        feats_list = []
        layer_ids_list = []
        prim_ids_list = [] if prim_ids is not None else None
        new_cu = [0]

        for i in range(M):
            s = int(cu_seqlens[i].item())
            e = int(cu_seqlens[i + 1].item())
            if s == e:
                new_cu.append(new_cu[-1])
                continue

            c = coords[
                s:e
            ]  # [n, 2] (已经是中心点，经过 to_vec_data) 或 [n, 4] (原始线段)
            f = feats[s:e]  # [n, D]
            l = layer_ids[s:e]  # [n]

            # 计算代表坐标：primitive_lib 经过 to_vec_data 后 coords 已是中心点 [n, 2]，直接使用；
            # 若传入原始线段坐标 [n, 4] 则取中点
            if c.shape[1] == 2:
                mid = c
            else:
                mid = (c[:, :2] + c[:, 2:4]) * 0.5

            # 网格量化
            grid_coord = torch.floor(mid / grid_size).long()  # [n, 2]
            grid_coord = grid_coord - grid_coord.min(dim=0).values  # 平移到从 0 开始

            # 将 2D 网格坐标映射为唯一整数 key
            max_g = int(grid_coord.max().item()) + 1
            key = grid_coord[:, 0] * max_g + grid_coord[:, 1]  # [n]

            # 按 key 排序，取每个 cell 的第一个点（确定性 test 模式）
            idx_sort = torch.argsort(key, stable=True)
            key_sorted = key[idx_sort]
            mask = torch.ones(key_sorted.shape[0], dtype=torch.bool, device=device)
            mask[1:] = key_sorted[1:] != key_sorted[:-1]
            idx_unique = idx_sort[mask]  # 每个 cell 的代表点索引

            coords_list.append(c[idx_unique])
            feats_list.append(f[idx_unique])
            layer_ids_list.append(l[idx_unique])
            if prim_ids is not None:
                prim_ids_list.append(prim_ids[s:e][idx_unique])
            new_cu.append(new_cu[-1] + idx_unique.shape[0])

        coords_out = (
            torch.cat(coords_list, dim=0)
            if coords_list
            else coords.new_empty(0, coords.shape[1])
        )
        feats_out = (
            torch.cat(feats_list, dim=0)
            if feats_list
            else feats.new_empty(0, feats.shape[1])
        )
        layer_ids_out = (
            torch.cat(layer_ids_list, dim=0)
            if layer_ids_list
            else layer_ids.new_empty(0)
        )
        cu_seqlens_out = torch.tensor(new_cu, dtype=cu_seqlens.dtype, device=device)
        prim_ids_out = torch.cat(prim_ids_list, dim=0) if prim_ids_list else None

        return coords_out, feats_out, layer_ids_out, cu_seqlens_out, prim_ids_out

    @torch.no_grad()
    def _get_data_dict(
        self,
        coords,
        feats,
        cu_seqlens,
        grid_size=0.01,
        prim_ids=None,
        layer_ids=None,
        sem_ids=None,
        cu_numprims=None,
        sample_mode="line",
    ):
        # ─── no_color_layer ablation gate ───────────────────────────────────
        # Defensive zeroing of color channels in `feats`. In the current 7-channel
        # pipeline (`feats[..., :7]` slice in to_vec_data), there are no color
        # columns to zero, so this branch is a no-op. If a future change keeps
        # the full 10-channel feats `[length, dx, dy, cx, cy, pcx, pcy, c_r, c_g, c_b]`,
        # this will mask out indices 7-9.
        if getattr(self, "no_color_layer", False) and feats.shape[-1] > 7:
            feats = feats.clone()
            feats[..., 7:] = 0.0
        # ────────────────────────────────────────────────────────────────────
        if sample_mode == "point":
            z_coords = torch.zeros(
                (coords.shape[0], 1), dtype=coords.dtype, device=coords.device
            )
            coords = torch.cat([coords, z_coords], dim=-1)
        elif sample_mode == "line":
            layer_ids_feats = torch.zeros(
                (coords.shape[0], 1), dtype=feats.dtype, device=feats.device
            )
            # Skip the per-batch normalized fill when no_color_layer=True →
            # the 3rd coord channel stays an all-zero column.
            if not getattr(self, "no_color_layer", False):
                for batch_idx in range(len(cu_seqlens) - 1):
                    eps = torch.finfo(torch.float32).eps
                    idx_start, idx_end = (
                        cu_seqlens[batch_idx],
                        cu_seqlens[batch_idx + 1],
                    )
                    layer_ids_batch = layer_ids[idx_start:idx_end]  # type: ignore
                    layer_id_min = layer_ids_batch.min()
                    layer_id_max = layer_ids_batch.max()
                    layer_ids_feats[idx_start:idx_end, 0] = (
                        layer_ids_batch - layer_id_min
                    ) / (layer_id_max - layer_id_min + eps) - 0.5
            coords = torch.cat([coords, layer_ids_feats], dim=-1)
        else:
            raise ValueError(f"Invalid sample mode: {sample_mode}")
        return dict(
            feat=feats,
            coord=coords,
            grid_size=grid_size,
            offset=cu_seqlens[1:],
            sem_ids=sem_ids,
            cu_numprims=cu_numprims,
        )

    def offset_to_batch(self, feat, coord, offset):
        """
        将连续的特征tensor根据offset转换为batch形式并进行padding

        Args:
            feat (torch.Tensor): 特征tensor，形状为[N, feature_dim]
            offset (torch.Tensor): batch offset信息，形状为[bs]，表示每个sample的点数

        Returns:
            torch.Tensor: 转换后的特征tensor，形状为[bs, seq_len, feature_dim]
        """
        bs = offset.shape[0] - 1
        feature_dim = feat.shape[1]

        # 计算每个sample的实际长度
        lengths = offset[1:] - offset[:-1]
        # if bs > 1:

        # 找到最大序列长度
        seq_len = lengths.max().item()

        # 创建输出tensor
        batch_feat = torch.zeros(
            bs, seq_len, feature_dim, dtype=feat.dtype, device=feat.device
        )
        batch_coord = torch.zeros(
            bs, seq_len, 2, dtype=coord.dtype, device=coord.device
        )
        batch_feat_key_padding_mask = torch.zeros(
            bs, seq_len, dtype=torch.bool, device=feat.device
        )

        # 为每个sample填充特征
        start_idx = 0
        for i in range(bs):
            end_idx = start_idx + lengths[i].item()
            actual_len = lengths[i].item()
            # 填充实际的特征，其余位置保持为0
            batch_feat[i, :actual_len] = feat[start_idx:end_idx]
            batch_coord[i, :actual_len] = coord[start_idx:end_idx]
            batch_feat_key_padding_mask[i, :actual_len] = True
            start_idx = end_idx

        return batch_feat, batch_coord, batch_feat_key_padding_mask

    def forward_on_codes(
        self,
        *,
        scale: Float["b np 3"],
        rotation: Float["b np 3"],
        translation: Float["b np 3"],
        # type_code: Int['b np'],
        semantic_id: Int["b np"],
        primitive_id: Int["b np"],
        flip_mode: Int["b np"],
        size: Float["b np 2"] | None = None,
        loss_reduction: str = "mean",
        return_cache=False,
        return_loss=True,
        append_eos=False,
        cache: LayerIntermediates | None = None,
        pc: Tensor | None = None,
        pc_embed: Tensor | None = None,
        coord: Tensor | None = None,
        feat: Tensor | None = None,
        offset: Tensor | None = None,
        grid_coord: Tensor | None = None,
        img: Tensor | None = None,
        is_train: bool = True,
        **kwargs,
    ):

        # if self.bg_pred:
        #     # 准备 primitive 级背景分类目标 - 移到 compute_primitive_bg_loss 内部
        #     # 使用 pc_cu_seqlens 而不是 vec_data['cu_numprims']，确保维度一致
        #     pass

        #### Input of the token might be pad with pad_id with someone define lengths
        # primitive_mask = reduce(scale != self.pad_id, 'b np 3 -> b np', 'all')
        # primitive_mask = reduce(type_code != self.pad_id, 'b np -> b np', 'all')\
        batch = primitive_id.shape[0]

        primitive_mask = reduce(primitive_id != self.pad_id, "b np -> b np", "all")

        #### Encode the primitive lib
        primitive_lib_embed = None

        if self.primitive_dir is not None:
            # primitive_lib_dict = {
            #     'coord': self.primitive_lib_coords,
            #     'feat': self.primitive_lib_feats,
            #     'offset': self.primitive_lib_offset,
            #     'grid_coord': self.primitive_lib_grid_coords
            # }

            # 对 primitive_lib 执行与输入数据相同的 VecGridSample 网格采样，减少编码显存开销
            plib_coords, plib_feats, plib_layer_ids, plib_cu_seqlens, plib_prim_ids = (
                self._vec_grid_sample_tensor(
                    self.primitive_lib_coords,
                    self.primitive_lib_feats,
                    self.primitive_lib_layer_ids,
                    self.primitive_lib_cu_seqlens,
                    prim_ids=self.primitive_lib_prim_ids,
                    grid_size=self.conditioner_grid_size,
                )
            )
            primitive_lib_dict = self._get_data_dict(
                plib_coords,
                plib_feats,
                plib_cu_seqlens,
                grid_size=self.conditioner_grid_size,
                layer_ids=plib_layer_ids,
                sample_mode="line",
            )
            # ── Lib forward path: 3 modes ──
            # (1) cond_grad_on_lib=True  → hybrid (BN eval + grad on, no cache)
            # (2) cache_lib_feats=True  → eval_no_grad once, cached afterwards
            # (3) baseline default      → eval_no_grad each step (current main)
            if self.cond_grad_on_lib:
                with bn_eval_only(self.conditioner):
                    with torch.autocast("cuda", enabled=False):
                        (
                            primitive_lib_feats,
                            primitive_lib_coords,
                            primitive_lib_cu_seqlens,
                            _,
                            _,
                            _,
                        ) = self.conditioner(
                            primitive_lib_dict, plib_cu_seqlens, plib_prim_ids
                        )
                (
                    primitive_lib_feats,
                    primitive_lib_coords,
                    primitive_lib_embed_key_padding_mask,
                ) = self.offset_to_batch(
                    primitive_lib_feats, primitive_lib_coords, primitive_lib_cu_seqlens
                )
            elif self.cache_lib_feats and self._lib_feats_cache_valid:
                # Reuse cached (post offset_to_batch) feats / coords / mask.
                primitive_lib_feats = self._lib_feats_cache
                primitive_lib_coords = self._lib_coords_cache
                primitive_lib_embed_key_padding_mask = self._lib_mask_cache
            else:
                with eval_no_grad(self.conditioner):
                    with torch.autocast("cuda", enabled=False):
                        (
                            primitive_lib_feats,
                            primitive_lib_coords,
                            primitive_lib_cu_seqlens,
                            _,
                            _,
                            _,
                        ) = self.conditioner(
                            primitive_lib_dict, plib_cu_seqlens, plib_prim_ids
                        )
                (
                    primitive_lib_feats,
                    primitive_lib_coords,
                    primitive_lib_embed_key_padding_mask,
                ) = self.offset_to_batch(
                    primitive_lib_feats, primitive_lib_coords, primitive_lib_cu_seqlens
                )
                if self.cache_lib_feats:
                    # Stash post-offset_to_batch tensors so subsequent steps skip the conditioner.
                    self._lib_feats_cache = primitive_lib_feats.detach()
                    self._lib_coords_cache = primitive_lib_coords.detach()
                    self._lib_mask_cache = primitive_lib_embed_key_padding_mask.detach()
                    self._lib_feats_cache_valid = True
            ### 预处理这些信息 ###
            primitive_lib_embed = self.to_cond_dim_primitive_lib(primitive_lib_feats)
            ##### 输入backbone前先进行预处理 #####
            # primitive_lib_embed, primitive_lib_coord, primitive_lib_embed_key_padding_mask = self.encode_primitive_lib(primitive_lib_dict)
            # ── Pool per-primitive tokens [num_prims, N_pts, dim] → [num_prims, dim] ──
            # All branches output the *unbatched* form. Line 3734 below applies
            # `.unsqueeze(0).repeat(batch,1,1)` for the prefix/contrastive paths.
            # The V2/V3 per-instance gather (in encode_sem_geom) consumes the
            # unbatched form directly.
            #
            # primitive_lib_embed: [num_prims, N_pts, dim]
            # primitive_lib_embed_key_padding_mask: [num_prims, N_pts]  (True = valid)
            valid = primitive_lib_embed_key_padding_mask.float()  # [P, N]
            lengths = valid.sum(dim=1, keepdim=True).clamp(min=1.0)  # [P, 1]
            if self.pooling_type == "mean":
                masked_embed = primitive_lib_embed * valid.unsqueeze(-1)
                primitive_lib_embed = masked_embed.sum(dim=1) / lengths  # [P, dim]
            elif self.pooling_type == "max":
                # Mask-aware max: invalid positions → -inf so they never win.
                masked = primitive_lib_embed.masked_fill(
                    ~primitive_lib_embed_key_padding_mask.unsqueeze(-1), float("-inf")
                )
                primitive_lib_embed = masked.max(dim=1).values  # [P, dim]
            elif self.pooling_type == "mean_max":
                # Simple addition (no extra params).
                masked_for_mean = primitive_lib_embed * valid.unsqueeze(-1)
                pooled_mean = masked_for_mean.sum(dim=1) / lengths
                masked_for_max = primitive_lib_embed.masked_fill(
                    ~primitive_lib_embed_key_padding_mask.unsqueeze(-1), float("-inf")
                )
                pooled_max = masked_for_max.max(dim=1).values
                primitive_lib_embed = pooled_mean + pooled_max  # [P, dim]
            elif self.pooling_type == "attn":
                # AttentionPooling.forward(x, mask=) expects mask=(B,N) True=padding,
                # so invert the validity mask we already have.
                pooled, _ = self.attention_pooling(
                    primitive_lib_embed,
                    mask=~primitive_lib_embed_key_padding_mask,
                )
                primitive_lib_embed = pooled.squeeze(1)  # [P, dim]
            elif self.pooling_type == "pma":
                primitive_lib_embed, _ = self.pma_pooling(
                    primitive_lib_embed, mask=~primitive_lib_embed_key_padding_mask
                )
                primitive_lib_embed += self.slot_embed(
                    torch.arange(self.n_seeds, device=self.device)
                    .view(1, self.n_seeds)
                    .expand(primitive_lib_embed.shape[1], self.n_seeds)
                )
                primitive_lib_embed = primitive_lib_embed.reshape(
                    primitive_lib_embed.shape[0] * self.n_seeds,
                    primitive_lib_embed.shape[-1],
                )
            else:
                raise ValueError(f"unknown pooling_type={self.pooling_type!r}")

        ### Encode Existing generate primitives through Primitive Encoder
        # Attribute-decoupled training must support a valid "noise-only"
        # sample: the conditioner still sees scene primitives, while every
        # foreground instance has been filtered from the AR target.  Running
        # encode_attr on an empty instance axis produces correctly shaped
        # empty GT tensors; the decoder then receives only SOS and is trained
        # to emit EOS immediately.  The legacy encoders are left on their
        # historical non-empty-only path.
        if scale.shape[1] > 0 or self.attr_decouple:
            if self.sem_geom_decouple or self.geom_sem_decouple:
                sem_embed, geom_embed, primitives_embeds, primitives_gt = (
                    self.encode_sem_geom(
                        scale=scale,
                        rotation=rotation,
                        translation=translation,
                        semantic_id=semantic_id,
                        primitive_id=primitive_id,
                        flip_mode=flip_mode,
                        primitive_mask=primitive_mask,
                        return_primitives=True,
                        primitive_geom_embed=primitive_lib_embed,
                        size=size,
                    )
                )
                #### 将上述的sem_embed和geom_embed组合成code的形式 ####
                """
                    进行交错合并: [sem_embed_0, geom_embed_0, sem_embed_1, geom_embed_1, ...]
                    1. 需要一个类型embedding来区分是sem_embed还是geom_embed
                    2. 还需要一个instance embedding来区分是哪个instance
                """
                if self.pretrain:
                    batch_size, num_primitives, dim = geom_embed.shape
                    codes = torch.zeros(
                        batch_size,
                        num_primitives,
                        dim,
                        device=self.device,
                        dtype=geom_embed.dtype,
                    )
                    codes = geom_embed
                    #### 添加embedding ####
                    attr_type = torch.zeros(
                        batch_size, num_primitives, dtype=torch.long, device=self.device
                    )
                    attr_instance_index = torch.arange(
                        num_primitives, device=self.device, dtype=torch.long
                    )
                    attr_instance_index = attr_instance_index.unsqueeze(0).repeat(
                        batch_size, 1
                    )

                    attr_type_embed = self.attr_type_embed(attr_type)
                    attr_instance_embed = self.attr_instance_embed(attr_instance_index)
                    codes = codes + attr_type_embed + attr_instance_embed
                else:
                    batch_size, num_primitives, dim = sem_embed.shape
                    codes = torch.zeros(
                        batch_size,
                        num_primitives * 2,
                        dim,
                        device=self.device,
                        dtype=sem_embed.dtype,
                    )
                    if self.sem_geom_decouple:
                        codes[:, 0::2, :] = sem_embed
                        codes[:, 1::2, :] = geom_embed
                    elif self.geom_sem_decouple:
                        codes[:, 0::2, :] = geom_embed
                        codes[:, 1::2, :] = sem_embed

                    #### 添加embedding ####
                    attr_type = torch.zeros(
                        batch_size,
                        2 * num_primitives,
                        dtype=torch.long,
                        device=self.device,
                    )
                    attr_type[:, 1::2] = (
                        1  # 偶数位置是 Semantic embedding, 奇数位置是 Geometry embedding
                    )
                    attr_instance_index = torch.arange(
                        num_primitives, device=self.device, dtype=torch.long
                    ).repeat_interleave(2)
                    attr_instance_index = attr_instance_index.unsqueeze(0).repeat(
                        batch_size, 1
                    )

                    attr_type_embed = self.attr_type_embed(attr_type)
                    attr_instance_embed = self.attr_instance_embed(attr_instance_index)
                    codes = codes + attr_type_embed + attr_instance_embed
                    """
                        sem_0, geom_0, sem_1, geom_1, ...
                    """
            elif self.attr_decouple:
                (
                    semantic_type_embed,
                    primitive_type_embed,
                    translation_embed,
                    flip_embed,
                    rotation_embed,
                    shape_embed,
                    primitives_embeds,
                    primitives_gt,
                ) = self.encode_attr(
                    scale=scale,
                    rotation=rotation,
                    translation=translation,
                    semantic_id=semantic_id,
                    primitive_id=primitive_id,
                    flip_mode=flip_mode,
                    primitive_mask=primitive_mask,
                    return_primitives=True,
                    primitive_geom_embed=primitive_lib_embed,
                    size=size,
                )
                batch_size, num_primitives, dim = semantic_type_embed.shape
                serialized_attr_embeds = []
                if self.attr_emit_semantic_token:
                    serialized_attr_embeds.append(semantic_type_embed)
                serialized_attr_embeds.extend(
                    [
                        primitive_type_embed,
                        translation_embed,
                        flip_embed,
                        rotation_embed,
                        shape_embed,
                    ]
                )
                if len(serialized_attr_embeds) != self.num_attr_tokens:
                    raise RuntimeError(
                        "attr token layout mismatch: "
                        f"{len(serialized_attr_embeds)} != {self.num_attr_tokens}"
                    )
                codes = torch.stack(serialized_attr_embeds, dim=2).reshape(
                    batch_size,
                    num_primitives * self.num_attr_tokens,
                    dim,
                )

                attr_type = torch.arange(
                    self.num_attr_tokens, device=self.device, dtype=torch.long
                ).repeat(batch_size, num_primitives)
                attr_instance_index = torch.arange(
                    num_primitives, device=self.device, dtype=torch.long
                ).repeat_interleave(self.num_attr_tokens)
                attr_instance_index = attr_instance_index.unsqueeze(0).repeat(
                    batch_size, 1
                )
                codes = (
                    codes
                    + self.attr_type_embed(attr_type)
                    + self.attr_instance_embed(attr_instance_index)
                )
                attr_token_mask = primitive_mask.repeat_interleave(
                    self.num_attr_tokens, dim=1
                )
                codes = codes.masked_fill(~attr_token_mask.unsqueeze(-1), 0.0)
            else:
                codes, primitives_embeds, primitives_gt = self.encode(
                    scale=scale,
                    rotation=rotation,
                    translation=translation,
                    # type_code=type_code,
                    semantic_id=semantic_id,
                    primitive_id=primitive_id,
                    flip_mode=flip_mode,
                    primitive_mask=primitive_mask,
                    return_primitives=True,
                    primitive_geom_embed=primitive_lib_embed,
                )
        else:
            codes = torch.empty(
                (scale.shape[0], 0, self.dim), dtype=torch.float32, device=self.device
            )
            primitives_embeds = None
            primitives_gt = None

        primitive_lib_embed = primitive_lib_embed.unsqueeze(0).repeat(batch, 1, 1)

        # handle shape conditions
        attn_context_kwargs = dict()
        if self.condition_on_shape:

            assert (
                exists(pc)
                ^ exists(pc_embed)
                ^ exists(coord)
                ^ exists(kwargs["vec_data"])
            ), "`pc` or `pc_embed` must be passed in"

            if exists(pc):
                if "michelangelo" in self.shape_condition_model_type:
                    pc_head, pc_embed = self.conditioner(shape=pc)
                    pc_embed = torch.cat(
                        [self.to_cond_dim_head(pc_head), self.to_cond_dim(pc_embed)],
                        dim=-2,
                    )
                else:
                    raise ValueError(
                        f"unknown shape_condition_model_type {self.shape_condition_model_type}"
                    )
            if exists(coord) or exists(kwargs["vec_data"]):
                if "point_transformer_v3" in self.shape_condition_model_type:
                    # data_dict = self._get_data_dict(kwargs['vec_data']['coords'], kwargs['vec_data']['feats'], kwargs['vec_data']['cu_seqlens'], grid_size=0.002, prim_ids=kwargs['vec_data']['prim_ids'], layer_ids=kwargs['vec_data']['layer_ids'], sem_ids=kwargs['vec_data']['sem_ids'], cu_numprims=kwargs['vec_data']['cu_numprims'], sample_mode="line")
                    data_dict = self._get_data_dict(
                        kwargs["vec_data"]["coords"],
                        kwargs["vec_data"]["feats"],
                        kwargs["vec_data"]["cu_seqlens"],
                        grid_size=self.conditioner_grid_size,
                        prim_ids=kwargs["vec_data"]["prim_ids"],
                        layer_ids=kwargs["vec_data"]["layer_ids"],
                        sem_ids=kwargs["vec_data"]["sem_ids"],
                        cu_numprims=kwargs["vec_data"]["cu_numprims"],
                        sample_mode="line",
                    )
                    with torch.autocast("cuda", enabled=False):
                        (
                            pc_feats,
                            pc_coords,
                            pc_cu_seqlens,
                            pooled_sem_ids,
                            prim_keep_mask,
                            prim_id_mapping,
                        ) = self.conditioner(
                            data_dict,
                            kwargs["vec_data"]["cu_seqlens"],
                            kwargs["vec_data"]["prim_ids"],
                        )
                    pc_feats, pc_coords, pc_embed_key_padding_mask = (
                        self.offset_to_batch(pc_feats, pc_coords, pc_cu_seqlens)
                    )
                    if self.shape_cond_with_perceiver_prefix:
                        # Perceiver Resampler: compress variable-length features → fixed K tokens
                        pc_embed = self.perceiver_resampler(
                            pc_feats, pc_embed_key_padding_mask
                        )  # [B, K, dim]
                        pc_embed = (
                            pc_embed + self.cond_prefix_pe.weight[None, :]
                        )  # add positional embedding
                        # pc_embed = self.to_cond_dim(pc_feats)
                    else:
                        pc_embed = self.to_cond_dim(pc_feats)
                    ### Pad the pc_embed into the same lengths
                    ### Add linear projection
            assert (
                pc_embed.shape[0] == codes.shape[0]
            ), "batch size of point cloud is not equal to the batch size of the primitive codes"

            # # 在您的代码第2777行之后添加
            # visualize_downsampled_instance_mask(
            #     pc_coord=pc_coord,
            #     downsampled_instance_masks=downsampled_instance_masks,
            #     save_path='./debug_vis/downsampled_visualization.png',
            #     batch_idx=0  # 可视化第一个样本
            # )

            pooled_pc_embed = pc_embed.mean(dim=1)  # (b, shape_condition_dim)

            # if self.bg_pred:
            #     mask_features = self.mask_features_head(pc_embed)
            # else:
            #     mask_features = None

            if self.shape_cond_with_cross_attn:
                attn_context_kwargs = dict(context=pc_embed)

            if self.coarse_adaptive_rmsnorm:
                attn_context_kwargs.update(condition=pooled_pc_embed)

        batch, seq_len, _ = codes.shape  # (b, np, dim)
        device = codes.device
        # assert seq_len <= self.max_seq_len, f'received codes of length {seq_len} but needs to be less than or equal to set max_seq_len {self.max_seq_len}'

        if append_eos:
            assert exists(codes)
            code_lens = primitive_mask.sum(dim=-1)
            codes = pad_tensor(codes)
            batch_arange = torch.arange(batch, device=device)
            batch_arange = rearrange(batch_arange, "... -> ... 1")
            code_lens = rearrange(code_lens, "... -> ... 1")
            codes[batch_arange, code_lens] = self.eos_token  # (b, np+1, dim)

        primitive_codes = codes  # (b, np, dim)

        primitive_codes_len = primitive_codes.shape[-2]

        (
            coarse_cache,
            coarse_gateloop_cache,
            coarse_post_gateloop_cache,
        ) = (
            cache if exists(cache) else ((None,) * 3)
        )

        if not exists(cache):
            sos = repeat(self.sos_token, "n d -> b n d", b=batch)
            len_sos = sos.shape[1]
            #### add bg implicit primitive
            if self.bg_pred:
                bg_implicit_primitives = self.bg_implicit_primitive.weight[
                    None, ...
                ].repeat(batch, 1, 1)
                len_bg_implicit_primitives = bg_implicit_primitives.shape[1]
                sos, _ = pack([bg_implicit_primitives, sos], "b * d")
            if self.primitive_cond_with_cat:
                # if self.use_dense_feat:
                # dense_features = dense_features + self.src_type_embed(torch.zeros((batch, 1), dtype=torch.long, device=self.device))
                # else:
                # pc_embed = pc_embed + self.src_type_embed(torch.zeros((batch, 1), dtype=torch.long, device=self.device))
                primitive_lib_embed = primitive_lib_embed + self.src_type_embed(
                    torch.ones((batch, 1), dtype=torch.long, device=self.device)
                )
            if self.add_pe and not self.shape_cond_with_perceiver_prefix:
                ## add positional encoding to the pc_embed (skipped in perceiver prefix mode, PE already added above)
                ## 先分batch考虑？
                pc_pos_embedding = pc_embed.new_zeros(
                    batch, pc_embed.shape[1], pc_embed.shape[2]
                )
                # pc_pos_embedding = self.pc_pe(pc_coords[:, :, :2], num_channels=pc_embed.shape[2], input_range=self.continuous_range_translation)
                # for b_i in range(batch):
                #     pc_mask_i = pc_embed_key_padding_mask[b_i]
                #     valid_pos = torch.where(pc_mask_i)[0]
                #     pc_coord_i = pc_coords[b_i][pc_mask_i]
                #     pc_pos_embedding_i = self.pc_pe(pc_coord_i[:, :2][None, ...], num_channels=pc_embed.shape[2])
                #     pc_pos_embedding[b_i][valid_pos] = pc_pos_embedding_i
                pc_pos_embedding = self.pc_pe(
                    pc_coords[:, :, :2], num_channels=pc_embed.shape[2]
                )
                pc_pos_embedding = pc_pos_embedding.masked_fill(
                    ~pc_embed_key_padding_mask.unsqueeze(-1), 0.0
                )
                pc_embed = pc_embed + pc_pos_embedding

            if self.shape_cond_with_cross_attn:
                attn_context_kwargs = dict(
                    context=pc_embed, context_mask=pc_embed_key_padding_mask
                )
            if self.primitive_cond_with_cat:
                len_primitive_lib_embed = primitive_lib_embed.shape[1]
                sos, _ = pack([primitive_lib_embed, sos], "b * d")
            if self.bg_fitting:
                if self.fg_fitting:
                    len_primitive_codes = primitive_codes.shape[1]
                    primitive_codes, packed_sos_shape = pack(
                        [sos, primitive_codes], "b * d"
                    )  # (b, n_sos+np, dim)
                else:
                    len_primitive_codes = 0
                    primitive_codes = sos
                # #### 只考虑对BG对fitting ####
                # len_primitive_codes = 0
                # primitive_codes = sos
            else:
                len_primitive_codes = primitive_codes.shape[1]
                primitive_codes, packed_sos_shape = pack(
                    [sos, primitive_codes], "b * d"
                )  # (b, n_sos+np, dim)
            ### Initialization of codes: pc_embed + sos_tokens + primitive_codes (Encoded by primitive encoder)

            if self.shape_cond_with_perceiver_prefix:
                # Prepend fixed-length perceiver prefix tokens to the front of the sequence.
                # Final layout: [perceiver_prefix (K) | primitive_lib_embed | (bg_implicit?) | sos | primitive_codes]
                len_pc_prefix = self.perceiver_num_latents
                primitive_codes, _ = pack([pc_embed, primitive_codes], "b * d")

        # ### 根据pc_embed_key_padding_mask来设置input_mask
        # input_mask = torch.ones((batch, primitive_codes.shape[1]), dtype=torch.bool, device=primitive_codes.device)
        # input_mask[:, len_primitive_lib_embed:len_primitive_lib_embed + len_pc_embed] = pc_embed_key_padding_mask

        if self.condition_on_shape:
            primitive_codes = self.shape_coarse_film_cond(
                primitive_codes, pooled_pc_embed
            )

        if exists(self.coarse_gateloop_block):
            primitive_codes, coarse_gateloop_cache = self.coarse_gateloop_block(
                primitive_codes, cache=coarse_gateloop_cache
            )

        if self.use_prefix_decoder:
            #### ADD the prefix-decoder
            #### prepare attn_mask, prefix_attn_len
            # assert self.shape_cond_with_cat and self.primitive_cond_with_cat
            if self.shape_cond_with_cross_attn:
                if self.bg_pred:
                    prefix_attn_len = (
                        len_primitive_lib_embed + len_bg_implicit_primitives
                    )
                else:
                    prefix_attn_len = len_primitive_lib_embed
            elif self.shape_cond_with_perceiver_prefix:
                # Perceiver prefix tokens are prepended; extend prefix_attn_len to cover them.
                base_prefix_len = 0
                if self.primitive_cond_with_cat:
                    base_prefix_len += len_primitive_lib_embed
                if self.bg_pred:
                    base_prefix_len += len_bg_implicit_primitives
                prefix_attn_len = len_pc_prefix + base_prefix_len
            # else:
            #     if self.bg_pred:
            #         prefix_attn_len = len_primitive_lib_embed + len_pc_embed + len_bg_implicit_primitives
            #     else:
            #         prefix_attn_len = len_pc_embed + len_primitive_lib_embed
            # prefix_attn_len = prefix_attn_len + 4
            if self.shape_cond_with_cross_attn:
                attended_primitive_codes, coarse_cache = (
                    self.decoder(  # (b, n_sos+np, dim)
                        primitive_codes,
                        cache=coarse_cache,
                        return_hiddens=True,
                        prefix_attn_len=prefix_attn_len,
                        **attn_context_kwargs,
                    )
                )
            elif self.shape_cond_with_perceiver_prefix:
                # Pure self-attention with fixed prefix; no cross-attn context needed.
                attended_primitive_codes, coarse_cache = (
                    self.decoder(  # (b, n_sos+np, dim)
                        primitive_codes,
                        cache=coarse_cache,
                        return_hiddens=True,
                        prefix_attn_len=prefix_attn_len,
                        **attn_context_kwargs,
                    )
                )
            else:
                #### prepare the mask for self-attention
                self_mask = torch.ones(
                    (batch, primitive_codes.shape[1]),
                    dtype=torch.bool,
                    device=primitive_codes.device,
                )
                if self.use_dense_feat:
                    self_mask[
                        :,
                        len_primitive_lib_embed : len_primitive_lib_embed
                        + len_pc_embed,
                    ] = dense_feat_key_padding_mask
                else:
                    self_mask[
                        :,
                        len_primitive_lib_embed : len_primitive_lib_embed
                        + len_pc_embed,
                    ] = pc_embed_key_padding_mask
                self_mask = self_mask.unsqueeze(1).unsqueeze(1)
                attended_primitive_codes, coarse_cache = (
                    self.decoder(  # (b, n_sos+np, dim)
                        primitive_codes,
                        cache=coarse_cache,
                        return_hiddens=True,
                        prefix_attn_len=prefix_attn_len,
                        attn_mask=self_mask,
                        **attn_context_kwargs,
                    )
                )
        else:
            attended_primitive_codes, coarse_cache = self.decoder(  # (b, n_sos+np, dim)
                primitive_codes,
                cache=coarse_cache,
                return_hiddens=True,
                **attn_context_kwargs,
            )

        if exists(self.coarse_post_gateloop_block):
            primitive_codes, coarse_post_gateloop_cache = (
                self.coarse_post_gateloop_block(
                    primitive_codes, cache=coarse_post_gateloop_cache
                )
            )

        if self.bg_fitting:
            if self.fg_fitting:
                embed = attended_primitive_codes[
                    :, -(primitive_codes_len + 1) :
                ]  # (b, np+1, dim)
            else:
                embed = None
        else:
            embed = attended_primitive_codes[
                :, -(primitive_codes_len + 1) :
            ]  # (b, np+1, dim)

        if self.bg_pred:
            # In perceiver prefix mode, prefix tokens are prepended, so offset by len_pc_prefix
            _bg_offset = (
                len_pc_prefix if self.shape_cond_with_perceiver_prefix else 0
            ) + len_primitive_lib_embed
            pred_bg_embed = attended_primitive_codes[
                :, _bg_offset : _bg_offset + len_bg_implicit_primitives
            ]
        else:
            pred_bg_embed = None

        if not return_loss:
            # 推理时也计算背景预测
            pred_primitive_bg = None
            if self.bg_pred and pred_bg_embed is not None:
                pc_embed_for_bg = self.pc_embed_to_bg_space(pc_embed)
                # 计算每个 primitive 与每个 bg query 的相似度
                # pc_embed_for_bg: (B, max_num_prims, dim)
                # pred_bg_embed: (B, num_bg_classes, dim)
                # 结果: (B, max_num_prims, num_bg_classes)
                pred_primitive_bg = torch.einsum(
                    "bpd,bcd->bpc", pc_embed_for_bg, pred_bg_embed
                )
                # 应用 sigmoid 获取概率
                pred_primitive_bg = torch.sigmoid(pred_primitive_bg)

            if not return_cache:
                return embed[:, -1:], pred_primitive_bg
            next_cache = (
                coarse_cache,
                coarse_gateloop_cache,
                coarse_post_gateloop_cache,
            )
            return embed[:, -1:], next_cache, pred_primitive_bg
        else:
            #### compute the mask loss
            # bg_mask_loss = torch.tensor(0.0, device=embed.device)
            bg_primitive_loss = torch.tensor(0.0, device=self.device)
            pred_primitive_bg = None  # 初始化
            pred_primitive_bg_prob = None
            gt_primitive_bg = None  # GT 背景类别（用于评估）
            bg_valid_mask = None  # 有效 primitive 的 mask
            if self.bg_pred:
                # bg_mask_loss, pred_masks = self.compute_bg_mask_loss(pred_bg_embed, dense_features, dense_feat_key_padding_mask, bg_targets)
                # 计算 primitive 级别的背景分类 loss
                # 将 pc_embed 映射到背景分类子空间
                pc_embed_for_bg = self.pc_embed_to_bg_space(pc_embed)

                # 根据 bg_matching 选择损失计算方法
                # 使用 pooled_sem_ids（primitive 级别），确保与 pc_cu_seqlens 一致
                if self.bg_matching is not None and self.bg_matching == "hungarian":
                    # bg_primitive_loss, pred_primitive_bg, bg_ce_loss = self.compute_primitive_bg_loss_hungarian(
                    #     pred_bg_embed, pc_embed_for_bg, pc_embed_key_padding_mask,
                    #     pooled_sem_ids, pc_cu_seqlens
                    # )
                    bg_primitive_loss, pred_primitive_bg, bg_ce_loss = (
                        self.compute_primitive_bg_loss_hungarian_vec(
                            pred_bg_embed,
                            pc_embed_for_bg,
                            pc_embed_key_padding_mask,
                            pooled_sem_ids,
                            pc_cu_seqlens,
                        )
                    )
                else:
                    # bg_primitive_loss, pred_primitive_bg = self.compute_primitive_bg_loss(
                    #     pred_bg_embed, pc_embed_for_bg, pc_embed_key_padding_mask,
                    #     pooled_sem_ids, pc_cu_seqlens
                    # )
                    bg_primitive_loss, pred_primitive_bg = (
                        self.compute_primitive_bg_loss_fast(
                            pred_bg_embed,
                            pc_embed_for_bg,
                            pc_embed_key_padding_mask,
                            pooled_sem_ids,
                            pc_cu_seqlens,
                        )
                    )
                    bg_ce_loss = 0
                if self.bg_fitting and not self.fg_fitting:
                    loss = 0.0
                    loss_semantic = torch.tensor(0.0, device=self.device)
                    loss_primitive = torch.tensor(0.0, device=self.device)
                    loss_translation = torch.tensor(0.0, device=self.device)
                    loss_rotation = torch.tensor(0.0, device=self.device)
                    loss_scale = torch.tensor(0.0, device=self.device)
                    loss_eos = torch.tensor(0.0, device=self.device)
                    loss_flip = torch.tensor(0.0, device=self.device)
                    loss_recon = torch.tensor(0.0, device=self.device)
                    loss_huber_translation = torch.tensor(0.0, device=self.device)
                    loss_huber_rotation = torch.tensor(0.0, device=self.device)
                    loss_huber_scale = torch.tensor(0.0, device=self.device)
                    loss_bg_mask = torch.tensor(0.0, device=self.device)
                    loss_inst_mask = torch.tensor(0.0, device=self.device)
                    loss_attn = torch.tensor(0.0, device=self.device)
                    loss_attn_constraint = torch.tensor(0.0, device=self.device)
                    loss += bg_primitive_loss
                    loss_dict = {
                        "loss_semantic": loss_semantic,
                        "loss_primitive": loss_primitive,
                        "loss_translation": loss_translation,
                        "loss_rotation": loss_rotation,
                        "loss_scale": loss_scale,
                        "loss_eos": loss_eos,
                        "loss_flip": loss_flip,
                        "loss_recon": loss_recon,
                        "loss_huber_translation": loss_huber_translation,
                        "loss_huber_rotation": loss_huber_rotation,
                        "loss_huber_scale": loss_huber_scale,
                        "loss_bg_mask": loss_bg_mask,
                        "loss_inst_mask": loss_inst_mask,
                        "loss_attn": loss_attn,
                        "loss_attn_constraint": loss_attn_constraint,
                        "loss_bg_primitive": bg_primitive_loss,
                        "loss": loss,
                        # 'loss_bg_ce': bg_ce_loss,
                    }
                    if is_train:
                        return loss, loss_dict
                    else:
                        pred_primitive_bg_prob = torch.sigmoid(pred_primitive_bg)

                        # 构建 GT 背景类别用于评估
                        # 使用 pooled_sem_ids（primitive 级别），与 pc_cu_seqlens 一致
                        batch_size = pred_primitive_bg.shape[0]
                        max_num_prims = pred_primitive_bg.shape[1]
                        num_bg_classes = pred_primitive_bg.shape[2]

                        # gt_primitive_bg: (B, max_num_prims) 存储每个 primitive 的 GT 背景类别，-1 表示非背景
                        gt_primitive_bg = torch.full(
                            (batch_size, max_num_prims),
                            -1,
                            dtype=torch.int32,
                            device=pred_primitive_bg.device,
                        )
                        # bg_valid_mask: (B, max_num_prims) 有效 primitive 的 mask
                        bg_valid_mask = pc_embed_key_padding_mask.clone()

                        for b_i in range(batch_size):
                            valid_mask = pc_embed_key_padding_mask[
                                b_i
                            ]  # (max_num_prims,)
                            num_valid = valid_mask.sum().item()

                            start_idx = pc_cu_seqlens[b_i].item()
                            end_idx = pc_cu_seqlens[b_i + 1].item()
                            sem_ids_bi = pooled_sem_ids[
                                start_idx:end_idx
                            ]  # (N_valid_prims,)

                            # 转换: semantic 30-34 → 0-4 (有效背景类), 其他 → -1 (非背景)
                            target_b = torch.full_like(
                                sem_ids_bi, -1, dtype=sem_ids_bi.dtype
                            )
                            bg_mask_target = (sem_ids_bi >= 30) & (sem_ids_bi <= 34)
                            target_b[bg_mask_target] = sem_ids_bi[bg_mask_target] - 30

                            # 将 target 填入 gt_primitive_bg 的有效位置
                            valid_indices = torch.where(valid_mask)[0][:num_valid]
                            gt_primitive_bg[b_i, valid_indices] = target_b
                        loss_dict = {
                            "loss_semantic": loss_semantic,
                            "loss_primitive": loss_primitive,
                            "loss_translation": loss_translation,
                            "loss_rotation": loss_rotation,
                            "loss_scale": loss_scale,
                            "loss_eos": loss_eos,
                            "loss_flip": loss_flip,
                            "loss_recon": loss_recon,
                            "loss_huber_translation": loss_huber_translation,
                            "loss_huber_rotation": loss_huber_rotation,
                            "loss_huber_scale": loss_huber_scale,
                            "loss_bg_mask": loss_bg_mask,
                            "loss_inst_mask": loss_inst_mask,
                            "loss_attn": loss_attn,
                            "loss_attn_constraint": loss_attn_constraint,
                            "loss_bg_primitive": bg_primitive_loss,
                            "loss": loss,
                            "pred_primitive_bg": pred_primitive_bg_prob,
                            "gt_primitive_bg": gt_primitive_bg,
                            "bg_valid_mask": bg_valid_mask,
                            # 用于可视化的 primitive-level 映射信息
                            "prim_keep_mask": prim_keep_mask,  # (P1+P2+...,) 原始哪些 primitive 被保留
                            "prim_id_mapping": prim_id_mapping,  # (P1+P2+...,) 原始 primitive ID → pooled primitive index
                            # 'loss_bg_ce': bg_ce_loss,
                        }
                        return loss_dict

            pred_primitive_embeds = embed[:, :-1]  # (b, np, dim)

            if self.sem_geom_decouple or self.geom_sem_decouple:
                loss = 0.0
                if primitives_embeds is None:
                    ### 表示此时只有背景元素 --- 只优化背景 ###
                    loss_semantic = torch.tensor(0.0, device=embed.device)
                    loss_primitive = torch.tensor(0.0, device=embed.device)
                    loss_translation = torch.tensor(0.0, device=embed.device)
                    loss_rotation = torch.tensor(0.0, device=embed.device)
                    loss_scale = torch.tensor(0.0, device=embed.device)
                    loss_eos = torch.tensor(0.0, device=embed.device)
                    loss_flip = torch.tensor(0.0, device=embed.device)
                    loss_recon = torch.tensor(0.0, device=embed.device)
                    loss_huber_translation = torch.tensor(0.0, device=embed.device)
                    loss_huber_rotation = torch.tensor(0.0, device=embed.device)
                    loss_huber_scale = torch.tensor(0.0, device=embed.device)
                    loss_bg_mask = torch.tensor(0.0, device=embed.device)
                    loss_inst_mask = torch.tensor(0.0, device=embed.device)
                    loss_attn = torch.tensor(0.0, device=embed.device)
                    loss_attn_constraint = torch.tensor(0.0, device=embed.device)
                    loss += bg_primitive_loss
                    loss_dict = {
                        "loss_semantic": loss_semantic,
                        "loss_primitive": loss_primitive,
                        "loss_translation": loss_translation,
                        "loss_rotation": loss_rotation,
                        "loss_scale": loss_scale,
                        "loss_eos": loss_eos,
                        "loss_flip": loss_flip,
                        "loss_recon": loss_recon,
                        "loss_huber_translation": loss_huber_translation,
                        "loss_huber_rotation": loss_huber_rotation,
                        "loss_huber_scale": loss_huber_scale,
                        "loss_bg_mask": loss_bg_mask,
                        "loss_inst_mask": loss_inst_mask,
                        "loss_attn": loss_attn,
                        "loss_attn_constraint": loss_attn_constraint,
                        "loss_bg_primitive": bg_primitive_loss,
                    }
                    if is_train:
                        return loss, loss_dict
                    else:
                        if self.bg_pred:
                            # 对 pred_primitive_bg 应用 sigmoid 得到概率（用于推理）
                            pred_primitive_bg_prob = torch.sigmoid(pred_primitive_bg)

                            # 构建 GT 背景类别用于评估
                            # 使用 pooled_sem_ids（primitive 级别），与 pc_cu_seqlens 一致
                            batch_size = pred_primitive_bg.shape[0]
                            max_num_prims = pred_primitive_bg.shape[1]
                            num_bg_classes = pred_primitive_bg.shape[2]

                            # gt_primitive_bg: (B, max_num_prims) 存储每个 primitive 的 GT 背景类别，-1 表示非背景
                            gt_primitive_bg = torch.full(
                                (batch_size, max_num_prims),
                                -1,
                                dtype=torch.int32,
                                device=pred_primitive_bg.device,
                            )
                            # bg_valid_mask: (B, max_num_prims) 有效 primitive 的 mask
                            bg_valid_mask = pc_embed_key_padding_mask.clone()

                            for b_i in range(batch_size):
                                valid_mask = pc_embed_key_padding_mask[
                                    b_i
                                ]  # (max_num_prims,)
                                num_valid = valid_mask.sum().item()

                                start_idx = pc_cu_seqlens[b_i].item()
                                end_idx = pc_cu_seqlens[b_i + 1].item()
                                sem_ids_bi = pooled_sem_ids[
                                    start_idx:end_idx
                                ]  # (N_valid_prims,)

                                # 转换: semantic 30-34 → 0-4 (有效背景类), 其他 → -1 (非背景)
                                target_b = torch.full_like(
                                    sem_ids_bi, -1, dtype=torch.int32
                                )
                                bg_mask_target = (sem_ids_bi >= 30) & (sem_ids_bi <= 34)
                                target_b[bg_mask_target] = (
                                    sem_ids_bi[bg_mask_target] - 30
                                )

                                # 将 target 填入 gt_primitive_bg 的有效位置
                                valid_indices = torch.where(valid_mask)[0][:num_valid]
                                gt_primitive_bg[b_i, valid_indices] = target_b

                            # 背景预测结果
                            loss_dict["pred_primitive_bg"] = (
                                pred_primitive_bg_prob,
                            )  # (B, max_num_prims, num_bg_classes) 概率值
                            loss_dict["gt_primitive_bg"] = (
                                gt_primitive_bg,
                            )  # (B, max_num_prims) GT 背景类别，-1 表示非背景
                            loss_dict["bg_valid_mask"] = (
                                bg_valid_mask,
                            )  # (B, max_num_prims) 有效 primitive 的 mask
                        return loss_dict

                gt_semantic_embed = primitives_embeds["semantic_id"]
                gt_primitive_embed = primitives_embeds["primitive_id"]
                gt_translation_embed = primitives_embeds["translation"]
                gt_rotation_embed = primitives_embeds["rotation"]
                gt_scale_embed = primitives_embeds["scale"]
                gt_flip_embed = primitives_embeds["flip_mode"]
                gt_semantic = primitives_gt["semantic_id"]
                gt_primitive = primitives_gt["primitive_id"]
                gt_translation = primitives_gt["translation"]
                gt_rotation = primitives_gt["rotation"]
                gt_scale = primitives_gt["scale"]
                gt_flip = primitives_gt["flip_mode"]
                gt_size = primitives_gt["size"] if self.size_pred else None

                gt_eos = torch.zeros(
                    (embed.shape[0], embed.shape[1] - 1),
                    dtype=torch.int64,
                    device=embed.device,
                )
                gt_eos = F.pad(gt_eos, (0, 1), value=0)
                if self.pretrain:
                    code_lens = primitive_mask.sum(dim=-1)
                else:
                    code_lens = primitive_mask.sum(dim=-1) * 2
                batch_arange = torch.arange(batch, device=device)
                batch_arange = rearrange(batch_arange, "... -> ... 1")
                code_lens = rearrange(code_lens, "... -> ... 1")
                gt_eos[batch_arange, code_lens] = 1

                if self.pretrain:
                    pred_geom_embeds = pred_primitive_embeds
                    translation_logits = rearrange(
                        self.to_translation_logits(pred_geom_embeds),
                        "b np (c nd) -> b np c nd",
                        nd=self.num_discrete_translation,
                    )
                    eos_logits = self.to_eos_logits(embed)
                    loss_translation = F.cross_entropy(
                        rearrange(translation_logits, "b np n c -> b c np n"),
                        gt_translation,
                        ignore_index=self.pad_id,
                        reduction="mean",
                    )
                    loss_eos = F.binary_cross_entropy_with_logits(
                        eos_logits.squeeze(-1), gt_eos.float(), reduction="mean"
                    )
                    probs_translation = translation_logits.softmax(-1)  # (B, N, 2, K)
                    pred_cont_translation = (
                        probs_translation * self.bin_centers_translation
                    ).sum(-1)[
                        primitive_mask
                    ]  # 期望值，可导
                    target_cont_translation = self.undiscretize_translation(
                        gt_translation.float()
                    )[primitive_mask]
                    huber_translation = F.smooth_l1_loss(
                        pred_cont_translation, target_cont_translation, reduction="mean"
                    )  # β=1 默认

                    loss += (
                        self.loss_weight["translation"] * loss_translation
                        + self.loss_weight["eos"] * loss_eos
                        + self.loss_weight["huber_translation"] * huber_translation
                    )

                    loss_dict = {
                        "loss_semantic": torch.tensor(0.0, device=embed.device),
                        "loss_primitive": torch.tensor(0.0, device=embed.device),
                        "loss_translation": loss_translation,
                        "loss_rotation": torch.tensor(0.0, device=embed.device),
                        "loss_scale": torch.tensor(0.0, device=embed.device),
                        "loss_eos": loss_eos,
                        "loss_flip": torch.tensor(0.0, device=embed.device),
                        "loss_recon": torch.tensor(0.0, device=embed.device),
                        "loss_huber_translation": huber_translation,
                        "loss_huber_rotation": torch.tensor(0.0, device=embed.device),
                        "loss_huber_scale": torch.tensor(0.0, device=embed.device),
                        "loss_bg_mask": torch.tensor(0.0, device=embed.device),
                        "loss_inst_mask": 0.0,
                        "loss_attn": 0.0,
                        "loss_attn_constraint": 0.0,
                    }
                    if is_train:
                        return loss, loss_dict
                    else:
                        ##### get the inference results #####
                        pred_discrete_translation = F.softmax(
                            translation_logits, dim=-1
                        ).argmax(dim=-1)
                        pred_cont_translation_unmask = (
                            probs_translation * self.bin_centers_translation
                        ).sum(-1)

                        results = {
                            "loss": loss,
                            "loss_semantic": torch.tensor(0.0, device=embed.device),
                            "loss_primitive": torch.tensor(0.0, device=embed.device),
                            "loss_translation": loss_translation,
                            "loss_rotation": torch.tensor(0.0, device=embed.device),
                            "loss_scale": torch.tensor(0.0, device=embed.device),
                            "loss_eos": loss_eos,
                            "loss_recon": torch.tensor(0.0, device=embed.device),
                            "loss_flip": torch.tensor(0.0, device=embed.device),
                            "loss_huber_translation": huber_translation,
                            "loss_huber_rotation": torch.tensor(
                                0.0, device=embed.device
                            ),
                            "loss_huber_scale": torch.tensor(0.0, device=embed.device),
                            "loss_bg_mask": torch.tensor(0.0, device=embed.device),
                            "gt_semantic": gt_semantic,
                            "gt_primitive": gt_primitive,
                            "gt_discrete_translation": gt_translation,
                            "gt_discrete_rotation": gt_rotation,
                            "gt_discrete_scale": gt_scale,
                            "gt_cont_translation": self.undiscretize_translation(
                                gt_translation.float()
                            ),
                            "gt_cont_rotation": self.undiscretize_rotation(
                                gt_rotation.float()
                            ),
                            "gt_cont_scale": self.undiscretize_scale(gt_scale.float()),
                            "gt_flip": gt_flip,
                            "gt_mask": primitive_mask,
                            "pred_mask": None,
                            "pred_semantic": None,
                            "pred_primitive": None,
                            "pred_discrete_translation": pred_discrete_translation,
                            "pred_discrete_rotation": None,
                            "pred_discrete_scale": None,
                            "pred_discrete_cont_translation": self.undiscretize_translation(
                                pred_discrete_translation.float()
                            ),
                            "pred_discrete_cont_rotation": None,
                            "pred_discrete_cont_scale": None,
                            "pred_cont_translation": pred_cont_translation_unmask,
                            "pred_cont_rotation": None,
                            "pred_cont_scale": None,
                            "pred_flip": None,
                            "semantic_logits": None,
                            "primitive_logits": None,
                            "translation_logits": translation_logits,
                            "rotation_logits": None,
                            "scale_logits": None,
                            "eos_logits": eos_logits,
                            "flip_logits": None,
                        }

                        return results

                if self.sem_geom_decouple:
                    pred_sem_embeds = pred_primitive_embeds[:, ::2]
                    pred_geom_embeds = pred_primitive_embeds[:, 1::2]
                elif self.geom_sem_decouple:
                    pred_geom_embeds = pred_primitive_embeds[:, ::2]
                    pred_sem_embeds = pred_primitive_embeds[:, 1::2]
                #### 单独计算loss ####
                ##### 处理semantic预测 #####
                semantic_logits = self.to_semantic_logits(pred_sem_embeds)
                ##### 之前的一个处理bug就是primitive和semantic是一起预测的，没有严格遵循Hierarchy Decoing的逻辑 ####
                # Use predicted semantic embed (straight-through) instead of GT,
                # so primitive prediction sees the same input as in AR inference.
                _st_tau = getattr(self, "gumbel_tau", 1.0)
                _ss_eps = getattr(self, "ss_eps", 0.0)
                _sem_onehot = st_gumbel_onehot(
                    semantic_logits, tau=_st_tau, stochastic=self.training
                )
                pred_semantic_embed_st = _sem_onehot @ self.semantic_embed.weight
                pred_semantic_embed_st = _ss_mix(
                    pred_semantic_embed_st, gt_semantic_embed, _ss_eps
                )
                sem_embed_packed, _ = pack(
                    [pred_sem_embeds, pred_semantic_embed_st], "b np *"
                )
                if self.primitive_contrastive_learning:
                    pred_primitive_embeds = self.pred_proj(sem_embed_packed)
                    lib_primitive_embeds = self.lib_proj(primitive_lib_embed)
                    pred_norm = F.normalize(pred_primitive_embeds, p=2, dim=-1)
                    lib_norm = F.normalize(lib_primitive_embeds, p=2, dim=-1)
                    logit_scale = self.logit_scale.exp().clamp(max=self.max_temperature)
                    primitive_logits = logit_scale * torch.einsum(
                        "bie,bpe->bip", pred_norm, lib_norm
                    )
                else:
                    primitive_logits = self.to_primitive_logits(sem_embed_packed)
                ### 修改这里primitive logits的预测逻辑 ###

                ##### 处理geom预测 #####
                if self.factorized_xy:
                    # Train: y conditioned on GT x bin (teacher forcing for the inner factorization).
                    _gt_tx_idx = gt_translation[..., 0].clamp(
                        min=0, max=self.num_discrete_translation - 1
                    )
                    translation_logits = self._factorized_2ch_logits(
                        pred_geom_embeds,
                        self.to_trans_x_logits,
                        self.to_trans_y_logits,
                        self.translation_embed,
                        cond_x_idx=_gt_tx_idx,
                    )
                else:
                    translation_logits = rearrange(
                        self.to_translation_logits(pred_geom_embeds),
                        "b np (c nd) -> b np c nd",
                        nd=self.num_discrete_translation,
                    )
                # Use Gumbel straight-through predicted translation embed instead of GT,
                # so the cascade matches AR inference (reduces intra-instance TF gap).
                _oh_tx = st_gumbel_onehot(
                    translation_logits[:, :, 0, :],
                    tau=_st_tau,
                    stochastic=self.training,
                )
                _oh_ty = st_gumbel_onehot(
                    translation_logits[:, :, 1, :],
                    tau=_st_tau,
                    stochastic=self.training,
                )
                pred_translation_embed_st = torch.cat(
                    [
                        _oh_tx @ self.translation_embed.weight,
                        _oh_ty @ self.translation_embed.weight,
                    ],
                    dim=-1,
                )
                pred_translation_embed_st = _ss_mix(
                    pred_translation_embed_st, gt_translation_embed, _ss_eps
                )
                embed_packed, _ = pack(
                    [pred_geom_embeds, pred_translation_embed_st], "b np *"
                )
                flip_logits = self.to_flip_logits(embed_packed)
                _flip_two_logits = torch.cat(
                    [-flip_logits, flip_logits], dim=-1
                )  # [B, np, 2]
                _oh_flip = st_gumbel_onehot(
                    _flip_two_logits, tau=_st_tau, stochastic=self.training
                )
                pred_flip_embed_st = _oh_flip @ self.flip_embed.weight
                pred_flip_embed_st = _ss_mix(pred_flip_embed_st, gt_flip_embed, _ss_eps)
                embed_packed, _ = pack([embed_packed, pred_flip_embed_st], "b np *")
                rotation_logits = rearrange(
                    self.to_rotation_logits(embed_packed),
                    "b np (c nd) -> b np c nd",
                    nd=self.num_discrete_rotation,
                )
                _oh_rot = st_gumbel_onehot(
                    rotation_logits[:, :, 0, :], tau=_st_tau, stochastic=self.training
                )
                pred_rotation_embed_st = _oh_rot @ self.rotation_embed.weight
                pred_rotation_embed_st = _ss_mix(
                    pred_rotation_embed_st, gt_rotation_embed, _ss_eps
                )
                embed_packed, _ = pack([embed_packed, pred_rotation_embed_st], "b np *")
                if self.size_pred:
                    if self.factorized_xy:
                        _gt_sw_idx = gt_size[..., 0].clamp(
                            min=0, max=self.num_discrete_size - 1
                        )
                        size_logits = self._factorized_2ch_logits(
                            embed_packed,
                            self.to_size_w_logits,
                            self.to_size_h_logits,
                            self.size_embed,
                            cond_x_idx=_gt_sw_idx,
                        )
                    else:
                        size_logits = rearrange(
                            self.to_size_logits(embed_packed),
                            "b np (c nd) -> b np c nd",
                            nd=self.num_discrete_size,
                        )
                    scale_logits = None
                else:
                    size_logits = None
                    scale_logits = rearrange(
                        self.to_scale_logits(embed_packed),
                        "b np (c nd) -> b np c nd",
                        nd=self.num_discrete_scale,
                    )

                eos_logits = self.to_eos_logits(embed)

                loss_eos = F.binary_cross_entropy_with_logits(
                    eos_logits.squeeze(-1), gt_eos.float(), reduction="mean"
                )
                loss_semantic = F.cross_entropy(
                    rearrange(semantic_logits, "b np c -> b c np"),
                    gt_semantic,
                    ignore_index=self.pad_id,
                    reduction="mean",
                )
                if self.primitive_contrastive_learning:
                    loss_primitive, hierarchical_loss = (
                        self.Hierarchical_primitive_criterion(
                            primitive_logits, gt_primitive, gt_semantic
                        )
                    )
                else:
                    loss_primitive = F.cross_entropy(
                        rearrange(primitive_logits, "b np c -> b c np"),
                        gt_primitive,
                        ignore_index=self.pad_id,
                        reduction="mean",
                    )
                if self.soft_ce:
                    loss_translation = self.soft_ce_with_kernel(
                        translation_logits,
                        gt_translation,
                        self.kernel_translation,
                        self.pad_id,
                    )
                    loss_rotation = self.soft_ce_with_kernel(
                        rotation_logits, gt_rotation, self.kernel_rotation, self.pad_id
                    )
                    loss_scale = self.soft_ce_with_kernel(
                        scale_logits, gt_scale, self.kernel_scale, self.pad_id
                    )
                else:
                    loss_translation = F.cross_entropy(
                        rearrange(translation_logits, "b np n c -> b c np n"),
                        gt_translation,
                        ignore_index=self.pad_id,
                        reduction="mean",
                    )
                    loss_rotation = F.cross_entropy(
                        rearrange(rotation_logits, "b np n c -> b c (np n)"),
                        gt_rotation,
                        ignore_index=self.pad_id,
                        reduction="mean",
                    )
                    # loss_scale = F.cross_entropy(rearrange(scale_logits, 'b np n c -> b c (np n)'),  gt_scale, ignore_index = self.pad_id, reduction='mean')
                    if self.size_pred:
                        loss_scale = F.cross_entropy(
                            rearrange(size_logits, "b np n c -> b c np n"),
                            gt_size,
                            ignore_index=self.pad_id,
                            reduction="mean",
                        )
                    else:
                        loss_scale = F.cross_entropy(
                            rearrange(scale_logits, "b np n c -> b c np n"),
                            gt_scale,
                            ignore_index=self.pad_id,
                            reduction="mean",
                        )

                pred_flip = flip_logits.squeeze(-1)[primitive_mask]
                loss_flip = F.binary_cross_entropy_with_logits(
                    pred_flip, gt_flip.float()[primitive_mask], reduction="mean"
                )

                loss += (
                    self.loss_weight["eos"] * loss_eos
                    + self.loss_weight["semantic_id"] * loss_semantic
                    + self.loss_weight["primitive_id"] * loss_primitive
                    + self.loss_weight["translation"] * loss_translation
                    + self.loss_weight["rotation"] * loss_rotation
                    + self.loss_weight["scale"] * loss_scale
                    + self.loss_weight["flip"] * loss_flip
                )

                huber_translation, huber_rotation, huber_scale = 0.0, 0.0, 0.0

                ### add huber loss
                if self.gumbel_loss and self.training:
                    probs_translation = F.gumbel_softmax(
                        translation_logits, tau=1.0, hard=True
                    )
                else:
                    probs_translation = translation_logits.softmax(-1)  # (B, N, 2, K)
                pred_cont_translation = (
                    probs_translation * self.bin_centers_translation
                ).sum(-1)[
                    primitive_mask
                ]  # 期望值，可导
                target_cont_translation = self.undiscretize_translation(
                    gt_translation.float()
                )[primitive_mask]
                huber_translation = F.smooth_l1_loss(
                    pred_cont_translation, target_cont_translation, reduction="mean"
                )  # β=1 默认

                if self.gumbel_loss and self.training:
                    probs_rotation = F.gumbel_softmax(
                        rotation_logits, tau=1.0, hard=True
                    )
                else:
                    probs_rotation = rotation_logits.softmax(-1)  # (B, N, 2, K)
                pred_cont_rotation = (probs_rotation * self.bin_centers_rotation).sum(
                    -1
                )[
                    primitive_mask
                ]  # 期望值，可导
                target_cont_rotation = self.undiscretize_rotation(gt_rotation.float())[
                    primitive_mask
                ]
                huber_rotation = (
                    F.smooth_l1_loss(
                        pred_cont_rotation,
                        target_cont_rotation.unsqueeze(-1),
                        reduction="mean",
                    )
                    / 180.0
                )

                if self.size_pred:
                    if self.gumbel_loss and self.training:
                        probs_size = F.gumbel_softmax(size_logits, tau=1.0, hard=True)
                    else:
                        probs_size = size_logits.softmax(-1)  # (B, N, 2, K)
                    pred_cont_size = (probs_size * self.bin_centers_size).sum(-1)[
                        primitive_mask
                    ]  # 期望值，可导
                    target_cont_size = self.undiscretize_size(gt_size.float())[
                        primitive_mask
                    ]
                    huber_scale = F.smooth_l1_loss(
                        pred_cont_size, target_cont_size, reduction="mean"
                    )
                    target_cont_scale = self.undiscretize_scale(gt_scale.float())[
                        primitive_mask
                    ]
                else:
                    probs_scale = scale_logits.softmax(-1)  # (B, N, 2, K)
                    pred_cont_scale = (probs_scale * self.bin_centers_scale).sum(-1)[
                        primitive_mask
                    ]  # 期望值，可导
                    target_cont_scale = self.undiscretize_scale(gt_scale.float())[
                        primitive_mask
                    ]
                    # huber_scale = F.smooth_l1_loss(pred_cont_scale, target_cont_scale.unsqueeze(-1), reduction='mean') / 1.5 # β=1 默认
                    huber_scale = (
                        F.smooth_l1_loss(
                            pred_cont_scale, target_cont_scale, reduction="mean"
                        )
                        / 1.5
                    )  # β=1 默认

                loss += (
                    self.loss_weight["huber_translation"] * huber_translation
                    + self.loss_weight["huber_rotation"] * huber_rotation
                    + self.loss_weight["huber_scale"] * huber_scale
                )

                if self.dist_loss:
                    loss_translation_dist = dist_loss_cdf(
                        translation_logits,
                        gt_translation,
                        pad_id=self.pad_id,
                        kind="w1",
                        q_from_kernel=self.kernel_translation,
                    )
                    loss_rotation_dist = dist_loss_cdf(
                        rotation_logits,
                        gt_rotation,
                        pad_id=self.pad_id,
                        kind="w1",
                        q_from_kernel=self.kernel_rotation,
                    )
                    loss_scale_dist = dist_loss_cdf(
                        scale_logits,
                        gt_scale,
                        pad_id=self.pad_id,
                        kind="w1",
                        q_from_kernel=self.kernel_scale,
                    )
                    loss += loss_translation_dist + loss_rotation_dist + loss_scale_dist

                # if self.loss_weight['reconstruction'] > 0:
                loss_recon = torch.tensor(0.0, device=embed.device)
                if self.loss_weight["reconstruction"] > 0:
                    loss_recon = self.compute_recon_loss(
                        translation_logits=None,
                        rotation_logits=None,
                        scale_logits=None,
                        gt_primitive_id=gt_primitive,
                        gt_translation=gt_translation,
                        gt_rotation=gt_rotation,
                        gt_scale=gt_scale,
                        gt_flip=gt_flip[primitive_mask],
                        primitive_mask=primitive_mask,
                        pc_coord=coord,
                        pc_primitive_index=None,
                        pc_offset=offset,
                        gt_primitive_coords=None,
                        pred_gumbel_cont_translation=pred_cont_translation,
                        pred_gumbel_cont_rotation=pred_cont_rotation,
                        pred_gumbel_cont_scale=(
                            target_cont_scale if self.size_pred else pred_cont_scale
                        ),
                        target_cont_rotation=target_cont_rotation,
                        target_cont_scale=target_cont_scale,
                        target_cont_translation=target_cont_translation,
                        pred_flip=gt_flip[primitive_mask],
                        # pred_flip = pred_flip,
                    )
                loss += self.loss_weight["reconstruction"] * loss_recon
                if self.bg_pred:
                    loss += bg_primitive_loss
                loss_dict = {
                    "loss_semantic": loss_semantic,
                    "loss_primitive": loss_primitive,
                    "loss_translation": loss_translation,
                    "loss_rotation": loss_rotation,
                    "loss_scale": loss_scale,
                    "loss_eos": loss_eos,
                    "loss_flip": loss_flip,
                    "loss_recon": loss_recon,
                    "loss_huber_translation": huber_translation,
                    "loss_huber_rotation": huber_rotation,
                    "loss_huber_scale": huber_scale,
                    # 'loss_bg_mask': bg_mask_loss,
                    "loss_bg_primitive": bg_primitive_loss,
                    "loss_inst_mask": 0.0,
                    "loss_attn": 0.0,
                    "loss_attn_constraint": 0.0,
                }
                if is_train:
                    return loss, loss_dict
                else:
                    ##### get the inference results #####
                    pred_eos = F.sigmoid(eos_logits.squeeze(-1)) > 0.5
                    mask = ~(pred_eos.float().cumsum(dim=-1) >= 1)[:, :-1]
                    pred_semantic = F.softmax(semantic_logits, dim=-1).argmax(dim=-1)
                    pred_primitive = F.softmax(primitive_logits, dim=-1).argmax(dim=-1)
                    pred_discrete_translation = F.softmax(
                        translation_logits, dim=-1
                    ).argmax(dim=-1)
                    pred_discrete_rotation = F.softmax(rotation_logits, dim=-1).argmax(
                        dim=-1
                    )
                    if self.size_pred:
                        pred_discrete_size = F.softmax(size_logits, dim=-1).argmax(
                            dim=-1
                        )
                        pred_discrete_scale = None
                    else:
                        pred_discrete_scale = F.softmax(scale_logits, dim=-1).argmax(
                            dim=-1
                        )
                        pred_discrete_size = None
                    pred_cont_translation_unmask = (
                        probs_translation * self.bin_centers_translation
                    ).sum(-1)
                    pred_cont_rotation_unmask = (
                        probs_rotation * self.bin_centers_rotation
                    ).sum(-1)
                    if self.size_pred:
                        pred_cont_size_unmask = (
                            probs_size * self.bin_centers_size
                        ).sum(-1)
                        pred_cont_scale_unmask = None
                    else:
                        pred_cont_scale_unmask = (
                            probs_scale * self.bin_centers_scale
                        ).sum(-1)
                        pred_cont_size_unmask = None
                    pred_flip = (torch.sigmoid(flip_logits) > 0.5).int()

                    if (
                        self.bg_pred
                    ):  # 对 pred_primitive_bg 应用 sigmoid 得到概率（用于推理）
                        pred_primitive_bg_prob = torch.sigmoid(pred_primitive_bg)

                        # 构建 GT 背景类别用于评估
                        # 使用 pooled_sem_ids（primitive 级别），与 pc_cu_seqlens 一致
                        batch_size = pred_primitive_bg.shape[0]
                        max_num_prims = pred_primitive_bg.shape[1]
                        num_bg_classes = pred_primitive_bg.shape[2]

                        # gt_primitive_bg: (B, max_num_prims) 存储每个 primitive 的 GT 背景类别，-1 表示非背景
                        gt_primitive_bg = torch.full(
                            (batch_size, max_num_prims),
                            -1,
                            dtype=torch.int32,
                            device=pred_primitive_bg.device,
                        )
                        # bg_valid_mask: (B, max_num_prims) 有效 primitive 的 mask
                        bg_valid_mask = pc_embed_key_padding_mask.clone()

                        for b_i in range(batch_size):
                            valid_mask = pc_embed_key_padding_mask[
                                b_i
                            ]  # (max_num_prims,)
                            num_valid = valid_mask.sum().item()

                            start_idx = pc_cu_seqlens[b_i].item()
                            end_idx = pc_cu_seqlens[b_i + 1].item()
                            sem_ids_bi = pooled_sem_ids[
                                start_idx:end_idx
                            ]  # (N_valid_prims,)

                            # 转换: semantic 30-34 → 0-4 (有效背景类), 其他 → -1 (非背景)
                            target_b = torch.full_like(
                                sem_ids_bi, -1, dtype=sem_ids_bi.dtype
                            )
                            bg_mask_target = (sem_ids_bi >= 30) & (sem_ids_bi <= 34)
                            target_b[bg_mask_target] = sem_ids_bi[bg_mask_target] - 30

                            # 将 target 填入 gt_primitive_bg 的有效位置
                            valid_indices = torch.where(valid_mask)[0][:num_valid]
                            gt_primitive_bg[b_i, valid_indices] = target_b

                    results = {
                        "loss": loss,
                        "loss_semantic": loss_semantic,
                        "loss_primitive": loss_primitive,
                        "loss_translation": loss_translation,
                        "loss_rotation": loss_rotation,
                        "loss_scale": loss_scale,
                        "loss_eos": loss_eos,
                        "loss_recon": loss_recon,
                        "loss_flip": loss_flip,
                        "loss_huber_translation": huber_translation,
                        "loss_huber_rotation": huber_rotation,
                        "loss_huber_scale": huber_scale,
                        "loss_bg_primitive": bg_primitive_loss,
                        "gt_semantic": gt_semantic,
                        "gt_primitive": gt_primitive,
                        "gt_discrete_translation": gt_translation,
                        "gt_discrete_rotation": gt_rotation,
                        "gt_discrete_scale": gt_scale,
                        "gt_cont_translation": self.undiscretize_translation(
                            gt_translation.float()
                        ),
                        "gt_cont_rotation": self.undiscretize_rotation(
                            gt_rotation.float()
                        ),
                        "gt_cont_scale": self.undiscretize_scale(gt_scale.float()),
                        "gt_cont_size": (
                            self.undiscretize_size(gt_size.float())
                            if self.size_pred
                            else None
                        ),
                        "gt_flip": gt_flip,
                        "gt_size": gt_size if self.size_pred else None,
                        "gt_mask": primitive_mask,
                        "pred_mask": mask,
                        "pred_semantic": pred_semantic,
                        "pred_primitive": pred_primitive,
                        "pred_discrete_translation": pred_discrete_translation,
                        "pred_discrete_rotation": pred_discrete_rotation,
                        "pred_discrete_scale": pred_discrete_scale,
                        "pred_discrete_size": pred_discrete_size,
                        "pred_discrete_cont_translation": self.undiscretize_translation(
                            pred_discrete_translation.float()
                        ),
                        "pred_discrete_cont_rotation": self.undiscretize_rotation(
                            pred_discrete_rotation.float()
                        ),
                        "pred_discrete_cont_scale": (
                            None
                            if self.size_pred
                            else self.undiscretize_scale(pred_discrete_scale.float())
                        ),
                        "pred_discrete_cont_size": (
                            self.undiscretize_size(pred_discrete_size.float())
                            if self.size_pred
                            else None
                        ),
                        "pred_cont_translation": pred_cont_translation_unmask,
                        "pred_cont_rotation": pred_cont_rotation_unmask,
                        "pred_cont_scale": pred_cont_scale_unmask,
                        "pred_cont_size": pred_cont_size_unmask,
                        "pred_flip": pred_flip,
                        "semantic_logits": semantic_logits,
                        "primitive_logits": primitive_logits,
                        "translation_logits": translation_logits,
                        "rotation_logits": rotation_logits,
                        "scale_logits": scale_logits,
                        "size_logits": size_logits if self.size_pred else None,
                        "eos_logits": eos_logits,
                        "flip_logits": flip_logits,
                        # 背景预测结果
                        "pred_primitive_bg": pred_primitive_bg_prob,  # (B, max_num_prims, num_bg_classes) 概率值
                        "gt_primitive_bg": gt_primitive_bg,  # (B, max_num_prims) GT 背景类别，-1 表示非背景
                        "bg_valid_mask": bg_valid_mask,  # (B, max_num_prims) 有效 primitive 的 mask
                        # 用于可视化的 primitive-level 映射信息
                        "prim_keep_mask": prim_keep_mask,  # (P1+P2+...,) 原始哪些 primitive 被保留
                        "prim_id_mapping": prim_id_mapping,  # (P1+P2+...,) 原始 primitive ID → pooled primitive index
                        ##### return instance-wise loss #####
                        "instance_semantic_loss": F.cross_entropy(
                            rearrange(semantic_logits, "b np c -> b c np"),
                            gt_semantic,
                            ignore_index=self.pad_id,
                            reduction="none",
                        ),
                        "instance_primitive_loss": F.cross_entropy(
                            rearrange(primitive_logits, "b np c -> b c np"),
                            gt_primitive,
                            ignore_index=self.pad_id,
                            reduction="none",
                        ),
                        "instance_translation_loss": F.cross_entropy(
                            rearrange(translation_logits, "b np n c -> b c np n"),
                            gt_translation,
                            ignore_index=self.pad_id,
                            reduction="none",
                        ),
                        "instance_rotation_loss": F.cross_entropy(
                            rearrange(rotation_logits, "b np n c -> b c (np n)"),
                            gt_rotation,
                            ignore_index=self.pad_id,
                            reduction="none",
                        ),
                        "instance_scale_loss": (
                            None
                            if self.size_pred
                            else F.cross_entropy(
                                rearrange(scale_logits, "b np n c -> b c np n"),
                                gt_scale,
                                ignore_index=self.pad_id,
                                reduction="none",
                            )
                        ),
                        "instance_flip_loss": (
                            F.binary_cross_entropy_with_logits(
                                pred_flip,
                                gt_flip.float()[primitive_mask],
                                reduction="none",
                            )
                            if self.bg_pred
                            else 0.0
                        ),
                        "instance_size_loss": (
                            F.cross_entropy(
                                rearrange(size_logits, "b np n c -> b c np n"),
                                gt_size,
                                ignore_index=self.pad_id,
                                reduction="none",
                            )
                            if self.size_pred
                            else 0.0
                        ),
                    }

                    return results

            if self.attr_decouple:
                return self._attr_decouple_loss_and_outputs(
                    embed=embed,
                    primitive_mask=primitive_mask,
                    primitives_embeds=primitives_embeds,
                    primitives_gt=primitives_gt,
                    primitive_lib_embed=primitive_lib_embed,
                    bg_primitive_loss=bg_primitive_loss,
                    is_train=is_train,
                    coord=coord,
                    offset=offset,
                    gt_cont_scale=scale,
                    gt_cont_rotation=rotation,
                    gt_cont_flip=flip_mode,
                    recon_source_segments=kwargs.get("recon_source_segments"),
                    recon_source_mask=kwargs.get("recon_source_mask"),
                    recon_valid_mask=kwargs.get("recon_valid_mask"),
                    recon_full_translation=kwargs.get("recon_full_translation"),
                    recon_clip_count_ratio=kwargs.get("recon_clip_count_ratio"),
                    recon_clip_length_ratio=kwargs.get("recon_clip_length_ratio"),
                    recon_is_partial=kwargs.get("recon_is_partial"),
                    recon_preclip_source_segments=kwargs.get(
                        "recon_preclip_source_segments"
                    ),
                    recon_preclip_source_mask=kwargs.get("recon_preclip_source_mask"),
                )

                # Legacy experimental branch retained below for checkpoint
                # archaeology; the production implementation returns above.
                loss = 0.0

                gt_semantic_embed = primitives_embeds["semantic_id"]
                gt_primitive_embed = primitives_embeds["primitive_id"]
                gt_translation_embed = primitives_embeds["translation"]
                gt_rotation_embed = primitives_embeds["rotation"]
                gt_scale_embed = primitives_embeds["scale"]
                gt_flip_embed = primitives_embeds["flip_mode"]
                gt_semantic = primitives_gt["semantic_id"]
                gt_primitive = primitives_gt["primitive_id"]
                gt_translation = primitives_gt["translation"]
                gt_rotation = primitives_gt["rotation"]
                gt_scale = primitives_gt["scale"]
                gt_flip = primitives_gt["flip_mode"]

                gt_eos = torch.zeros(
                    (embed.shape[0], embed.shape[1] - 1),
                    dtype=torch.int64,
                    device=embed.device,
                )
                gt_eos = F.pad(gt_eos, (0, 1), value=0)
                if self.pretrain:
                    if self.learnable_anchor_embedding:
                        code_lens = primitive_mask.sum(dim=-1) * 2
                    else:
                        code_lens = primitive_mask.sum(dim=-1)
                else:
                    if self.sem_prim_pretrain:
                        code_lens = primitive_mask.sum(dim=-1) * 2
                    elif self.lm_loss:
                        code_lens = primitive_mask.sum(dim=-1) * 7
                    else:
                        if self.learnable_anchor_embedding:
                            code_lens = primitive_mask.sum(dim=-1) * 7
                        else:
                            code_lens = primitive_mask.sum(dim=-1) * 6
                batch_arange = torch.arange(batch, device=device)
                batch_arange = rearrange(batch_arange, "... -> ... 1")
                code_lens = rearrange(code_lens, "... -> ... 1")
                gt_eos[batch_arange, code_lens] = 1

                if self.pretrain:
                    if self.learnable_anchor_embedding:
                        pred_anchor_embeds = pred_primitive_embeds[:, 0::2]
                        pred_geom_embeds = pred_primitive_embeds[:, 1::2]
                        pred_embed, _ = pack(
                            [pred_anchor_embeds, pred_geom_embeds], "b np *"
                        )
                        translation_logits = rearrange(
                            self.to_translation_logits(pred_embed),
                            "b np (c nd) -> b np c nd",
                            nd=self.num_discrete_translation,
                        )
                        eos_logits = self.to_eos_logits(embed)
                        loss_translation = F.cross_entropy(
                            rearrange(translation_logits, "b np n c -> b c np n"),
                            gt_translation,
                            ignore_index=self.pad_id,
                            reduction="mean",
                        )
                        loss_eos = F.binary_cross_entropy_with_logits(
                            eos_logits.squeeze(-1), gt_eos.float(), reduction="mean"
                        )
                        probs_translation = translation_logits.softmax(
                            -1
                        )  # (B, N, 2, K)
                        pred_cont_translation = (
                            probs_translation * self.bin_centers_translation
                        ).sum(-1)[
                            primitive_mask
                        ]  # 期望值，可导
                        target_cont_translation = self.undiscretize_translation(
                            gt_translation.float()
                        )[primitive_mask]
                        huber_translation = F.smooth_l1_loss(
                            pred_cont_translation,
                            target_cont_translation,
                            reduction="mean",
                        )  # β=1 默认

                        loss += (
                            self.loss_weight["translation"] * loss_translation
                            + self.loss_weight["eos"] * loss_eos
                            + self.loss_weight["huber_translation"] * huber_translation
                        )

                        loss_dict = {
                            "loss_semantic": torch.tensor(0.0, device=embed.device),
                            "loss_primitive": torch.tensor(0.0, device=embed.device),
                            "loss_translation": loss_translation,
                            "loss_rotation": torch.tensor(0.0, device=embed.device),
                            "loss_scale": torch.tensor(0.0, device=embed.device),
                            "loss_eos": loss_eos,
                            "loss_flip": torch.tensor(0.0, device=embed.device),
                            "loss_recon": torch.tensor(0.0, device=embed.device),
                            "loss_huber_translation": huber_translation,
                            "loss_huber_rotation": torch.tensor(
                                0.0, device=embed.device
                            ),
                            "loss_huber_scale": torch.tensor(0.0, device=embed.device),
                            "loss_bg_mask": torch.tensor(0.0, device=embed.device),
                            "loss_inst_mask": 0.0,
                            "loss_attn": 0.0,
                            "loss_attn_constraint": 0.0,
                        }
                    else:
                        pred_geom_embeds = pred_primitive_embeds
                        translation_logits = rearrange(
                            self.to_translation_logits(pred_geom_embeds),
                            "b np (c nd) -> b np c nd",
                            nd=self.num_discrete_translation,
                        )
                        eos_logits = self.to_eos_logits(embed)
                        loss_translation = F.cross_entropy(
                            rearrange(translation_logits, "b np n c -> b c np n"),
                            gt_translation,
                            ignore_index=self.pad_id,
                            reduction="mean",
                        )
                        loss_eos = F.binary_cross_entropy_with_logits(
                            eos_logits.squeeze(-1), gt_eos.float(), reduction="mean"
                        )
                        probs_translation = translation_logits.softmax(
                            -1
                        )  # (B, N, 2, K)
                        pred_cont_translation = (
                            probs_translation * self.bin_centers_translation
                        ).sum(-1)[
                            primitive_mask
                        ]  # 期望值，可导
                        target_cont_translation = self.undiscretize_translation(
                            gt_translation.float()
                        )[primitive_mask]
                        huber_translation = F.smooth_l1_loss(
                            pred_cont_translation,
                            target_cont_translation,
                            reduction="mean",
                        )  # β=1 默认

                        loss += (
                            self.loss_weight["translation"] * loss_translation
                            + self.loss_weight["eos"] * loss_eos
                            + self.loss_weight["huber_translation"] * huber_translation
                        )

                        loss_dict = {
                            "loss_semantic": torch.tensor(0.0, device=embed.device),
                            "loss_primitive": torch.tensor(0.0, device=embed.device),
                            "loss_translation": loss_translation,
                            "loss_rotation": torch.tensor(0.0, device=embed.device),
                            "loss_scale": torch.tensor(0.0, device=embed.device),
                            "loss_eos": loss_eos,
                            "loss_flip": torch.tensor(0.0, device=embed.device),
                            "loss_recon": torch.tensor(0.0, device=embed.device),
                            "loss_huber_translation": huber_translation,
                            "loss_huber_rotation": torch.tensor(
                                0.0, device=embed.device
                            ),
                            "loss_huber_scale": torch.tensor(0.0, device=embed.device),
                            "loss_bg_mask": torch.tensor(0.0, device=embed.device),
                            "loss_inst_mask": 0.0,
                            "loss_attn": 0.0,
                            "loss_attn_constraint": 0.0,
                        }
                    if is_train:
                        return loss, loss_dict
                    else:
                        ##### get the inference results #####
                        pred_discrete_translation = F.softmax(
                            translation_logits, dim=-1
                        ).argmax(dim=-1)
                        pred_cont_translation_unmask = (
                            probs_translation * self.bin_centers_translation
                        ).sum(-1)

                        results = {
                            "loss": loss,
                            "loss_semantic": torch.tensor(0.0, device=embed.device),
                            "loss_primitive": torch.tensor(0.0, device=embed.device),
                            "loss_translation": loss_translation,
                            "loss_rotation": torch.tensor(0.0, device=embed.device),
                            "loss_scale": torch.tensor(0.0, device=embed.device),
                            "loss_eos": loss_eos,
                            "loss_recon": torch.tensor(0.0, device=embed.device),
                            "loss_flip": torch.tensor(0.0, device=embed.device),
                            "loss_huber_translation": huber_translation,
                            "loss_huber_rotation": torch.tensor(
                                0.0, device=embed.device
                            ),
                            "loss_huber_scale": torch.tensor(0.0, device=embed.device),
                            "loss_bg_mask": torch.tensor(0.0, device=embed.device),
                            "gt_semantic": gt_semantic,
                            "gt_primitive": gt_primitive,
                            "gt_discrete_translation": gt_translation,
                            "gt_discrete_rotation": gt_rotation,
                            "gt_discrete_scale": gt_scale,
                            "gt_cont_translation": self.undiscretize_translation(
                                gt_translation.float()
                            ),
                            "gt_cont_rotation": self.undiscretize_rotation(
                                gt_rotation.float()
                            ),
                            "gt_cont_scale": self.undiscretize_scale(gt_scale.float()),
                            "gt_flip": gt_flip,
                            "gt_mask": primitive_mask,
                            "pred_mask": None,
                            "pred_semantic": None,
                            "pred_primitive": None,
                            "pred_discrete_translation": pred_discrete_translation,
                            "pred_discrete_rotation": None,
                            "pred_discrete_scale": None,
                            "pred_discrete_cont_translation": self.undiscretize_translation(
                                pred_discrete_translation.float()
                            ),
                            "pred_discrete_cont_rotation": None,
                            "pred_discrete_cont_scale": None,
                            "pred_cont_translation": pred_cont_translation_unmask,
                            "pred_cont_rotation": None,
                            "pred_cont_scale": None,
                            "pred_flip": None,
                            "semantic_logits": None,
                            "primitive_logits": None,
                            "translation_logits": translation_logits,
                            "rotation_logits": None,
                            "scale_logits": None,
                            "eos_logits": eos_logits,
                            "flip_logits": None,
                        }

                        return results

                if self.sem_prim_pretrain:
                    pred_sem_embeds = pred_primitive_embeds[:, 0::2]
                    pred_prim_embeds = pred_primitive_embeds[:, 1::2]
                    semantic_logits = self.to_semantic_logits(pred_sem_embeds)
                    primitive_logits = self.to_primitive_logits(pred_prim_embeds)
                    loss_semantic = F.cross_entropy(
                        rearrange(semantic_logits, "b np c -> b c np"),
                        gt_semantic,
                        ignore_index=self.pad_id,
                        reduction="mean",
                    )
                    loss_primitive = F.cross_entropy(
                        rearrange(primitive_logits, "b np c -> b c np"),
                        gt_primitive,
                        ignore_index=self.pad_id,
                        reduction="mean",
                    )
                    loss += (
                        self.loss_weight["semantic_id"] * loss_semantic
                        + self.loss_weight["primitive_id"] * loss_primitive
                    )

                    if self.lm_loss:
                        #### prepare the lm target ####
                        gt_lm = torch.zeros(
                            batch,
                            num_primitives * 2,
                            dtype=torch.long,
                            device=self.device,
                        )
                        # sem_targets = gt_semantic.masked_fill(~primitive_mask, self.pad_id)
                        # prim_targets = gt_primitive.masked_fill(~primitive_mask, self.pad_id)
                        gt_lm[:, 0::2] = gt_semantic
                        gt_lm[:, 1::2] = gt_primitive + self.num_semantic
                        gt_lm[:, 0::2] = gt_lm[:, 0::2].masked_fill(
                            ~primitive_mask, self.pad_id
                        )
                        gt_lm[:, 1::2] = gt_lm[:, 1::2].masked_fill(
                            ~primitive_mask, self.pad_id
                        )
                        lm_logits = self.lm_head(pred_primitive_embeds)
                        loss_lm = F.cross_entropy(
                            rearrange(lm_logits, "b np c -> b c np"),
                            gt_lm,
                            ignore_index=self.pad_id,
                            reduction="mean",
                        )
                        loss += 0.1 * loss_lm

                    loss_dict = {
                        "loss_semantic": loss_semantic,
                        "loss_primitive": loss_primitive,
                        "loss_translation": torch.tensor(0.0, device=embed.device),
                        "loss_rotation": torch.tensor(0.0, device=embed.device),
                        "loss_scale": torch.tensor(0.0, device=embed.device),
                        "loss_eos": torch.tensor(0.0, device=embed.device),
                        "loss_flip": torch.tensor(0.0, device=embed.device),
                        "loss_recon": torch.tensor(0.0, device=embed.device),
                        "loss_huber_translation": torch.tensor(
                            0.0, device=embed.device
                        ),
                        "loss_huber_rotation": torch.tensor(0.0, device=embed.device),
                        "loss_huber_scale": torch.tensor(0.0, device=embed.device),
                        "loss_bg_mask": torch.tensor(0.0, device=embed.device),
                        "loss_inst_mask": 0.0,
                        "loss_attn": 0.0,
                        "loss_attn_constraint": 0.0,
                        "loss_lm": loss_lm,
                    }
                    if is_train:
                        return loss, loss_dict
                if self.lm_loss:
                    pred_sem_embeds = pred_primitive_embeds[:, 0::7]
                    pred_prim_embeds = pred_primitive_embeds[:, 1::7]
                    pred_trans_x_embeds = pred_primitive_embeds[:, 2::7]
                    pred_trans_y_embeds = pred_primitive_embeds[:, 3::7]
                    pred_flip_embeds = pred_primitive_embeds[:, 4::7]
                    pred_rot_embeds = pred_primitive_embeds[:, 5::7]
                    pred_scale_embeds = pred_primitive_embeds[:, 6::7]
                    #### 单独计算loss ####
                    ##### 处理semantic预测 #####
                    semantic_logits = self.to_semantic_logits(pred_sem_embeds)
                    primitive_logits = self.to_primitive_logits(pred_prim_embeds)
                    ##### 处理geom预测 #####
                    # translation_logits = rearrange(self.to_translation_logits(pred_trans_embeds), 'b np (c nd) -> b np c nd', nd=self.num_discrete_translation)
                    translation_logits_x = self.to_translation_logits(
                        pred_trans_x_embeds
                    )
                    translation_logits_y = self.to_translation_logits(
                        pred_trans_y_embeds
                    )
                    # translation_logits = (translation_logits_x, translation_logits_y)
                    flip_logits = self.to_flip_logits(pred_flip_embeds)
                    rotation_logits = rearrange(
                        self.to_rotation_logits(pred_rot_embeds),
                        "b np (c nd) -> b np c nd",
                        nd=self.num_discrete_rotation,
                    )
                    scale_logits = rearrange(
                        self.to_scale_logits(pred_scale_embeds),
                        "b np (c nd) -> b np c nd",
                        nd=self.num_discrete_scale,
                    )
                    eos_logits = self.to_eos_logits(embed)
                    ##### prepare for the LM loss #####
                    gt_lm = torch.zeros(
                        batch, num_primitives * 7, dtype=torch.long, device=self.device
                    )
                    gt_lm[:, 0::7] = gt_semantic
                    gt_lm[:, 1::7] = gt_primitive + self.num_semantic
                    gt_lm[:, 2::7] = (
                        gt_translation[:, :, 0] + self.num_semantic + self.num_primitive
                    )
                    gt_lm[:, 3::7] = (
                        gt_translation[:, :, 1] + self.num_semantic + self.num_primitive
                    )
                    gt_lm[:, 4::7] = (
                        gt_flip
                        + self.num_semantic
                        + self.num_primitive
                        + self.num_discrete_translation
                    )
                    gt_lm[:, 5::7] = (
                        gt_rotation
                        + self.num_semantic
                        + self.num_primitive
                        + self.num_discrete_translation
                        + 1
                    )
                    gt_lm[:, 6::7] = (
                        gt_scale
                        + self.num_semantic
                        + self.num_primitive
                        + self.num_discrete_translation
                        + 1
                        + self.num_discrete_rotation
                    )
                    gt_lm[:, 0::7] = gt_lm[:, 0::7].masked_fill(
                        ~primitive_mask, self.pad_id
                    )
                    gt_lm[:, 1::7] = gt_lm[:, 1::7].masked_fill(
                        ~primitive_mask, self.pad_id
                    )
                    gt_lm[:, 2::7] = gt_lm[:, 2::7].masked_fill(
                        ~primitive_mask, self.pad_id
                    )
                    gt_lm[:, 3::7] = gt_lm[:, 3::7].masked_fill(
                        ~primitive_mask, self.pad_id
                    )
                    gt_lm[:, 4::7] = gt_lm[:, 4::7].masked_fill(
                        ~primitive_mask, self.pad_id
                    )
                    gt_lm[:, 5::7] = gt_lm[:, 5::7].masked_fill(
                        ~primitive_mask, self.pad_id
                    )
                    gt_lm[:, 6::7] = gt_lm[:, 6::7].masked_fill(
                        ~primitive_mask, self.pad_id
                    )
                    lm_logits = self.lm_head(pred_primitive_embeds)
                    loss_lm = F.cross_entropy(
                        rearrange(lm_logits, "b np c -> b c np"),
                        gt_lm,
                        ignore_index=self.pad_id,
                        reduction="mean",
                    )
                    loss += 0.1 * loss_lm

                if self.learnable_anchor_embedding:
                    pred_anchor_embeds = pred_primitive_embeds[:, 0::7]
                    pred_translation_embeds = pred_primitive_embeds[:, 1::7]
                    pred_flip_embeds = pred_primitive_embeds[:, 2::7]
                    pred_rotation_embeds = pred_primitive_embeds[:, 3::7]
                    pred_scale_embeds = pred_primitive_embeds[:, 4::7]
                    pred_semantic_type_embeds = pred_primitive_embeds[:, 5::7]
                    pred_primitive_type_embeds = pred_primitive_embeds[:, 6::7]

                    pred_translation_embeds, _ = pack(
                        [pred_anchor_embeds, pred_translation_embeds], "b np *"
                    )
                    pred_flip_embeds, _ = pack(
                        [pred_anchor_embeds, pred_flip_embeds], "b np *"
                    )
                    pred_rotation_embeds, _ = pack(
                        [pred_anchor_embeds, pred_rotation_embeds], "b np *"
                    )
                    pred_scale_embeds, _ = pack(
                        [pred_anchor_embeds, pred_scale_embeds], "b np *"
                    )
                    pred_semantic_type_embeds, _ = pack(
                        [pred_anchor_embeds, pred_semantic_type_embeds], "b np *"
                    )
                    pred_primitive_type_embeds, _ = pack(
                        [pred_anchor_embeds, pred_primitive_type_embeds], "b np *"
                    )

                    translation_logits = rearrange(
                        self.to_translation_logits(pred_translation_embeds),
                        "b np (c nd) -> b np c nd",
                        nd=self.num_discrete_translation,
                    )
                    flip_logits = self.to_flip_logits(pred_flip_embeds)
                    rotation_logits = rearrange(
                        self.to_rotation_logits(pred_rotation_embeds),
                        "b np (c nd) -> b np c nd",
                        nd=self.num_discrete_rotation,
                    )
                    scale_logits = rearrange(
                        self.to_scale_logits(pred_scale_embeds),
                        "b np (c nd) -> b np c nd",
                        nd=self.num_discrete_scale,
                    )
                    semantic_logits = self.to_semantic_logits(pred_semantic_type_embeds)
                    primitive_logits = self.to_primitive_logits(
                        pred_primitive_type_embeds
                    )

                    eos_logits = self.to_eos_logits(embed)
                # else:
                #     # pred_sem_embeds = pred_primitive_embeds[:, ::2]
                #     # pred_geom_embeds = pred_primitive_embeds[:, 1::2]
                #     pred_sem_embeds = pred_primitive_embeds[:, 0::6]
                #     pred_prim_embeds = pred_primitive_embeds[:, 1::6]
                #     pred_trans_embeds = pred_primitive_embeds[:, 2::6]
                #     pred_flip_embeds = pred_primitive_embeds[:, 3::6]
                #     pred_rot_embeds = pred_primitive_embeds[:, 4::6]
                #     pred_scale_embeds = pred_primitive_embeds[:, 5::6]
                #     #### 单独计算loss ####
                #     ##### 处理semantic预测 #####
                #     semantic_logits = self.to_semantic_logits(pred_sem_embeds)
                #     primitive_logits = self.to_primitive_logits(pred_prim_embeds)
                #     ##### 处理geom预测 #####
                #     translation_logits = rearrange(self.to_translation_logits(pred_trans_embeds), 'b np (c nd) -> b np c nd', nd=self.num_discrete_translation)
                #     flip_logits = self.to_flip_logits(pred_flip_embeds)
                #     rotation_logits = rearrange(self.to_rotation_logits(pred_rot_embeds), 'b np (c nd) -> b np c nd', nd=self.num_discrete_rotation)
                #     scale_logits = rearrange(self.to_scale_logits(pred_scale_embeds), 'b np (c nd) -> b np c nd', nd=self.num_discrete_scale)
                #     eos_logits = self.to_eos_logits(embed)

                loss_eos = F.binary_cross_entropy_with_logits(
                    eos_logits.squeeze(-1), gt_eos.float(), reduction="mean"
                )
                loss_semantic = F.cross_entropy(
                    rearrange(semantic_logits, "b np c -> b c np"),
                    gt_semantic,
                    ignore_index=self.pad_id,
                    reduction="mean",
                )
                loss_primitive = F.cross_entropy(
                    rearrange(primitive_logits, "b np c -> b c np"),
                    gt_primitive,
                    ignore_index=self.pad_id,
                    reduction="mean",
                )
                if self.lm_loss:
                    loss_translation_x = F.cross_entropy(
                        rearrange(translation_logits_x, "b np c -> b c np"),
                        gt_translation[:, :, 0],
                        ignore_index=self.pad_id,
                        reduction="mean",
                    )
                    loss_translation_y = F.cross_entropy(
                        rearrange(translation_logits_y, "b np c -> b c np"),
                        gt_translation[:, :, 1],
                        ignore_index=self.pad_id,
                        reduction="mean",
                    )
                    loss_translation = (loss_translation_x + loss_translation_y) / 2.0
                else:
                    loss_translation = F.cross_entropy(
                        rearrange(translation_logits, "b np n c -> b c np n"),
                        gt_translation,
                        ignore_index=self.pad_id,
                        reduction="mean",
                    )
                loss_rotation = F.cross_entropy(
                    rearrange(rotation_logits, "b np n c -> b c (np n)"),
                    gt_rotation,
                    ignore_index=self.pad_id,
                    reduction="mean",
                )
                loss_scale = F.cross_entropy(
                    rearrange(scale_logits, "b np n c -> b c (np n)"),
                    gt_scale,
                    ignore_index=self.pad_id,
                    reduction="mean",
                )
                pred_flip = flip_logits.squeeze(-1)[primitive_mask]
                loss_flip = F.binary_cross_entropy_with_logits(
                    pred_flip, gt_flip.float()[primitive_mask], reduction="mean"
                )

                loss += (
                    self.loss_weight["eos"] * loss_eos
                    + self.loss_weight["semantic_id"] * loss_semantic
                    + self.loss_weight["primitive_id"] * loss_primitive
                    + self.loss_weight["translation"] * loss_translation
                    + self.loss_weight["rotation"] * loss_rotation
                    + self.loss_weight["scale"] * loss_scale
                    + self.loss_weight["flip"] * loss_flip
                )

                huber_translation, huber_rotation, huber_scale = 0.0, 0.0, 0.0

                ### add huber loss
                if self.lm_loss:
                    probs_translation_x = translation_logits_x.softmax(-1)
                    probs_translation_y = translation_logits_y.softmax(-1)
                    pred_cont_translation_x = (
                        probs_translation_x * self.bin_centers_translation
                    ).sum(-1)[primitive_mask]
                    pred_cont_translation_y = (
                        probs_translation_y * self.bin_centers_translation
                    ).sum(-1)[primitive_mask]
                    target_cont_translation_x = self.undiscretize_translation(
                        gt_translation[:, :, 0].float()
                    )[primitive_mask]
                    target_cont_translation_y = self.undiscretize_translation(
                        gt_translation[:, :, 1].float()
                    )[primitive_mask]
                    huber_translation_x = F.smooth_l1_loss(
                        pred_cont_translation_x,
                        target_cont_translation_x,
                        reduction="mean",
                    )
                    huber_translation_y = F.smooth_l1_loss(
                        pred_cont_translation_y,
                        target_cont_translation_y,
                        reduction="mean",
                    )
                    huber_translation = (
                        huber_translation_x + huber_translation_y
                    ) / 2.0
                    pred_cont_translation = torch.stack(
                        [pred_cont_translation_x, pred_cont_translation_y], dim=-1
                    )
                    target_cont_translation = torch.stack(
                        [target_cont_translation_x, target_cont_translation_y], dim=-1
                    )
                else:
                    probs_translation = translation_logits.softmax(-1)  # (B, N, 2, K)
                    pred_cont_translation = (
                        probs_translation * self.bin_centers_translation
                    ).sum(-1)[
                        primitive_mask
                    ]  # 期望值，可导
                    target_cont_translation = self.undiscretize_translation(
                        gt_translation.float()
                    )[primitive_mask]
                    huber_translation = F.smooth_l1_loss(
                        pred_cont_translation, target_cont_translation, reduction="mean"
                    )  # β=1 默认

                probs_rotation = rotation_logits.softmax(-1)  # (B, N, 2, K)
                pred_cont_rotation = (probs_rotation * self.bin_centers_rotation).sum(
                    -1
                )[
                    primitive_mask
                ]  # 期望值，可导
                target_cont_rotation = self.undiscretize_rotation(gt_rotation.float())[
                    primitive_mask
                ]
                huber_rotation = (
                    F.smooth_l1_loss(
                        pred_cont_rotation,
                        target_cont_rotation.unsqueeze(-1),
                        reduction="mean",
                    )
                    / 180.0
                )

                probs_scale = scale_logits.softmax(-1)  # (B, N, 2, K)
                pred_cont_scale = (probs_scale * self.bin_centers_scale).sum(-1)[
                    primitive_mask
                ]  # 期望值，可导
                target_cont_scale = self.undiscretize_scale(gt_scale.float())[
                    primitive_mask
                ]
                huber_scale = (
                    F.smooth_l1_loss(
                        pred_cont_scale,
                        target_cont_scale.unsqueeze(-1),
                        reduction="mean",
                    )
                    / 1.5
                )  # β=1 默认

                loss += (
                    self.loss_weight["huber_translation"] * huber_translation
                    + self.loss_weight["huber_rotation"] * huber_rotation
                    + self.loss_weight["huber_scale"] * huber_scale
                )

                # if self.loss_weight['reconstruction'] > 0:
                loss_recon = 0
                loss_recon = self.compute_recon_loss(
                    translation_logits=(
                        translation_logits_x if self.lm_loss else translation_logits
                    ),
                    rotation_logits=rotation_logits,
                    scale_logits=scale_logits,
                    gt_primitive_id=gt_primitive,
                    gt_translation=gt_translation,
                    gt_rotation=gt_rotation,
                    gt_scale=gt_scale,
                    # gt_flip=gt_flip,
                    primitive_mask=primitive_mask,
                    pc_coord=coord,
                    pc_primitive_index=kwargs["primitive_index"],
                    pc_offset=offset,
                    gt_primitive_coords=kwargs["gt_primitive_coords"],
                    pred_gumbel_cont_translation=pred_cont_translation,
                    pred_gumbel_cont_rotation=pred_cont_rotation,
                    pred_gumbel_cont_scale=pred_cont_scale,
                    target_cont_rotation=target_cont_rotation,
                    target_cont_scale=target_cont_scale,
                    target_cont_translation=target_cont_translation,
                    pred_flip=gt_flip[primitive_mask],
                    # pred_flip = pred_flip,
                )
                loss += self.loss_weight["reconstruction"] * loss_recon
                loss_dict = {
                    "loss_semantic": loss_semantic,
                    "loss_primitive": loss_primitive,
                    "loss_translation": loss_translation,
                    "loss_rotation": loss_rotation,
                    "loss_scale": loss_scale,
                    "loss_eos": loss_eos,
                    "loss_flip": loss_flip,
                    "loss_recon": loss_recon,
                    "loss_huber_translation": huber_translation,
                    "loss_huber_rotation": huber_rotation,
                    "loss_huber_scale": huber_scale,
                    "loss_bg_primitive": bg_primitive_loss,
                    "loss_inst_mask": 0.0,
                    "loss_attn": 0.0,
                    "loss_attn_constraint": 0.0,
                    "loss_lm": loss_lm if self.lm_loss else 0.0,
                }

                return loss, loss_dict

            loss = 0.0

            ### Use the primitive_pack_embedding
            gt_semantic_embed = primitives_embeds["semantic_id"]
            gt_primitive_embed = primitives_embeds["primitive_id"]
            gt_translation_embed = primitives_embeds["translation"]
            gt_rotation_embed = primitives_embeds["rotation"]
            gt_scale_embed = primitives_embeds["scale"]
            if self.pred_flip:
                gt_flip_embed = primitives_embeds["flip_mode"]

            ### Prepare for the output logits
            # type_logits = self.to_type_logits(pred_primitive_embeds)
            semantic_logits = self.to_semantic_logits(pred_primitive_embeds)
            primitive_logits = self.to_primitive_logits(pred_primitive_embeds)
            embed_packed, _ = pack(
                [pred_primitive_embeds, gt_semantic_embed, gt_primitive_embed], "b np *"
            )
            translation_logits = rearrange(
                self.to_translation_logits(embed_packed),
                "b np (c nd) -> b np c nd",
                nd=self.num_discrete_translation,
            )
            embed_packed, _ = pack([embed_packed, gt_translation_embed], "b np *")
            if self.pred_flip:
                flip_logits = self.to_flip_logits(embed_packed)
                embed_packed, _ = pack([embed_packed, gt_flip_embed], "b np *")
            rotation_logits = rearrange(
                self.to_rotation_logits(embed_packed),
                "b np (c nd) -> b np c nd",
                nd=self.num_discrete_rotation,
            )
            embed_packed, _ = pack([embed_packed, gt_rotation_embed], "b np *")
            scale_logits = rearrange(
                self.to_scale_logits(embed_packed),
                "b np (c nd) -> b np c nd",
                nd=self.num_discrete_scale,
            )
            eos_logits = self.to_eos_logits(embed)

            ### Prepare for the GT
            # gt_type = primitives_gt['type_code']
            gt_semantic = primitives_gt["semantic_id"]
            gt_primitive = primitives_gt["primitive_id"]
            gt_translation = primitives_gt["translation"]
            gt_rotation = primitives_gt["rotation"]
            gt_scale = primitives_gt["scale"]
            gt_eos = torch.zeros_like(
                gt_semantic, dtype=torch.int64, device=gt_semantic.device
            )
            gt_eos = F.pad(gt_eos, (0, 1), value=0)
            code_lens = primitive_mask.sum(dim=-1)
            batch_arange = torch.arange(batch, device=device)
            batch_arange = rearrange(batch_arange, "... -> ... 1")
            code_lens = rearrange(code_lens, "... -> ... 1")
            gt_eos[batch_arange, code_lens] = 1

            ### Computing the loss
            # loss_types = F.cross_entropy(rearrange(type_logits, 'b np c -> b c np'),  gt_type, ignore_index = self.pad_id, reduction='mean')
            # loss_eos = F.binary_cross_entropy_with_logits(reduce(eos_logits, 'b np 1 -> b np', 'all'),  gt_eos.float(), reduction='mean')
            ### Loss_eos也需要添加一个mask padding的操作
            loss_eos = F.binary_cross_entropy_with_logits(
                eos_logits.squeeze(-1), gt_eos.float(), reduction="mean"
            )
            loss_semantic = F.cross_entropy(
                rearrange(semantic_logits, "b np c -> b c np"),
                gt_semantic,
                ignore_index=self.pad_id,
                reduction="mean",
            )
            # if not self.use_geom_embed:
            loss_primitive = F.cross_entropy(
                rearrange(primitive_logits, "b np c -> b c np"),
                gt_primitive,
                ignore_index=self.pad_id,
                reduction="mean",
            )
            # else:
            #     loss_primitive = F.cross_entropy(
            #             rearrange(pointer_logits, 'b np c -> b c np'),
            #             gt_primitive,
            #             ignore_index=self.pad_id,
            #             reduction='mean'
            #         )
            # loss_primitive = masked_cross_entropy_with_logits(primitive_logits, gt_primitive, gt_semantic, self.primitive2semantic, self.semantic2primitive, ignore_index = self.pad_id, reduction='mean')

            loss_translation = F.cross_entropy(
                rearrange(translation_logits, "b np n c -> b c np n"),
                gt_translation,
                ignore_index=self.pad_id,
                reduction="mean",
            )
            loss_rotation = F.cross_entropy(
                rearrange(rotation_logits, "b np n c -> b c (np n)"),
                gt_rotation,
                ignore_index=self.pad_id,
                reduction="mean",
            )
            loss_scale = F.cross_entropy(
                rearrange(scale_logits, "b np n c -> b c (np n)"),
                gt_scale,
                ignore_index=self.pad_id,
                reduction="mean",
            )

            loss += (
                self.loss_weight["semantic_id"] * loss_semantic
                + self.loss_weight["primitive_id"] * loss_primitive
                + self.loss_weight["translation"] * loss_translation
                + self.loss_weight["rotation"] * loss_rotation
                + self.loss_weight["scale"] * loss_scale
                + self.loss_weight["eos"] * loss_eos
            )
            loss_flip = 0
            if self.loss_weight["flip"] > 0.0 and self.pred_flip:
                gt_flip = primitives_gt["flip_mode"]
                # loss_flip = F.cross_entropy(rearrange(flip_logits, 'b np c -> b c np'),  gt_flip, ignore_index = self.pad_id, reduction='mean')
                pred_flip = flip_logits.squeeze(-1)[primitive_mask]
                loss_flip = F.binary_cross_entropy_with_logits(
                    pred_flip, gt_flip.float()[primitive_mask], reduction="mean"
                )
                loss += self.loss_weight["flip"] * loss_flip

            huber_translation, huber_rotation, huber_scale = 0.0, 0.0, 0.0

            ### add huber loss
            probs_translation = translation_logits.softmax(-1)  # (B, N, 2, K)
            pred_cont_translation = (
                probs_translation * self.bin_centers_translation
            ).sum(-1)[
                primitive_mask
            ]  # 期望值，可导
            target_cont_translation = self.undiscretize_translation(
                gt_translation.float()
            )[primitive_mask]
            huber_translation = F.smooth_l1_loss(
                pred_cont_translation, target_cont_translation, reduction="mean"
            )  # β=1 默认

            probs_rotation = rotation_logits.softmax(-1)  # (B, N, 2, K)
            pred_cont_rotation = (probs_rotation * self.bin_centers_rotation).sum(-1)[
                primitive_mask
            ]  # 期望值，可导
            target_cont_rotation = self.undiscretize_rotation(gt_rotation.float())[
                primitive_mask
            ]
            # 🔄 使用环形距离的 Huber loss，处理角度的周期性
            # if self.circular_rotate:
            #     huber_rotation = self.circular_smooth_l1_loss(pred_cont_rotation, target_cont_rotation.unsqueeze(-1), period=360.0, reduction='mean') / 180.0
            # else:
            huber_rotation = (
                F.smooth_l1_loss(
                    pred_cont_rotation,
                    target_cont_rotation.unsqueeze(-1),
                    reduction="mean",
                )
                / 180.0
            )

            probs_scale = scale_logits.softmax(-1)  # (B, N, 2, K)
            pred_cont_scale = (probs_scale * self.bin_centers_scale).sum(-1)[
                primitive_mask
            ]  # 期望值，可导
            target_cont_scale = self.undiscretize_scale(gt_scale.float())[
                primitive_mask
            ]
            huber_scale = (
                F.smooth_l1_loss(
                    pred_cont_scale, target_cont_scale.unsqueeze(-1), reduction="mean"
                )
                / 1.5
            )  # β=1 默认

            ### Predict primitives -> differentiable sampling (Gumbel-Softmax) -> sampling points -> compute chamfer distance between predicted points and gt points
            # loss = loss_types + loss_translation + loss_rotation + loss_scale + loss_eos
            # loss = loss_semantic + loss_primitive + loss_translation + loss_rotation + loss_scale + loss_eos
            loss += (
                self.loss_weight["huber_translation"] * huber_translation
                + self.loss_weight["huber_rotation"] * huber_rotation
                + self.loss_weight["huber_scale"] * huber_scale
            )

            # if self.loss_weight['reconstruction'] > 0:
            loss_recon = 0
            loss_recon = self.compute_recon_loss(
                translation_logits=translation_logits,
                rotation_logits=rotation_logits,
                scale_logits=scale_logits,
                gt_primitive_id=gt_primitive,
                gt_translation=gt_translation,
                gt_rotation=gt_rotation,
                gt_scale=gt_scale,
                # gt_flip=gt_flip,
                primitive_mask=primitive_mask,
                pc_coord=coord,
                pc_primitive_index=kwargs["primitive_index"],
                pc_offset=offset,
                gt_primitive_coords=kwargs["gt_primitive_coords"],
                pred_gumbel_cont_translation=pred_cont_translation,
                pred_gumbel_cont_rotation=pred_cont_rotation,
                pred_gumbel_cont_scale=pred_cont_scale,
                target_cont_rotation=target_cont_rotation,
                target_cont_scale=target_cont_scale,
                target_cont_translation=target_cont_translation,
                pred_flip=gt_flip[primitive_mask],
                # pred_flip = pred_flip,
            )
            loss += self.loss_weight["reconstruction"] * loss_recon

            if self.bg_pred:
                loss += bg_primitive_loss

            loss_attn = torch.zeros(
                [],
                device=attended_primitive_codes.device,
                dtype=attended_primitive_codes.dtype,
            )
            loss_dict = {
                "loss_semantic": loss_semantic,
                "loss_primitive": loss_primitive,
                "loss_translation": loss_translation,
                "loss_rotation": loss_rotation,
                "loss_scale": loss_scale,
                "loss_eos": loss_eos,
                "loss_flip": loss_flip,
                "loss_recon": loss_recon,
                "loss_huber_translation": huber_translation,
                "loss_huber_rotation": huber_rotation,
                "loss_huber_scale": huber_scale,
                "loss_bg_primitive": bg_primitive_loss,
            }

            return loss, loss_dict

    def _attr_decouple_loss_and_outputs(
        self,
        *,
        embed,
        primitive_mask,
        primitives_embeds,
        primitives_gt,
        primitive_lib_embed,
        bg_primitive_loss,
        is_train,
        coord=None,
        offset=None,
        gt_cont_scale=None,
        gt_cont_rotation=None,
        gt_cont_flip=None,
        recon_source_segments=None,
        recon_source_mask=None,
        recon_valid_mask=None,
        recon_full_translation=None,
        recon_clip_count_ratio=None,
        recon_clip_length_ratio=None,
        recon_is_partial=None,
        recon_preclip_source_segments=None,
        recon_preclip_source_mask=None,
    ):
        """Loss/evaluation head for the attribute-decoupled decoder.

        ``embed`` contains one boundary state followed by either six baseline
        tokens or five semantic-free tokens per padded instance. Boundary
        states are the only states supervised by EOS; attribute losses are
        evaluated on their corresponding preceding state.
        """
        del primitives_embeds, coord, offset
        stride = self.num_attr_tokens
        expected_stride = 6 if self.attr_emit_semantic_token else 5
        if stride != expected_stride:
            raise RuntimeError(
                f"attr_decouple expects {expected_stride} tokens, got {stride}"
            )

        pred_h = embed[:, :-1]
        if self.attr_emit_semantic_token:
            h_sem = pred_h[:, 0::stride]
            h_prim = pred_h[:, 1::stride]
            first_geom_index = 2
        else:
            # With no serialized semantic token, the instance-boundary state
            # directly predicts primitive. B can still supervise semantic from
            # this same state without feeding a semantic token into history.
            h_sem = pred_h[:, 0::stride]
            h_prim = h_sem
            first_geom_index = 1
        h_trans = pred_h[:, first_geom_index::stride]
        h_flip = pred_h[:, first_geom_index + 1 :: stride]
        h_rot = pred_h[:, first_geom_index + 2 :: stride]
        h_shape = pred_h[:, first_geom_index + 3 :: stride]

        gt_semantic = primitives_gt["semantic_id"]
        gt_primitive = primitives_gt["primitive_id"]
        gt_translation = primitives_gt["translation"]
        gt_rotation = primitives_gt["rotation"]
        gt_scale = primitives_gt["scale"]
        gt_size = primitives_gt["size"]
        gt_flip = primitives_gt["flip_mode"]

        semantic_logits = (
            self.to_semantic_logits(h_sem) if self.attr_semantic_supervision else None
        )
        if self.attr_primitive_semantic_condition == "gt":
            safe_semantic = gt_semantic.masked_fill(~primitive_mask, 0)
            semantic_cond = self.semantic_embed(safe_semantic).masked_fill(
                ~primitive_mask.unsqueeze(-1), 0.0
            )
            primitive_head_input = torch.cat([h_prim, semantic_cond], dim=-1)
        else:
            primitive_head_input = h_prim
        if self.primitive_contrastive_learning:
            pred_primitive_embeds = self.pred_proj(primitive_head_input)
            lib_primitive_embeds = self.lib_proj(primitive_lib_embed)
            pred_norm = F.normalize(pred_primitive_embeds, p=2, dim=-1)
            lib_norm = F.normalize(lib_primitive_embeds, p=2, dim=-1)
            logit_scale = self.logit_scale.exp().clamp(max=self.max_temperature)
            primitive_logits = logit_scale * torch.einsum(
                "bie,bpe->bip", pred_norm, lib_norm
            )
        else:
            primitive_logits = self.to_primitive_logits(primitive_head_input)

        if self.factorized_xy:
            translation_logits = self._factorized_2ch_logits(
                h_trans,
                self.to_trans_x_logits,
                self.to_trans_y_logits,
                self.translation_embed,
                cond_x_idx=gt_translation[..., 0].clamp(
                    min=0, max=self.num_discrete_translation - 1
                ),
            )
        else:
            translation_logits = rearrange(
                self.to_translation_logits(h_trans),
                "b np (c nd) -> b np c nd",
                nd=self.num_discrete_translation,
            )

        flip_logits = self.to_flip_logits(h_flip)
        rotation_logits = rearrange(
            self.to_rotation_logits(h_rot),
            "b np nd -> b np 1 nd",
        )
        if self.size_pred:
            if self.factorized_xy:
                size_logits = self._factorized_2ch_logits(
                    h_shape,
                    self.to_size_w_logits,
                    self.to_size_h_logits,
                    self.size_embed,
                    cond_x_idx=gt_size[..., 0].clamp(
                        min=0, max=self.num_discrete_size - 1
                    ),
                )
            else:
                size_logits = rearrange(
                    self.to_size_logits(h_shape),
                    "b np (c nd) -> b np c nd",
                    nd=self.num_discrete_size,
                )
            scale_logits = None
            shape_logits = size_logits
            gt_shape = gt_size
            shape_centers = self.bin_centers_size
            undiscretize_shape = self.undiscretize_size
        else:
            size_logits = None
            scale_logits = rearrange(
                self.to_scale_logits(h_shape),
                "b np (c nd) -> b np c nd",
                nd=self.num_discrete_scale,
            )
            shape_logits = scale_logits
            gt_shape = gt_scale
            shape_centers = self.bin_centers_scale
            undiscretize_shape = self.undiscretize_scale

        zero = embed.sum() * 0.0
        primitive_loss_parts = {
            "loss_fine": zero.detach(),
            "loss_coarse": zero.detach(),
        }
        if primitive_mask.any():
            if self.attr_semantic_supervision:
                loss_semantic = F.cross_entropy(
                    rearrange(semantic_logits, "b np c -> b c np"),
                    gt_semantic,
                    ignore_index=self.pad_id,
                )
            else:
                loss_semantic = zero
            if self.primitive_contrastive_learning:
                loss_primitive, primitive_loss_parts = (
                    self.Hierarchical_primitive_criterion(
                        primitive_logits, gt_primitive, gt_semantic
                    )
                )
            else:
                loss_primitive = F.cross_entropy(
                    rearrange(primitive_logits, "b np c -> b c np"),
                    gt_primitive,
                    ignore_index=self.pad_id,
                )
            loss_translation = F.cross_entropy(
                rearrange(translation_logits, "b np n c -> b c np n"),
                gt_translation,
                ignore_index=self.pad_id,
            )
            if (
                self.rotation_symmetry_mode != "off"
                and self.rotation_circular_label_sigma > 0
            ):
                loss_rotation = self.soft_ce_with_kernel(
                    rotation_logits,
                    gt_rotation.unsqueeze(-1),
                    self.kernel_rotation_symmetry,
                    self.pad_id,
                )
            else:
                loss_rotation = F.cross_entropy(
                    rearrange(rotation_logits, "b np n c -> b c (np n)"),
                    gt_rotation,
                    ignore_index=self.pad_id,
                )
            loss_shape = F.cross_entropy(
                rearrange(shape_logits, "b np n c -> b c np n"),
                gt_shape,
                ignore_index=self.pad_id,
            )
            loss_flip = F.binary_cross_entropy_with_logits(
                flip_logits.squeeze(-1)[primitive_mask],
                gt_flip.float()[primitive_mask],
            )
        else:
            loss_semantic = loss_primitive = loss_translation = zero
            loss_rotation = loss_shape = loss_flip = zero

        # EOS exists only at instance boundaries, never between attributes.
        eos_logits_all = self.to_eos_logits(embed).squeeze(-1)
        eos_logits = eos_logits_all[:, ::stride]
        num_instances = primitive_mask.sum(dim=-1)
        boundary_ids = torch.arange(eos_logits.shape[1], device=embed.device)
        eos_valid = boundary_ids.unsqueeze(0) <= num_instances.unsqueeze(1)
        eos_target = boundary_ids.unsqueeze(0) == num_instances.unsqueeze(1)
        loss_eos = F.binary_cross_entropy_with_logits(
            eos_logits[eos_valid], eos_target.float()[eos_valid]
        )

        if getattr(self, "continuous_decode_mode", "legacy") == "soft_expectation":
            (
                pred_cont_translation_all,
                pred_cont_rotation_phase_all,
                pred_cont_rotation_all,
                pred_cont_shape_all,
            ) = self._attr_soft_decode_continuous(
                translation_logits,
                rotation_logits,
                shape_logits,
                shape_centers,
                gt_primitive,
            )
            # Share the same tensors and graph: no second decode or sampling.
            recon_pred_translation_all = pred_cont_translation_all
            recon_pred_rotation_phase_all = pred_cont_rotation_phase_all
            recon_pred_rotation_all = pred_cont_rotation_all
            recon_pred_scale_all = pred_cont_shape_all
        else:
            trans_probs = (
                F.gumbel_softmax(translation_logits, tau=1.0, hard=True, dim=-1)
                if self.gumbel_loss and self.training
                else translation_logits.softmax(-1)
            )
            # Phase experiments deliberately use one deterministic differentiable
            # circular decoder for train, validation, direct Huber, and recon.  The
            # force_one control receives the exact same decoder as audited N.
            rot_probs = (
                rotation_logits.softmax(-1)
                if self.rotation_symmetry_mode != "off"
                else (
                    F.gumbel_softmax(rotation_logits, tau=1.0, hard=True, dim=-1)
                    if self.gumbel_loss and self.training
                    else rotation_logits.softmax(-1)
                )
            )
            shape_probs = (
                F.gumbel_softmax(shape_logits, tau=1.0, hard=True, dim=-1)
                if self.gumbel_loss and self.training
                else shape_logits.softmax(-1)
            )
            pred_cont_translation_all = (
                trans_probs * self.bin_centers_translation
            ).sum(-1)
            pred_cont_rotation_phase_all = self._stable_circular_rotation_expectation(
                rot_probs
            )
            pred_cont_rotation_all = self._rotation_phase_to_physical(
                pred_cont_rotation_phase_all, gt_primitive
            )
            pred_cont_shape_all = (shape_probs * shape_centers).sum(-1)
            # Reconstruction either retains the historical deterministic soft
            # expectation or uses deterministic straight-through hard argmax.  The
            # latter matches formal AR's forward bins without Gumbel noise.
            recon_trans_probs = self._attr_recon_decode_probabilities(
                translation_logits
            )
            recon_rot_probs = self._attr_recon_decode_probabilities(rotation_logits)
            # Shape decoding intentionally stays unchanged.  In the current R2
            # protocol reconstruction uses detached GT scale, so predicted scale
            # has no reconstruction-loss gradient.
            recon_shape_probs = shape_logits.softmax(-1)
            recon_pred_translation_all = (
                recon_trans_probs * self.bin_centers_translation
            ).sum(-1)
            recon_pred_rotation_phase_all = self._stable_circular_rotation_expectation(
                recon_rot_probs
            )
            recon_pred_rotation_all = self._rotation_phase_to_physical(
                recon_pred_rotation_phase_all, gt_primitive
            )
            if not self.size_pred and self.scale_log:
                recon_pred_scale_all = torch.exp(
                    (
                        recon_shape_probs
                        * torch.log(self.bin_centers_scale.clamp_min(1e-8))
                    ).sum(-1)
                )
            else:
                recon_pred_scale_all = (recon_shape_probs * shape_centers).sum(-1)
        # The CE target is necessarily clamped to the configured bins, whereas
        # the continuous auxiliary losses must retain the original augmented
        # transform.  The reconstruction payload carries that full translation
        # after all augmentations and before any visibility recentering.
        target_cont_translation_all = (
            recon_full_translation.float()
            if recon_full_translation is not None
            else self.undiscretize_translation(gt_translation.float())
        )
        raw_target_cont_rotation_all = (
            gt_cont_rotation.float()
            if gt_cont_rotation is not None
            else self.undiscretize_rotation(gt_rotation.float())
        )
        target_cont_rotation_phase_all = self._rotation_gt_phase(
            raw_target_cont_rotation_all, gt_primitive
        )
        target_cont_rotation_all = self._rotation_gt_to_physical(
            raw_target_cont_rotation_all, gt_primitive
        )
        target_cont_shape_all = (
            gt_cont_scale.float()
            if (not self.size_pred and gt_cont_scale is not None)
            else undiscretize_shape(gt_shape.float())
        )
        if not self.size_pred and self.scale_log:
            # A linear expectation of log-spaced scale centers is biased toward
            # large bins.  Use exp(E[log(scale)]) consistently for continuous
            # scale supervision and for reconstruction.
            pred_cont_shape_all = recon_pred_scale_all

        if primitive_mask.any():
            huber_translation = F.smooth_l1_loss(
                pred_cont_translation_all[primitive_mask],
                target_cont_translation_all[primitive_mask],
            )
            rotation_target_phase = target_cont_rotation_phase_all[
                primitive_mask
            ].unsqueeze(-1)
            rotation_residual_phase = (
                pred_cont_rotation_phase_all[primitive_mask]
                - rotation_target_phase
                + 180.0
            ).remainder(360.0) - 180.0
            rotation_orders = (
                self.rotation_symmetry_orders[
                    gt_primitive[primitive_mask]
                    .long()
                    .clamp(min=0, max=self.num_primitive - 1)
                ]
                .to(rotation_residual_phase.dtype)
                .unsqueeze(-1)
            )
            rotation_residual = rotation_residual_phase / rotation_orders
            huber_rotation = (
                F.smooth_l1_loss(
                    rotation_residual,
                    torch.zeros_like(rotation_residual),
                )
                / 180.0
            )
            if not self.size_pred and self.scale_log:
                huber_shape = F.smooth_l1_loss(
                    torch.log(pred_cont_shape_all[primitive_mask].clamp_min(1e-8)),
                    torch.log(target_cont_shape_all[primitive_mask].clamp_min(1e-8)),
                )
            else:
                huber_shape = F.smooth_l1_loss(
                    pred_cont_shape_all[primitive_mask],
                    target_cont_shape_all[primitive_mask],
                )
            if not self.size_pred:
                huber_shape = huber_shape / 1.5
        else:
            huber_translation = huber_rotation = huber_shape = zero

        recon_stats = {
            "candidate": 0,
            "geometry_valid": 0,
            "partial_valid": 0,
            "full_valid": 0,
            "loss_full": zero,
            "loss_partial": zero,
            "full_effective_count": zero.detach(),
            "partial_effective_count": zero.detach(),
            "loss_translation": zero,
            "loss_rotation": zero,
            "rotation_valid": 0,
            "rotation_effective_count": zero.detach(),
        }
        if self.loss_weight["reconstruction"] > 0:
            loss_recon, recon_stats = self._compute_attr_full_recon_loss(
                pred_translation=recon_pred_translation_all,
                pred_rotation=recon_pred_rotation_all,
                pred_scale=(None if self.size_pred else recon_pred_scale_all),
                gt_primitive_id=gt_primitive,
                gt_scale=gt_cont_scale,
                gt_rotation=target_cont_rotation_all,
                gt_flip=gt_cont_flip,
                primitive_mask=primitive_mask,
                recon_source_segments=recon_source_segments,
                recon_source_mask=recon_source_mask,
                recon_valid_mask=recon_valid_mask,
                recon_full_translation=recon_full_translation,
                recon_clip_count_ratio=recon_clip_count_ratio,
                recon_clip_length_ratio=recon_clip_length_ratio,
                recon_is_partial=recon_is_partial,
                recon_preclip_source_segments=(recon_preclip_source_segments),
                recon_preclip_source_mask=recon_preclip_source_mask,
            )
        else:
            loss_recon = zero
        recon_weight = self.loss_weight["reconstruction"] * self.recon_weight_factor()
        loss = (
            self.loss_weight["semantic_id"] * loss_semantic
            + self.loss_weight["primitive_id"] * loss_primitive
            + self.loss_weight["translation"] * loss_translation
            + self.loss_weight["rotation"] * loss_rotation
            + self.loss_weight["scale"] * loss_shape
            + self.loss_weight["flip"] * loss_flip
            + self.loss_weight["eos"] * loss_eos
            + self.loss_weight["huber_translation"] * huber_translation
            + self.loss_weight["huber_rotation"] * huber_rotation
            + self.loss_weight["huber_scale"] * huber_shape
            + recon_weight * loss_recon
        )
        if self.bg_pred:
            loss = loss + bg_primitive_loss

        loss_dict = {
            "loss_semantic": loss_semantic,
            "loss_primitive": loss_primitive,
            "loss_primitive_fine": primitive_loss_parts.get("loss_fine", zero),
            "loss_primitive_coarse": primitive_loss_parts.get("loss_coarse", zero),
            "loss_translation": loss_translation,
            "loss_rotation": loss_rotation,
            "loss_scale": loss_shape,
            "loss_eos": loss_eos,
            "loss_flip": loss_flip,
            "loss_recon": loss_recon,
            "loss_recon_full": recon_stats["loss_full"],
            "loss_recon_partial": recon_stats["loss_partial"],
            "loss_recon_translation": recon_stats.get("loss_translation", zero),
            "loss_recon_rotation": recon_stats.get("loss_rotation", zero),
            "recon_weight": zero + recon_weight,
            "recon_partial_weight": zero + self.recon_partial_weight,
            "recon_rotation_weight": (
                zero + getattr(self, "recon_rotation_weight", 1.0)
            ),
            "recon_candidate_instances": zero + recon_stats["candidate"],
            "recon_valid_instances": zero + recon_stats["geometry_valid"],
            "recon_partial_instances": zero + recon_stats.get("partial_valid", 0),
            "recon_full_instances": zero + recon_stats.get("full_valid", 0),
            "recon_rotation_valid_instances": (
                zero + recon_stats.get("rotation_valid", 0)
            ),
            "recon_rotation_effective_count": (
                zero + recon_stats.get("rotation_effective_count", 0)
            ),
            "recon_partial_effective_count": (
                zero + recon_stats["partial_effective_count"]
            ),
            "recon_full_effective_count": (zero + recon_stats["full_effective_count"]),
            "loss_huber_translation": huber_translation,
            "loss_huber_rotation": huber_rotation,
            "loss_huber_scale": huber_shape,
            "loss_bg_primitive": bg_primitive_loss,
            "loss_inst_mask": zero,
            "loss_attn": zero,
            "loss_attn_constraint": zero,
        }
        if is_train:
            return loss, loss_dict

        pred_eos = torch.sigmoid(eos_logits) > 0.5
        pred_mask = ~(pred_eos.float().cumsum(dim=-1) >= 1)[:, :-1]
        pred_primitive = primitive_logits.argmax(dim=-1)
        if self.attr_semantic_supervision:
            pred_semantic = semantic_logits.argmax(dim=-1)
        else:
            pred_semantic = self.Hierarchical_primitive_criterion.primitive_id2semantic[
                pred_primitive.clamp(min=0, max=self.num_primitive - 1)
            ]
        pred_discrete_translation = translation_logits.argmax(dim=-1)
        pred_discrete_rotation = rotation_logits.argmax(dim=-1).squeeze(-1)
        pred_discrete_rotation_phase = self.undiscretize_rotation(
            pred_discrete_rotation.float()
        )
        pred_discrete_cont_rotation = self._rotation_phase_to_physical(
            pred_discrete_rotation_phase, gt_primitive
        )
        pred_discrete_shape = shape_logits.argmax(dim=-1)
        pred_flip = (torch.sigmoid(flip_logits.squeeze(-1)) > 0.5).long()

        return {
            "loss": loss,
            **loss_dict,
            "gt_semantic": gt_semantic,
            "gt_primitive": gt_primitive,
            "gt_discrete_translation": gt_translation,
            "gt_discrete_rotation": gt_rotation,
            "gt_discrete_scale": gt_scale,
            "gt_cont_translation": target_cont_translation_all,
            "gt_cont_rotation": target_cont_rotation_all,
            "gt_cont_rotation_phase": target_cont_rotation_phase_all,
            "gt_cont_scale": (
                gt_cont_scale.float()
                if gt_cont_scale is not None
                else self.undiscretize_scale(gt_scale.float())
            ),
            "gt_cont_size": (
                self.undiscretize_size(gt_size.float()) if self.size_pred else None
            ),
            "gt_flip": gt_flip,
            "gt_size": gt_size if self.size_pred else None,
            "gt_mask": primitive_mask,
            "pred_mask": pred_mask,
            "pred_semantic": pred_semantic,
            "pred_primitive": pred_primitive,
            "pred_discrete_translation": pred_discrete_translation,
            "pred_discrete_rotation": pred_discrete_rotation,
            "pred_discrete_scale": None if self.size_pred else pred_discrete_shape,
            "pred_discrete_size": pred_discrete_shape if self.size_pred else None,
            "pred_discrete_cont_translation": self.undiscretize_translation(
                pred_discrete_translation.float()
            ),
            "pred_discrete_cont_rotation": pred_discrete_cont_rotation,
            "pred_discrete_cont_rotation_phase": pred_discrete_rotation_phase,
            "pred_discrete_cont_scale": (
                None
                if self.size_pred
                else self.undiscretize_scale(pred_discrete_shape.float())
            ),
            "pred_discrete_cont_size": (
                self.undiscretize_size(pred_discrete_shape.float())
                if self.size_pred
                else None
            ),
            "pred_cont_translation": pred_cont_translation_all,
            "pred_cont_rotation": pred_cont_rotation_all,
            "pred_cont_rotation_phase": pred_cont_rotation_phase_all,
            "pred_cont_scale": None if self.size_pred else pred_cont_shape_all,
            "pred_cont_size": pred_cont_shape_all if self.size_pred else None,
            "pred_flip": pred_flip,
            "semantic_logits": semantic_logits,
            "primitive_logits": primitive_logits,
            "translation_logits": translation_logits,
            "rotation_logits": rotation_logits,
            "scale_logits": scale_logits,
            "size_logits": size_logits,
            "eos_logits": eos_logits,
            "flip_logits": flip_logits,
            "pred_primitive_bg": None,
            "gt_primitive_bg": None,
            "bg_valid_mask": None,
        }

    def compute_primitive_bg_loss_fast(
        self, pred_bg_embed, pc_embed, pc_embed_key_padding_mask, sem_ids, pc_cu_seqlens
    ):
        """
        向量化版本：移除 For 循环，大幅提升 GPU 利用率
        """
        # 1. 预测值计算 & 展平
        # pred_bg_embed: (B, num_bg_classes, dim)
        # pc_embed: (B, max_num_prims, dim)
        # pred_primitive_bg: (B, max_num_prims, num_bg_classes)
        pred_primitive_bg = torch.einsum("bpd,bcd->bpc", pc_embed, pred_bg_embed)

        num_bg_classes = pred_bg_embed.shape[1]

        # 2. 利用 Mask 提取有效预测 (Filter Valid Primitives)
        # 根据你的定义：True 表示有效位 (Valid)
        # mask_flat: (B * max_num_prims)
        mask_flat = pc_embed_key_padding_mask.flatten()

        # pred_flat: (B * max_num_prims, num_bg_classes)
        pred_flat = pred_primitive_bg.flatten(0, 1)

        # pred_valid: (Total_Valid_Prims, num_bg_classes)
        # 这里直接选出了所有 Batch 中有效的点，拼接在一起
        pred_valid = pred_flat[mask_flat]

        # 3. 构建 Target (Vectorized)
        # sem_ids 已经是 packed 的 (Total_Valid_Prims,)，直接使用即可，无需切片
        # 注意：确保 sem_ids 的长度等于 mask_flat.sum()，这在你的数据 pipeline 中应该是一致的

        # 将 Semantic ID 映射到 Background ID (0 ~ 4)
        target_cls = sem_ids - 30

        # 找出属于背景的 Mask (Valid Background Primitives)
        # 假设背景类 ID 是 30-34
        is_bg = (target_cls >= 0) & (target_cls < num_bg_classes)

        # 初始化全 0 的 One-hot Target
        # target_onehot: (Total_Valid_Prims, num_bg_classes)
        target_onehot = torch.zeros_like(pred_valid)

        # 只有那些确实是背景的点，才将其对应的类别位置置为 1
        # 利用 boolean indexing 进行快速赋值
        if is_bg.any():
            # 选出是背景的那些行
            bg_rows = pred_valid[is_bg]
            # 选出对应的类别索引
            bg_cols = target_cls[is_bg]

            # 构造索引进行赋值
            # 我们需要设置 target_onehot[i, c] = 1，其中 i 是 is_bg 为 True 的索引
            # 使用 scatter 或者高级索引
            # 这种写法最高效：直接在筛选出的视图上操作可能不生效，所以用全局索引
            valid_indices = torch.nonzero(is_bg).squeeze(1)
            target_onehot[valid_indices, bg_cols] = 1.0

        # 4. 计算 Loss
        # 你的 JIT 函数期望输入形状为 (num_classes, N_points)
        # 因此我们需要转置 (Transpose)
        pred_t = pred_valid.t()  # (num_bg_classes, Total_Valid_Prims)
        target_t = target_onehot.t()  # (num_bg_classes, Total_Valid_Prims)

        # 计算 Loss
        # 注意：这里计算的是整个 Batch 所有点的平均 Loss
        # 如果你想严格保持 "先对每个 Sample 求平均，再对 Batch 求平均"，向量化会比较麻烦。
        # 但通常在 Point Cloud 任务中，直接对 Total Points 求平均更稳定。

        # 替换为你现在的 Focal Loss (强烈建议) 或原本的 CE
        # loss_ce = sigmoid_focal_loss_jit(pred_t, target_t, num_bg_classes)
        # loss_ce = sigmoid_ce_loss_jit(pred_t, target_t, num_bg_classes)
        loss_ce = sigmoid_focal_loss_jit(pred_t, target_t, num_bg_classes)
        loss_dice = dice_loss_jit(pred_t, target_t, num_bg_classes)

        # 返回 total loss 和原始预测 (用于 debug 或可视化)
        return loss_ce + loss_dice, pred_primitive_bg

    def compute_primitive_bg_loss_hungarian_vec(
        self, pred_bg_embed, pc_embed, pc_embed_key_padding_mask, sem_ids, pc_cu_seqlens
    ):
        """
        向量化版本的 Loss 计算
        """
        batch, num_bg_classes, dim = pred_bg_embed.shape

        # 1. Prediction (Batch 级)
        pred_logits = self.bg_class_pred(pred_bg_embed)  # (B, Q, num_classes + 1)
        # (B, Q, max_num_prims)
        pred_masks = torch.einsum("bcd,bpd->bcp", pred_bg_embed, pc_embed)

        # 保存用于可视化的结果
        pred_primitive_bg = pred_masks.transpose(1, 2)

        # 2. Target Construction (Batch 级) & Prepare for Matcher
        targets = []

        for b_i in range(batch):
            # 获取有效位
            valid_mask = pc_embed_key_padding_mask[b_i]  # True = Valid
            # 校验数据对齐 (Debug 用)
            seq_len = pc_cu_seqlens[b_i + 1] - pc_cu_seqlens[b_i]
            assert (
                seq_len == valid_mask.sum()
            ), f"Data Mismatch: pc_cu_seqlens says {seq_len}, mask says {valid_mask.sum()}"

            # 获取当前 sample 的 primitives
            start_idx = pc_cu_seqlens[b_i].item()
            end_idx = pc_cu_seqlens[b_i + 1].item()
            sem_ids_bi = sem_ids[start_idx:end_idx]  # (N_valid,)

            # 构建 Labels (Semantic -> Instance style)
            # 假设 30-34 是背景
            bg_mask_bool = (sem_ids_bi >= 30) & (sem_ids_bi <= 34)
            if bg_mask_bool.sum() == 0:
                targets.append(
                    {
                        "labels": torch.tensor(
                            [], device=pred_bg_embed.device, dtype=torch.long
                        ),
                        "masks": torch.tensor([], device=pred_bg_embed.device),
                    }
                )
                continue

            valid_bg_ids = sem_ids_bi[bg_mask_bool] - 30
            unique_labels = torch.unique(valid_bg_ids).to(torch.long)

            # 构建 Target Masks
            # t_masks: (num_targets, N_valid)
            num_valid = valid_mask.sum().item()
            t_masks = torch.zeros(
                (len(unique_labels), num_valid),
                dtype=pred_masks.dtype,
                device=pred_masks.device,
            )

            for i, label in enumerate(unique_labels):
                # 这里的逻辑是：找出所有等于 label 的点，置为 1
                # 注意：sem_ids_bi 已经是只包含 Valid 点的了，所以可以直接对应
                t_masks[i] = ((sem_ids_bi - 30) == label).float()

            targets.append(
                {
                    "labels": unique_labels,  # (num_gt,)
                    "masks": t_masks,  # (num_gt, N_valid_points)
                }
            )

        # 3. Perform Matching (Batch 级调用)
        # 注意：Matcher 内部还是会有循环，但这是无法避免的（除非不用 scipy）
        # 我们传递 "masked" 的 pred_masks 给 matcher 可能会很麻烦因为每个 sample 点数不一样
        # 技巧：我们只在 Matcher 内部做 slice，外部保持 Padding 形式传进去可能比较难
        # 修正：为了配合 variable length masks，我们最好还是在 Matcher 外部把 Mask 准备好
        # 这里为了简化，我们仅传给 Matcher 需要的信息。
        # 由于每个 sample 的点数不一样，直接传 pred_masks (B, Q, Max) 是最方便的，
        # 但 Targets 里的 masks 是 (G, Valid)。
        # -> 需要在 Matcher 里把 pred_masks[b] 用 valid_mask 切片。

        # 临时修改 Matcher 接口不太好，我们这里手动做一下适配：
        matcher_outputs_masks = []
        for b_i in range(batch):
            matcher_outputs_masks.append(
                pred_masks[b_i][:, pc_embed_key_padding_mask[b_i]]
            )

        # Hack: 修改 Matcher 让它接受 List[Tensor] 作为 pred_masks 可能会破坏结构
        # 简单做法：Matcher 里加一行 valid_mask 处理。
        # 假设我们稍微改一下 Matcher 逻辑 (见上文建议)，或者在这里传入 slice 好的

        # 咱们沿用你原本的逻辑，但在外部 match 完
        # 为了不改动 Matcher 太大，我们这里还是得循环调用 Matcher 或者修改 Matcher 内部
        # 鉴于 Matcher 必须 loop，我们就在这里 loop 匹配，拿到 indices

        indices = []
        for b_i in range(batch):
            if len(targets[b_i]["labels"]) == 0:
                indices.append(
                    (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
                )
                continue

            # 准备单个 sample 的 output
            # 只取有效的 points 参与计算 cost
            valid_mask = pc_embed_key_padding_mask[b_i]
            out_m = pred_masks[b_i][:, valid_mask]  # (Q, N_valid)
            out_p = pred_logits[b_i : b_i + 1]  # (1, Q, C+1)

            tgt_t = [targets[b_i]]  # list of 1 dict

            # 临时构造 output dict
            curr_out = {"pred_logits": out_p, "pred_masks": out_m.unsqueeze(0)}

            # 调用 Matcher (注意 Matcher 现在需要能处理 1 个 batch)
            match_res = self.bg_matcher(curr_out, tgt_t)
            indices.append(match_res[0])

        # 4. Compute Loss (Vectorized - 核心优化)
        # 我们要将 batch 中所有的正样本 Gather 起来，一次性算 Loss

        # 4.1 Classification Loss (所有 Query 都参与)
        # 构建 Target Classes (B, Q) 初始化为 "No Object" (最后一类)
        target_classes = torch.full(
            pred_logits.shape[:2],
            num_bg_classes,  # no_object class index
            dtype=torch.long,
            device=pred_logits.device,
        )

        # 填入匹配到的类别
        # idx 是 (batch_idx, query_idx) 的元组
        idx = self._get_src_permutation_idx(indices)

        # target_classes_o: 把 batch 里所有 target label 拼起来
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )

        # 赋值：只有匹配到的 Query 才有具体的背景类别，其他保持 No Object
        target_classes[idx] = target_classes_o

        # 计算全局 CE Loss (一次性)
        # pred_logits: (B, Q, C+1) -> flatten -> (B*Q, C+1)
        # target_classes: (B, Q) -> flatten -> (B*Q)
        empty_weight = torch.ones(num_bg_classes + 1, device=pred_logits.device)
        empty_weight[-1] = 0.1  # no-object 类的权重较低
        loss_ce = F.cross_entropy(
            pred_logits.transpose(1, 2),  # (B, C+1, Q) expected by CE if multi-dim
            target_classes,
            empty_weight,  # 预先定义好权重
        )

        # 4.2 Mask Loss (只计算匹配到的 Query)
        # src_masks: 选出所有匹配的 Query 预测的 Mask
        # 这比较麻烦，因为每个 Sample 的点数不一样，不能简单的 Stack。
        # 必须把它们 Flatten 成一个巨大的 1D 向量或者 List 处理。
        # 但为了利用 batch_dice_loss_jit，我们需要 Tensor。

        # 方案：由于点数不定，我们只能将所有 Valid Points 拼成一条长龙？
        # 不行，Mask Loss 是 (N, Points)。
        # 妥协方案：在这里循环 batch 计算 Mask Loss 是可以接受的，因为无法 batching (Different shapes)。
        # 或者是：Cat all (Matched Query, Valid Points) pairs? No, points count differs.

        total_mask_loss = 0
        total_dice_loss = 0
        num_total_pos = 0

        for b_i, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue

            valid_mask = pc_embed_key_padding_mask[b_i]

            # 取出匹配的预测 Mask (Num_Matched, N_valid)
            src_masks = pred_masks[b_i][src_idx][:, valid_mask]
            # 取出匹配的目标 Mask (Num_Matched, N_valid)
            tgt_masks = targets[b_i]["masks"][tgt_idx]

            # 这里的 batch_dice_loss_jit 只需要处理 (M, N) 的输入，非常快
            num_masks = len(src_idx)
            total_mask_loss += (
                sigmoid_ce_loss_jit(src_masks, tgt_masks, 1) * num_masks
            )  # 先乘回来，最后除总数
            total_dice_loss += dice_loss_jit(src_masks, tgt_masks, 1) * num_masks
            num_total_pos += num_masks

        if num_total_pos > 0:
            total_mask_loss = total_mask_loss / num_total_pos
            total_dice_loss = total_dice_loss / num_total_pos

        total_loss = loss_ce + 2.0 * total_mask_loss + 2.0 * total_dice_loss  # 权重可调

        return total_loss, pred_primitive_bg, loss_ce

    def _get_src_permutation_idx(self, indices):
        # 辅助函数：将 list of (src, tgt) 转为 tensor 索引
        # return: (batch_idx_tensor, src_idx_tensor)
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(
        self,
        *,  ### represent the input of the model must be input with arguments
        scale: Float["b np 3"],
        rotation: Float["b np 3"],
        translation: Float["b np 3"],
        # type_code: Int['b np'],
        semantic_id: Int["b np"],
        primitive_id: Int["b np"],
        flip_mode: Int["b np"],
        cache: LayerIntermediates | None = None,
        pc: Tensor | None = None,
        pc_embed: Tensor | None = None,
        loss_reduction: str = "mean",
        return_cache=False,
        return_loss=True,
        append_eos=False,
        coord: Tensor | None = None,
        feat: Tensor | None = None,
        offset: Tensor | None = None,
        grid_coord: Tensor | None = None,
        img: Tensor | None = None,
        is_train: bool = True,
        **kwargs,
    ):
        return self.forward_on_codes(
            scale=scale,
            rotation=rotation,
            translation=translation,
            # type_code = type_code,
            semantic_id=semantic_id,
            primitive_id=primitive_id,
            flip_mode=flip_mode,
            cache=cache,
            pc=pc,
            pc_embed=pc_embed,
            loss_reduction=loss_reduction,
            return_cache=return_cache,
            return_loss=return_loss,
            append_eos=append_eos,
            coord=coord,
            feat=feat,
            offset=offset,
            grid_coord=grid_coord,
            img=img,
            is_train=is_train,
            **kwargs,
        )

    @torch.no_grad()
    def _compute_pred_primitive_bg_for_generate(
        self, pc_embed, pc_mask, plib, pooled_pc_embed
    ):
        """
        Match ``forward`` background probabilities (B, P, num_bg_classes) for
        prefix ``[lib | bg implicit | sos]`` with cross-attention. Returns zeros
        when the architecture is not the cross-attn, non-perceiver path.
        """
        if not self.bg_pred or pc_embed is None:
            return None
        B, P, _D = pc_embed.shape
        device = pc_embed.device
        if (
            not (self.use_prefix_decoder and self.shape_cond_with_cross_attn)
            or self.shape_cond_with_perceiver_prefix
        ):
            return torch.zeros(
                B, P, self.num_bg_classes, device=device, dtype=pc_embed.dtype
            )
        batch = B
        sos0 = repeat(self.sos_token, "n d -> b n d", b=batch)
        bg_implicit = self.bg_implicit_primitive.weight[None, ...].repeat(batch, 1, 1)
        len_bg = bg_implicit.shape[1]
        seq, _ = pack([bg_implicit, sos0], "b * d")
        if plib is not None:
            if self.primitive_cond_with_cat:
                pl = plib + self.src_type_embed(
                    torch.ones((batch, 1), dtype=torch.long, device=device)
                )
            else:
                pl = plib
            len_pl = pl.shape[1]
            seq, _ = pack([pl, seq], "b * d")
        else:
            len_pl = 0
        empty = torch.empty(batch, 0, self.dim, device=device, dtype=seq.dtype)
        primitive_codes, _ = pack([seq, empty], "b * d")
        if self.condition_on_shape and pooled_pc_embed is not None:
            primitive_codes = self.shape_coarse_film_cond(
                primitive_codes, pooled_pc_embed
            )
        if exists(self.coarse_gateloop_block):
            primitive_codes, _ = self.coarse_gateloop_block(primitive_codes, cache=None)
        prefix_attn_len = len_pl + len_bg
        attended, _ = self.decoder(
            primitive_codes,
            cache=None,
            return_hiddens=True,
            context=pc_embed,
            context_mask=pc_mask,
            prefix_attn_len=prefix_attn_len,
        )
        pred_bg_embed = attended[:, len_pl : len_pl + len_bg, :]
        pc_f = self.pc_embed_to_bg_space(pc_embed)
        return torch.sigmoid(torch.einsum("bpd,bcd->bpc", pc_f, pred_bg_embed))

    def _empty_sem_geom_pred_dict(self, T, device, pred_primitive_bg):
        B = 1
        z = self.pad_id
        o = {
            "primitives": [],
            "pred_semantic": torch.full((B, T), z, dtype=torch.long, device=device),
            "pred_primitive": torch.full((B, T), z, dtype=torch.long, device=device),
            "pred_mask": torch.zeros((B, T), dtype=torch.bool, device=device),
            "pred_flip": torch.zeros((B, T), dtype=torch.long, device=device),
            "pred_discrete_translation": torch.full(
                (B, T, 2), z, dtype=torch.long, device=device
            ),
            "pred_discrete_rotation": torch.full(
                (B, T), z, dtype=torch.long, device=device
            ),
            "pred_discrete_cont_translation": torch.zeros(
                (B, T, 2), device=device, dtype=torch.float32
            ),
            "pred_discrete_cont_rotation": torch.zeros(
                (B, T, 1), device=device, dtype=torch.float32
            ),
            "pred_cont_translation": torch.zeros(
                (B, T, 2), device=device, dtype=torch.float32
            ),
            "pred_cont_rotation": torch.zeros(
                (B, T, 1), device=device, dtype=torch.float32
            ),
        }
        if self.size_pred:
            o["pred_discrete_scale"] = None
            o["pred_discrete_size"] = torch.full(
                (B, T, 2), z, dtype=torch.long, device=device
            )
            o["pred_discrete_cont_size"] = torch.zeros(
                (B, T, 2), device=device, dtype=torch.float32
            )
            o["pred_cont_size"] = torch.zeros(
                (B, T, 2), device=device, dtype=torch.float32
            )
            o["pred_discrete_cont_scale"] = None
            o["pred_cont_scale"] = None
        else:
            o["pred_discrete_size"] = None
            o["pred_discrete_cont_size"] = None
            o["pred_cont_size"] = None
            o["pred_discrete_scale"] = torch.full(
                (B, T, 2), z, dtype=torch.long, device=device
            )
            o["pred_discrete_cont_scale"] = torch.zeros(
                (B, T, 2), device=device, dtype=torch.float32
            )
            o["pred_cont_scale"] = torch.zeros(
                (B, T, 2), device=device, dtype=torch.float32
            )
        o["pred_primitive_bg"] = pred_primitive_bg
        return o

    @torch.no_grad()
    def generate(
        self,
        vec_data=None,
        pc_embed=None,
        max_primitives: int = 256,
        gt_attrs=None,
        oracle_mode: str | None = None,
    ):
        """
        Autoregressive primitive generation.

        Provide either vec_data (raw point-cloud dict, production path) or
        pc_embed (pre-encoded tensor, test/bypass path).

        Args:
            vec_data:        dict with keys coords, feats, cu_seqlens, layer_ids,
                             prim_ids, sem_ids, cu_numprims.  All tensors on CUDA.
            pc_embed:        Optional pre-encoded shape embedding [B, S, dim].
                             When provided, skips _encode_shape_for_generate().
            max_primitives:  Maximum number of instances to generate.
            gt_attrs:        Optional raw batch dict (parallel_decode path only).
                             When provided, each returned primitive dict gains a
                             'debug' key with per-attribute discrete index comparison
                             (pred_idx, gt_idx, match, diff).  Debug-only.

        Returns:
            For ``sem_geom_decouple`` / ``geom_sem_decouple``: a dict with
            ``primitives`` (list of per-instance attribute dicts) and batched
            ``pred_*`` fields consistent with training-time inference. Other
            branches may return None. Each per-instance entry has: semantic_id,
            primitive_id, translation, rotation, scale/size, flip_mode.
        """
        was_training = self.training
        self.eval()
        try:
            return self._generate_impl(
                vec_data, pc_embed, max_primitives, gt_attrs, oracle_mode
            )
        finally:
            if was_training:
                self.train()

    @torch.no_grad()
    def generate_token0_paired(
        self,
        vec_data,
        gt_attrs,
        oracle_mode: str,
        max_primitives: int = 256,
    ):
        """Decode a control/oracle pair from one shared shape encoding.

        Row 0 is an untouched AR control and row 1 receives the requested
        token-0 intervention.  Keeping both rows in the same decoder calls
        removes encoder order sampling and cross-process CUDA drift from the
        causal comparison.
        """
        if not self.attr_decouple:
            raise ValueError("paired token-0 generation requires attr_decouple=True")
        if oracle_mode not in {"attr_token0_sem", "attr_token0_all"}:
            raise ValueError(f"unsupported paired oracle mode: {oracle_mode!r}")
        was_training = self.training
        self.eval()
        try:
            encoded, encoded_mask, primitive_lib = self._encode_shape_for_generate(
                vec_data
            )
            if encoded.shape[0] != 1:
                raise ValueError(
                    f"paired token-0 generation requires input batch size 1, got {encoded.shape[0]}"
                )
            pooled = encoded.mean(dim=1)
            pred_bg = self._compute_pred_primitive_bg_for_generate(
                encoded, encoded_mask, primitive_lib, pooled
            )

            def pair(tensor):
                return None if tensor is None else tensor.repeat_interleave(2, dim=0)

            paired_gt = {key: pair(value) for key, value in gt_attrs.items()}
            # A padding semantic marks row 0 as oracle-invalid. All torch.where
            # interventions therefore leave that row's predictions untouched.
            paired_gt["semantic_id"][0, 0] = self.pad_id
            return self._generate_attr_decouple_batched(
                pair(encoded),
                pair(encoded_mask),
                pair(primitive_lib),
                max_primitives,
                pred_primitive_bg_pre=pair(pred_bg),
                gt_attrs=paired_gt,
                oracle_mode=oracle_mode,
            )
        finally:
            if was_training:
                self.train()

    def _generate_impl(
        self, vec_data, pc_embed, max_primitives, gt_attrs=None, oracle_mode=None
    ):
        assert (vec_data is None) != (
            pc_embed is None
        ), "Provide exactly one of vec_data or pc_embed"

        if vec_data is not None:
            pc_embed_enc, pc_mask, plib_embed = self._encode_shape_for_generate(
                vec_data
            )
        else:
            pc_embed_enc = pc_embed
            pc_mask = None
            plib_embed = None

        pooled = pc_embed_enc.mean(dim=1) if exists(pc_embed_enc) else None
        pr_bg = self._compute_pred_primitive_bg_for_generate(
            pc_embed_enc, pc_mask, plib_embed, pooled
        )

        if self.attr_decouple:
            attr_oracle_modes = {
                None,
                "attr_token0_sem",
                "attr_token0_all",
                "attr_gt_previous_free_current",
            }
            if oracle_mode not in attr_oracle_modes:
                raise NotImplementedError(
                    "attr_decouple generation only supports token-0 oracle modes "
                    f"{sorted(x for x in attr_oracle_modes if x is not None)}, "
                    f"got {oracle_mode!r}"
                )
            if oracle_mode is not None and gt_attrs is None:
                raise ValueError(f"{oracle_mode} requires gt_attrs")
            if oracle_mode is None and gt_attrs is not None:
                raise ValueError("gt_attrs requires an attr_decouple oracle_mode")
            return self._generate_attr_decouple_batched(
                pc_embed_enc,
                pc_mask,
                plib_embed,
                max_primitives,
                pred_primitive_bg_pre=pr_bg,
                gt_attrs=gt_attrs,
                oracle_mode=oracle_mode,
            )
        if self.sem_geom_decouple or self.geom_sem_decouple:
            B_run = pc_embed_enc.shape[0] if exists(pc_embed_enc) else 1
            if B_run == 1:
                out = self._generate_sem_geom(
                    pc_embed_enc,
                    pc_mask,
                    plib_embed,
                    max_primitives,
                    pred_primitive_bg_pre=pr_bg,
                    oracle_mode=oracle_mode,
                    gt_attrs=gt_attrs,
                )
                # Normalize ``primitives`` field to list-of-list (length B=1)
                # so downstream consumers can always do ``primitives[b]``.
                if isinstance(out, dict):
                    p = out.get("primitives", None)
                    if isinstance(p, list) and (len(p) == 0 or isinstance(p[0], dict)):
                        out["primitives"] = [p]
                return out
            else:
                if oracle_mode is not None or gt_attrs is not None:
                    raise NotImplementedError(
                        f"oracle / gt_attrs paths only support B=1, got B={B_run}. "
                        f"Use --batch-size 1 with oracle modes."
                    )
                return self._generate_sem_geom_batched(
                    pc_embed_enc,
                    pc_mask,
                    plib_embed,
                    max_primitives,
                    pred_primitive_bg_pre=pr_bg,
                )
        else:
            return None

    def _generate_attr_decouple_batched(
        self,
        pc_embed,
        pc_embed_key_padding_mask,
        primitive_lib_embed,
        max_primitives: int = 256,
        pred_primitive_bg_pre=None,
        gt_attrs=None,
        oracle_mode: str | None = None,
    ) -> dict:
        """Pure AR generation for the attribute-decoupled sequence.

        History is either ``semantic -> primitive -> ...`` or the semantic-free
        ``primitive -> ...`` layout. EOS is queried only after a complete
        instance, matching the training boundary convention.
        """
        T = self.max_seq_len
        max_primitives = min(int(max_primitives), T)
        z = self.pad_id
        B = pc_embed.shape[0] if pc_embed is not None else 1
        device = next(self.parameters()).device
        gt_previous_free_current = oracle_mode == "attr_gt_previous_free_current"
        if gt_previous_free_current:
            if B != 1:
                raise ValueError(
                    "GT-previous/free-current diagnosis requires batch size 1"
                )
            required = {
                "semantic_id",
                "primitive_id",
                "translation",
                "flip_mode",
                "rotation",
                "size" if self.size_pred else "scale",
            }
            missing = sorted(required.difference(gt_attrs))
            if missing:
                raise KeyError(f"missing GT-history fields: {missing}")
            gt_count = int((gt_attrs["primitive_id"][0] != self.pad_id).sum().item())
            max_primitives = min(max_primitives, gt_count)

        seq_parts = []
        prefix_len = 0
        if self.shape_cond_with_perceiver_prefix:
            seq_parts.append(pc_embed)
            prefix_len += pc_embed.shape[1]
        if primitive_lib_embed is not None and self.primitive_cond_with_cat:
            plib_tokens = primitive_lib_embed + self.src_type_embed(
                torch.ones((B, 1), dtype=torch.long, device=device)
            )
            seq_parts.append(plib_tokens)
            prefix_len += plib_tokens.shape[1]
        if self.bg_pred:
            bg_tokens = self.bg_implicit_primitive.weight.unsqueeze(0).expand(B, -1, -1)
            seq_parts.append(bg_tokens)
            prefix_len += bg_tokens.shape[1]
        seq_parts.append(self.sos_token.unsqueeze(0).expand(B, -1, -1))
        x_seq = torch.cat(seq_parts, dim=1)
        pooled_pc = pc_embed.mean(dim=1) if pc_embed is not None else None

        def run_decoder(seq):
            decoder_input = seq
            if self.condition_on_shape:
                decoder_input = self.shape_coarse_film_cond(decoder_input, pooled_pc)
            if exists(self.coarse_gateloop_block):
                decoder_input, _ = self.coarse_gateloop_block(decoder_input)
            decoder_kwargs = dict(return_hiddens=True)
            if self.use_prefix_decoder:
                decoder_kwargs["prefix_attn_len"] = prefix_len
            if self.shape_cond_with_cross_attn:
                decoder_kwargs.update(
                    context=pc_embed,
                    context_mask=pc_embed_key_padding_mask,
                )
            hidden, _ = self.decoder(decoder_input, **decoder_kwargs)
            return hidden

        def make_token(projected, attr_id, instance_id):
            attr_ids = torch.full((B, 1), attr_id, dtype=torch.long, device=device)
            instance_ids = torch.full(
                (B, 1),
                min(instance_id, self.max_seq_len - 1),
                dtype=torch.long,
                device=device,
            )
            return (
                projected
                + self.attr_type_embed(attr_ids)
                + self.attr_instance_embed(instance_ids)
            )

        hidden = run_decoder(x_seq)
        current_h = hidden[:, -1:, :]
        done = torch.sigmoid(self.to_eos_logits(current_h)).reshape(B) > 0.9
        eos_would_stop = torch.zeros(B, T + 1, dtype=torch.bool, device=device)
        eos_would_stop[:, 0] = done
        if gt_previous_free_current:
            done = torch.zeros_like(done)
        done_at = torch.zeros(B, dtype=torch.long, device=device)

        oracle_token0 = oracle_mode in {"attr_token0_sem", "attr_token0_all"}
        oracle_all_token0 = oracle_mode == "attr_token0_all"
        if oracle_token0:
            required = {
                "semantic_id",
                "primitive_id",
                "translation",
                "flip_mode",
                "rotation",
            }
            required.add("size" if self.size_pred else "scale")
            missing = sorted(required.difference(gt_attrs))
            if missing:
                raise KeyError(f"missing attr token-0 oracle GT fields: {missing}")
            if gt_attrs["semantic_id"].shape[0] != B:
                raise ValueError(
                    "attr token-0 oracle batch mismatch: "
                    f"GT={gt_attrs['semantic_id'].shape[0]} encoded={B}"
                )
            oracle_valid0 = gt_attrs["semantic_id"][:, 0].long() != self.pad_id
            # A non-empty GT scene must reach its forced first token even if the
            # model's initial EOS query fires. EOS after that complete token is
            # deliberately left autoregressive.
            done = done & ~oracle_valid0
        else:
            oracle_valid0 = torch.zeros(B, dtype=torch.bool, device=device)

        buf_sem = torch.full((B, T), z, dtype=torch.long, device=device)
        buf_prim = torch.full((B, T), z, dtype=torch.long, device=device)
        buf_flip = torch.zeros((B, T), dtype=torch.long, device=device)
        buf_trans = torch.full((B, T, 2), z, dtype=torch.long, device=device)
        buf_rot = torch.full((B, T), z, dtype=torch.long, device=device)
        buf_shape = torch.full((B, T, 2), z, dtype=torch.long, device=device)
        buf_sem_prob = torch.zeros((B, T), dtype=torch.float32, device=device)
        buf_prim_prob = torch.zeros((B, T), dtype=torch.float32, device=device)
        buf_trans_cont = torch.zeros((B, T, 2), dtype=torch.float32, device=device)
        buf_rot_cont = torch.zeros((B, T, 1), dtype=torch.float32, device=device)
        buf_shape_cont = torch.zeros((B, T, 2), dtype=torch.float32, device=device)

        steps_run = 0
        for inst_i in range(max_primitives):
            if bool(done.all().item()):
                break
            steps_run = inst_i + 1
            active = ~done

            force_token0 = oracle_token0 and inst_i == 0
            force_full_token0 = oracle_all_token0 and inst_i == 0
            semantic_logits = None
            semantic_pred = None
            semantic_raw = None
            if (
                self.attr_semantic_supervision
                or self.attr_primitive_semantic_condition == "gt"
            ):
                semantic_logits = self.to_semantic_logits(current_h.squeeze(1))
                semantic_pred = semantic_logits.argmax(dim=-1)
                if force_token0:
                    semantic_pred = torch.where(
                        oracle_valid0,
                        gt_attrs["semantic_id"][:, 0].long(),
                        semantic_pred,
                    )
                semantic_raw = self.semantic_embed(semantic_pred.unsqueeze(1))

            if self.attr_emit_semantic_token:
                semantic_token = make_token(self.sem_proj_in(semantic_raw), 0, inst_i)
                x_seq = torch.cat([x_seq, semantic_token], dim=1)
                h_primitive = run_decoder(x_seq)[:, -1:, :]
                primitive_attr_id = 1
            else:
                h_primitive = current_h
                primitive_attr_id = 0

            if self.attr_primitive_semantic_condition == "gt":
                primitive_input = torch.cat([h_primitive, semantic_raw], dim=-1)
            else:
                primitive_input = h_primitive
            if self.primitive_contrastive_learning and primitive_lib_embed is not None:
                pred_primitive_embed = self.pred_proj(primitive_input)
                lib_primitive_embed = self.lib_proj(primitive_lib_embed)
                primitive_logits = self.logit_scale.exp().clamp(
                    max=self.max_temperature
                ) * torch.einsum(
                    "bie,bpe->bip",
                    F.normalize(pred_primitive_embed, p=2, dim=-1),
                    F.normalize(lib_primitive_embed, p=2, dim=-1),
                )
                primitive_pred = primitive_logits.argmax(dim=-1).squeeze(1)
            else:
                primitive_logits = self.to_primitive_logits(primitive_input)
                primitive_pred = primitive_logits.argmax(dim=-1).squeeze(1)
            if force_full_token0:
                primitive_pred = torch.where(
                    oracle_valid0,
                    gt_attrs["primitive_id"][:, 0].long(),
                    primitive_pred,
                )
            primitive_base = torch.remainder(primitive_pred, self.num_primitive)
            if not self.attr_semantic_supervision:
                semantic_pred = (
                    self.Hierarchical_primitive_criterion.primitive_id2semantic[
                        primitive_base
                    ]
                )
            if self.attr_replace_primitive_id_with_geom:
                if primitive_lib_embed is None:
                    raise RuntimeError(
                        "attr_replace_primitive_id_with_geom=True requires "
                        "primitive template geometry during AR generation"
                    )
                primitive_identity_id = primitive_pred.clamp(
                    max=primitive_lib_embed.shape[1] - 1
                ).unsqueeze(1)
            else:
                primitive_identity_id = primitive_base.unsqueeze(1)
            primitive_raw = self._attr_primitive_identity_embed(
                primitive_identity_id,
                torch.ones_like(primitive_identity_id, dtype=torch.bool),
                primitive_lib_embed,
            )
            primitive_projected = self.prim_proj_in(primitive_raw)
            if (
                self.use_lib_in_geom
                and not self.attr_replace_primitive_id_with_geom
                and primitive_lib_embed is not None
            ):
                lib_index = primitive_pred.clamp(max=primitive_lib_embed.shape[1] - 1)
                selected_lib = primitive_lib_embed[
                    torch.arange(B, device=device), lib_index
                ].unsqueeze(1)
                primitive_projected = (
                    primitive_projected
                    + self.beta_lib_geom * self.proj_lib_to_geom(selected_lib)
                )
            primitive_token = make_token(primitive_projected, primitive_attr_id, inst_i)
            x_seq = torch.cat([x_seq, primitive_token], dim=1)
            h_translation = run_decoder(x_seq)[:, -1:, :]

            if self.factorized_xy:
                translation_logits = self._factorized_2ch_logits(
                    h_translation,
                    self.to_trans_x_logits,
                    self.to_trans_y_logits,
                    self.translation_embed,
                    cond_x_idx=None,
                )
            else:
                translation_logits = rearrange(
                    self.to_translation_logits(h_translation),
                    "b t (c nd) -> b t c nd",
                    c=2,
                    nd=self.num_discrete_translation,
                )
            translation_pred = translation_logits.argmax(dim=-1).squeeze(1)
            if force_full_token0:
                translation_pred = torch.where(
                    oracle_valid0.unsqueeze(-1),
                    gt_attrs["translation"][:, 0].long(),
                    translation_pred,
                )
            translation_raw = rearrange(
                self.translation_embed(translation_pred.unsqueeze(1)),
                "b n c d -> b n (c d)",
            )
            translation_token = make_token(
                self.trans_proj_in(translation_raw),
                primitive_attr_id + 1,
                inst_i,
            )
            x_seq = torch.cat([x_seq, translation_token], dim=1)
            h_flip = run_decoder(x_seq)[:, -1:, :]

            flip_logits = self.to_flip_logits(h_flip)
            flip_pred = (torch.sigmoid(flip_logits).reshape(B) > 0.5).long()
            if force_full_token0:
                flip_pred = torch.where(
                    oracle_valid0,
                    gt_attrs["flip_mode"][:, 0].long(),
                    flip_pred,
                )
            flip_raw = self.flip_embed(flip_pred.unsqueeze(1))
            flip_token = make_token(
                self.flip_proj_in(flip_raw), primitive_attr_id + 2, inst_i
            )
            x_seq = torch.cat([x_seq, flip_token], dim=1)
            h_rotation = run_decoder(x_seq)[:, -1:, :]

            rotation_logits = self.to_rotation_logits(h_rotation)
            rotation_pred = rotation_logits.argmax(dim=-1).squeeze(1)
            if force_full_token0:
                rotation_pred = torch.where(
                    oracle_valid0,
                    gt_attrs["rotation"][:, 0].long(),
                    rotation_pred,
                )
            rotation_raw = self.rotation_embed(rotation_pred.unsqueeze(1))
            rotation_token = make_token(
                self.rot_proj_in(rotation_raw), primitive_attr_id + 3, inst_i
            )
            x_seq = torch.cat([x_seq, rotation_token], dim=1)
            h_shape = run_decoder(x_seq)[:, -1:, :]

            if self.size_pred:
                if self.factorized_xy:
                    shape_logits = self._factorized_2ch_logits(
                        h_shape,
                        self.to_size_w_logits,
                        self.to_size_h_logits,
                        self.size_embed,
                        cond_x_idx=None,
                    )
                else:
                    shape_logits = rearrange(
                        self.to_size_logits(h_shape),
                        "b t (c nd) -> b t c nd",
                        c=2,
                        nd=self.num_discrete_size,
                    )
                shape_table = self.size_embed
                shape_projector = self.shape_proj_in
                shape_centers = self.bin_centers_size
            else:
                shape_logits = rearrange(
                    self.to_scale_logits(h_shape),
                    "b t (c nd) -> b t c nd",
                    c=2,
                    nd=self.num_discrete_scale,
                )
                shape_table = self.scale_embed
                shape_projector = self.shape_proj_in
                shape_centers = self.bin_centers_scale
            shape_pred = shape_logits.argmax(dim=-1).squeeze(1)
            if force_full_token0:
                shape_key = "size" if self.size_pred else "scale"
                shape_pred = torch.where(
                    oracle_valid0.unsqueeze(-1),
                    gt_attrs[shape_key][:, 0].long(),
                    shape_pred,
                )
            shape_raw = rearrange(
                shape_table(shape_pred.unsqueeze(1)), "b n c d -> b n (c d)"
            )
            shape_token = make_token(
                shape_projector(shape_raw), primitive_attr_id + 4, inst_i
            )
            x_seq = torch.cat([x_seq, shape_token], dim=1)
            current_h = run_decoder(x_seq)[:, -1:, :]

            if semantic_logits is not None:
                sem_prob = (
                    semantic_logits.softmax(-1)
                    .gather(1, semantic_pred.unsqueeze(1))
                    .squeeze(1)
                )
            else:
                primitive_probabilities = primitive_logits.squeeze(1).softmax(-1)
                semantic_probabilities = torch.zeros(
                    B,
                    self.num_semantic,
                    device=device,
                    dtype=primitive_probabilities.dtype,
                )
                semantic_probabilities.index_add_(
                    1,
                    self.Hierarchical_primitive_criterion.primitive_id2semantic,
                    primitive_probabilities,
                )
                sem_prob = semantic_probabilities.gather(
                    1, semantic_pred.unsqueeze(1)
                ).squeeze(1)
            prim_prob = (
                primitive_logits.squeeze(1)
                .softmax(-1)
                .gather(1, primitive_pred.unsqueeze(1))
                .squeeze(1)
            )
            translation_cont = (
                (translation_logits.softmax(-1) * self.bin_centers_translation)
                .sum(-1)
                .squeeze(1)
            )
            rotation_phase_cont = self._stable_circular_rotation_expectation(
                rotation_logits.softmax(-1)
            )
            rotation_cont = self._rotation_phase_to_physical(
                rotation_phase_cont, primitive_base
            )
            shape_probabilities = shape_logits.softmax(-1)
            if not self.size_pred and self.scale_log:
                shape_cont = torch.exp(
                    (
                        shape_probabilities
                        * torch.log(self.bin_centers_scale.clamp_min(1e-8))
                    ).sum(-1)
                ).squeeze(1)
            else:
                shape_cont = (shape_probabilities * shape_centers).sum(-1).squeeze(1)

            if force_full_token0:
                valid2 = oracle_valid0.unsqueeze(-1)
                if "translation_cont" in gt_attrs:
                    translation_cont = torch.where(
                        valid2,
                        gt_attrs["translation_cont"][:, 0].float(),
                        translation_cont,
                    )
                if "rotation_cont" in gt_attrs:
                    rotation_cont = torch.where(
                        valid2,
                        gt_attrs["rotation_cont"][:, 0].float().reshape(B, 1),
                        rotation_cont.reshape(B, 1),
                    ).reshape(B, 1)
                shape_cont_key = "size_cont" if self.size_pred else "scale_cont"
                if shape_cont_key in gt_attrs:
                    shape_cont = torch.where(
                        valid2, gt_attrs[shape_cont_key][:, 0].float(), shape_cont
                    )

            buf_sem[:, inst_i] = torch.where(active, semantic_pred, buf_sem[:, inst_i])
            buf_prim[:, inst_i] = torch.where(
                active, primitive_pred, buf_prim[:, inst_i]
            )
            buf_flip[:, inst_i] = torch.where(active, flip_pred, buf_flip[:, inst_i])
            buf_rot[:, inst_i] = torch.where(active, rotation_pred, buf_rot[:, inst_i])
            active2 = active.unsqueeze(-1)
            buf_trans[:, inst_i] = torch.where(
                active2, translation_pred, buf_trans[:, inst_i]
            )
            buf_shape[:, inst_i] = torch.where(
                active2, shape_pred, buf_shape[:, inst_i]
            )
            buf_sem_prob[:, inst_i] = torch.where(
                active, sem_prob, buf_sem_prob[:, inst_i]
            )
            buf_prim_prob[:, inst_i] = torch.where(
                active, prim_prob, buf_prim_prob[:, inst_i]
            )
            buf_trans_cont[:, inst_i] = torch.where(
                active2, translation_cont, buf_trans_cont[:, inst_i]
            )
            buf_rot_cont[:, inst_i] = torch.where(
                active2, rotation_cont, buf_rot_cont[:, inst_i]
            )
            buf_shape_cont[:, inst_i] = torch.where(
                active2, shape_cont, buf_shape_cont[:, inst_i]
            )

            newly_done = (
                torch.sigmoid(self.to_eos_logits(current_h)).reshape(B) > 0.5
            ) & active
            eos_would_stop[:, inst_i + 1] = newly_done
            done_at = torch.where(
                newly_done, torch.full_like(done_at, inst_i + 1), done_at
            )
            done = done | newly_done

            if gt_previous_free_current:
                # The just-recorded tuple is fully free. Replace only its six
                # history tokens with GT before predicting the next instance.
                # This avoids intra-instance GT leakage into the recorded tuple.
                gi = inst_i
                g_sem = gt_attrs["semantic_id"][:, gi].long()
                g_prim = gt_attrs["primitive_id"][:, gi].long()
                g_tr = gt_attrs["translation"][:, gi].long()
                g_flip = gt_attrs["flip_mode"][:, gi].long()
                g_rot = gt_attrs["rotation"][:, gi].long()
                shape_key = "size" if self.size_pred else "scale"
                g_shape = gt_attrs[shape_key][:, gi].long()
                gt_sem_proj = self.sem_proj_in(self.semantic_embed(g_sem.unsqueeze(1)))
                prim_ids = (
                    g_prim
                    if self.attr_replace_primitive_id_with_geom
                    else torch.remainder(g_prim, self.num_primitive)
                ).unsqueeze(1)
                gt_prim_raw = self._attr_primitive_identity_embed(
                    prim_ids,
                    torch.ones_like(prim_ids, dtype=torch.bool),
                    primitive_lib_embed,
                )
                gt_prim_proj = self.prim_proj_in(gt_prim_raw)
                if (
                    self.use_lib_in_geom
                    and not self.attr_replace_primitive_id_with_geom
                    and primitive_lib_embed is not None
                ):
                    lib_index = g_prim.clamp(max=primitive_lib_embed.shape[1] - 1)
                    gt_prim_proj = (
                        gt_prim_proj
                        + self.beta_lib_geom
                        * self.proj_lib_to_geom(
                            primitive_lib_embed[
                                torch.arange(B, device=device), lib_index
                            ].unsqueeze(1)
                        )
                    )
                gt_tr_proj = self.trans_proj_in(
                    rearrange(
                        self.translation_embed(g_tr.unsqueeze(1)),
                        "b n c d -> b n (c d)",
                    )
                )
                gt_flip_proj = self.flip_proj_in(self.flip_embed(g_flip.unsqueeze(1)))
                gt_rot_proj = self.rot_proj_in(self.rotation_embed(g_rot.unsqueeze(1)))
                shape_embed = self.size_embed if self.size_pred else self.scale_embed
                gt_shape_proj = self.shape_proj_in(
                    rearrange(shape_embed(g_shape.unsqueeze(1)), "b n c d -> b n (c d)")
                )
                gt_projected = (
                    [gt_sem_proj] if self.attr_emit_semantic_token else []
                ) + [gt_prim_proj, gt_tr_proj, gt_flip_proj, gt_rot_proj, gt_shape_proj]
                gt_block = torch.cat(
                    [make_token(proj, ai, gi) for ai, proj in enumerate(gt_projected)],
                    dim=1,
                )
                if gt_block.shape[1] != self.num_attr_tokens:
                    raise RuntimeError("GT history token count mismatch")
                x_seq = torch.cat([x_seq[:, : -self.num_attr_tokens], gt_block], dim=1)
                current_h = run_decoder(x_seq)[:, -1:, :]
                eos_would_stop[:, inst_i + 1] = (
                    torch.sigmoid(self.to_eos_logits(current_h)).reshape(B) > 0.5
                )
                done = torch.zeros_like(done)

        done_at = torch.where(~done, torch.full_like(done_at, steps_run), done_at)
        if gt_previous_free_current:
            done_at[:] = steps_run
        pred_mask = torch.arange(T, device=device).unsqueeze(0) < done_at.unsqueeze(1)
        pred_discrete_cont_translation = self.undiscretize_translation(
            buf_trans.float()
        )
        pred_discrete_rotation_phase = self.undiscretize_rotation(buf_rot.float())
        pred_discrete_cont_rotation = self._rotation_phase_to_physical(
            pred_discrete_rotation_phase, buf_prim
        ).unsqueeze(-1)
        if self.size_pred:
            pred_discrete_cont_shape = self.undiscretize_size(buf_shape.float())
        else:
            pred_discrete_cont_shape = self.undiscretize_scale(buf_shape.float())

        # The full-token oracle represents an actually correct first instance,
        # not merely its nearest discrete-bin centre. Keep history discrete (as
        # in training), while exposing exact continuous GT to the downstream
        # projector for token 0.
        if oracle_all_token0:
            valid2 = oracle_valid0.unsqueeze(-1)
            if "translation_cont" in gt_attrs:
                pred_discrete_cont_translation[:, 0] = torch.where(
                    valid2,
                    gt_attrs["translation_cont"][:, 0].float(),
                    pred_discrete_cont_translation[:, 0],
                )
            if "rotation_cont" in gt_attrs:
                pred_discrete_cont_rotation[:, 0] = torch.where(
                    valid2,
                    gt_attrs["rotation_cont"][:, 0].float().reshape(B, 1),
                    pred_discrete_cont_rotation[:, 0],
                )
            shape_cont_key = "size_cont" if self.size_pred else "scale_cont"
            if shape_cont_key in gt_attrs:
                pred_discrete_cont_shape[:, 0] = torch.where(
                    valid2,
                    gt_attrs[shape_cont_key][:, 0].float(),
                    pred_discrete_cont_shape[:, 0],
                )

        primitives = []
        for batch_i in range(B):
            sample = []
            for inst_i in range(int(done_at[batch_i].item())):
                item = {
                    "semantic_id": int(buf_sem[batch_i, inst_i].item()),
                    "primitive_id": int(buf_prim[batch_i, inst_i].item()),
                    "translation": pred_discrete_cont_translation[
                        batch_i, inst_i
                    ].tolist(),
                    "rotation": float(
                        pred_discrete_cont_rotation[batch_i, inst_i, 0].item()
                    ),
                    "flip_mode": int(buf_flip[batch_i, inst_i].item()),
                }
                if self.size_pred:
                    item["size"] = pred_discrete_cont_shape[batch_i, inst_i].tolist()
                else:
                    item["scale"] = pred_discrete_cont_shape[batch_i, inst_i].tolist()
                sample.append(item)
            primitives.append(sample)

        output = {
            "primitives": primitives,
            "pred_semantic": buf_sem,
            "pred_primitive": buf_prim,
            "pred_mask": pred_mask,
            "pred_flip": buf_flip,
            "pred_discrete_translation": buf_trans,
            "pred_discrete_rotation": buf_rot,
            "pred_discrete_cont_translation": pred_discrete_cont_translation,
            "pred_discrete_cont_rotation": pred_discrete_cont_rotation,
            "pred_discrete_cont_rotation_phase": pred_discrete_rotation_phase.unsqueeze(
                -1
            ),
            "pred_cont_translation": buf_trans_cont,
            "pred_cont_rotation": buf_rot_cont,
            "pred_semantic_prob": buf_sem_prob,
            "pred_primitive_prob": buf_prim_prob,
            "pred_primitive_bg": pred_primitive_bg_pre,
            "oracle_mode": oracle_mode,
            "eos_would_stop": eos_would_stop,
        }
        if self.size_pred:
            output.update(
                pred_discrete_size=buf_shape,
                pred_discrete_cont_size=pred_discrete_cont_shape,
                pred_cont_size=buf_shape_cont,
                pred_discrete_scale=None,
                pred_discrete_cont_scale=None,
                pred_cont_scale=None,
            )
        else:
            output.update(
                pred_discrete_scale=buf_shape,
                pred_discrete_cont_scale=pred_discrete_cont_shape,
                pred_cont_scale=buf_shape_cont,
                pred_discrete_size=None,
                pred_discrete_cont_size=None,
                pred_cont_size=None,
            )
        return output

    def _pool_primitive_lib(self, plib_feats_b, plib_mask, chunk: int | None = None):
        """Project + pool padded library features ``[P, N_max, enc_dim]`` → ``[L, dim]``.

        Pooling is per-primitive independent, so we process primitives in chunks
        and concatenate, never materializing ``to_cond_dim_primitive_lib`` over
        the whole padded library (shape ``[P, N_max, dim]``, fp32 — the dominant
        VRAM transient in the AR path). The result is numerically identical to
        the single-shot pooling in ``forward_on_codes``.
        """
        P = plib_feats_b.shape[0]
        if chunk is None:
            chunk = int(getattr(self, "_plib_pool_chunk", 8) or P)
        chunk = max(1, min(chunk, P))
        out = []
        for s in range(0, P, chunk):
            feats = plib_feats_b[s : s + chunk]  # [c, N, enc_dim]
            mask = plib_mask[s : s + chunk]  # [c, N]
            emb = self.to_cond_dim_primitive_lib(feats)  # [c, N, dim]
            valid = mask.float()
            lengths = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
            if self.pooling_type == "mean":
                pe = (emb * valid.unsqueeze(-1)).sum(dim=1) / lengths
            elif self.pooling_type == "max":
                pe = (
                    emb.masked_fill(~mask.unsqueeze(-1), float("-inf"))
                    .max(dim=1)
                    .values
                )
            elif self.pooling_type == "mean_max":
                pe_mean = (emb * valid.unsqueeze(-1)).sum(dim=1) / lengths
                pe_max = (
                    emb.masked_fill(~mask.unsqueeze(-1), float("-inf"))
                    .max(dim=1)
                    .values
                )
                pe = pe_mean + pe_max
            elif self.pooling_type == "attn":
                pooled, _ = self.attention_pooling(emb, mask=~mask)
                pe = pooled.squeeze(1)
            elif self.pooling_type == "pma":
                pe, _ = self.pma_pooling(emb, mask=~mask)
                pe = pe + self.slot_embed(
                    torch.arange(self.n_seeds, device=self.device)
                    .view(1, self.n_seeds)
                    .expand(pe.shape[1], self.n_seeds)
                )
                pe = pe.reshape(pe.shape[0] * self.n_seeds, pe.shape[-1])
            else:
                raise ValueError(f"unknown pooling_type={self.pooling_type!r}")
            out.append(pe)
        return torch.cat(out, dim=0)

    @torch.no_grad()
    def _encode_shape_for_generate(self, vec_data: dict):
        """
        Encode input vec_data through PointTransformerV3 and the primitive library.
        Mirrors the shape-encoding block in _forward_on_codes_diffhead.

        Args:
            vec_data: dict with keys coords, feats, cu_seqlens, layer_ids,
                      prim_ids, sem_ids, cu_numprims — tensors on CUDA.

        Returns:
            pc_embed:                  Tensor [B, S, dim] or None
            pc_embed_key_padding_mask: Tensor [B, S] bool  or None
            primitive_lib_embed:       Tensor [B, L, dim] or None
        """
        device = next(self.parameters()).device
        batch = vec_data["cu_seqlens"].shape[0] - 1

        # ── Primitive library ─────────────────────────────────────────────────
        # The library is fixed at inference (grid_size constant, no augmentation),
        # so its pooled embedding is identical for every window. We compute it
        # once and cache the pre-repeat [L, dim] tensor; subsequent generate()
        # calls reuse it. This removes the dominant per-window VRAM transient
        # (the [P, N_max, dim] fp32 projection over the whole library) and the
        # repeated encode cost. See _pool_primitive_lib for the chunked pooling.
        primitive_lib_embed = None
        if self.primitive_dir is not None:
            cached = getattr(self, "_plib_pooled_cache", None)
            if cached is None:
                (
                    plib_coords,
                    plib_feats,
                    plib_layer_ids,
                    plib_cu_seqlens,
                    plib_prim_ids,
                ) = self._vec_grid_sample_tensor(
                    self.primitive_lib_coords,
                    self.primitive_lib_feats,
                    self.primitive_lib_layer_ids,
                    self.primitive_lib_cu_seqlens,
                    prim_ids=self.primitive_lib_prim_ids,
                    grid_size=self.conditioner_grid_size,
                )
                plib_dict = self._get_data_dict(
                    plib_coords,
                    plib_feats,
                    plib_cu_seqlens,
                    grid_size=self.conditioner_grid_size,
                    layer_ids=plib_layer_ids,
                    sample_mode="line",
                )
                with eval_no_grad(self.conditioner):
                    with torch.autocast("cuda", enabled=False):
                        plib_feats_enc, plib_coords_enc, plib_cu_enc, _, _, _ = (
                            self.conditioner(
                                plib_dict,
                                plib_cu_seqlens,
                                plib_prim_ids,
                            )
                        )
                plib_feats_b, _, plib_mask = self.offset_to_batch(
                    plib_feats_enc,
                    plib_coords_enc,
                    plib_cu_enc,
                )
                # Mirror the pooling block in forward_on_codes, but project +
                # pool one chunk of primitives at a time so we never materialize
                # to_cond_dim_primitive_lib(plib_feats_b) for the whole library.
                cached = self._pool_primitive_lib(plib_feats_b, plib_mask)
                self._plib_pooled_cache = cached.detach()
            primitive_lib_embed = cached.unsqueeze(0).repeat(batch, 1, 1)  # [B, L, dim]

        # ── Shape via PointTransformerV3 ──────────────────────────────────────
        pc_embed = None
        pc_embed_key_padding_mask = None
        if (
            self.condition_on_shape
            and "point_transformer_v3" in self.shape_condition_model_type
        ):
            data_dict = self._get_data_dict(
                vec_data["coords"],
                vec_data["feats"],
                vec_data["cu_seqlens"],
                grid_size=self.conditioner_grid_size,
                layer_ids=vec_data["layer_ids"],
                prim_ids=vec_data["prim_ids"],
                sem_ids=vec_data["sem_ids"],
                cu_numprims=vec_data["cu_numprims"],
                sample_mode="line",
            )
            with torch.autocast("cuda", enabled=False):
                pc_feats, pc_coords, pc_cu, _, _, _ = self.conditioner(
                    data_dict,
                    vec_data["cu_seqlens"],
                    vec_data["prim_ids"],
                )
            pc_feats_b, pc_coords_b, pc_mask = self.offset_to_batch(
                pc_feats, pc_coords, pc_cu
            )
            pc_embed = self.to_cond_dim(pc_feats_b)
            pc_embed_key_padding_mask = pc_mask

            # Add coordinate-based PE to pc_embed (matching training paths)
            if self.add_pe:
                pc_pos_embedding = self.pc_pe(
                    pc_coords_b[:, :, :2], num_channels=pc_embed.shape[2]
                )
                pc_pos_embedding = pc_pos_embedding.masked_fill(
                    ~pc_mask.unsqueeze(-1), 0.0
                )
                pc_embed = pc_embed + pc_pos_embedding

        return pc_embed, pc_embed_key_padding_mask, primitive_lib_embed

    def _build_gt_sem_tok(self, gt_attrs, inst_i, inst_idx, B, device):
        """Build a sem token from GT attributes for oracle_context injection into x_seq."""
        gt_sem = gt_attrs["semantic_id"][0, inst_i].reshape(B).to(device).long()
        gt_prim = gt_attrs["primitive_id"][0, inst_i].reshape(B).to(device).long()
        gt_prim_embed = self.primitive_embed(gt_prim.reshape(B, 1))
        gt_sem_in = torch.cat(
            [self.semantic_embed(gt_sem.reshape(B, 1)), gt_prim_embed], dim=-1
        )
        gt_sem_embed_i = self.sem_proj_in(gt_sem_in)
        return (
            gt_sem_embed_i
            + self.attr_type_embed(torch.zeros((B, 1), dtype=torch.long, device=device))
            + self.attr_instance_embed(
                torch.full((B, 1), inst_idx, dtype=torch.long, device=device)
            )
        )

    def _build_gt_geom_tok(self, gt_attrs, inst_i, inst_idx, B, device):
        """Build a geom token from GT attributes for oracle_context injection into x_seq."""
        gt_tx = gt_attrs["translation"][0, inst_i, 0].reshape(B).to(device).long()
        gt_ty = gt_attrs["translation"][0, inst_i, 1].reshape(B).to(device).long()
        gt_trans_idx = torch.stack([gt_tx, gt_ty], dim=-1).unsqueeze(1)  # [B,1,2]
        gt_trans_emb = rearrange(
            self.translation_embed(gt_trans_idx), "b np c d -> b np (c d)"
        )
        gt_flip = gt_attrs["flip_mode"][0, inst_i].reshape(B, 1).to(device).long()
        gt_flip_emb = self.flip_embed(gt_flip)
        gt_rot = gt_attrs["rotation"][0, inst_i].reshape(B).to(device).long()
        gt_rot_emb = self.rotation_embed(gt_rot.unsqueeze(1))
        if self.size_pred:
            gt_sx = gt_attrs["size"][0, inst_i, 0].reshape(B).to(device).long()
            gt_sy = gt_attrs["size"][0, inst_i, 1].reshape(B).to(device).long()
            gt_sz_idx = torch.stack([gt_sx, gt_sy], dim=-1).unsqueeze(1)
            gt_sz_emb = rearrange(self.size_embed(gt_sz_idx), "b np c d -> b np (c d)")
            gt_geom_packed = torch.cat(
                [gt_trans_emb, gt_flip_emb, gt_rot_emb, gt_sz_emb], dim=-1
            )
        else:
            gt_s0 = gt_attrs["scale"][0, inst_i, 0].reshape(B).to(device).long()
            gt_s1 = gt_attrs["scale"][0, inst_i, 1].reshape(B).to(device).long()
            gt_sc_idx = torch.stack([gt_s0, gt_s1], dim=-1).unsqueeze(1)
            gt_sc_emb = rearrange(self.scale_embed(gt_sc_idx), "b np c d -> b np (c d)")
            gt_geom_packed = torch.cat(
                [gt_trans_emb, gt_flip_emb, gt_rot_emb, gt_sc_emb], dim=-1
            )
        gt_geom_embed_i = self.geom_proj_in(gt_geom_packed)
        return (
            gt_geom_embed_i
            + self.attr_type_embed(torch.ones((B, 1), dtype=torch.long, device=device))
            + self.attr_instance_embed(
                torch.full((B, 1), inst_idx, dtype=torch.long, device=device)
            )
        )

    @torch.no_grad()
    def _generate_sem_geom(
        self,
        pc_embed,
        pc_embed_key_padding_mask,
        primitive_lib_embed,
        max_primitives: int = 256,
        pred_primitive_bg_pre=None,
        oracle_mode: str | None = None,
        gt_attrs=None,
    ) -> dict:
        """
        Serial AR for ``sem_geom_decouple``: interleaved ``[sem_i, geom_i]`` tokens
        and classification heads (no DiffHead, no ``parallel_decode``,
        no ``block_causal_sem_geom``). Matches the pure-causal training path.

        oracle_mode controls oracle ablation behaviour (all modes require gt_attrs):
          None / 'ar'          — standard autoregressive, no oracle
          'oracle_eos'         — force exactly gt_attrs['n_instances'] primitives
          'oracle_context'     — inject GT sem_tok / geom_tok into x_seq each step;
                                 per-step attributes are still AR-predicted
          'oracle_sem'         — force GT semantic_id in cascade & result
          'oracle_sem_prim'    — force GT semantic_id + primitive_id in cascade & result
          'oracle_geom_cascade'— use GT embeds as context inside the geom cascade
                                 (flip conditioned on GT trans, rot on GT flip, …);
                                 recorded geom attributes are still AR-predicted
          'oracle_all'         — force GT for every attribute (theoretical upper bound)

        gt_attrs keys (int64 tensors, B=1):
          'semantic_id'  [B, np]    GT semantic class indices
          'primitive_id' [B, np]    GT primitive type indices
          'translation'  [B, np, 2] GT discrete (tx_idx, ty_idx)
          'flip_mode'    [B, np]    GT flip  (0 or 1)
          'rotation'     [B, np]    GT discrete rotation index
          'scale'        [B, np, 2] GT discrete scale  (when not size_pred)
          'size'         [B, np, 2] GT discrete size   (when size_pred)
          'n_instances'  int        number of GT instances  (required for oracle_eos)

        Returns a dict with ``primitives`` (list of per-instance dicts) and batched
        ``pred_*`` tensors shaped like training-time inference (``B=1``, length
        ``max_seq_len``).
        """
        # ── Oracle flags ─────────────────────────────────────────────────────────
        _has_gt = gt_attrs is not None
        _force_sem = oracle_mode in (
            "oracle_sem",
            "oracle_sem_prim",
            "oracle_geom_cascade",
            "oracle_all",
        )
        _force_prim = oracle_mode in (
            "oracle_sem_prim",
            "oracle_geom_cascade",
            "oracle_all",
        )
        _force_geom = oracle_mode in ("oracle_geom_cascade", "oracle_all")
        _force_context = oracle_mode in ("oracle_context", "oracle_all")
        _force_eos = oracle_mode in ("oracle_eos", "oracle_all")
        _n_gt = (
            int(gt_attrs["n_instances"])
            if (_has_gt and "n_instances" in gt_attrs)
            else max_primitives
        )
        # ─────────────────────────────────────────────────────────────────────────
        T = self.max_seq_len
        z = self.pad_id
        B = pc_embed.shape[0] if pc_embed is not None else 1
        assert B == 1, f"generate() only supports B=1, got B={B}"
        device = next(self.parameters()).device
        plib_src = primitive_lib_embed
        L = 0
        sos = self.sos_token.unsqueeze(0).expand(B, -1, -1)
        if plib_src is not None and self.primitive_cond_with_cat:
            plib = plib_src + self.src_type_embed(
                torch.ones((B, 1), dtype=torch.long, device=device)
            )
            L = plib.shape[1]
            x_seq = torch.cat([plib, sos], dim=1)
        elif plib_src is not None:
            plib = plib_src
            L = plib.shape[1]
            x_seq = torch.cat([plib, sos], dim=1)
        else:
            x_seq = sos

        def run_decoder(seq):
            s = seq
            if exists(self.coarse_gateloop_block):
                s, _ = self.coarse_gateloop_block(s)
            if self.use_prefix_decoder and self.shape_cond_with_cross_attn:
                h, _ = self.decoder(
                    s,
                    context=pc_embed,
                    context_mask=pc_embed_key_padding_mask,
                    return_hiddens=True,
                    prefix_attn_len=L,
                )
            else:
                h, _ = self.decoder(s, return_hiddens=True)
            return h

        hidden = run_decoder(x_seq)
        cur_sem_h = hidden[:, -1:, :]
        # Stricter than 0.5: random init often false-triggers EOS at SOS; trained
        # empty-scene models still use a high logit here.
        if torch.sigmoid(self.to_eos_logits(cur_sem_h)).mean().item() > 0.9:
            return self._empty_sem_geom_pred_dict(T, device, pred_primitive_bg_pre)

        all_instances = []
        sem_l, prim_l = [], []
        sem_prob_l, prim_prob_l = [], []  # softmax prob at the sampled class
        trans_d_l, rot_d_l, flip_l = [], [], []
        trans_soft_l, rot_soft_l = [], []
        scale_d_l, scale_soft_l = [], []
        size_d_l, size_soft_l = [], []

        for inst_i in tqdm(range(max_primitives)):
            _gt_valid = _has_gt and inst_i < _n_gt  # True when GT exists for this step

            # ── Semantics ────────────────────────────────────────────────
            sem_logits = self.to_semantic_logits(cur_sem_h.squeeze(1))
            sem_pred = sem_logits.argmax(dim=-1)
            # Oracle: force GT semantic
            if _force_sem and _gt_valid:
                sem_pred = (
                    gt_attrs["semantic_id"][0, inst_i].reshape(B).to(device).long()
                )
            sem_id_2d = sem_pred.reshape(B, 1)
            sem_type_embed = self.semantic_embed(sem_id_2d)
            sem_embed_packed = torch.cat([cur_sem_h, sem_type_embed], dim=-1)

            # ── Primitive type ────────────────────────────────────────────
            if self.primitive_contrastive_learning and plib_src is not None:
                pred_prim_proj = self.pred_proj(sem_embed_packed)
                lib_prim_proj = self.lib_proj(primitive_lib_embed)
                pred_norm = F.normalize(pred_prim_proj, p=2, dim=-1)
                lib_norm = F.normalize(lib_prim_proj, p=2, dim=-1)
                logit_scale = self.logit_scale.exp().clamp(max=self.max_temperature)
                primitive_logits = logit_scale * torch.einsum(
                    "bie,bpe->bip", pred_norm, lib_norm
                )
                prim_pred = primitive_logits.argmax(dim=-1)
            else:
                primitive_logits = self.to_primitive_logits(sem_embed_packed)
                prim_pred = primitive_logits.argmax(dim=-1)
            # Oracle: force GT primitive
            if _force_prim and _gt_valid:
                prim_pred = (
                    gt_attrs["primitive_id"][0, inst_i].reshape(B).to(device).long()
                )

            # ── Build sem_tok and append to sequence ──────────────────────
            # 检索序号 prim_pred ∈ [0, P) 用于记录/渲染;回喂 AR 的 primitive token 只用
            # 基类 id(< num_primitive),因 self.primitive_embed = nn.Embedding(num_primitive)。
            # 合并库(P>76)按 76-块对齐编号(见 make_libB_combined.py),故 id % num_primitive = 基类 id。
            # 对 P<=76 的普通 run,remainder 恒等,零影响。
            prim_feed = torch.remainder(prim_pred, self.num_primitive)
            prim_id_2d = prim_feed.reshape(B, 1)
            prim_type_embed = self.primitive_embed(prim_id_2d)
            sem_in = torch.cat(
                [self.semantic_embed(sem_id_2d), prim_type_embed], dim=-1
            )
            sem_embed_i = self.sem_proj_in(sem_in)
            inst_idx = min(inst_i, self.max_seq_len - 1)
            sem_tok = (
                sem_embed_i
                + self.attr_type_embed(
                    torch.zeros((B, 1), dtype=torch.long, device=device)
                )
                + self.attr_instance_embed(
                    torch.full((B, 1), inst_idx, dtype=torch.long, device=device)
                )
            )
            # Oracle: replace sem_tok with GT token in x_seq
            if _force_context and _gt_valid:
                sem_tok = self._build_gt_sem_tok(gt_attrs, inst_i, inst_idx, B, device)
            x_seq = torch.cat([x_seq, sem_tok], dim=1)

            hidden = run_decoder(x_seq)
            sem_h = hidden[:, -1:, :]

            # ── Geometry cascade ──────────────────────────────────────────
            if self.factorized_xy:
                # Pure-AR: x first, y conditioned on argmax x (cond_x_idx=None).
                translation_logits = self._factorized_2ch_logits(
                    sem_h,
                    self.to_trans_x_logits,
                    self.to_trans_y_logits,
                    self.translation_embed,
                    cond_x_idx=None,
                )
            else:
                translation_logits = rearrange(
                    self.to_translation_logits(sem_h),
                    "b t (c nd) -> b t c nd",
                    c=2,
                    nd=self.num_discrete_translation,
                )
            tx_pred = translation_logits[:, 0, 0, :].argmax(dim=-1)
            ty_pred = translation_logits[:, 0, 1, :].argmax(dim=-1)

            # ── Translation embed ─────────────────────────────────────────
            # cond: drives the next cascade step (GT if oracle_geom_cascade)
            # tok:  written into x_seq geom_tok, must match recorded output
            if _force_geom and _gt_valid:
                _tx_cond = (
                    gt_attrs["translation"][0, inst_i, 0].reshape(B).to(device).long()
                )
                _ty_cond = (
                    gt_attrs["translation"][0, inst_i, 1].reshape(B).to(device).long()
                )
            else:
                _tx_cond, _ty_cond = tx_pred, ty_pred
            trans_idx_cond = torch.stack([_tx_cond, _ty_cond], dim=-1).unsqueeze(1)
            translation_type_embed_cond = rearrange(
                self.translation_embed(trans_idx_cond), "b np c d -> b np (c d)"
            )
            # tok embed always uses model prediction (consistent with _rec_tx/ty)
            trans_idx_tok = torch.stack([tx_pred, ty_pred], dim=-1).unsqueeze(1)
            translation_type_embed_tok = rearrange(
                self.translation_embed(trans_idx_tok), "b np c d -> b np (c d)"
            )

            embed_packed = torch.cat([sem_h, translation_type_embed_cond], dim=-1)
            flip_logits = self.to_flip_logits(embed_packed)
            flip_pred_2d = (torch.sigmoid(flip_logits) > 0.5).long().squeeze(-1)

            # ── Flip embed ────────────────────────────────────────────────
            if _force_geom and _gt_valid:
                _flip_cond = (
                    gt_attrs["flip_mode"][0, inst_i].reshape(B, 1).to(device).long()
                )
            else:
                _flip_cond = flip_pred_2d
            flip_type_embed_cond = self.flip_embed(_flip_cond)
            # tok embed always uses model prediction
            flip_type_embed_tok = self.flip_embed(flip_pred_2d)

            embed_packed = torch.cat([embed_packed, flip_type_embed_cond], dim=-1)
            rotation_logits = rearrange(
                self.to_rotation_logits(embed_packed),
                "b t (c nd) -> b t c nd",
                c=1,
                nd=self.num_discrete_rotation,
            )
            rot_pred = rotation_logits[:, 0, 0, :].argmax(dim=-1)

            # ── Rotation embed ────────────────────────────────────────────
            if _force_geom and _gt_valid:
                _rot_cond = gt_attrs["rotation"][0, inst_i].reshape(B).to(device).long()
            else:
                _rot_cond = rot_pred
            rot_type_embed_cond = self.rotation_embed(_rot_cond.unsqueeze(1))
            # tok embed always uses model prediction
            rot_type_embed_tok = self.rotation_embed(rot_pred.unsqueeze(1))

            embed_packed = torch.cat([embed_packed, rot_type_embed_cond], dim=-1)
            if self.size_pred:
                if self.factorized_xy:
                    size_logits = self._factorized_2ch_logits(
                        embed_packed,
                        self.to_size_w_logits,
                        self.to_size_h_logits,
                        self.size_embed,
                        cond_x_idx=None,
                    )
                else:
                    size_logits = rearrange(
                        self.to_size_logits(embed_packed),
                        "b t (c nd) -> b t c nd",
                        c=2,
                        nd=self.num_discrete_size,
                    )
                sx_pred = size_logits[:, 0, 0, :].argmax(dim=-1)
                sy_pred = size_logits[:, 0, 1, :].argmax(dim=-1)
                size_idx = torch.stack([sx_pred, sy_pred], dim=-1).unsqueeze(1)
                size_type_embed = rearrange(
                    self.size_embed(size_idx), "b np c d -> b np (c d)"
                )
                geom_packed = torch.cat(
                    [
                        translation_type_embed_tok,
                        flip_type_embed_tok,
                        rot_type_embed_tok,
                        size_type_embed,
                    ],
                    dim=-1,
                )
            else:
                scale_logits = rearrange(
                    self.to_scale_logits(embed_packed),
                    "b t (c nd) -> b t c nd",
                    c=2,
                    nd=self.num_discrete_scale,
                )
                s0_pred = scale_logits[:, 0, 0, :].argmax(dim=-1)
                s1_pred = scale_logits[:, 0, 1, :].argmax(dim=-1)
                scale_idx = torch.stack([s0_pred, s1_pred], dim=-1).unsqueeze(1)
                scale_type_embed = rearrange(
                    self.scale_embed(scale_idx), "b np c d -> b np (c d)"
                )
                geom_packed = torch.cat(
                    [
                        translation_type_embed_tok,
                        flip_type_embed_tok,
                        rot_type_embed_tok,
                        scale_type_embed,
                    ],
                    dim=-1,
                )

            pt = F.softmax(translation_logits, dim=-1)
            trans_soft_l.append(
                (pt * self.bin_centers_translation).sum(-1)[0, 0].clone()
            )
            pr = F.softmax(rotation_logits, dim=-1)
            rot_soft_l.append(
                (pr * self.bin_centers_rotation).sum(-1)[0, 0].clone().view(1)
            )
            # Record: use GT for forced attributes, AR-predicted otherwise
            _rec_sem = (
                int(gt_attrs["semantic_id"][0, inst_i].item())
                if (_force_sem and _gt_valid)
                else int(sem_pred.item())
            )
            _rec_prim = (
                int(gt_attrs["primitive_id"][0, inst_i].item())
                if (_force_prim and _gt_valid)
                else int(prim_pred.view(-1)[0].item())
            )
            _rec_tx = (
                int(gt_attrs["translation"][0, inst_i, 0].item())
                if (oracle_mode == "oracle_all" and _gt_valid)
                else int(tx_pred.item())
            )
            _rec_ty = (
                int(gt_attrs["translation"][0, inst_i, 1].item())
                if (oracle_mode == "oracle_all" and _gt_valid)
                else int(ty_pred.item())
            )
            _rec_flip = (
                int(gt_attrs["flip_mode"][0, inst_i].item())
                if (oracle_mode == "oracle_all" and _gt_valid)
                else int(flip_pred_2d[0, 0].item())
            )
            _rec_rot = (
                int(gt_attrs["rotation"][0, inst_i].item())
                if (oracle_mode == "oracle_all" and _gt_valid)
                else int(rot_pred.item())
            )
            sem_l.append(_rec_sem)
            prim_l.append(_rec_prim)
            # softmax prob at the SAMPLED token (= model's confidence in the
            # chosen class). Captured before any oracle override so the value
            # always reflects the AR forward pass.
            with torch.no_grad():
                _sem_p_full = F.softmax(sem_logits, dim=-1)
                _sem_prob = float(_sem_p_full[0, _rec_sem].item())
                _prim_p_full = F.softmax(primitive_logits, dim=-1).reshape(-1)
                _prim_prob = float(_prim_p_full[_rec_prim].item())
            sem_prob_l.append(_sem_prob)
            prim_prob_l.append(_prim_prob)
            trans_d_l.append([_rec_tx, _rec_ty])
            rot_d_l.append(_rec_rot)
            flip_l.append(_rec_flip)
            if self.size_pred:
                ps = F.softmax(size_logits, dim=-1)
                size_soft_l.append((ps * self.bin_centers_size).sum(-1)[0, 0].clone())
                _rec_sx = (
                    int(gt_attrs["size"][0, inst_i, 0].item())
                    if (oracle_mode == "oracle_all" and _gt_valid)
                    else int(sx_pred.item())
                )
                _rec_sy = (
                    int(gt_attrs["size"][0, inst_i, 1].item())
                    if (oracle_mode == "oracle_all" and _gt_valid)
                    else int(sy_pred.item())
                )
                size_d_l.append([_rec_sx, _rec_sy])
            else:
                psc = F.softmax(scale_logits, dim=-1)
                scale_soft_l.append(
                    (psc * self.bin_centers_scale).sum(-1)[0, 0].clone()
                )
                _rec_s0 = (
                    int(gt_attrs["scale"][0, inst_i, 0].item())
                    if (oracle_mode == "oracle_all" and _gt_valid)
                    else int(s0_pred.item())
                )
                _rec_s1 = (
                    int(gt_attrs["scale"][0, inst_i, 1].item())
                    if (oracle_mode == "oracle_all" and _gt_valid)
                    else int(s1_pred.item())
                )
                scale_d_l.append([_rec_s0, _rec_s1])

            # Continuous values for result dict (oracle_all uses GT discrete → continuous)
            tx_val = float(
                self.undiscretize_translation(
                    torch.tensor(_rec_tx, dtype=torch.float).view(1)
                ).item()
            )
            ty_val = float(
                self.undiscretize_translation(
                    torch.tensor(_rec_ty, dtype=torch.float).view(1)
                ).item()
            )
            rot_val = float(
                self.undiscretize_rotation(
                    torch.tensor(_rec_rot, dtype=torch.float).view(1)
                ).item()
            )
            if self.size_pred:
                sx_val = float(
                    self.undiscretize_size(
                        torch.tensor(_rec_sx, dtype=torch.float).view(1)
                    ).item()
                )
                sy_val = float(
                    self.undiscretize_size(
                        torch.tensor(_rec_sy, dtype=torch.float).view(1)
                    ).item()
                )
            else:
                sx_val = float(
                    self.undiscretize_scale(
                        torch.tensor(_rec_s0, dtype=torch.float).view(1)
                    ).item()
                )
                sy_val = float(
                    self.undiscretize_scale(
                        torch.tensor(_rec_s1, dtype=torch.float).view(1)
                    ).item()
                )

            result = {
                "semantic_id": _rec_sem,
                "primitive_id": _rec_prim,
                "translation": [tx_val, ty_val],
                "rotation": rot_val,
                "flip_mode": _rec_flip,
                "oracle_mode": oracle_mode,  # tag for downstream analysis
            }
            if self.size_pred:
                result["size"] = [sx_val, sy_val]
            else:
                result["scale"] = [sx_val, sy_val]
            all_instances.append(result)

            # ── Build geom_tok and append to sequence ─────────────────────
            geom_embed_i = self.geom_proj_in(geom_packed)
            geom_tok = (
                geom_embed_i
                + self.attr_type_embed(
                    torch.ones((B, 1), dtype=torch.long, device=device)
                )
                + self.attr_instance_embed(
                    torch.full((B, 1), inst_idx, dtype=torch.long, device=device)
                )
            )
            # Oracle: replace geom_tok with GT token in x_seq
            if _force_context and _gt_valid:
                geom_tok = self._build_gt_geom_tok(
                    gt_attrs, inst_i, inst_idx, B, device
                )
            x_seq = torch.cat([x_seq, geom_tok], dim=1)
            hidden = run_decoder(x_seq)
            cur_sem_h = hidden[:, -1:, :]

            # ── EOS check ─────────────────────────────────────────────────
            if _force_eos:
                if inst_i >= _n_gt - 1:
                    break
            else:
                if torch.sigmoid(self.to_eos_logits(cur_sem_h)).mean().item() > 0.5:
                    break

        n = len(all_instances)
        if n == 0:
            return self._empty_sem_geom_pred_dict(T, device, pred_primitive_bg_pre)

        sem_t = F.pad(
            torch.tensor(sem_l, device=device, dtype=torch.long).unsqueeze(0),
            (0, T - n),
            value=z,
        )
        prim_t = F.pad(
            torch.tensor(prim_l, device=device, dtype=torch.long).unsqueeze(0),
            (0, T - n),
            value=z,
        )
        pred_mask = torch.zeros(1, T, dtype=torch.bool, device=device)
        pred_mask[0, :n] = True
        flip_t = F.pad(
            torch.tensor(flip_l, device=device, dtype=torch.long).unsqueeze(0),
            (0, T - n),
            value=0,
        )

        trans_d_core = torch.tensor(
            trans_d_l, device=device, dtype=torch.long
        ).unsqueeze(0)
        pred_discrete_cont_translation = self.undiscretize_translation(
            trans_d_core.float()
        )
        pred_discrete_translation = F.pad(trans_d_core, (0, 0, 0, T - n), value=z)
        pred_discrete_cont_translation = F.pad(
            pred_discrete_cont_translation, (0, 0, 0, T - n), value=0.0
        )

        rot_d_core = torch.tensor(rot_d_l, device=device, dtype=torch.long).unsqueeze(0)
        urot = self.undiscretize_rotation(rot_d_core.float())
        if urot.dim() == 2:
            urot = urot.unsqueeze(-1)
        pred_discrete_rotation = F.pad(rot_d_core, (0, T - n), value=z)
        pred_discrete_cont_rotation = F.pad(urot, (0, 0, 0, T - n), value=0.0)

        pred_cont_translation = F.pad(
            torch.stack(trans_soft_l, dim=0).unsqueeze(0), (0, 0, 0, T - n), value=0.0
        )
        pred_cont_rotation = F.pad(
            torch.stack(rot_soft_l, dim=0).unsqueeze(0), (0, 0, 0, T - n), value=0.0
        )

        sem_prob_t = F.pad(
            torch.tensor(sem_prob_l, device=device, dtype=torch.float32).unsqueeze(0),
            (0, T - n),
            value=0.0,
        )
        prim_prob_t = F.pad(
            torch.tensor(prim_prob_l, device=device, dtype=torch.float32).unsqueeze(0),
            (0, T - n),
            value=0.0,
        )

        out = {
            "primitives": all_instances,
            "pred_semantic": sem_t,
            "pred_primitive": prim_t,
            "pred_mask": pred_mask,
            "pred_flip": flip_t,
            "pred_discrete_translation": pred_discrete_translation,
            "pred_discrete_rotation": pred_discrete_rotation,
            "pred_discrete_cont_translation": pred_discrete_cont_translation,
            "pred_discrete_cont_rotation": pred_discrete_cont_rotation,
            "pred_cont_translation": pred_cont_translation,
            "pred_cont_rotation": pred_cont_rotation,
            "pred_semantic_prob": sem_prob_t,
            "pred_primitive_prob": prim_prob_t,
        }
        if self.size_pred:
            sz_core = torch.tensor(size_d_l, device=device, dtype=torch.long).unsqueeze(
                0
            )
            pred_discrete_cont_size = F.pad(
                self.undiscretize_size(sz_core.float()), (0, 0, 0, T - n), value=0.0
            )
            out["pred_discrete_size"] = F.pad(sz_core, (0, 0, 0, T - n), value=z)
            out["pred_discrete_cont_size"] = pred_discrete_cont_size
            out["pred_cont_size"] = F.pad(
                torch.stack(size_soft_l, dim=0).unsqueeze(0),
                (0, 0, 0, T - n),
                value=0.0,
            )
            out["pred_discrete_scale"] = None
            out["pred_discrete_cont_scale"] = None
            out["pred_cont_scale"] = None
        else:
            sc_core = torch.tensor(
                scale_d_l, device=device, dtype=torch.long
            ).unsqueeze(0)
            pred_discrete_cont_scale = F.pad(
                self.undiscretize_scale(sc_core.float()), (0, 0, 0, T - n), value=0.0
            )
            out["pred_discrete_scale"] = F.pad(sc_core, (0, 0, 0, T - n), value=z)
            out["pred_discrete_cont_scale"] = pred_discrete_cont_scale
            out["pred_cont_scale"] = F.pad(
                torch.stack(scale_soft_l, dim=0).unsqueeze(0),
                (0, 0, 0, T - n),
                value=0.0,
            )
            out["pred_discrete_size"] = None
            out["pred_discrete_cont_size"] = None
            out["pred_cont_size"] = None
        out["pred_primitive_bg"] = pred_primitive_bg_pre
        return out

    def _generate_sem_geom_batched(
        self,
        pc_embed,
        pc_embed_key_padding_mask,
        primitive_lib_embed,
        max_primitives: int = 256,
        pred_primitive_bg_pre=None,
    ) -> dict:
        """Batched AR generation for ``sem_geom_decouple`` / ``geom_sem_decouple``.

        Same logic as ``_generate_sem_geom`` but B>1 capable:
          * Per-sample EOS tracking via ``done`` / ``done_at`` masks (each sample
            terminates independently; samples already done keep cruising through
            the decoder so we don't have to slice rows, but their outputs are
            ignored beyond their own EOS step).
          * No ``.item()`` / per-step Python lists — accumulates into
            ``(B, max_primitives, ...)`` tensor buffers, one ``.cpu()`` at the end.
          * Oracle / ``gt_attrs`` not supported (caller must dispatch B=1 to the
            original ``_generate_sem_geom`` for those paths).
          * Returns ``primitives`` as **list-of-list** (length B), plus the
            ``pred_*`` batched tensors of shape ``(B, T, ...)`` consistent with
            training-time inference.
        """
        T = self.max_seq_len
        z = self.pad_id
        B = pc_embed.shape[0] if pc_embed is not None else 1
        device = next(self.parameters()).device
        plib_src = primitive_lib_embed
        L = 0
        sos = self.sos_token.unsqueeze(0).expand(B, -1, -1)
        if plib_src is not None and self.primitive_cond_with_cat:
            plib = plib_src + self.src_type_embed(
                torch.ones((B, 1), dtype=torch.long, device=device)
            )
            L = plib.shape[1]
            x_seq = torch.cat([plib, sos], dim=1)
        elif plib_src is not None:
            plib = plib_src
            L = plib.shape[1]
            x_seq = torch.cat([plib, sos], dim=1)
        else:
            x_seq = sos

        def run_decoder(seq):
            s = seq
            if exists(self.coarse_gateloop_block):
                s, _ = self.coarse_gateloop_block(s)
            if self.use_prefix_decoder and self.shape_cond_with_cross_attn:
                h, _ = self.decoder(
                    s,
                    context=pc_embed,
                    context_mask=pc_embed_key_padding_mask,
                    return_hiddens=True,
                    prefix_attn_len=L,
                )
            else:
                h, _ = self.decoder(s, return_hiddens=True)
            return h

        hidden = run_decoder(x_seq)
        cur_sem_h = hidden[:, -1:, :]

        # Per-sample early-EOS at SOS (replaces global .mean().item() > 0.9)
        eos_p_init = torch.sigmoid(self.to_eos_logits(cur_sem_h)).reshape(B)  # (B,)
        done = eos_p_init > 0.9  # (B,) bool
        done_at = torch.zeros(B, dtype=torch.long, device=device)  # (B,) int
        if bool(done.all().item()):
            return self._empty_sem_geom_batched_dict(
                B, T, device, pred_primitive_bg_pre
            )

        # Pre-allocate accumulator buffers (B, max_primitives, ...)
        buf_sem = torch.zeros(B, max_primitives, dtype=torch.long, device=device)
        buf_prim = torch.zeros(B, max_primitives, dtype=torch.long, device=device)
        buf_flip = torch.zeros(B, max_primitives, dtype=torch.long, device=device)
        buf_tx = torch.zeros(B, max_primitives, dtype=torch.long, device=device)
        buf_ty = torch.zeros(B, max_primitives, dtype=torch.long, device=device)
        buf_rot = torch.zeros(B, max_primitives, dtype=torch.long, device=device)
        buf_sem_prob = torch.zeros(
            B, max_primitives, device=device, dtype=torch.float32
        )
        buf_prim_prob = torch.zeros(
            B, max_primitives, device=device, dtype=torch.float32
        )
        buf_trans_soft = torch.zeros(
            B, max_primitives, 2, device=device, dtype=torch.float32
        )
        buf_rot_soft = torch.zeros(
            B, max_primitives, 1, device=device, dtype=torch.float32
        )
        if self.size_pred:
            buf_sx = torch.zeros(B, max_primitives, dtype=torch.long, device=device)
            buf_sy = torch.zeros(B, max_primitives, dtype=torch.long, device=device)
            buf_size_soft = torch.zeros(
                B, max_primitives, 2, device=device, dtype=torch.float32
            )
        else:
            buf_s0 = torch.zeros(B, max_primitives, dtype=torch.long, device=device)
            buf_s1 = torch.zeros(B, max_primitives, dtype=torch.long, device=device)
            buf_scale_soft = torch.zeros(
                B, max_primitives, 2, device=device, dtype=torch.float32
            )

        n_steps_run = 0
        for inst_i in tqdm(range(max_primitives)):
            n_steps_run = inst_i + 1
            update_mask = ~done  # (B,)

            # ── Semantics ────────────────────────────────────────────
            sem_logits = self.to_semantic_logits(cur_sem_h.squeeze(1))  # (B, n_sem)
            sem_pred = sem_logits.argmax(dim=-1)  # (B,)
            sem_id_2d = sem_pred.reshape(B, 1)
            sem_type_embed = self.semantic_embed(sem_id_2d)
            sem_embed_packed = torch.cat([cur_sem_h, sem_type_embed], dim=-1)

            # ── Primitive type ────────────────────────────────────────
            if self.primitive_contrastive_learning and plib_src is not None:
                pred_prim_proj = self.pred_proj(sem_embed_packed)
                lib_prim_proj = self.lib_proj(primitive_lib_embed)
                pred_norm = F.normalize(pred_prim_proj, p=2, dim=-1)
                lib_norm = F.normalize(lib_prim_proj, p=2, dim=-1)
                logit_scale = self.logit_scale.exp().clamp(max=self.max_temperature)
                primitive_logits = logit_scale * torch.einsum(
                    "bie,bpe->bip", pred_norm, lib_norm
                )
                prim_pred = primitive_logits.argmax(dim=-1).reshape(B)
            else:
                primitive_logits = self.to_primitive_logits(sem_embed_packed)
                prim_pred = primitive_logits.argmax(dim=-1).reshape(B)

            # 见 _generate_sem_geom 处注释:回喂 primitive token 用基类 id(id % num_primitive),
            # prim_pred(完整检索序号 0..P-1)仍用于 buf_prim 记录 / softmax gather / 输出。
            prim_feed = torch.remainder(prim_pred, self.num_primitive)
            prim_id_2d = prim_feed.reshape(B, 1)
            prim_type_embed = self.primitive_embed(prim_id_2d)
            sem_in = torch.cat(
                [self.semantic_embed(sem_id_2d), prim_type_embed], dim=-1
            )
            sem_embed_i = self.sem_proj_in(sem_in)
            inst_idx = min(inst_i, self.max_seq_len - 1)
            sem_tok = (
                sem_embed_i
                + self.attr_type_embed(
                    torch.zeros((B, 1), dtype=torch.long, device=device)
                )
                + self.attr_instance_embed(
                    torch.full((B, 1), inst_idx, dtype=torch.long, device=device)
                )
            )
            x_seq = torch.cat([x_seq, sem_tok], dim=1)

            hidden = run_decoder(x_seq)
            sem_h = hidden[:, -1:, :]

            # ── Geometry cascade ──────────────────────────────────────
            if self.factorized_xy:
                translation_logits = self._factorized_2ch_logits(
                    sem_h,
                    self.to_trans_x_logits,
                    self.to_trans_y_logits,
                    self.translation_embed,
                    cond_x_idx=None,
                )
            else:
                translation_logits = rearrange(
                    self.to_translation_logits(sem_h),
                    "b t (c nd) -> b t c nd",
                    c=2,
                    nd=self.num_discrete_translation,
                )
            tx_pred = translation_logits[:, 0, 0, :].argmax(dim=-1)  # (B,)
            ty_pred = translation_logits[:, 0, 1, :].argmax(dim=-1)
            trans_idx_tok = torch.stack([tx_pred, ty_pred], dim=-1).unsqueeze(1)
            translation_type_embed_tok = rearrange(
                self.translation_embed(trans_idx_tok), "b np c d -> b np (c d)"
            )

            embed_packed = torch.cat([sem_h, translation_type_embed_tok], dim=-1)
            flip_logits = self.to_flip_logits(embed_packed)
            flip_pred_2d = (
                (torch.sigmoid(flip_logits) > 0.5).long().squeeze(-1)
            )  # (B, 1)
            flip_type_embed_tok = self.flip_embed(flip_pred_2d)

            embed_packed = torch.cat([embed_packed, flip_type_embed_tok], dim=-1)
            rotation_logits = rearrange(
                self.to_rotation_logits(embed_packed),
                "b t (c nd) -> b t c nd",
                c=1,
                nd=self.num_discrete_rotation,
            )
            rot_pred = rotation_logits[:, 0, 0, :].argmax(dim=-1)  # (B,)
            rot_type_embed_tok = self.rotation_embed(rot_pred.unsqueeze(1))

            embed_packed = torch.cat([embed_packed, rot_type_embed_tok], dim=-1)
            if self.size_pred:
                if self.factorized_xy:
                    size_logits = self._factorized_2ch_logits(
                        embed_packed,
                        self.to_size_w_logits,
                        self.to_size_h_logits,
                        self.size_embed,
                        cond_x_idx=None,
                    )
                else:
                    size_logits = rearrange(
                        self.to_size_logits(embed_packed),
                        "b t (c nd) -> b t c nd",
                        c=2,
                        nd=self.num_discrete_size,
                    )
                sx_pred = size_logits[:, 0, 0, :].argmax(dim=-1)  # (B,)
                sy_pred = size_logits[:, 0, 1, :].argmax(dim=-1)
                size_idx = torch.stack([sx_pred, sy_pred], dim=-1).unsqueeze(1)
                size_type_embed = rearrange(
                    self.size_embed(size_idx), "b np c d -> b np (c d)"
                )
                geom_packed = torch.cat(
                    [
                        translation_type_embed_tok,
                        flip_type_embed_tok,
                        rot_type_embed_tok,
                        size_type_embed,
                    ],
                    dim=-1,
                )
            else:
                scale_logits = rearrange(
                    self.to_scale_logits(embed_packed),
                    "b t (c nd) -> b t c nd",
                    c=2,
                    nd=self.num_discrete_scale,
                )
                s0_pred = scale_logits[:, 0, 0, :].argmax(dim=-1)  # (B,)
                s1_pred = scale_logits[:, 0, 1, :].argmax(dim=-1)
                scale_idx = torch.stack([s0_pred, s1_pred], dim=-1).unsqueeze(1)
                scale_type_embed = rearrange(
                    self.scale_embed(scale_idx), "b np c d -> b np (c d)"
                )
                geom_packed = torch.cat(
                    [
                        translation_type_embed_tok,
                        flip_type_embed_tok,
                        rot_type_embed_tok,
                        scale_type_embed,
                    ],
                    dim=-1,
                )

            # ── Record predictions to buffers (only update non-done samples) ──
            sem_prob_step = (
                F.softmax(sem_logits, dim=-1)
                .gather(1, sem_pred.unsqueeze(1))
                .squeeze(1)
            )  # (B,)
            prim_prob_step = (
                F.softmax(primitive_logits.reshape(B, -1), dim=-1)
                .gather(1, prim_pred.unsqueeze(1))
                .squeeze(1)
            )  # (B,)
            pt = F.softmax(translation_logits, dim=-1)
            trans_soft_step = (pt * self.bin_centers_translation).sum(-1)[
                :, 0, :
            ]  # (B, 2)
            pr = F.softmax(rotation_logits, dim=-1)
            rot_soft_step = (pr * self.bin_centers_rotation).sum(-1)[:, 0, :]  # (B, 1)

            buf_sem[:, inst_i] = torch.where(update_mask, sem_pred, buf_sem[:, inst_i])
            buf_prim[:, inst_i] = torch.where(
                update_mask, prim_pred, buf_prim[:, inst_i]
            )
            buf_tx[:, inst_i] = torch.where(update_mask, tx_pred, buf_tx[:, inst_i])
            buf_ty[:, inst_i] = torch.where(update_mask, ty_pred, buf_ty[:, inst_i])
            buf_flip[:, inst_i] = torch.where(
                update_mask, flip_pred_2d.reshape(B), buf_flip[:, inst_i]
            )
            buf_rot[:, inst_i] = torch.where(update_mask, rot_pred, buf_rot[:, inst_i])
            buf_sem_prob[:, inst_i] = torch.where(
                update_mask, sem_prob_step, buf_sem_prob[:, inst_i]
            )
            buf_prim_prob[:, inst_i] = torch.where(
                update_mask, prim_prob_step, buf_prim_prob[:, inst_i]
            )

            mask2 = update_mask.unsqueeze(-1)  # (B, 1)
            buf_trans_soft[:, inst_i, :] = torch.where(
                mask2, trans_soft_step, buf_trans_soft[:, inst_i, :]
            )
            buf_rot_soft[:, inst_i, :] = torch.where(
                mask2, rot_soft_step, buf_rot_soft[:, inst_i, :]
            )

            if self.size_pred:
                buf_sx[:, inst_i] = torch.where(update_mask, sx_pred, buf_sx[:, inst_i])
                buf_sy[:, inst_i] = torch.where(update_mask, sy_pred, buf_sy[:, inst_i])
                ps = F.softmax(size_logits, dim=-1)
                size_soft_step = (ps * self.bin_centers_size).sum(-1)[:, 0, :]
                buf_size_soft[:, inst_i, :] = torch.where(
                    mask2, size_soft_step, buf_size_soft[:, inst_i, :]
                )
            else:
                buf_s0[:, inst_i] = torch.where(update_mask, s0_pred, buf_s0[:, inst_i])
                buf_s1[:, inst_i] = torch.where(update_mask, s1_pred, buf_s1[:, inst_i])
                psc = F.softmax(scale_logits, dim=-1)
                scale_soft_step = (psc * self.bin_centers_scale).sum(-1)[:, 0, :]
                buf_scale_soft[:, inst_i, :] = torch.where(
                    mask2, scale_soft_step, buf_scale_soft[:, inst_i, :]
                )

            # ── Build geom_tok and append to sequence ─────────────────
            geom_embed_i = self.geom_proj_in(geom_packed)
            geom_tok = (
                geom_embed_i
                + self.attr_type_embed(
                    torch.ones((B, 1), dtype=torch.long, device=device)
                )
                + self.attr_instance_embed(
                    torch.full((B, 1), inst_idx, dtype=torch.long, device=device)
                )
            )
            x_seq = torch.cat([x_seq, geom_tok], dim=1)
            hidden = run_decoder(x_seq)
            cur_sem_h = hidden[:, -1:, :]

            # ── Per-sample EOS check ──────────────────────────────────
            eos_p = torch.sigmoid(self.to_eos_logits(cur_sem_h)).reshape(B)  # (B,)
            newly_done = (eos_p > 0.5) & (~done)
            done_at = torch.where(
                newly_done,
                torch.full_like(done_at, inst_i + 1),
                done_at,
            )
            done = done | newly_done
            if bool(done.all().item()):
                break

        # Samples that never EOS'd → done_at = n_steps_run
        still_active = ~done
        done_at = torch.where(
            still_active,
            torch.full_like(done_at, n_steps_run),
            done_at,
        )

        # ── Build output tensors (B, T, ...) ────────────────────────────
        # Pad buffers from (B, max_primitives) → (B, T)
        pad_right = max(0, T - max_primitives)
        slice_w = min(T, max_primitives)

        def _pad_1d(buf, pad_value):
            if pad_right > 0:
                return F.pad(buf, (0, pad_right), value=pad_value)
            return buf[:, :T]

        def _pad_2d(buf, pad_value):
            # buf: (B, max_primitives, K)
            if pad_right > 0:
                return F.pad(buf, (0, 0, 0, pad_right), value=pad_value)
            return buf[:, :T, :]

        pred_semantic = _pad_1d(buf_sem, z)
        pred_primitive = _pad_1d(buf_prim, z)
        pred_flip = _pad_1d(buf_flip, 0)
        sem_prob_full = _pad_1d(buf_sem_prob, 0.0)
        prim_prob_full = _pad_1d(buf_prim_prob, 0.0)

        trans_d_core = torch.stack([buf_tx, buf_ty], dim=-1)  # (B, max_primitives, 2)
        pred_discrete_translation = _pad_2d(trans_d_core, z)
        pred_discrete_cont_translation = _pad_2d(
            self.undiscretize_translation(trans_d_core.float()),
            0.0,
        )
        pred_cont_translation = _pad_2d(buf_trans_soft, 0.0)

        rot_d_core = buf_rot  # (B, max_primitives)
        urot = self.undiscretize_rotation(rot_d_core.float())
        if urot.dim() == 2:
            urot = urot.unsqueeze(-1)  # (B, max_primitives, 1)
        pred_discrete_rotation = _pad_1d(rot_d_core, z)
        pred_discrete_cont_rotation = _pad_2d(urot, 0.0)
        pred_cont_rotation = _pad_2d(buf_rot_soft, 0.0)

        # pred_mask: True where step < done_at[i]
        arange_T = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
        pred_mask = arange_T < done_at.unsqueeze(1)  # (B, T)

        out: Dict[str, Any] = {
            "pred_semantic": pred_semantic,
            "pred_primitive": pred_primitive,
            "pred_mask": pred_mask,
            "pred_flip": pred_flip,
            "pred_discrete_translation": pred_discrete_translation,
            "pred_discrete_rotation": pred_discrete_rotation,
            "pred_discrete_cont_translation": pred_discrete_cont_translation,
            "pred_discrete_cont_rotation": pred_discrete_cont_rotation,
            "pred_cont_translation": pred_cont_translation,
            "pred_cont_rotation": pred_cont_rotation,
            "pred_semantic_prob": sem_prob_full,
            "pred_primitive_prob": prim_prob_full,
        }
        if self.size_pred:
            sz_core = torch.stack([buf_sx, buf_sy], dim=-1)
            out["pred_discrete_size"] = _pad_2d(sz_core, z)
            out["pred_discrete_cont_size"] = _pad_2d(
                self.undiscretize_size(sz_core.float()),
                0.0,
            )
            out["pred_cont_size"] = _pad_2d(buf_size_soft, 0.0)
            out["pred_discrete_scale"] = None
            out["pred_discrete_cont_scale"] = None
            out["pred_cont_scale"] = None
        else:
            sc_core = torch.stack([buf_s0, buf_s1], dim=-1)
            out["pred_discrete_scale"] = _pad_2d(sc_core, z)
            out["pred_discrete_cont_scale"] = _pad_2d(
                self.undiscretize_scale(sc_core.float()),
                0.0,
            )
            out["pred_cont_scale"] = _pad_2d(buf_scale_soft, 0.0)
            out["pred_discrete_size"] = None
            out["pred_discrete_cont_size"] = None
            out["pred_cont_size"] = None
        out["pred_primitive_bg"] = pred_primitive_bg_pre

        # ── Build per-sample ``primitives`` list-of-list (one .cpu() pass) ──
        sem_cpu = buf_sem.cpu().numpy()
        prim_cpu = buf_prim.cpu().numpy()
        flip_cpu = buf_flip.cpu().numpy()
        done_at_cpu = done_at.cpu().numpy()
        cont_t = self.undiscretize_translation(trans_d_core.float()).cpu().numpy()
        cont_r = urot.cpu().numpy()
        if self.size_pred:
            cont_sz = (
                self.undiscretize_size(torch.stack([buf_sx, buf_sy], dim=-1).float())
                .cpu()
                .numpy()
            )
        else:
            cont_sc = (
                self.undiscretize_scale(torch.stack([buf_s0, buf_s1], dim=-1).float())
                .cpu()
                .numpy()
            )

        primitives_per_sample: List[List[Dict[str, Any]]] = []
        for b in range(B):
            n_b = int(done_at_cpu[b])
            sample: List[Dict[str, Any]] = []
            for j in range(n_b):
                rec = {
                    "semantic_id": int(sem_cpu[b, j]),
                    "primitive_id": int(prim_cpu[b, j]),
                    "translation": [float(cont_t[b, j, 0]), float(cont_t[b, j, 1])],
                    "rotation": float(cont_r[b, j, 0]),
                    "flip_mode": int(flip_cpu[b, j]),
                    "oracle_mode": None,
                }
                if self.size_pred:
                    rec["size"] = [float(cont_sz[b, j, 0]), float(cont_sz[b, j, 1])]
                else:
                    rec["scale"] = [float(cont_sc[b, j, 0]), float(cont_sc[b, j, 1])]
                sample.append(rec)
            primitives_per_sample.append(sample)
        out["primitives"] = primitives_per_sample
        return out

    def _empty_sem_geom_batched_dict(self, B, T, device, pred_primitive_bg):
        """Batched analogue of ``_empty_sem_geom_pred_dict`` for B>1."""
        z = self.pad_id
        o: Dict[str, Any] = {
            "primitives": [[] for _ in range(B)],
            "pred_semantic": torch.full((B, T), z, dtype=torch.long, device=device),
            "pred_primitive": torch.full((B, T), z, dtype=torch.long, device=device),
            "pred_mask": torch.zeros((B, T), dtype=torch.bool, device=device),
            "pred_flip": torch.zeros((B, T), dtype=torch.long, device=device),
            "pred_discrete_translation": torch.full(
                (B, T, 2), z, dtype=torch.long, device=device
            ),
            "pred_discrete_rotation": torch.full(
                (B, T), z, dtype=torch.long, device=device
            ),
            "pred_discrete_cont_translation": torch.zeros(
                (B, T, 2), device=device, dtype=torch.float32
            ),
            "pred_discrete_cont_rotation": torch.zeros(
                (B, T, 1), device=device, dtype=torch.float32
            ),
            "pred_cont_translation": torch.zeros(
                (B, T, 2), device=device, dtype=torch.float32
            ),
            "pred_cont_rotation": torch.zeros(
                (B, T, 1), device=device, dtype=torch.float32
            ),
            "pred_semantic_prob": torch.zeros(
                (B, T), device=device, dtype=torch.float32
            ),
            "pred_primitive_prob": torch.zeros(
                (B, T), device=device, dtype=torch.float32
            ),
        }
        if self.size_pred:
            o["pred_discrete_size"] = torch.full(
                (B, T, 2), z, dtype=torch.long, device=device
            )
            o["pred_discrete_cont_size"] = torch.zeros(
                (B, T, 2), device=device, dtype=torch.float32
            )
            o["pred_cont_size"] = torch.zeros(
                (B, T, 2), device=device, dtype=torch.float32
            )
            o["pred_discrete_scale"] = None
            o["pred_discrete_cont_scale"] = None
            o["pred_cont_scale"] = None
        else:
            o["pred_discrete_scale"] = torch.full(
                (B, T, 2), z, dtype=torch.long, device=device
            )
            o["pred_discrete_cont_scale"] = torch.zeros(
                (B, T, 2), device=device, dtype=torch.float32
            )
            o["pred_cont_scale"] = torch.zeros(
                (B, T, 2), device=device, dtype=torch.float32
            )
            o["pred_discrete_size"] = None
            o["pred_discrete_cont_size"] = None
            o["pred_cont_size"] = None
        o["pred_primitive_bg"] = pred_primitive_bg
        return o


def pad_tensor(tensor):
    if tensor.dim() == 3:
        bs, seq_len, dim = tensor.shape
        padding = torch.zeros((bs, 1, dim), dtype=tensor.dtype, device=tensor.device)
    elif tensor.dim() == 2:
        bs, seq_len = tensor.shape
        padding = torch.zeros((bs, 1), dtype=tensor.dtype, device=tensor.device)
    else:
        raise ValueError("Unsupported tensor shape: {}".format(tensor.shape))

    return torch.cat([tensor, padding], dim=1)


def apply_transformation(pc, scale, rotation_vector, translation):
    bs, np, num_points, _ = pc.shape
    scaled_pc = pc * scale.unsqueeze(2)

    rotation_matrix = euler_angles_to_matrix(rotation_vector.view(-1, 3), "XYZ").view(
        bs, np, 3, 3
    )  # euler tmp
    rotated_pc = torch.einsum("bnij,bnpj->bnpi", rotation_matrix, scaled_pc)

    transformed_pc = rotated_pc + translation.unsqueeze(2)

    return transformed_pc


def random_sample_pc(pc, max_lens, n_points=10000):
    bs = max_lens.shape[0]
    max_len = max_lens.max().item() * n_points

    random_values = torch.rand(bs, max_len, device=max_lens.device)
    mask = torch.arange(max_len).expand(bs, max_len).to(max_lens.device) < (
        max_lens * n_points
    )
    masked_random_values = random_values * mask.float()
    _, indices = torch.topk(masked_random_values, n_points, dim=1)

    return pc[torch.arange(bs).unsqueeze(1), indices]
