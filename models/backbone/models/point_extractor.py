"""
Point Cloud Feature Extractor using PointTransformerV3

Author: Based on Xiaoyang Wu's implementation
Adapted for standalone use as feature extractor
"""

import torch
import torch.nn as nn
import os
from collections import OrderedDict
from models.backbone.models.point_transformer_v3 import PointTransformerV3
from models.backbone.utils import Point


class PointFeatureExtractor(nn.Module):
    """
    Point Cloud Feature Extractor using PointTransformerV3 backbone

    This class wraps the PointTransformerV3 model and provides a clean interface
    for extracting features from point clouds for downstream tasks.
    """

    def __init__(
        self,
        in_channels=6,
        backbone_out_channels=64,
        enc_channels=512,
        dec_channels=64,
        bg_pred=False,
        freeze_backbone=False,
        use_dense_feat=False,
        backbone_config=None,
    ):
        """
        Args:
            in_channels (int): Number of input channels (default: 6 for xyz+rgb)
            backbone_out_channels (int): Expected output channels from backbone
            freeze_backbone (bool): Whether to freeze backbone parameters
            backbone_config (dict): Custom configuration for PointTransformerV3
        """
        super().__init__()
        self.bg_pred = bg_pred
        self.use_dense_feat = use_dense_feat
        # Default backbone configuration
        default_config = dict(
            use_dense_feat=use_dense_feat,
            bg_pred=bg_pred,
            in_channels=in_channels,
            order=("z", "z-trans"),
            stride=(2, 2, 2, 2),
            enc_depths=(2, 2, 2, 6, 2),
            enc_channels=(32, 64, 128, 256, 512),
            enc_num_head=(2, 4, 8, 16, 32),
            enc_patch_size=(48, 48, 48, 48, 48),
            dec_depths=(2, 2, 2, 2),
            dec_channels=(64, 64, 128, 256),
            dec_num_head=(4, 4, 8, 16),
            dec_patch_size=(48, 48, 48, 48),
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            drop_path=0.3,
            pre_norm=True,
            shuffle_orders=True,
            enable_rpe=False,
            enable_flash=False,  # Default to False for better compatibility
            upcast_attention=True,
            upcast_softmax=True,
            cls_mode=False,
        )

        # Update with custom config if provided
        if backbone_config is not None:
            default_config.update(backbone_config)

        # Create backbone
        self.backbone = PointTransformerV3(**default_config)
        self.freeze_backbone = freeze_backbone
        self.backbone_out_channels = backbone_out_channels

        self.enc_channels = enc_channels
        self.dec_channels = dec_channels

        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        super(PointFeatureExtractor, self).train(mode)
        if mode and not self.freeze_backbone:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()

    def load_weights(
        self,
        weight_path,
        strict=True,
        map_location=None,
        backbone_only=True,
        key_mapping=None,
    ):
        """
        Load pre-trained weights from a checkpoint file

        Args:
            weight_path (str): Path to the weight file (.pth or .ckpt)
            strict (bool): Whether to strictly enforce that the keys in state_dict
                         match the keys returned by this module's state_dict()
            map_location: Device to map the weights to (e.g., 'cpu', 'cuda:0')
            backbone_only (bool): Whether to load only backbone weights (ignore head weights)
            key_mapping (dict): Custom key mapping for weight names. If None, will try
                              common mappings automatically

        Returns:
            dict: Loading result information including missing/unexpected keys
        """
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Weight file not found: {weight_path}")

        # Load checkpoint
        print(f"Loading weights from: {weight_path}")

        if map_location is None:
            map_location = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint = torch.load(weight_path, map_location=map_location)

        # Extract state_dict from checkpoint
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Apply key mapping for compatibility
        state_dict = self._map_state_dict_keys(state_dict, key_mapping, backbone_only)

        # Load weights
        if backbone_only:
            # Only load backbone weights
            result = self.backbone.load_state_dict(state_dict, strict=strict)
        else:
            # Load full model weights
            result = self.load_state_dict(state_dict, strict=strict)

        print(f"Successfully loaded weights!")
        if result.missing_keys:
            print(f"Missing keys: {result.missing_keys}")
        if result.unexpected_keys:
            print(f"Unexpected keys: {result.unexpected_keys}")

        return {
            "missing_keys": result.missing_keys,
            "unexpected_keys": result.unexpected_keys,
            "loaded_from": weight_path,
        }

    def _map_state_dict_keys(self, state_dict, key_mapping=None, backbone_only=True):
        """
        Map state_dict keys to match the current model structure

        Args:
            state_dict (dict): Original state dictionary
            key_mapping (dict): Custom key mapping
            backbone_only (bool): Whether to extract only backbone weights

        Returns:
            dict: Mapped state dictionary
        """
        new_state_dict = OrderedDict()

        # Default key mappings for common patterns
        default_mappings = [
            # From Pointcept format to standalone format
            (r"^backbone\.", ""),  # Remove backbone prefix
            (r"^model\.backbone\.", ""),  # Remove model.backbone prefix
            (r"^module\.backbone\.", ""),  # Remove module.backbone prefix
            (r"^segmentor\.backbone\.", ""),  # Remove segmentor.backbone prefix
        ]

        # Add custom mappings if provided
        if key_mapping:
            if isinstance(key_mapping, dict):
                # Convert dict to list of tuples
                custom_mappings = [(k, v) for k, v in key_mapping.items()]
            else:
                custom_mappings = key_mapping
            default_mappings.extend(custom_mappings)

        # Apply mappings
        for old_key, value in state_dict.items():
            new_key = old_key

            # Apply default mappings
            import re

            for pattern, replacement in default_mappings:
                new_key = re.sub(pattern, replacement, new_key)

            # Filter for backbone only if requested
            if backbone_only:
                # Skip non-backbone keys (like classifier heads)
                skip_patterns = [
                    r"^cls_head",
                    r"^classifier",
                    r"^head",
                    r"^segment_head",
                    r"^decode_head",
                ]

                should_skip = False
                for pattern in skip_patterns:
                    if re.match(pattern, new_key):
                        should_skip = True
                        break

                if should_skip:
                    continue

            new_state_dict[new_key] = value

        return new_state_dict

    def offset_to_batch(self, feat, coord, offset):
        """
        将连续的特征tensor根据offset转换为batch形式并进行padding

        Args:
            feat (torch.Tensor): 特征tensor，形状为[N, feature_dim]
            offset (torch.Tensor): batch offset信息，形状为[bs]，表示每个sample的点数

        Returns:
            torch.Tensor: 转换后的特征tensor，形状为[bs, seq_len, feature_dim]
        """
        bs = offset.shape[0]
        feature_dim = feat.shape[1]

        # 计算每个sample的实际长度
        lengths = offset.clone()
        if bs > 1:
            lengths[1:] = offset[1:] - offset[:-1]

        # 找到最大序列长度
        seq_len = lengths.max().item()

        # 创建输出tensor
        batch_feat = torch.zeros(
            bs, seq_len, feature_dim, dtype=feat.dtype, device=feat.device
        )
        batch_coord = torch.zeros(
            bs, seq_len, 3, dtype=coord.dtype, device=coord.device
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

    def instance_masks_to_list(self, instance_masks, offset):
        """
        将拼接在一起的instance_masks根据offset转换为list格式

        Args:
            instance_masks (torch.Tensor): 降采样后的instance masks，形状为[N_inst, N_total]
            offset (torch.Tensor): batch offset信息，形状为[bs]，表示每个sample的点数累积和

        Returns:
            List[torch.Tensor]: 长度为bs的list，每个元素形状为[N_inst, N_sample_i]
        """
        bs = offset.shape[0]
        n_inst = instance_masks.shape[0]

        # 计算每个sample的实际长度
        lengths = offset.clone()
        if bs > 1:
            lengths[1:] = offset[1:] - offset[:-1]

        # 根据offset分割instance_masks
        masks_list = []
        start_idx = 0
        for i in range(bs):
            end_idx = start_idx + lengths[i].item()
            # 提取当前样本的mask: [N_inst, N_sample_i]
            sample_mask = instance_masks[:, start_idx:end_idx]
            masks_list.append(sample_mask)
            start_idx = end_idx

        return masks_list

    def save_weights(self, save_path, include_config=True):
        """
        Save current model weights

        Args:
            save_path (str): Path to save the weights
            include_config (bool): Whether to include model configuration
        """
        checkpoint = {
            "state_dict": self.state_dict(),
            "backbone_out_channels": self.backbone_out_channels,
        }

        if include_config:
            # Try to extract backbone config
            try:
                checkpoint["backbone_config"] = self.backbone.get_config()
            except:
                print("Warning: Could not extract backbone config")

        torch.save(checkpoint, save_path)
        print(f"Weights saved to: {save_path}")

    def mapping_to_list(self, point_mapping, src_offset, tgt_offset):
        """
        将映射关系转换为 batch 形式的列表

        Args:
            point_mapping (dict): 包含映射信息的字典，其中:
                - 'mapping': [N_tgt] -> src点的全局索引
                - 'inverse_mapping': [N_src] -> tgt点的全局索引
            src_offset (torch.Tensor): 源点云（原始点云）的累积offset，形状 [bs]
            tgt_offset (torch.Tensor): 目标点云（下采样后）的累积offset，形状 [bs]

        Returns:
            list: 长度为 bs 的列表，每个元素是该样本的映射张量（局部索引）
        """
        tgt_point_mapping = []
        assert len(src_offset) == len(
            tgt_offset
        ), "src_offset and tgt_offset must have the same length"
        bs = src_offset.shape[0]

        # offset 是累积值，例如 [100, 250, 400] 表示三个样本
        # 第一个样本: [0, 100), 第二个样本: [100, 250), 第三个样本: [250, 400)
        for i in range(bs):
            # 计算当前 batch 的起始和结束位置
            src_start_idx = 0 if i == 0 else src_offset[i - 1].item()
            src_end_idx = src_offset[i].item()

            tgt_start_idx = 0 if i == 0 else tgt_offset[i - 1].item()
            tgt_end_idx = tgt_offset[i].item()

            # 提取当前 batch 的映射关系
            # point_mapping['mapping'][tgt_idx] 表示目标点 tgt_idx 对应的源点的全局索引
            batch_mapping = point_mapping["mapping"][tgt_start_idx:tgt_end_idx].clone()

            # 将全局索引转换为当前 batch 内的局部索引
            batch_mapping -= src_start_idx

            # 验证索引范围是否正确
            src_num = src_end_idx - src_start_idx
            assert batch_mapping.min() >= 0 and batch_mapping.max() < src_num, (
                f"Batch {i}: mapping index out of range. "
                f"Expected [0, {src_num}), got [{batch_mapping.min()}, {batch_mapping.max()}]"
            )

            tgt_point_mapping.append(batch_mapping)

        return tgt_point_mapping

    def inverse_mapping_to_list(self, point_mapping, src_offset, tgt_offset):
        """
        将反向映射关系转换为 batch 形式的列表（原始点 -> 下采样点）

        Args:
            point_mapping (dict): 包含映射信息的字典，其中:
                - 'inverse_mapping': [N_src] -> tgt点的全局索引
            src_offset (torch.Tensor): 源点云（原始点云）的累积offset，形状 [bs]
            tgt_offset (torch.Tensor): 目标点云（下采样后）的累积offset，形状 [bs]

        Returns:
            list: 长度为 bs 的列表，每个元素是该样本的反向映射张量（局部索引）
        """
        src_point_mapping = []
        assert len(src_offset) == len(
            tgt_offset
        ), "src_offset and tgt_offset must have the same length"
        bs = src_offset.shape[0]

        for i in range(bs):
            # 计算当前 batch 的起始和结束位置
            src_start_idx = 0 if i == 0 else src_offset[i - 1].item()
            src_end_idx = src_offset[i].item()

            tgt_start_idx = 0 if i == 0 else tgt_offset[i - 1].item()
            tgt_end_idx = tgt_offset[i].item()

            # 提取当前 batch 的反向映射关系
            # point_mapping['inverse_mapping'][src_idx] 表示源点 src_idx 对应的目标点的全局索引
            batch_inverse_mapping = point_mapping["inverse_mapping"][
                src_start_idx:src_end_idx
            ].clone()

            # 将全局索引转换为当前 batch 内的局部索引
            batch_inverse_mapping -= tgt_start_idx

            # 验证索引范围是否正确
            tgt_num = tgt_end_idx - tgt_start_idx
            assert (
                batch_inverse_mapping.min() >= 0
                and batch_inverse_mapping.max() < tgt_num
            ), (
                f"Batch {i}: inverse mapping index out of range. "
                f"Expected [0, {tgt_num}), got [{batch_inverse_mapping.min()}, {batch_inverse_mapping.max()}]"
            )

            src_point_mapping.append(batch_inverse_mapping)

        return src_point_mapping

    def forward(self, input_dict, return_mapping=False, is_primitive_lib=False):
        """
        Forward pass through the feature extractor

        Args:
            input_dict (dict): Input dictionary containing:
                - coord: Point coordinates (N, 3)
                - feat: Point features (N, C)
                - offset: Batch offset information (B,)
                - grid_size: Grid size for voxelization (optional)
                - instance_masks: Instance masks (optional) [n_inst, N_total]
            return_point (bool): Whether to return the Point object

        Returns:
            tuple: feat, coord, feat_key_padding_mask, dense_feat, dense_coord,
                   dense_feat_key_padding_mask, tgt_point_mapping, downsampled_instance_masks
        """
        # Create Point object from input
        point = Point(input_dict)

        # Extract features using backbone
        if self.freeze_backbone:
            with torch.no_grad():
                feat_dict = self.backbone(
                    point,
                    return_mapping=return_mapping,
                    is_primitive_lib=is_primitive_lib,
                )
        else:
            feat_dict = self.backbone(
                point, return_mapping=return_mapping, is_primitive_lib=is_primitive_lib
            )

        feat = feat_dict["features"]
        offset = feat_dict["offset"]
        coord = feat_dict["coordinates"]

        # multi_scale_feat = feat_dict['multi_scale_features']
        # feat, offset = multi_scale_feat[-3]['features'], multi_scale_feat[-3]['offset']
        # 将feat从[N, 512]转换为[bs, seq_len, 512]
        feat, coord, feat_key_padding_mask = self.offset_to_batch(feat, coord, offset)

        tgt_point_mapping = None
        if return_mapping:
            #### 映射关系转化为list
            point_mapping = feat_dict["point_mapping"]
            src_offset = input_dict["offset"]
            tgt_offset = feat_dict["offset"]
            tgt_point_mapping = self.mapping_to_list(
                point_mapping, src_offset, tgt_offset
            )

        dense_feat, dense_coord, dense_feat_key_padding_mask = None, None, None
        if (self.bg_pred or self.use_dense_feat) and not is_primitive_lib:
            dense_feat = feat_dict["dense_features"]
            dense_offset = feat_dict["dense_offset"]
            dense_coord = feat_dict["dense_coordinates"]
            dense_feat, dense_coord, dense_feat_key_padding_mask = self.offset_to_batch(
                dense_feat, dense_coord, dense_offset
            )

        # 获取降采样后的instance_masks（如果存在）
        downsampled_instance_masks = None
        if "instance_masks" in feat_dict:
            downsampled_instance_masks_tensor = feat_dict[
                "instance_masks"
            ]  # [N_inst, N_total]
            # 将拼接的tensor转换为list格式
            downsampled_instance_masks = self.instance_masks_to_list(
                downsampled_instance_masks_tensor, offset
            )

        return (
            feat,
            coord,
            feat_key_padding_mask,
            dense_feat,
            dense_coord,
            dense_feat_key_padding_mask,
            tgt_point_mapping,
            downsampled_instance_masks,
        )

    def extract_features(self, input_dict):
        """
        Simplified interface for feature extraction

        Args:
            input_dict (dict): Input point cloud data

        Returns:
            torch.Tensor: Extracted features (N, C)
        """
        result = self.forward(input_dict, return_point=False)
        return result["features"]

    def get_feature_dim(self):
        """
        Get the dimension of extracted features

        Returns:
            int: Feature dimension
        """
        # This would be determined by the final decoder output
        # For the default config, it should be dec_channels[0] = 64
        return self.backbone_out_channels


def create_point_extractor(pretrained_path=None, **kwargs):
    """
    Factory function to create predefined point extractors

    Args:
        config_name (str): Name of predefined configuration
        pretrained_path (str): Path to pretrained weights (optional)
        **kwargs: Additional arguments to override default config

    Returns:
        PointFeatureExtractor: Configured feature extractor
    """
    config = dict(
        bg_pred=kwargs["bg_pred"],
        use_dense_feat=kwargs["use_dense_feat"],
        freeze_backbone=kwargs["freeze_backbone"],
        in_channels=6,
        backbone_out_channels=64,
        enc_channels=512,
        backbone_config=dict(
            in_channels=6,
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            stride=(2, 2, 2, 2),
            enc_depths=(2, 2, 2, 6, 2),
            enc_channels=(32, 64, 128, 256, 512),
            enc_num_head=(2, 4, 8, 16, 32),
            enc_patch_size=(1024, 1024, 1024, 1024, 1024),
            dec_depths=(2, 2, 2, 2),
            dec_channels=(64, 64, 128, 256),
            dec_num_head=(4, 4, 8, 16),
            dec_patch_size=(1024, 1024, 1024, 1024),
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            drop_path=0.3,
            shuffle_orders=True,
            pre_norm=True,
            enable_rpe=False,
            enable_flash=True,
            upcast_attention=False,
            upcast_softmax=False,
            cls_mode=False,
        ),
    )
    config.update(kwargs)

    # Create extractor
    extractor = PointFeatureExtractor(**config)

    # Load pretrained weights if provided
    if pretrained_path is not None:
        extractor.load_weights(pretrained_path, strict=False, backbone_only=True)

    # extractor.eval()

    return extractor
