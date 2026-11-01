"""
Point Transformer - V3 Mode1

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from functools import partial
import torch
import torch.nn as nn

from models.backbone.utils import PointModule, PointSequential, Point
from models.backbone.models.pt_components import (
    Block,
    SerializedPooling,
    SerializedUnpooling,
    Embedding,
)


class PointTransformerV3(PointModule):
    def __init__(
        self,
        in_channels=6,
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
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        bg_pred=False,
        use_dense_feat=False,
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.cls_mode = cls_mode
        self.shuffle_orders = shuffle_orders
        self.bg_pred = bg_pred
        self.use_dense_feat = use_dense_feat
        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.cls_mode or self.num_stages == len(dec_depths) + 1
        assert self.cls_mode or self.num_stages == len(dec_channels) + 1
        assert self.cls_mode or self.num_stages == len(dec_num_head) + 1
        assert self.cls_mode or self.num_stages == len(dec_patch_size) + 1

        # norm layers
        bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        ln_layer = nn.LayerNorm
        # activation layers
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        # encoder
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # decoder
        if not self.cls_mode:
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = PointSequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    SerializedUnpooling(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="up",
                )
                for i in range(dec_depths[s]):
                    dec.add(
                        Block(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=enable_rpe,
                            enable_flash=enable_flash,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                        ),
                        name=f"block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")

    @staticmethod
    def get_downsampled_to_original_mapping(point):
        """
        获取下采样后的点到原始点云的映射关系

        Args:
            point: 经过下采样后的 Point 对象（通常是 encoder 的输出）

        Returns:
            torch.Tensor: 形状为 [N_downsampled]，其中每个值表示该下采样点对应原始点云中的哪个点
                         如果一个下采样点对应多个原始点，返回的是该 cluster 的代表点索引
            或者
            dict: 如果需要更详细的信息，返回包含以下键的字典：
                - 'mapping': [N_downsampled] 下采样点到原始点的映射（cluster head indices）
                - 'inverse_mapping': [N_original] 原始点到下采样点的映射
                - 'num_stages': 经过的下采样阶段数
        """
        if "pooling_parent" not in point.keys():
            # 没有经过下采样，返回恒等映射
            num_points = point.feat.shape[0]
            return torch.arange(num_points, device=point.feat.device)

        # 收集所有层级的 pooling_inverse 信息
        inverse_mappings = []
        current_point = point

        # 从当前层回溯到最初的输入
        while "pooling_parent" in current_point.keys():
            if "pooling_inverse" in current_point.keys():
                inverse_mappings.append(current_point["pooling_inverse"])
            current_point = current_point["pooling_parent"]

        # 反转列表，使其从最早的下采样到最后的下采样
        inverse_mappings.reverse()

        if len(inverse_mappings) == 0:
            # 没有找到任何 pooling_inverse 信息
            num_points = point.feat.shape[0]
            return torch.arange(num_points, device=point.feat.device)

        # 从原始点云开始，逐层追踪到下采样后的点
        # inverse_mappings[i][j] 表示第 i 层下采样中，原始点 j 对应的下采样点索引

        # 构建完整的映射关系
        # final_inverse[i] 表示原始点 i 对应最终下采样后的哪个点
        final_inverse = inverse_mappings[0]
        for inv_map in inverse_mappings[1:]:
            # 将当前层的映射应用到之前累积的映射上
            final_inverse = inv_map[final_inverse]

        # 现在 final_inverse[i] 表示原始点 i 对应下采样后的点索引
        # 但我们需要反过来：下采样后的点 j 对应原始点中的哪些点

        # 获取下采样后的点数
        num_downsampled = point.feat.shape[0]

        # 为每个下采样点找到对应的原始点（取 cluster 中的第一个点作为代表）
        mapping = torch.zeros(
            num_downsampled, dtype=torch.long, device=point.feat.device
        )
        for original_idx in range(len(final_inverse)):
            downsampled_idx = final_inverse[original_idx].item()
            if mapping[downsampled_idx] == 0 or original_idx == 0:
                mapping[downsampled_idx] = original_idx

        return {
            "mapping": mapping,  # [N_downsampled] -> 原始点索引
            "inverse_mapping": final_inverse,  # [N_original] -> 下采样点索引
            "num_stages": len(inverse_mappings),
            "num_original_points": len(final_inverse),
            "num_downsampled_points": num_downsampled,
        }

    @staticmethod
    def get_original_points_for_each_cluster(point):
        """
        获取每个下采样点对应的所有原始点的索引列表

        Args:
            point: 经过下采样后的 Point 对象

        Returns:
            List[List[int]]: 长度为 N_downsampled 的列表，
                            每个元素是一个列表，包含该下采样点对应的所有原始点索引
        """
        mapping_info = PointTransformerV3.get_downsampled_to_original_mapping(point)

        if isinstance(mapping_info, torch.Tensor):
            # 没有经过下采样
            return [[i] for i in range(len(mapping_info))]

        inverse_mapping = mapping_info[
            "inverse_mapping"
        ]  # [N_original] -> 下采样点索引
        num_downsampled = mapping_info["num_downsampled_points"]

        # 为每个下采样点收集所有对应的原始点
        clusters = [[] for _ in range(num_downsampled)]
        for original_idx in range(len(inverse_mapping)):
            downsampled_idx = inverse_mapping[original_idx].item()
            clusters[downsampled_idx].append(original_idx)

        return clusters

    def forward(self, data_dict, return_mapping=False, is_primitive_lib=False):
        """
        Args:
            data_dict: 输入数据字典
            return_mapping: 是否返回下采样点到原始点的映射关系

        Returns:
            feat_dict: 特征字典
            如果 return_mapping=True，还会包含 'point_mapping' 键
        """
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.embedding(point)
        point = self.enc(point)

        feat_dict = dict()
        feat_dict["features"] = point.feat
        feat_dict["offset"] = point.offset
        feat_dict["coordinates"] = point.coord

        # 如果需要，返回映射信息
        if return_mapping:
            feat_dict["point_mapping"] = self.get_downsampled_to_original_mapping(point)

        # 返回降采样后的instance_masks（如果存在）
        if "instance_masks" in point.keys():
            feat_dict["instance_masks"] = point.instance_masks

        if (self.bg_pred or self.use_dense_feat) and not is_primitive_lib:
            dense_point = self.dec(point)
            feat_dict["dense_features"] = dense_point.feat
            feat_dict["dense_offset"] = dense_point.offset
            feat_dict["dense_coordinates"] = dense_point.coord

        return feat_dict
        # ### Get the hierarchical features
        # if not self.cls_mode:
        #     point = self.dec(point)
        # return point
