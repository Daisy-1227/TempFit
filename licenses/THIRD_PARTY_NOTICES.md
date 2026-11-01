# Third-party notices

项目作者仅对其有权许可的自有部分采用 MIT。以下第三方部分不由项目 MIT 重新许可。原文件内的版权与来源注释予以保留。

当前目录用于本地代码审查，许可适用范围尚有未决项；README 致谢不能替代许可证、版权声明或必要授权。

| 组件 | 当前代码位置 | 证据和待办 |
| --- | --- | --- |
| PointTransformerV3 / Pointcept | `models/backbone/` | 本地注明改编自独立 PTv3 仓库；已保存的 PTv3 许可为 CC BY-NC 4.0，Pointcept 为 MIT。具体本地版本和逐文件适用范围待核实，不能统一标为 MIT。 |
| SymPointV2 Hungarian evaluator | `evaluation/instance_eval_hungarian.py`、`evaluation/pq_protocol.py` | 原摘录来源已记录，再分发许可尚未确认；保留供本地 review，公开前需确认、替换或排除相关模块。 |
| Hilbert / Z-order | `models/backbone/models/serialization/`、`models/backbone/utils/serialization/` | 保留原作者声明；Hilbert 的 MIT 参考已保存，准确版本及 Z-order 完整声明待补。 |
| Registry | `models/primitive/utils/registry.py` | 保留 OpenMMLab 版权；MMCV Apache-2.0 为待确认版本的参考。 |
| 位置编码 | `models/primitive/positional_embedding.py` | 历史登记包含 Facebook / 3DETR 来源线索；精确派生关系待核实，不能将参考许可视为适用结论。 |
| Slot attention | `models/primitive/slot_attn.py` | SLATE 与 BO-QSA 混合来源；SLATE MIT 参考已保存，BO-QSA 摘录许可待核实。 |
| masking / SPOT utilities | `models/primitive/utils/spot_utils.py` | OCL、SLATE、DeiT 多来源线索；逐段适用许可待核实。 |

`licenses/upstream/` 保存已取得的许可参考全文，`licenses/UPSTREAM_REFERENCES.json` 记录来源 URL、日期、内容哈希和未决状态。上游当前许可快照不等于本地复制版本的许可证认定。未取得许可的项目不创建虚构 LICENSE。

依赖 wheel 的声明快照位于 `environment/licenses/`，索引为 `environment/DEPENDENCY_NOTICES.json`。本目录不分发 wheel 或其他第三方二进制；声明提取也不代表已审核其二进制再分发条件。
