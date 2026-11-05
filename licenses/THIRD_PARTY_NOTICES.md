# Third-party notices

Third-party components retain their own licenses and copyright notices.

| Component | Location | License and source |
| --- | --- | --- |
| PointTransformerV3 | `models/backbone/models/model.py` and adapted backbone components | MIT; detached implementation at `13dfa65c2603e4f238ee7ca844dbf1dfa2003504` |
| Pointcept | `models/backbone/` helpers | MIT |
| Hilbert serialization | Both backbone `serialization/hilbert.py` copies | numpy-hilbert-curve: MIT |
| Z-order serialization | Both backbone `serialization/z_order.py` copies | ocnn-pytorch: MIT; Copyright (c) 2022 Peng-Shuai Wang |
| Registry | `models/primitive/utils/registry.py` | MMCV: Apache-2.0 |
| Positional embedding | `models/primitive/positional_embedding.py` | 3DETR: Apache-2.0; Copyright (c) Facebook, Inc. and its affiliates |
| Instance evaluator | `evaluation/instance_eval_hungarian.py` | Recorded SymPointV2 attribution; exact source and license remain unconfirmed |

The bundled PTv3 implementation predates the upstream license change on
2026-10-30. Its historical MIT license is included in `upstream/`.

Source references, comparison versions and unresolved items are recorded in
`UPSTREAM_REFERENCES.json`. License texts are in
`upstream/`. Adaptations include the backbone integration, positional embedding,
registry and instance evaluator.

Dependency notices are in `environment/licenses/`, with package versions and
source information in `environment/DEPENDENCY_NOTICES.json`.
