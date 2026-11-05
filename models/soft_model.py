import torch
from torch import nn
from models import create_model
from models.prompt import TargetSelector, TrainablePrompt, assignment_loss


class SoftModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = create_model("attr_decouple", cfg["model"])
        self.selector = TargetSelector()
        self.assignment_weight = cfg["soft_guidance"]["assignment_weight"]

    def clear_cache(self):
        self.model._plib_pooled_cache = None
        self.model._lib_feats_cache_valid = False

    def counts(self, sample):
        # Count instances for preclip_full reconstruction with instance_mean reduction.
        m = self.model
        assert (
            m.recon_partial_source_mode == "preclip_full"
            and m.recon_gt_geometry_threshold < 0
        )
        assert (
            m.recon_instance_reduction == "instance_mean"
            and m.recon_partial_weight == 1
        )
        n = len(sample["primitive_id"])
        ids = torch.as_tensor(
            sample["primitive_id"], device=m.attr_recon_template_mask.device
        ).long()
        available = m.attr_recon_template_mask[ids].any(-1).cpu()
        valid = (
            sample["recon_valid_mask"].bool()
            & sample["recon_preclip_source_mask"].any(-1)
            & available
        )
        return n, n + 1, int(valid.sum())

    def forward(
        self, batch, coords, targets, denominators, real=1.0, beta=2.0, audit=False
    ):
        captures = []
        bn_before = (
            [
                (m, m.num_batches_tracked.clone())
                for m in self.model.modules()
                if isinstance(m, nn.modules.batchnorm._BatchNorm)
                and m.num_batches_tracked is not None
            ]
            if audit and self.training
            else []
        )
        handle = (
            self.model.conditioner.register_forward_hook(
                lambda m, a, o: captures.append(o)
            )
            if audit
            else None
        )
        try:
            with TrainablePrompt(
                self.model, self.selector, coords, beta=beta
            ) as prompt:
                r = self.model(**batch, is_train=False)
                selection = assignment_loss(prompt.logits, targets)
                z = prompt.logits[:, : targets.shape[1]]
                if audit:
                    assert len(captures) == 2, "Expected template + scene forward only"
                    f, c, cu, sem, keep, mapping = captures[-1]
                    assert torch.allclose(c, coords[0], atol=2e-6, rtol=0)
                    assert keep.all() and torch.equal(
                        mapping.long(), torch.arange(len(c), device=c.device)
                    )
                    assert all(
                        int(m.num_batches_tracked - before) == 1
                        for m, before in bn_before
                    ), "BN updated outside scene pass"
        finally:
            if handle:
                handle.remove()
        n = targets.shape[1]
        nr = r["recon_full_effective_count"] + r["recon_partial_effective_count"]
        expected = (
            (batch["primitive_id"] >= 0)
            & batch["recon_valid_mask"].bool()
            & batch["recon_preclip_source_mask"].any(-1)
            & self.model.attr_recon_template_mask[
                batch["primitive_id"].clamp_min(0).long()
            ].any(-1)
        ).sum()
        assert nr == expected, ("recon denominator drift", nr, expected)
        we = self.model.loss_weight["eos"]
        wr = r["recon_weight"]
        attrs = r["loss"] - we * r["loss_eos"] - wr * r["loss_recon"]
        # Caller multiplies by world_size to cancel DDP's gradient averaging.
        objective = real * (
            attrs * n / denominators[0].clamp_min(1)
            + we * r["loss_eos"] * (n + 1) / denominators[1].clamp_min(1)
            + wr * r["loss_recon"] * nr / denominators[2].clamp_min(1)
            + self.assignment_weight * selection * n / denominators[0].clamp_min(1)
        )
        metrics = {
            k: v.detach()
            for k, v in r.items()
            if torch.is_tensor(v)
            and v.numel() == 1
            and (k.startswith("loss") or k.startswith("recon"))
        }
        metrics.update(
            assignment=selection.detach(),
            attr_sum=attrs.detach() * n * real,
            eos_sum=r["loss_eos"].detach() * (n + 1) * real,
            recon_sum=r["loss_recon"].detach() * nr * real,
            assignment_sum=selection.detach() * n * real,
            n=torch.tensor(n * real, device=objective.device),
            ne=torch.tensor((n + 1) * real, device=objective.device),
            nr=nr.detach() * real,
            scenes=torch.tensor(real, device=objective.device),
        )
        # DDP sees only the consumed objective graph. Diagnostics are all detached.
        detached = {k: v.detach() for k, v in r.items() if torch.is_tensor(v)}
        return dict(
            objective=objective,
            metrics=metrics,
            predictions=detached,
            selector_logits=z.detach(),
        )
