import torch
from torch import nn
from torch.nn import functional as F
from models.soft_model import SoftModel as SingleSoftModel
from models.prompt import TrainablePrompt


def assignment_loss(logits, target, tokens, instances):
    logits = logits[:, : target.shape[1]]
    assert logits.shape == target.shape
    if not instances.any():
        return (
            logits.masked_fill(~(tokens[:, None, :] & instances[:, :, None]), 0).sum()
            * 0
        )
    # Never evaluate BCE on arbitrary padded values (even NaN/Inf test sentinels).
    valid = tokens[:, None, :] & instances[:, :, None]
    z = logits.masked_fill(~valid, 0)
    t = target.masked_fill(~valid, 0)
    positive = t.sum(-1)
    negative = (tokens[:, None, :].float() - t).sum(-1)
    assert (positive[instances] > 0).all()
    ce = F.binary_cross_entropy_with_logits(z, t, reduction="none").masked_fill(
        ~valid, 0
    )
    balanced = (ce * t).sum(-1) / positive.clamp_min(1)
    balanced += (ce * (1 - t)).sum(-1) / negative.clamp_min(1)
    balanced /= 1 + (negative > 0).float()
    p = z.sigmoid().masked_fill(~valid, 0)
    dice = 1 - (2 * (p * t).sum(-1) + 1e-6) / (p.sum(-1) + positive + 1e-6)
    return (balanced + dice)[instances].mean()


class BatchedPrompt(TrainablePrompt):
    def forward(self, x, **kwargs):
        assert kwargs.get("cache") is None and "cross_attn_bias" not in kwargs
        self.prefix = int(kwargs["prefix_attn_len"])
        context = kwargs["context"]
        valid = kwargs["context_mask"]
        assert context.shape[:2] == self.coords.shape[:2] == valid.shape
        assert valid.any(-1).all()
        self.valid = valid
        self.phase = "global"
        self.global_hidden = self.original(x, **kwargs)[0]
        self.queries = self.global_hidden[:, self.prefix :: 6]
        # Padded keys/coordinates cannot affect valid logits. Invalid query rows are excluded only in loss.
        self.logits = self.selector(
            self.queries,
            context.masked_fill(~valid[..., None], 0),
            self.coords.masked_fill(~valid[..., None], 0),
        )
        probability = self.logits.sigmoid().masked_fill(~valid[:, None, :], 0)
        expanded = probability.repeat_interleave(6, dim=1)[
            :, : x.shape[1] - self.prefix
        ]
        bias = F.pad(self.beta * expanded[:, None], (self.mem, 0, self.prefix, 0))
        assert bias.shape == (x.shape[0], 1, x.shape[1], context.shape[1] + self.mem)
        assert torch.isfinite(bias).all()
        self.phase = "guided"
        result = self.original(x, **dict(kwargs, cross_attn_bias=bias))
        self.phase = None
        self.calls += 1
        return result

    def audit(self, module, args, kwargs):
        if self.phase != "guided":
            return
        mask, bias = kwargs.get("mask"), kwargs.get("attn_bias")
        assert mask is not None and mask.dtype == torch.bool and bias is not None
        # x-transformers broadcasts a B x 1 x 1 x (memory+tokens) validity mask.
        expected = F.pad(self.valid, (self.mem, 0), value=True)[:, None, None, :]
        assert torch.equal(mask, expected.expand_as(mask)), (
            "attention mask mismatch",
            mask.shape,
            expected.shape,
        )
        assert (bias[..., : self.mem] == 0).all() and (
            bias[:, :, : self.prefix] == 0
        ).all()
        assert (
            bias[..., self.mem :].masked_select(
                ~self.valid[:, None, None, :].expand_as(bias[..., self.mem :])
            )
            == 0
        ).all()
        self.attention_checks += 1


class SoftModel(SingleSoftModel):
    def forward(
        self, batch, coords, targets, denominators, real=1.0, beta=2.0, audit=False
    ):
        values = targets["values"]
        im = targets["instances"]
        tm = targets["tokens"]
        real *= targets["real"]
        assert torch.equal(im, batch["primitive_id"] >= 0)
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
        captures = []
        handle = (
            self.model.conditioner.register_forward_hook(
                lambda m, a, o: captures.append(o)
            )
            if audit
            else None
        )
        try:
            with BatchedPrompt(self.model, self.selector, coords, beta=beta) as prompt:
                r = self.model(**batch, is_train=False)
                assert torch.equal(prompt.valid, tm)
                selection = assignment_loss(prompt.logits, values, tm, im)
                logits = prompt.logits[:, : values.shape[1]]
                if audit:
                    assert len(captures) == 2
                    f, c, cu, sem, keep, mapping = captures[-1]
                    _, pc, valid = self.model.offset_to_batch(f, c, cu)
                    assert torch.equal(valid, tm) and torch.allclose(
                        pc[tm], coords[tm], atol=2e-6, rtol=0
                    )
                    assert keep.all()
                    expected = torch.cat(
                        [torch.arange(int(n), device=c.device) for n in tm.sum(-1)]
                    )
                    assert torch.equal(
                        mapping.long(), expected
                    ), "Cross-scene ownership mismatch"
                    assert all(
                        int(m.num_batches_tracked - v) == 1 for m, v in bn_before
                    )
        finally:
            if handle:
                handle.remove()
        n = im.sum()
        ne = n + im.shape[0]
        nr = r["recon_full_effective_count"] + r["recon_partial_effective_count"]
        expected = (
            im
            & batch["recon_valid_mask"].bool()
            & batch["recon_preclip_source_mask"].any(-1)
            & self.model.attr_recon_template_mask[
                batch["primitive_id"].clamp_min(0).long()
            ].any(-1)
        ).sum()
        assert nr == expected, ("recon denominator mismatch", nr, expected)
        we = self.model.loss_weight["eos"]
        wr = r["recon_weight"]
        attrs = r["loss"] - we * r["loss_eos"] - wr * r["loss_recon"]
        objective = real * (
            attrs * n / denominators[0].clamp_min(1)
            + we * r["loss_eos"] * ne / denominators[1].clamp_min(1)
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
            eos_sum=r["loss_eos"].detach() * ne * real,
            recon_sum=r["loss_recon"].detach() * nr * real,
            assignment_sum=selection.detach() * n * real,
            n=n * real,
            ne=ne * real,
            nr=nr.detach() * real,
            scenes=objective.new_tensor(im.shape[0] * real),
        )
        return dict(
            objective=objective,
            metrics=metrics,
            selector_logits=logits.detach(),
            predictions={k: v.detach() for k, v in r.items() if torch.is_tensor(v)},
        )
