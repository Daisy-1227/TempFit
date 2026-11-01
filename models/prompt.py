import hashlib
import torch
from torch import nn
from torch.nn import functional as F

from models.selector import TargetSelector


HEAD_ROOTS = {
    "to_semantic_logits",
    "pred_proj",
    "lib_proj",
    "logit_scale",
    "to_trans_x_logits",
    "to_trans_y_logits",
    "to_flip_logits",
    "to_rotation_logits",
    "to_scale_logits",
    "to_eos_logits",
}


def configure(model, arm):
    model.eval().requires_grad_(False)
    if arm != "A":
        for name, parameter in model.named_parameters():
            if name.split(".")[0] in HEAD_ROOTS | {"decoder"}:
                parameter.requires_grad_(True)
    return {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}


def training_mode(model):
    # Root flag controls the original Gumbel/Huber training definition.
    # Frozen feature extractors, input embeddings and GateLoop remain in eval.
    model.eval()
    model.training = True
    for root in HEAD_ROOTS | {"decoder"}:
        module = getattr(model, root, None)
        if isinstance(module, nn.Module):
            module.train()


def digest(model, frozen_only=False):
    parameters = dict(model.named_parameters())
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if frozen_only and name in parameters and parameters[name].requires_grad:
            continue
        a = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str((a.shape, a.dtype)).encode())
        h.update(a.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def assignment_loss(logits, targets):
    """Per-instance balanced BCE + Dice; absent groups are not fabricated."""
    if targets.shape[1] == 0:
        return logits.sum() * 0.0
    logits = logits[:, : targets.shape[1]]
    assert logits.shape == targets.shape
    targets = targets.float()
    positive = targets.sum(-1)
    negative = (1 - targets).sum(-1)
    assert (positive > 0).all()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    balanced = (ce * targets).sum(-1) / positive + (ce * (1 - targets)).sum(
        -1
    ) / negative.clamp_min(1)
    balanced = balanced / (1 + (negative > 0).float())
    probability = logits.sigmoid()
    dice = 1 - (2 * (probability * targets).sum(-1) + 1e-6) / (
        probability.sum(-1) + positive + 1e-6
    )
    return (balanced + dice).mean()


class TrainablePrompt:
    def __init__(self, model, selector, coords, beta=2.0, mode="tf"):
        assert mode in {"tf", "ar"}
        assert model.attr_decouple and model.num_attr_tokens == 6
        assert not model.bg_pred and not model.shape_cond_with_perceiver_prefix
        self.model, self.selector, self.coords = model, selector, coords
        self.beta, self.mode = float(beta), mode
        self.calls = self.eos_checks = self.attention_checks = 0
        self.blocks = [
            b
            for t, (_, b, _) in zip(model.decoder.layer_types, model.decoder.layers)
            if t == "c"
        ]
        assert len(self.blocks) == 6
        assert len({b.num_mem_kv for b in self.blocks}) == 1
        self.mem = self.blocks[0].num_mem_kv
        self.phase = None

    def __enter__(self):
        self.original = self.model.decoder.forward
        self.model.decoder.forward = self.forward
        self.handles = [self.model.to_eos_logits.register_forward_pre_hook(self.eos)]
        for block in self.blocks:
            self.handles.append(
                block.attend.register_forward_pre_hook(self.audit, with_kwargs=True)
            )
        return self

    def __exit__(self, *args):
        self.model.decoder.forward = self.original
        for handle in self.handles:
            handle.remove()

    def forward(self, x, **kwargs):
        assert kwargs.get("cache") is None and "cross_attn_bias" not in kwargs
        self.prefix = int(kwargs["prefix_attn_len"])
        context = kwargs["context"]
        valid = kwargs["context_mask"]
        assert x.shape[0] == 1, "Initial fitting protocol uses one scene per microbatch"
        assert valid.all(), "Unpadded single-scene memory is required"
        assert context.shape[:2] == self.coords.shape[:2]
        self.phase = "global"
        global_result = self.original(x, **kwargs)
        self.global_hidden = global_result[0]
        self.queries = self.global_hidden[:, self.prefix :: 6]
        self.logits = self.selector(self.queries, context, self.coords)
        probability = self.logits.sigmoid()
        expanded = probability.repeat_interleave(6, dim=1)[
            :, : x.shape[1] - self.prefix
        ]
        bias = F.pad(self.beta * expanded[:, None], (self.mem, 0, self.prefix, 0))
        assert bias.shape == (x.shape[0], 1, x.shape[1], context.shape[1] + self.mem)
        assert (
            torch.isfinite(bias).all()
            and (bias >= 0).all()
            and (bias <= self.beta + 1e-6).all()
        )
        self.phase = "guided"
        guided_result = self.original(x, **dict(kwargs, cross_attn_bias=bias))
        self.phase = None
        self.calls += 1
        return guided_result

    def eos(self, module, args):
        # TF receives all attribute/boundary rows; AR receives the last row.
        global_eos = (
            self.global_hidden[:, self.prefix :]
            if self.mode == "tf"
            else self.global_hidden[:, -1:]
        )
        assert global_eos.shape == args[0].shape
        self.eos_checks += 1
        return (global_eos,)  # deliberately NOT detached

    def audit(self, module, args, kwargs):
        if self.phase != "guided":
            return
        mask, bias = kwargs.get("mask"), kwargs.get("attn_bias")
        assert mask is not None and mask.dtype == torch.bool and mask.all()
        assert bias is not None and (bias[..., : self.mem] == 0).all()
        assert (bias[:, :, : self.prefix] == 0).all()
        self.attention_checks += 1


def frozen_tf_inputs(model, batch):
    """Cache ONLY frozen pre-decoder tensors and loss targets, never queries.

    Validity depends on configure(): decoder/output heads are the only unfrozen
    original modules. Native forward is also returned for equivalence testing.
    """
    recorded = {}
    original_head = model._attr_decouple_loss_and_outputs

    def decoder_hook(module, args, kwargs):
        recorded["x"] = args[0].detach().clone()
        recorded["kwargs"] = kwargs

    def head(**kwargs):
        recorded["head"] = {
            k: v for k, v in kwargs.items() if k not in {"embed", "is_train"}
        }
        return original_head(**kwargs)

    handle = model.decoder.register_forward_pre_hook(decoder_hook, with_kwargs=True)
    model._attr_decouple_loss_and_outputs = head
    try:
        # no_grad, not inference_mode: cached inputs must support weight grads.
        with torch.no_grad():
            native = model(**batch, is_train=False)
    finally:
        model._attr_decouple_loss_and_outputs = original_head
        handle.remove()
    assert (
        recorded["x"].shape[1]
        == int(recorded["kwargs"]["prefix_attn_len"])
        + 6 * batch["primitive_id"].shape[1]
        + 1
    )
    return recorded, native


def tf_from_cache(model, cache):
    output = model.decoder(cache["x"], **cache["kwargs"])[0]
    prefix = int(cache["kwargs"]["prefix_attn_len"])
    return model._attr_decouple_loss_and_outputs(
        embed=output[:, prefix:], is_train=False, **cache["head"]
    )
