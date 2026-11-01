from contextvars import ContextVar
import types
import torch

_full_audit = ContextVar("tempfit_full_attention_audit", default=True)


def install(raw, cfg):
    spec = cfg.get("runtime_optimization", {})
    if not spec.get("enabled", False):
        return None
    assert not hasattr(raw, "_speed_state"), "Optimizations already installed"
    state = dict(
        template_hits=0,
        template_misses=0,
        full_audit_forwards=0,
        lightweight_forwards=0,
        forward_calls=0,
    )
    raw._speed_state = state
    if spec.get("cpu_counts", True):
        counts_cache = {}

        def counts(self, sample):
            m = self.model
            assert (
                m.recon_partial_source_mode == "preclip_full"
                and m.recon_gt_geometry_threshold < 0
            )
            assert (
                m.recon_instance_reduction == "instance_mean"
                and m.recon_partial_weight == 1
            )
            mask = m.attr_recon_template_mask
            key = (id(mask), mask._version, mask.device, mask.dtype)
            if counts_cache.get("key") != key:
                counts_cache.update(
                    key=key, tensor=mask, available=mask.detach().any(-1).cpu()
                )
            ids = torch.as_tensor(sample["primitive_id"]).long()
            assert ids.device.type == "cpu"
            valid = (
                sample["recon_valid_mask"].bool()
                & sample["recon_preclip_source_mask"].any(-1)
                & counts_cache["available"][ids]
            )
            n = len(ids)
            return n, n + 1, int(valid.sum())

        raw.counts = types.MethodType(counts, raw)
    if spec.get("template_geometry_cache", True):
        original = raw.model._vec_grid_sample_tensor
        geometry_cache = {}

        def cached(coords, feats, layer_ids, cu_seqlens, prim_ids=None, grid_size=0.01):
            values = (coords, feats, layer_ids, cu_seqlens, prim_ids)
            expected = tuple(
                getattr(raw.model, n)
                for n in (
                    "primitive_lib_coords",
                    "primitive_lib_feats",
                    "primitive_lib_layer_ids",
                    "primitive_lib_cu_seqlens",
                    "primitive_lib_prim_ids",
                )
            )
            # Only the fixed library is eligible. A future scene call falls back.
            if not all(a is b for a, b in zip(values, expected)):
                return original(
                    coords,
                    feats,
                    layer_ids,
                    cu_seqlens,
                    prim_ids=prim_ids,
                    grid_size=grid_size,
                )
            tensors = [v for v in values if torch.is_tensor(v)]
            assert not any(t.requires_grad for t in tensors)
            key = (
                tuple(
                    (
                        id(t),
                        t._version,
                        tuple(t.shape),
                        tuple(t.stride()),
                        t.device,
                        t.dtype,
                    )
                    for t in tensors
                ),
                float(grid_size),
            )
            if geometry_cache.get("key") != key:
                result = original(
                    coords,
                    feats,
                    layer_ids,
                    cu_seqlens,
                    prim_ids=prim_ids,
                    grid_size=grid_size,
                )
                assert all(not t.requires_grad for t in result if torch.is_tensor(t))
                geometry_cache.update(key=key, inputs=values, result=result)
                state["template_misses"] += 1
            else:
                state["template_hits"] += 1
            return geometry_cache["result"]

        raw.model._vec_grid_sample_tensor = cached
    interval = int(spec.get("full_attention_audit_every_micros", 256))
    assert interval >= 1
    from models.batched import BatchedPrompt

    if not hasattr(BatchedPrompt, "_speed_original_audit"):
        BatchedPrompt._speed_original_audit = BatchedPrompt.audit

        def audit(self, module, args, kwargs):
            if _full_audit.get():
                return BatchedPrompt._speed_original_audit(self, module, args, kwargs)
            if self.phase == "guided":
                self.attention_checks += 1

        BatchedPrompt.audit = audit
    original_forward = raw.forward

    def forward(self, *args, **kwargs):
        requested = kwargs.get("audit", args[6] if len(args) > 6 else False)
        full = bool(requested or state["forward_calls"] % interval == 0)
        state["forward_calls"] += 1
        state["full_audit_forwards" if full else "lightweight_forwards"] += 1
        token = _full_audit.set(full)
        try:
            return original_forward(*args, **kwargs)
        finally:
            _full_audit.reset(token)

    raw.forward = types.MethodType(forward, raw)
    return state
