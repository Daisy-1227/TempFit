def create_model(variant, model_config):
    if variant != "attr_decouple":
        raise ValueError("Supported model variant: attr_decouple")
    from models.primitive.primitive_transformer import PrimitiveTransformerDiscrete

    kwargs = dict(model_config)
    kwargs.pop("name", None)
    kwargs.update(
        sem_geom_decouple=False,
        geom_sem_decouple=False,
        attr_decouple=True,
        attr_emit_semantic_token=True,
        attr_primitive_semantic_condition="gt",
        attr_semantic_supervision=True,
        primitive_coarse_loss_weight=1.0,
    )
    return PrimitiveTransformerDiscrete(**kwargs)


def build_model(cfg):
    from models.batched import SoftModel

    return SoftModel(cfg)
