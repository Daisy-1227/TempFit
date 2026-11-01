from data import register_dataset, DatasetSplits
from datasets.vec_data.floorplancad.floorplancad import FloorPlanCAD, IndustryCAD


@register_dataset("floorplancad")
def build(dataset_args: dict):
    train_dataset = FloorPlanCAD(**dataset_args, split="train")
    val_dataset = FloorPlanCAD(**dataset_args, split="val")
    test_dataset = FloorPlanCAD(**dataset_args, split="test")
    data_collator = FloorPlanCAD.collate_fn
    return (
        DatasetSplits(train=train_dataset, val=val_dataset, test=test_dataset),
        data_collator,
    )


@register_dataset("industrycad")
def build(dataset_args: dict):
    train_dataset = IndustryCAD(**dataset_args, split="train")
    test_dataset = IndustryCAD(**dataset_args, split="test")
    data_collator = IndustryCAD.collate_fn
    return (
        DatasetSplits(train=train_dataset, val=test_dataset, test=test_dataset),
        data_collator,
    )
