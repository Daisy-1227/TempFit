import copy
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from util.io import read
from datasets.instance_scale import AuditedDataset, IndependentInstanceScale
from datasets.full_transforms import (
    DropOutOfRangeTranslation,
    EmptyInputAfterTranslationFilter,
)
from datasets.metadata import metadata
from datasets.primitive_dataset import create_dataset
from datasets.transform import TRANSFORMS


def arr(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


@TRANSFORMS.register_module()
class FullSourceInRange:
    """No clipping. Preserve every coordinate of every retained source instance."""

    def __init__(self, continuous_range=(-2.0, 2.0)):
        self.filter = DropOutOfRangeTranslation(continuous_range)

    def __call__(self, data):
        svg = data["svg_data"]
        before = np.asarray(svg.coords).reshape(-1, 2, 2)
        prim = np.asarray(svg.primitive_ids)
        own = np.asarray(svg.instance_ids)[prim]
        translation = np.asarray(data["translation"])
        keep_gt = (
            (translation >= self.filter.lower) & (translation <= self.filter.upper)
        ).all(1)
        keep_line = np.ones(len(own), dtype=bool)
        fg = own >= 0
        keep_line[fg] = keep_gt[own[fg]]
        expected = before[keep_line].copy()
        expected_translation = translation[keep_gt].copy()
        input_gt = len(translation)
        input_lines = len(before)
        data = self.filter(data)
        svg = data["svg_data"]
        coords = np.asarray(svg.coords).reshape(-1, 2, 2)
        np.testing.assert_array_equal(coords, expected)
        np.testing.assert_array_equal(data["translation"], expected_translation)
        n = len(data["translation"])
        owners = np.asarray(svg.instance_ids)[np.asarray(svg.primitive_ids)]
        fg = owners >= 0
        assert np.isfinite(coords).all()
        counts = np.bincount(owners[fg], minlength=n)
        lengths = np.bincount(
            owners[fg],
            weights=np.linalg.norm(coords[fg, 1] - coords[fg, 0], axis=1),
            minlength=n,
        )
        assert (
            len(counts) == n and (counts > 0).all()
        ), "Retained GT without source geometry"
        audit = data.setdefault("_recon_audit", {})
        audit.update(
            full_translation_pre_clip=np.array(data["translation"], copy=True),
            clip_input_segment_counts=counts,
            clip_input_segment_lengths=lengths,
            preclip_full_coords=coords[fg].astype(np.float32, copy=True),
            preclip_full_line_instance_ids=owners[fg].astype(np.int32, copy=True),
            clip_removed_instances=0,
            clip_removed_segments=0,
            visibility_filter_input_segment_counts_all=counts.copy(),
            visibility_filter_semantic_id_all=np.array(data["semantic_id"], copy=True),
            visibility_filter_ignored_instances=0,
            visibility_filter_noise_segments=0,
            visibility_filter_noise_length=0.0,
            full_only_input_gt=input_gt,
            full_only_input_lines=input_lines,
            full_only_before_grid_gt=n,
            full_only_before_grid_lines=len(coords),
            full_only_kept_geometry_exact=True,
            full_only_outside_unit_lines=int(
                (np.abs(coords) > 1).any(axis=(1, 2)).sum()
            ),
            full_only_outside_translation_box_lines=int(
                (np.abs(coords) > 2).any(axis=(1, 2)).sum()
            ),
        )
        for key in (
            "clip_count_ratio",
            "clip_length_ratio",
            "clip_bbox_ratio",
            "visibility_filter_count_ratio_all",
            "visibility_filter_length_ratio_all",
            "visibility_filter_bbox_ratio_all",
        ):
            audit[key] = np.ones(n, dtype=np.float32)
        return data


class FullOnlyPrimitiveDataset(AuditedDataset):
    def __getitem__(self, idx, _invalid_retry_count=0):
        try:
            return super().__getitem__(idx, _invalid_retry_count)
        except EmptyInputAfterTranslationFilter:
            return self._handle_invalid_sample(
                idx,
                _invalid_retry_count,
                "empty_input_after_full_source_translation_filter",
                retryable=self.split == "train",
            )

    def _transform(self, svg, gt, crop_window, *, deterministic_fallback=False):
        if not deterministic_fallback:
            return super()._transform(
                svg, gt, crop_window, deterministic_fallback=False
            )
        # The production fallback whitelist omits our new transform. Do not use it:
        # preserve sample identity, disable stochastic augmentation, execute filter/snapshot.
        original_chain = self.transform.transforms
        original_instance = self.instance_transform
        suffix = [
            t
            for t in original_chain
            if type(t).__name__ in ("FullSourceInRange", "VecGridSample", "ToTensor")
        ]
        assert sum(isinstance(t, FullSourceInRange) for t in suffix) == 1
        grids = [(t, t.mode) for t in suffix if type(t).__name__ == "VecGridSample"]
        try:
            self.instance_transform = None
            self.transform.transforms = suffix
            for t, _ in grids:
                t.mode = "test"
            result = super()._transform(
                svg, gt, crop_window, deterministic_fallback=False
            )
            result[2]["_recon_audit"]["full_only_deterministic_fallback"] = True
            return result
        finally:
            self.instance_transform = original_instance
            self.transform.transforms = original_chain
            for t, mode in grids:
                t.mode = mode


class SceneDataset(Dataset):
    def __init__(self, cfg, split="train", capture=False, deterministic=False):
        self.cfg, self.split, self.epoch = cfg, split, 0
        manifest = read(cfg["experiment"]["manifest"])
        self.names = manifest[split]
        from evaluation.seeds import preprocessing

        self.preprocessing_seeds = preprocessing(manifest) if split == "val" else {}
        dc = copy.deepcopy(cfg["dataset_" + split])
        self.repeat = dc["repeat"] if split == "train" else 1
        dc.update(repeat=1, shuffle=False, demo=False, recon_audit=True)
        assert dc["filter_translation"] is False, "Do not use legacy +/-15000 prefilter"
        assert (
            dc["continuous_range_translation"]
            == cfg["model"]["continuous_range_translation"]
        )
        if deterministic:
            dc["instance_transform"] = None
            dc["transform"] = [
                t
                for t in dc["transform"]
                if t["type"] in ("FullSourceInRange", "VecGridSample", "ToTensor")
            ]
            for t in dc["transform"]:
                if t["type"] == "VecGridSample":
                    t["mode"] = "test"
        self.ds = create_dataset(dc)
        self.ds.__class__ = FullOnlyPrimitiveDataset
        self.ds.retry_events = []
        lookup = {str(name): i for i, name in enumerate(self.ds.data_filename)}
        if split == "train":
            assert set(lookup) == set(self.names), "Train coverage/version drift"
        self.indices = [lookup[name] for name in self.names]
        chain = self.ds.transform.transforms
        assert not any(type(t).__name__ == "VecClipping" for t in chain)
        loc = [i for i, t in enumerate(chain) if isinstance(t, FullSourceInRange)]
        assert len(loc) == 1
        self.scale = None
        if split == "train" and not deterministic:
            self.scale = IndependentInstanceScale(
                **cfg["instance_scale"], capture=capture
            )
            chain.insert(loc[0], self.scale)

    def __len__(self):
        return len(self.names) * self.repeat

    def __getitem__(self, key):
        index, real = key if isinstance(key, tuple) else (int(key), True)
        local = index % len(self.names)
        states = random.getstate(), np.random.get_state(), torch.get_rng_state()
        s = int(self.cfg["experiment"]["data_seed"] + self.epoch * 1000003 + index)
        if self.split == "val":
            s = self.preprocessing_seeds[self.names[local]]
        random.seed(s)
        np.random.seed(s % 2**32)
        torch.random.default_generator.manual_seed(s)
        if self.scale:
            self.scale.reset(s + 700000001)
        self.ds.retry_events = []
        try:
            sample = self.ds[self.indices[local]]
            assert (
                Path(sample["filename"]).name == self.names[local]
            ), "Scene substitution prohibited"
            assert not arr(sample["recon_is_partial"]).any()
            t = arr(sample["translation"])
            assert ((t >= -2) & (t <= 2)).all()
            assert sample["_recon_audit"]["full_only_kept_geometry_exact"]
            return dict(
                sample=sample,
                metadata=metadata(sample),
                index=index,
                real=bool(real),
                scale_events=copy.deepcopy(self.scale.events) if self.scale else [],
                retries=copy.deepcopy(self.ds.retry_events),
            )
        finally:
            random.setstate(states[0])
            np.random.set_state(states[1])
            torch.set_rng_state(states[2])


def batch(data, device):
    from datasets.batch import batch as base_batch, groups

    # Source audit and before-grid ownership are not model inputs. Apart from
    # avoiding ragged debug collation, this prevents any diagnostic GT leakage.
    cleaned = [
        dict(d, sample={k: v for k, v in d["sample"].items() if not k.startswith("_")})
        for d in groups(data)
    ]
    return base_batch(cleaned, device)
