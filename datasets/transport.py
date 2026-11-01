import copy
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class StorageBytes:
    data: np.ndarray


@dataclass
class TensorView:
    storage: StorageBytes
    dtype: torch.dtype
    shape: tuple
    stride: tuple
    offset: int


def _walk(value, memo, storages, decoding):
    key = id(value)
    if key in memo:
        return memo[key]
    if decoding and isinstance(value, TensorView):
        skey = id(value.storage)
        if skey not in storages:
            array = value.storage.data
            if not array.flags.writeable:
                array = array.copy()
            storages[skey] = torch.from_numpy(array).untyped_storage()
        result = torch.empty(0, dtype=value.dtype).set_(
            storages[skey], value.offset, value.shape, value.stride
        )
    elif torch.is_tensor(value):
        if decoding:
            raise TypeError("Unexpected torch tensor in encoded payload")
        if (
            type(value) is not torch.Tensor
            or value.device.type != "cpu"
            or value.requires_grad
            or value.layout != torch.strided
            or value.is_quantized
            or value.is_conj()
            or value.is_neg()
        ):
            raise TypeError(
                "Transport requires ordinary dense CPU data tensors without autograd/conjugate bits"
            )
        storage = value.untyped_storage()
        skey = storage._cdata
        if skey not in storages:
            array = (
                torch.empty(0, dtype=torch.uint8)
                .set_(storage, 0, (storage.nbytes(),), (1,))
                .numpy()
            )
            storages[skey] = StorageBytes(array)
        result = TensorView(
            storages[skey],
            value.dtype,
            tuple(value.shape),
            tuple(value.stride()),
            value.storage_offset(),
        )
    elif isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Object arrays are not supported by training transport")
        result = value
    elif isinstance(value, dict):
        result = copy.copy(value)
        result.clear()
        memo[key] = result
        result.update((k, _walk(v, memo, storages, decoding)) for k, v in value.items())
    elif isinstance(value, list):
        result = []
        memo[key] = result
        result.extend(_walk(v, memo, storages, decoding) for v in value)
    elif isinstance(value, tuple):
        fields = [_walk(v, memo, storages, decoding) for v in value]
        result = type(value)(*fields) if hasattr(value, "_fields") else tuple(fields)
    elif hasattr(value, "__dict__") and not isinstance(value, type):
        result = copy.copy(value)
        memo[key] = result
        result.__dict__ = {
            k: _walk(v, memo, storages, decoding) for k, v in value.__dict__.items()
        }
    else:
        result = value
    memo[key] = result
    return result


def encode(value):
    return _walk(value, {}, {}, False)


def decode(value):
    return _walk(value, {}, {}, True)


def compact_group(items):
    # Augmentation, assertions, metadata and exposure summaries have already run
    # in Dataset.__getitem__. batch() discards private keys in the old path too.
    return [
        dict(
            item,
            sample={k: v for k, v in item["sample"].items() if k != "_recon_audit"},
        )
        for item in items
    ]


def transport_collate(items):
    # Top-level function works with both fork and spawn worker startup.
    return encode(compact_group(items))
