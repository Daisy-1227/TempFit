from datetime import timedelta

import torch.distributed as dist

from util.guard import ProtectedStop


_control_group = None


def initialize():
    global _control_group
    if dist.is_initialized() and dist.get_world_size() > 1:
        # Validation and checkpoint I/O can take longer than a training collective.
        _control_group = dist.new_group(backend="gloo", timeout=timedelta(hours=24))


def gather_objects(value):
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return [value]
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value, group=_control_group)
    return values


def on_main(accelerator, function, broadcast_result=False):
    result, error = None, None
    message = [None]
    if accelerator.is_main_process:
        try:
            result = function()
            message[0] = (None, result if broadcast_result else None)
        except Exception as exc:
            error = exc
            message[0] = ((type(exc).__name__, str(exc)), None)
    if accelerator.num_processes > 1:
        dist.broadcast_object_list(message, src=0, group=_control_group)
    failure, shared = message[0]
    if failure:
        if error is not None:
            raise error
        kind, detail = failure
        if kind == "ProtectedStop":
            raise ProtectedStop(detail)
        raise RuntimeError(f"Main process {kind}: {detail}")
    return shared if broadcast_result else result


def check_resources(accelerator, guard, boundary=False):
    def check():
        if boundary:
            return guard.boundary()
        guard.check()
    return on_main(accelerator, check, broadcast_result=boundary)
