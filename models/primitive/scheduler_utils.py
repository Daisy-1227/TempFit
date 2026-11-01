# scheduler_utils.py
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Optional, Type, Dict, Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR

from accelerate import Accelerator


def exists(x) -> bool:
    return x is not None


def ConstantLRScheduler(optimizer: Optimizer) -> _LRScheduler:
    """A scheduler that keeps LR constant (stateful & serializable)."""
    return LambdaLR(optimizer, lr_lambda=lambda _: 1.0)


class LinearWarmupDampener:
    """
    A tiny warmup helper that mimics the 'warmup.dampening' pattern:
    use it as:

        with warmup.dampening():
            scheduler.step()

    After scheduler.step() sets lr, we scale it by warmup_factor in __exit__.
    """

    def __init__(self, optimizer: Optimizer, warmup_period: int):
        self.optimizer = optimizer
        self.warmup_period = int(max(0, warmup_period))
        self.step_idx = 0

    def state_dict(self) -> Dict[str, Any]:
        return {"warmup_period": self.warmup_period, "step_idx": self.step_idx}

    def load_state_dict(self, sd: Dict[str, Any]):
        self.warmup_period = int(sd.get("warmup_period", self.warmup_period))
        self.step_idx = int(sd.get("step_idx", 0))

    def _factor(self) -> float:
        if self.warmup_period <= 0:
            return 1.0
        # step_idx starts at 0; first update => factor = 1/warmup_period
        return min(1.0, float(self.step_idx + 1) / float(self.warmup_period))

    @contextmanager
    def dampening(self):
        try:
            yield
        finally:
            f = self._factor()
            # scheduler.step() has already set absolute lr; now scale it once.
            for g in self.optimizer.param_groups:
                g["lr"] = g["lr"] * f
            self.step_idx += 1


class OptimizerWithWarmupSchedule(nn.Module):
    """
    Fix for Accelerate + grad accumulation + OneCycleLR:

    - Only prepare optimizer with accelerator.
    - Create scheduler AFTER optimizer is prepared.
    - Do NOT accelerator.prepare() the scheduler.
    - Call .step() only when you actually want an optimizer update.

    This keeps the unit of 'scheduler.step()' = one optimizer update.
    """

    def __init__(
        self,
        accelerator: Accelerator,
        optimizer: Optimizer,
        scheduler: Optional[Type[_LRScheduler]] = None,
        scheduler_kwargs: Dict[str, Any] | None = None,
        warmup_steps: int = 0,
        max_grad_norm: Optional[float] = None,
    ):
        super().__init__()
        self.accelerator = accelerator
        self.max_grad_norm = max_grad_norm

        if scheduler_kwargs is None:
            scheduler_kwargs = {}

        # 1) Prepare optimizer only
        self.optimizer: Optimizer = accelerator.prepare(optimizer)

        # 2) Optional warmup dampener (usually 0 when using OneCycleLR)
        self.warmup = (
            LinearWarmupDampener(self.optimizer, warmup_steps)
            if warmup_steps > 0
            else None
        )

        # 3) Create scheduler after optimizer is prepared; DO NOT prepare scheduler
        if exists(scheduler):
            self.scheduler: _LRScheduler = scheduler(self.optimizer, **scheduler_kwargs)
        else:
            self.scheduler = ConstantLRScheduler(self.optimizer)

    def state_dict(self) -> Dict[str, Any]:
        pkg = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        if exists(self.warmup):
            pkg["warmup"] = self.warmup.state_dict()
        return pkg

    def load_state_dict(self, pkg: Dict[str, Any]):
        self.optimizer.load_state_dict(pkg["optimizer"])
        self.scheduler.load_state_dict(pkg["scheduler"])
        if exists(self.warmup) and "warmup" in pkg:
            self.warmup.load_state_dict(pkg["warmup"])

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_lr(self) -> float:
        # safe helper for logging
        return float(self.optimizer.param_groups[0].get("lr", 0.0))

    def step(self):
        # IMPORTANT: call this ONLY when you want an optimizer update
        if exists(self.max_grad_norm):
            params = []
            for g in self.optimizer.param_groups:
                params.extend(g["params"])
            self.accelerator.clip_grad_norm_(params, self.max_grad_norm)

        self.optimizer.step()

        # accelerate will set this when it skipped due to overflow, etc.
        if not getattr(self.accelerator, "optimizer_step_was_skipped", False):
            ctx = nullcontext() if self.warmup is None else self.warmup.dampening()
            with ctx:
                self.scheduler.step()
