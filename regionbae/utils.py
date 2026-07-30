"""Seeding, LR scheduler factory, and an optional-W&B logger shim.

``seed_everything`` and ``make_scheduler`` are copied verbatim from the research
code (``src_v2/utils.py``). ``get_logger`` is new: the original code called
``wandb`` unconditionally, which would make the public pipeline unusable without
a W&B account. The shim is a no-op unless W&B logging is explicitly requested.
"""

import numpy as np
import torch

from . import lr_scheduler as lr


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_scheduler(optimizer, dataloader_len, lr_scheduler_choice, learning_rate):
    if lr_scheduler_choice == 0:
        return None
    steps_per_epoch = max(1, dataloader_len)
    t_0 = max(1, steps_per_epoch // 2)
    return lr.CustomCosineAnnealingWarmUpRestarts(
        optimizer,
        T_0=t_0,
        T_up=min(10, t_0),
        T_mult=2,
        eta_max=max(learning_rate, 1e-4),
        gamma=0.5,
    )


class NoOpLogger:
    """Stand-in used when W&B logging is disabled or wandb is not installed."""

    def watch(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass

    def finish(self, *args, **kwargs):
        pass


def get_logger(use_wandb=False, **init_kwargs):
    """Return a wandb-like logger. Falls back to a no-op when unavailable."""
    if not use_wandb:
        return NoOpLogger()
    try:
        import wandb
    except ImportError:
        print('[WARN] wandb is not installed; continuing without experiment logging')
        return NoOpLogger()
    if init_kwargs:
        wandb.init(**init_kwargs)
    return wandb
