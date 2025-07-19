from typing import Union
import torch
from torch.optim.lr_scheduler import _LRScheduler


class WarmupLR(_LRScheduler):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            warmup_steps: Union[int, float] = 25000,
            last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def __repr__(self):
        return f"{self.__class__.__name__}(warmup_steps={self.warmup_steps})"

    def get_lr(self):
        step_num = self.last_epoch + 1
        if self.warmup_steps == 0:
            return [
                lr * step_num ** -0.5
                for lr in self.base_lrs
            ]
        else:
            return [
                lr
                * self.warmup_steps ** 0.5
                * min(step_num ** -0.5, step_num * self.warmup_steps ** -1.5)
                for lr in self.base_lrs
            ]
    def set_step(self, step: int):
        self.last_epoch = step