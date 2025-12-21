import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


class CosineWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    """
    A learning rate scheduler that combines linear warmup followed by cosine annealing.

    This scheduler increases the learning rate linearly from a very small value to the
    base learning rate over the warmup period, and then decays it using cosine annealing
    down to a minimum learning rate over the remaining training steps.
    """

    def __init__(
        self, optimizer, total_steps, warmup_ratio=0.1, min_lr=1e-5, last_epoch=-1
    ):
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            total_steps (int): Total number of training steps.
            warmup_ratio (float, optional): Fraction of total steps used for linear warmup.
                Defaults to 0.1.
            min_lr (float, optional): Minimum learning rate reached at the end of cosine
                annealing. Defaults to 1e-5.
            last_epoch (int, optional): The index of the last epoch. Default: -1.
        """
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = int(warmup_ratio * total_steps)
        self.cosine_steps = total_steps - self.warmup_steps

        warmup = LinearLR(
            optimizer, start_factor=1e-8, end_factor=1.0, total_iters=self.warmup_steps
        )
        cosine = CosineAnnealingLR(optimizer, T_max=self.cosine_steps, eta_min=min_lr)

        self.scheduler = SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[self.warmup_steps]
        )
        super().__init__(optimizer, last_epoch)

    def step(self):
        self.scheduler.step()

    def get_last_lr(self):
        return self.scheduler.get_last_lr()
