"""
学习率调度器
"""
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, StepLR
from timm.scheduler import CosineLRScheduler, PlateauLRScheduler


def build_scheduler(cfg, optimizer, steps_per_epoch):
    """构建学习率调度器"""
    scheduler_name = getattr(cfg.train, 'scheduler', 'cosine')

    if scheduler_name == 'cosine':
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=cfg.train.max_epoch * steps_per_epoch,
            lr_min=cfg.train.lr * 0.01,
            warmup_lr_init=cfg.train.warmup.lr,
            warmup_t=cfg.train.warmup.epoch * steps_per_epoch,
            cycle_limit=1,
            t_in_epochs=False,
        )
    elif scheduler_name == 'step':
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_name == 'multistep':
        milestones = getattr(cfg.train, 'milestones', [60, 80])
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    else:
        scheduler = None

    return scheduler
