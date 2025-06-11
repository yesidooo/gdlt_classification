# train.py (完整版)
import os
import argparse
from tqdm import tqdm
import yaml
from contextlib import suppress

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from config import Config
from dataset import BaseDataset
from model import build_classifier, build_loss
from utils.logger import get_logger
from utils.metrics import AverageMeter
from utils.ema import ModelEma
from utils.scheduler import build_scheduler


def train_one_epoch(epoch, cfg, model, train_loader, optimizer, loss_fn, device,
                    lr_scheduler=None, scaler=None, model_ema=None, logger=None):
    """训练一个epoch"""
    model.train()

    losses = AverageMeter()
    accuracies = AverageMeter()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Train Epoch {epoch}')

    for batch_idx, data in pbar:
        images = data['image'].to(device, non_blocking=True)
        labels = data['label'].to(device, non_blocking=True)

        optimizer.zero_grad()

        # 混合精度训练
        if cfg.train.use_amp and scaler is not None:
            with autocast():
                outputs = model(images)
                loss = loss_fn(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

        # 更新EMA
        if model_ema is not None:
            model_ema.update(model)

        # 计算准确率
        _, predicted = outputs.max(1)
        accuracy = predicted.eq(labels).float().mean()

        losses.update(loss.item(), images.size(0))
        accuracies.update(accuracy.item(), images.size(0))

        # 更新进度条
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'Loss': f'{losses.avg:.4f}',
            'Acc': f'{accuracies.avg:.4f}',
            'LR': f'{current_lr:.6f}'
        })

        # 学习率调度
        if lr_scheduler is not None and hasattr(lr_scheduler, 'step_update'):
            lr_scheduler.step_update(epoch * len(train_loader) + batch_idx)

    if logger:
        logger.info(f'Train Epoch {epoch}: Avg Loss: {losses.avg:.4f}, Avg Acc: {accuracies.avg:.4f}')

    return losses.avg, accuracies.avg


@torch.no_grad()
def validate(epoch, cfg, model, val_loader, loss_fn, device, logger=None):
    """验证"""
    model.eval()

    losses = AverageMeter()
    accuracies = AverageMeter()

    all_predictions = []
    all_targets = []

    pbar = tqdm(val_loader, desc=f'Val Epoch {epoch}')

    for data in pbar:
        images = data['image'].to(device, non_blocking=True)
        labels = data['label'].to(device, non_blocking=True)

        if cfg.train.use_amp:
            with autocast():
                outputs = model(images)
                loss = loss_fn(outputs, labels)
        else:
            outputs = model(images)
            loss = loss_fn(outputs, labels)

        _, predicted = outputs.max(1)
        accuracy = predicted.eq(labels).float().mean()

        losses.update(loss.item(), images.size(0))
        accuracies.update(accuracy.item(), images.size(0))

        all_predictions.extend(predicted.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

        pbar.set_postfix({
            'Loss': f'{losses.avg:.4f}',
            'Acc': f'{accuracies.avg:.4f}'
        })

    # 计算详细指标
    f1 = f1_score(all_targets, all_predictions, average='macro')
    precision = precision_score(all_targets, all_predictions, average='macro')
    recall = recall_score(all_targets, all_predictions, average='macro')

    if logger:
        logger.info(f'Val Epoch {epoch}: Loss: {losses.avg:.4f}, Acc: {accuracies.avg:.4f}, '
                    f'F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')

    return losses.avg, accuracies.avg, f1, precision, recall


def save_checkpoint(state, checkpoint_path, filename):
    """保存检查点"""
    filepath = os.path.join(checkpoint_path, filename)
    torch.save(state, filepath)


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """加载检查点"""
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_acc = checkpoint.get('best_acc', 0.0)
        return epoch, best_acc
    else:
        return 0, 0.0


def main(args):
    # 加载配置
    cfg = Config(args.config)
    if args.exp_name:
        cfg.update_exp(args.exp_name)

    # 设置随机种子
    if cfg.train.random_seed is not None:
        torch.manual_seed(cfg.train.random_seed)
        torch.cuda.manual_seed_all(cfg.train.random_seed)

    # 设置训练优化
    if cfg.train.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if cfg.train.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 设置日志
    logger = get_logger(cfg.log_path)
    logger.info(f'Starting experiment: {cfg.experiments.exp_name}')
    logger.info(f'Device: {device}')

    # 构建数据集
    train_dataset = BaseDataset(cfg, mode='train')
    val_dataset = BaseDataset(cfg, mode='val')

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_works,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_works,
        pin_memory=True
    )

    # 构建模型
    model = build_classifier(cfg)
    model.to(device)

    # 构建损失函数
    loss_fn = build_loss(cfg)

    # 构建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay
    )

    # 构建学习率调度器
    lr_scheduler = build_scheduler(cfg, optimizer, len(train_loader))

    # 混合精度训练
    scaler = GradScaler() if cfg.train.use_amp else None

    # EMA
    model_ema = ModelEma(model) if cfg.train.use_ema else None

    # 恢复训练
    start_epoch = 0
    best_acc = 0.0
    if cfg.experiments.resume:
        resume_path = os.path.join(cfg.checkpoint_path, 'latest.pth')
        if os.path.exists(resume_path):
            start_epoch, best_acc = load_checkpoint(model, optimizer, resume_path, device)
            logger.info(f'Resumed from epoch {start_epoch}, best acc: {best_acc:.4f}')

    # 训练循环
    for epoch in range(start_epoch + 1, cfg.train.max_epoch + 1):
        # 训练
        train_loss, train_acc = train_one_epoch(
            epoch, cfg, model, train_loader, optimizer, loss_fn, device,
            lr_scheduler, scaler, model_ema, logger
        )

        # 验证
        val_model = model_ema.module if model_ema else model
        val_loss, val_acc, f1, precision, recall = validate(
            epoch, cfg, val_model, val_loader, loss_fn, device, logger
        )

        # 保存检查点
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc

        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
            'val_acc': val_acc,
            'val_loss': val_loss,
            'f1_score': f1,
            'precision': precision,
            'recall': recall
        }

        save_checkpoint(state, cfg.checkpoint_path, 'latest.pth')
        if is_best:
            save_checkpoint(state, cfg.checkpoint_path, 'best.pth')

        # 定期保存
        if epoch % 10 == 0:
            save_checkpoint(state, cfg.checkpoint_path, f'epoch_{epoch:03d}.pth')

        # 学习率调度
        if lr_scheduler is not None and not hasattr(lr_scheduler, 'step_update'):
            lr_scheduler.step()

    logger.info(f'Training completed. Best accuracy: {best_acc:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--config', default='config/config.yaml', help='config file path')
    parser.add_argument('--exp_name', default=None, help='experiment name')
    args = parser.parse_args()
    main(args)