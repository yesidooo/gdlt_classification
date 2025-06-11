import os
import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
import numpy as np
from tqdm import tqdm

from config import Config
from dataset import BaseDataset
from model import build_classifier, build_loss
from utils.logger import get_logger
from utils.metrics import AverageMeter


@torch.no_grad()
def validate_model(cfg, model, val_loader, loss_fn, device, class_names=None, logger=None):
    """完整的模型验证"""
    model.eval()

    losses = AverageMeter()
    all_predictions = []
    all_targets = []
    all_probabilities = []

    pbar = tqdm(val_loader, desc='Validating')

    for data in pbar:
        images = data['image'].to(device, non_blocking=True)
        labels = data['label'].to(device, non_blocking=True)

        # 前向传播
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # 获取概率和预测
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)

        losses.update(loss.item(), images.size(0))

        all_predictions.extend(predicted.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())

        pbar.set_postfix({'Loss': f'{losses.avg:.4f}'})

    # 计算指标
    accuracy = accuracy_score(all_targets, all_predictions)
    f1_macro = f1_score(all_targets, all_predictions, average='macro')
    f1_micro = f1_score(all_targets, all_predictions, average='micro')
    f1_weighted = f1_score(all_targets, all_predictions, average='weighted')
    precision_macro = precision_score(all_targets, all_predictions, average='macro')
    recall_macro = recall_score(all_targets, all_predictions, average='macro')

    # 每类别指标
    f1_per_class = f1_score(all_targets, all_predictions, average=None)
    precision_per_class = precision_score(all_targets, all_predictions, average=None)
    recall_per_class = recall_score(all_targets, all_predictions, average=None)

    results = {
        'loss': losses.avg,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_per_class': f1_per_class,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'predictions': all_predictions,
        'targets': all_targets,
        'probabilities': all_probabilities
    }

    # 打印结果
    if logger:
        logger.info(f'Validation Results:')
        logger.info(f'Loss: {losses.avg:.4f}')
        logger.info(f'Accuracy: {accuracy:.4f}')
        logger.info(f'F1 (macro): {f1_macro:.4f}')
        logger.info(f'F1 (micro): {f1_micro:.4f}')
        logger.info(f'F1 (weighted): {f1_weighted:.4f}')
        logger.info(f'Precision (macro): {precision_macro:.4f}')
        logger.info(f'Recall (macro): {recall_macro:.4f}')

        if class_names:
            logger.info('\nPer-class Results:')
            for i, name in enumerate(class_names):
                logger.info(f'{name}: F1={f1_per_class[i]:.4f}, '
                            f'Precision={precision_per_class[i]:.4f}, '
                            f'Recall={recall_per_class[i]:.4f}')

        # 分类报告
        target_names = class_names if class_names else [f'Class_{i}' for i in range(len(np.unique(all_targets)))]
        report = classification_report(all_targets, all_predictions, target_names=target_names)
        logger.info(f'\nClassification Report:\n{report}')

    return results


def load_model(cfg, checkpoint_path, device):
    """加载训练好的模型"""
    model = build_classifier(cfg)

    if os.path.isdir(checkpoint_path):
        # 寻找最好的检查点
        best_checkpoint = os.path.join(checkpoint_path, 'best.pth')
        if os.path.exists(best_checkpoint):
            checkpoint_path = best_checkpoint
        else:
            # 寻找最新的检查点
            checkpoints = [f for f in os.listdir(checkpoint_path) if f.endswith('.pth')]
            if checkpoints:
                checkpoint_path = os.path.join(checkpoint_path, sorted(checkpoints)[-1])

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f'Loaded model from: {checkpoint_path}')
    if 'val_acc' in checkpoint:
        print(f'Model validation accuracy: {checkpoint["val_acc"]:.4f}')

    return model


def main(args):
    # 加载配置
    cfg = Config(args.config)
    if args.exp_name:
        cfg.update_exp(args.exp_name)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 设置日志
    logger = get_logger(cfg.log_path, name='validate')
    logger.info(f'Starting validation for experiment: {cfg.experiments.exp_name}')

    # 加载模型
    model = load_model(cfg, args.checkpoint, device)

    # 构建验证数据集
    val_dataset = BaseDataset(cfg, mode='val')
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 构建损失函数
    loss_fn = build_loss(cfg)

    # 类别名称
    class_names = getattr(cfg, 'class_names', None)

    # 执行验证
    results = validate_model(cfg, model, val_loader, loss_fn, device, class_names, logger)

    # 保存结果
    if args.save_results:
        import json
        output_file = args.output if args.output else 'validation_results.json'

        # 将numpy数组转换为列表以便JSON序列化
        json_results = {
            'loss': results['loss'],
            'accuracy': results['accuracy'],
            'f1_macro': results['f1_macro'],
            'f1_micro': results['f1_micro'],
            'f1_weighted': results['f1_weighted'],
            'precision_macro': results['precision_macro'],
            'recall_macro': results['recall_macro'],
            'f1_per_class': results['f1_per_class'].tolist(),
            'precision_per_class': results['precision_per_class'].tolist(),
            'recall_per_class': results['recall_per_class'].tolist(),
        }

        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        logger.info(f'Results saved to {output_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validation script')
    parser.add_argument('--config', default='config/config.yaml', help='config file path')
    parser.add_argument('--checkpoint', required=True, help='checkpoint path')
    parser.add_argument('--exp_name', default=None, help='experiment name')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--save_results', action='store_true', help='save results to file')
    parser.add_argument('--output', default=None, help='output file path')
    args = parser.parse_args()
    main(args)
