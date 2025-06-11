import os
import argparse
import torch
import glob
from torch.utils.data import DataLoader
import torch.nn.functional as F
import cv2
from PIL import Image
import numpy as np

from config import Config
from model import build_classifier
from utils.logger import get_logger


class InferenceDataset:
    """推理数据集"""

    def __init__(self, image_path, transforms=None):
        if os.path.isfile(image_path):
            self.image_paths = [image_path]
        elif os.path.isdir(image_path):
            self.image_paths = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                self.image_paths.extend(glob.glob(os.path.join(image_path, ext)))
        else:
            raise ValueError(f"Invalid path: {image_path}")

        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transforms:
            image = self.transforms(image)

        return image, image_path


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

    return model


@torch.no_grad()
def inference(cfg, model, dataloader, device, class_names=None, save_results=False):
    """执行推理"""
    model.eval()
    results = []

    for batch_idx, (images, image_paths) in enumerate(dataloader):
        images = images.to(device)

        # 前向传播
        outputs = model(images)
        probabilities = F.softmax(outputs, dim=1)

        # 获取预测结果
        scores, predictions = torch.max(probabilities, dim=1)

        for i, (prob, pred, score, img_path) in enumerate(zip(
                probabilities.cpu().numpy(),
                predictions.cpu().numpy(),
                scores.cpu().numpy(),
                image_paths
        )):
            result = {
                'image_path': img_path,
                'prediction': int(pred),
                'confidence': float(score),
                'probabilities': prob.tolist()
            }

            if class_names:
                result['class_name'] = class_names[pred]

            results.append(result)

            # 打印结果
            class_name = class_names[pred] if class_names else f'Class {pred}'
            print(f'{os.path.basename(img_path)}: {class_name} (confidence: {score:.4f})')

    return results


def main(args):
    # 加载配置
    cfg = Config(args.config)
    if args.exp_name:
        cfg.update_exp(args.exp_name)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = load_model(cfg, args.checkpoint, device)

    # 构建数据变换
    from torchvision import transforms
    inference_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 构建数据集和数据加载器
    dataset = InferenceDataset(args.input, inference_transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 类别名称
    class_names = getattr(cfg, 'class_names', None)

    # 执行推理
    results = inference(cfg, model, dataloader, device, class_names, args.save_results)

    # 保存结果
    if args.save_results:
        import json
        output_file = args.output if args.output else 'inference_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'Results saved to {output_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference script')
    parser.add_argument('--config', default='config/config.yaml', help='config file path')
    parser.add_argument('--checkpoint', required=True, help='checkpoint path')
    parser.add_argument('--input', required=True, help='input image or directory')
    parser.add_argument('--output', default=None, help='output file path')
    parser.add_argument('--exp_name', default=None, help='experiment name')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--save_results', action='store_true', help='save results to file')
    args = parser.parse_args()
    main(args)
