import os
import argparse
import torch
import glob
import torch.nn.functional as F
import cv2
from PIL import Image
import numpy as np
import json

from config import Config
from model import build_classifier
from utils.logger import get_logger


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


def get_image_paths(input_path):
    """获取图片路径列表"""
    if os.path.isfile(input_path):
        return [input_path]
    elif os.path.isdir(input_path):
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']:
            image_paths.extend(glob.glob(os.path.join(input_path, ext)))
        return image_paths
    else:
        raise ValueError(f"Invalid path: {input_path}")


def preprocess_image(image_path, transforms):
    """预处理单张图片"""
    try:
        image = Image.open(image_path).convert('RGB')
        if transforms:
            image = transforms(image)
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


@torch.no_grad()
def inference_single_image(model, image_tensor, device, class_names=None):
    """对单张图片进行推理"""
    model.eval()

    # 添加batch维度
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    # 前向传播
    outputs = model(image_tensor)
    probabilities = F.softmax(outputs, dim=1)

    # 获取预测结果
    scores, predictions = torch.max(probabilities, dim=1)

    prob = probabilities.cpu().numpy()[0]
    pred = predictions.cpu().numpy()[0]
    score = scores.cpu().numpy()[0]

    result = {
        'prediction': int(pred),
        'confidence': float(score),
        'probabilities': prob.tolist()
    }

    if class_names:
        result['class_name'] = class_names[pred]

    return result


def inference_batch_images(model, image_paths, transforms, device, class_names=None, batch_size=32):
    """批量推理多张图片"""
    results = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        valid_paths = []

        # 加载并预处理批次中的图片
        for img_path in batch_paths:
            image = preprocess_image(img_path, transforms)
            if image is not None:
                batch_images.append(image)
                valid_paths.append(img_path)

        if not batch_images:
            continue

        # 将图片堆叠成批次
        batch_tensor = torch.stack(batch_images).to(device)

        # 批次推理
        outputs = model(batch_tensor)
        probabilities = F.softmax(outputs, dim=1)
        scores, predictions = torch.max(probabilities, dim=1)

        # 处理结果
        for j, img_path in enumerate(valid_paths):
            prob = probabilities[j].cpu().numpy()
            pred = predictions[j].cpu().numpy()
            score = scores[j].cpu().numpy()

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


def inference_single_path(cfg, model, image_path, transforms, device, class_names=None):
    """推理单个图片路径"""
    image = preprocess_image(image_path, transforms)
    if image is None:
        return None

    result = inference_single_image(model, image, device, class_names)
    result['image_path'] = image_path

    # 打印结果
    class_name = result.get('class_name', f"Class {result['prediction']}")
    print(f'{os.path.basename(image_path)}: {class_name} (confidence: {result["confidence"]:.4f})')

    return result


def main(args):
    # 加载配置
    cfg = Config(args.config)
    if args.exp_name:
        cfg.update_exp(args.exp_name)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 加载模型
    print('Loading model...')
    model = load_model(cfg, args.checkpoint, device)

    # 构建数据变换
    from torchvision import transforms
    inference_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 类别名称
    class_names = getattr(cfg, 'class_names', None)

    # 获取图片路径
    image_paths = get_image_paths(args.input)
    print(f'Found {len(image_paths)} images')

    results = []

    if len(image_paths) == 1:
        # 单张图片推理
        result = inference_single_path(cfg, model, image_paths[0], inference_transforms, device, class_names)
        if result:
            results.append(result)
    else:
        # 批量图片推理
        results = inference_batch_images(model, image_paths, inference_transforms, device, class_names, args.batch_size)

    # 保存结果
    if args.save_results and results:
        output_file = args.output if args.output else 'inference_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f'Results saved to {output_file}')

    return results


def inference_from_array(cfg, model, image_array, device, class_names=None):
    """从numpy数组推理（便于外部调用）"""
    from torchvision import transforms

    # 构建变换
    inference_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 如果是BGR格式，转换为RGB
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

    # 预处理
    image_tensor = inference_transforms(image_array)

    # 推理
    result = inference_single_image(model, image_tensor, device, class_names)

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference script')
    parser.add_argument('--config', default='config/config.yaml', help='config file path')
    parser.add_argument('--checkpoint', required=True, help='checkpoint path')
    parser.add_argument('--input', required=True, help='input image or directory')
    parser.add_argument('--output', default=None, help='output file path')
    parser.add_argument('--exp_name', default=None, help='experiment name')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for multiple images')
    parser.add_argument('--save_results', action='store_true', help='save results to file')
    args = parser.parse_args()
    main(args)