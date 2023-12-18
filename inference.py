import os

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import cv2
import argparse

from dataset import Cat_Dog_Infer
from config import Config
from model import build_classifier

def load_checkpoint(model, checkpoint_path):
    checkpoint = os.listdir(checkpoint_path)[-1]
    checkpoint = os.path.join(checkpoint_path, checkpoint)
    checkpoint_dict = torch.load(checkpoint)
    model.load_state_dict(checkpoint_dict['model'])
    return model

def inference(cfg, loader, model):
    model.eval()
    with torch.no_grad():
        for idx, (img, img_path) in enumerate(loader):
            pred = model(img)
            print(pred.data.numpy())
            pred = F.softmax(pred, 1)
            score, cls = torch.max(pred, dim=1)
            txt = f'cls: {cfg.cls[cls.item()]}, score: {score.item()}'
            vis = True
            if vis:
                img = cv2.imread(img_path[0])
                cv2.putText(img, txt, (50, 50), 0, 1, (255, 0, 0), 2)
                cv2.imshow(f'inference_img_{idx}', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    print('inference finished')

def main(exp_name):
    cfg = Config()
    cfg.update_exp(exp_name)
    dataset = Cat_Dog_Infer(cfg.infer_img_path)
    loader = DataLoader(dataset, batch_size=1, num_workers=0)
    model = build_classifier(cfg)
    model = load_checkpoint(model, cfg.checkpoint_path)
    inference(cfg, loader, model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default='exp', help='experiment name')
    args = parser.parse_args()