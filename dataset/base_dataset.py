"""
@auther: llf
"""
import os
import cv2
import glob
import pandas as pd
from PIL import Image

from torchvision import transforms

from aug_common import *


class BaseDataset:
    """
    dataset基类
    """
    def __init__(self, cfg, mode='train'):
        self.reader = cfg['reader']
        self.dataset_name = cfg['dataset_name']
        self.dataset_path = cfg['dataset_path']
        self.mode = mode

        self.transforms = self.build_transforms(cfg['augments'][self.mode]['common'])
        self.mete_datas = self.get_mete_datas()

    def get_mete_datas(self, ):
        file_path = glob.glob(f'{self.dataset_path}/{self.mode}*')[0]
        _, ext = os.path.splitext(file_path)
        mete_datas = dict(images=[], labels=[])
        if ext == '.txt':
            f = open(file_path, 'r')
            lines = f.readlines()
            for line in lines:
                image, label = line.split(' ')
                mete_datas['images'].append(image)
                mete_datas['labels'].append(int(label))
            f.close()

        elif ext == '.csv':
            df = pd.read_csv(file_path)
            for idx, row in df.iterrows():
                mete_datas['images'].append(row[0])
                mete_datas['labels'].append(row[1])

        else:
            raise NotImplementedError(f'format {ext} is not support so far')

        return mete_datas

    @staticmethod
    def build_transforms(aug_cfg: dict):
        trans_items = []
        for aug, cfg in aug_cfg.items():
            trans_items.append(eval(aug)(*cfg))
        return transforms.Compose(trans_items)

    def __len__(self):
        return len(self.mete_datas['images'])

    def __getitem__(self, idx):
        image, label = self.mete_datas['images'][idx], self.mete_datas['labels'][idx]
        if self.reader == cv2:
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = Image.open(image)
        image = self.transforms(image)
        return {'image': image, 'label': label}


if __name__ == '__main__':
    raise NotImplementedError
