"""
@auther: llf
"""
import os
import cv2
import glob
import pandas as pd
from PIL import Image
from torchvision.transforms import transforms


class BaseDataset:
    """
    dataset基类
    """
    def __init__(self, cfg, mode='train'):
        self.reader = cfg['reader']
        self.dataset_name = cfg['dataset_name']
        self.dataset_path = cfg['dataset_path']
        self.mode = mode

        self.transforms = self.build_transforms(cfg['augments'])
        self.mete_data = self.get_mete_data()

    def get_mete_data(self, ):
        file_path = glob.glob(f'{self.dataset_path}/{self.mode}*')[0]
        _, ext = os.path.splitext(file_path)
        mete_data = dict(images=[], labels=[])
        if ext == '.txt':
            f = open(file_path, 'r')
            lines = f.readlines()
            for line in lines:
                image, label = line.split(' ')
                mete_data['images'].append(image)
                mete_data['label'].append(int(label))
            f.close()

        elif ext == '.csv':
            df = pd.read_csv(file_path)
            for idx, row in df.iterrows():
                mete_data['images'].append(row[0])
                mete_data['label'].append(row[1])

        else:
            raise NotImplementedError(f'format {ext} is not support so far')

        return mete_data

    def build_transforms(self, aug_cfg):

        return None

    def __len__(self):
        return len(self.mete_data['img'])

    def __getitem__(self, idx):
        img, label = self.mete_data['img'][idx], self.mete_data['label'][idx]
        img = Image.open(img)
        img = self.transforms(img)
        return {'img': img, 'label': label}


if __name__ == '__main__':
    raise NotImplementedError
