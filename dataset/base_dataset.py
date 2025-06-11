"""
@author: llf
"""
import os
import cv2
import glob
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import aug_common


class BaseDataset(Dataset):
    """
    dataset基类
    """

    def __init__(self, cfg, mode='train'):
        self.reader = cfg['data']['reader']
        self.dataset_name = cfg['data']['dataset_name']
        self.dataset_path = cfg['data']['dataset_path']
        self.mode = mode

        self.transforms = self.build_transforms(cfg['data']['augments'][self.mode]['common'])
        self.mete_datas = self.get_mete_datas()

    def get_mete_datas(self):
        file_path = glob.glob(f'{self.dataset_path}/{self.mode}*')[0]
        _, ext = os.path.splitext(file_path)
        mete_datas = dict(images=[], labels=[])

        if ext == '.txt':
            with open(file_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    image, label = line.strip().split(' ')
                    mete_datas['images'].append(os.path.join(self.dataset_path, image))
                    mete_datas['labels'].append(int(label))

        elif ext == '.csv':
            df = pd.read_csv(file_path)
            for idx, row in df.iterrows():
                mete_datas['images'].append(os.path.join(self.dataset_path, row[0]))
                mete_datas['labels'].append(int(row[1]))

        else:
            raise NotImplementedError(f'format {ext} is not support so far')

        return mete_datas

    @staticmethod
    def build_transforms(aug_cfg: dict):
        trans_items = []
        for aug, cfg in aug_cfg.items():
            if hasattr(aug_common, aug):
                aug_func = getattr(aug_common, aug)
                if cfg is None:
                    trans_items.append(aug_func())
                else:
                    trans_items.append(aug_func(**cfg))
            else:
                raise Exception(f'{aug} not support yet')
        return transforms.Compose(trans_items)

    def __len__(self):
        return len(self.mete_datas['images'])

    def __getitem__(self, idx):
        image_path, label = self.mete_datas['images'][idx], self.mete_datas['labels'][idx]

        if self.reader == 'cv2':
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = Image.open(image_path).convert('RGB')

        image = self.transforms(image)
        return {'image': image, 'label': label}


if __name__ == '__main__':
    raise NotImplementedError
