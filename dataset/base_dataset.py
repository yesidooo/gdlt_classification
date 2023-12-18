import os
import pandas as pd
from PIL import Image
import torch
from torchvision.transforms import transforms

class Cat_Dog:
    def __init__(self, dataset_name='cat_dog', dataset_path='', mode='train'):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.mode = mode
        if self.mode == 'train':
            train_csv = os.path.join(self.dataset_path, 'train.csv')
            self.data = pd.read_csv(train_csv)
            self.transforms = transforms.Compose([
                                                 transforms.Resize((418, 418)),
                                                 transforms.ToTensor()])
        if self.mode == 'val':
            val_csv = os.path.join(self.dataset_path, 'val.csv')
            self.data = pd.read_csv(val_csv)
            self.transforms = transforms.Compose([transforms.Resize([418, 418]),
                                                 transforms.ToTensor()])

    def __len__(self):
        return len(self.data['img'])

    def __getitem__(self, idx):
        img, label = self.data['img'][idx], self.data['label'][idx]
        img = Image.open(img)
        img = self.transforms(img)
        return {'img': img, 'label': label}

class Cat_Dog_Infer(Cat_Dog):
    def __init__(self, infer_imgs_path):
        self.infer_imgs_path = infer_imgs_path
        infer_imgs = os.listdir(self.infer_imgs_path)
        self.data = [os.path.join(self.infer_imgs_path, i) for i in infer_imgs]
        self.transforms = transforms.Compose([transforms.Resize([418, 418]),
                                              transforms.ToTensor()])

    def __len__(self):
        return len(self.data)//100

    def __getitem__(self, idx):
        img_path = self.data[idx]
        img = Image.open(img_path)
        img = self.transforms(img)
        return img, img_path

if __name__ == '__main__':
    dataset_path = '../kaggle'
    dataset = Cat_Dog(dataset_path=dataset_path)
    d = dataset(0)