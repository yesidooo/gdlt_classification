import random

from torchvision import transforms


__all__ = [
    'ToPILImage', 'ToTensor', 'PILToTensor', 'SRColorJitter', 'SRGaussianBlur', 'SRRandomAffine',
    'RandomHorizontalFlip', 'RandomVerticalFlip', 'Normalize', 'RandomErasing'
]


class ToPILImage(transforms.ToPILImage):

    def __init__(self, *args, **kwargs):
        super(ToPILImage).__init__(*args, **kwargs)


class ToTensor(transforms.ToTensor):
    def __init__(self, ):
        super(ToTensor).__init__()


class PILToTensor(transforms.PILToTensor):
    def __init__(self, ):
        super(PILToTensor).__init__()


class SRColorJitter(transforms.ColorJitter):
    """在原始ColorJitter基础上封装以一定概率进行此变换的功能
    """
    def __init__(self, prob=0.5, *args, **kwargs):
        super(SRColorJitter).__init__(*args, **kwargs)
        self.prob = prob

    def forward(self, img):
        rand = random.random()
        if rand < self.prob:
            return super().forward(img)
        else:
            return img


class SRGaussianBlur(transforms.GaussianBlur):
    """在原始GaussianBlur基础上封装以一定概率进行此变换的功能
    """
    def __init__(self, prob=0.5, *args, **kwargs):
        super(SRGaussianBlur).__init__(*args, **kwargs)
        self.prob = prob

    def forward(self, img):
        rand = random.random()
        if rand < self.prob:
            return super().forward(img)
        else:
            return img


class SRRandomAffine(transforms.RandomAffine):
    """在原始RandomAffine基础上封装以一定概率进行此变换的功能
    """
    def __init__(self, prob=0.5, *args, **kwargs):
        super(SRRandomAffine).__init__(*args, **kwargs)
        self.prob = prob

    def forward(self, img):
        rand = random.random()
        if rand < self.prob:
            return super().forward(img)
        else:
            return img


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):

    def __init__(self, *args, **kwargs):
        super(RandomHorizontalFlip).__init__(*args, **kwargs)


class RandomVerticalFlip(transforms.RandomVerticalFlip):

    def __init__(self, *args, **kwargs):
        super(RandomVerticalFlip).__init__(*args, **kwargs)


class Normalize(transforms.Normalize):

    def __init__(self, *args, **kwargs):
        super(Normalize).__init__(*args, **kwargs)


class RandomErasing(transforms.RandomErasing):

    def __init__(self, *args, **kwargs):
        super(RandomErasing).__init__(*args, **kwargs)