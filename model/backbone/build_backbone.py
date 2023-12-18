from .resnet import *
from .vit import *

def build_backbone(backbone_mode='resnet18', pretrained=True, pretrained_path=None):
    if backbone_mode == 'resnet18':
        return resnet18(pretrained, pretrained_path)