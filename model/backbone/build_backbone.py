import torch
import timm

from torchvision import models


def build_backbone(cfg):
    """
    根据配置文件加载 backbone (from torchvision or timm), 并根据配置加载预训练权重
    :param cfg:
    :return: model with/without pretrained weights
    """
    backbone = cfg.get('backbone', None)
    pretrained = cfg.get('pretrained', False)
    pretrained_path = cfg.get('pretrained_path', None)

    if pretrained_path is not None:
        pretrained = False

    if hasattr(models, backbone):
        backbone = getattr(models, backbone)(pretrained=pretrained)
    elif hasattr(timm.models, backbone):
        backbone = getattr(timm.models, backbone)(pretrained=pretrained)
    else:
        raise NotImplementedError(f'{backbone} not support so far')

    if pretrained_path is not None:
        weights = torch.load(pretrained_path, map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
        try:
            backbone = backbone.load_state_dict(weights, strict=True)
        except ValueError as e:
            print(str(e))
            backbone = backbone.load_state_dict(weights, strict=False)

    return backbone
