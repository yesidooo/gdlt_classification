"""
@auther: llf
"""
from torch import nn

from .backbone import build_backbone


class BaseClassifier(nn.Module):
    def __init__(self, backbone, num_classes=3):
        super(BaseClassifier).__init__()
        self.backbone = backbone
        self.num_classes = num_classes

        self._adj_head()

    def forward(self, x):
        out = self.backbone(x)
        out = self.head(out)
        return out

    def _adj_head(self, ):
        head_name, head_module = self.backbone.named_children()[-1]
        if isinstance(head_module, nn.Linear):
            in_channel = head_module.in_features
            self.backbone.head_name = nn.Linear(in_channel, self.num_classes)

        elif isinstance(head_module, nn.Sequential):
            in_channel = head_module[-1].in_features
            head_module[-1] = nn.Linear(in_channel, self.num_classes)

        else:
            raise NotImplementedError('head type can not be recognized, please check')

    def load_checkpoint(self, ):
        raise NotImplementedError()


def build_classifier(cfg):

    backbone = build_backbone(cfg)
    classifier = BaseClassifier(backbone, cfg['num_classes'])

    return classifier
