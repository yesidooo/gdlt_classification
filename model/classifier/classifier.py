from torch import nn

from ..backbone import build_backbone

class Classifier(nn.Module):
    def __init__(self, backbone, num_classes=3):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(1000, num_classes)

    def forward(self, x):
        out = self.backbone(x)
        out = self.head(out)
        return out

def build_classifier(cfg):
    backbnoe = build_backbone(cfg.backbone_mode, cfg.pretrained, cfg.pretrained_path)
    classifier = Classifier(backbnoe, cfg.num_classes)
    return classifier