from torch import nn
from .backbone import build_backbone


class BaseClassifier(nn.Module):
    def __init__(self, cfg):
        super(BaseClassifier).__init__()
        self.backbone = build_backbone(cfg.model)
        self.num_classes = cfg.model.num_classes
        self._adj_head()

    def forward(self, x):
        features = self.backbone(x)
        if hasattr(self, 'head'):
            out = self.head(features)
        else:
            out = features
        return out

    def _adj_head(self):
        """调整分类头"""
        # 获取最后一层的名称和模块
        last_child_name = list(self.backbone.named_children())[-1][0]
        last_child = list(self.backbone.children())[-1]

        if isinstance(last_child, nn.Linear):
            in_features = last_child.in_features
            setattr(self.backbone, last_child_name, nn.Linear(in_features, self.num_classes))

        elif isinstance(last_child, nn.Sequential) and isinstance(last_child[-1], nn.Linear):
            in_features = last_child[-1].in_features
            last_child[-1] = nn.Linear(in_features, self.num_classes)

        elif hasattr(self.backbone, 'classifier') and isinstance(self.backbone.classifier, nn.Linear):
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(in_features, self.num_classes)

        elif hasattr(self.backbone, 'fc') and isinstance(self.backbone.fc, nn.Linear):
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, self.num_classes)

        else:
            # 如果无法自动调整，添加一个新的分类头
            if hasattr(self.backbone, 'num_features'):
                in_features = self.backbone.num_features
            elif hasattr(self.backbone, 'feature_info'):
                in_features = self.backbone.feature_info[-1]['num_chs']
            else:
                # 通过前向传播获取特征维度
                import torch
                with torch.no_grad():
                    dummy_input = torch.randn(1, 3, 224, 224)
                    features = self.backbone(dummy_input)
                    in_features = features.shape[-1]

            self.head = nn.Linear(in_features, self.num_classes)


def build_classifier(cfg):
    classifier = BaseClassifier(cfg)
    return classifier
