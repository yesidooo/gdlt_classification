from .ce_loss import *

def build_loss(loss_mode='labelsmoothingcrossentropy'):
    if loss_mode == 'labelsmoothingcrossentropy':
        return LabelSmoothingCrossEntropy(smoothing=0.1)
    elif loss_mode == 'softtargetcrossentropy':
        return SoftTargetCrossEntropy()