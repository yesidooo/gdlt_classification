from . import loss_func


def build_loss(cfg):
    loss_cfg = cfg.model.loss_func
    loss_name = list(loss_cfg.keys())[0]
    loss_params = loss_cfg[loss_name] if loss_cfg[loss_name] else {}

    if hasattr(loss_func, loss_name):
        loss = getattr(loss_func, loss_name)(**loss_params)
    else:
        raise Exception(f'{loss_name} not support yet')
    return loss
