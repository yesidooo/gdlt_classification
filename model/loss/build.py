import loss_func


def build_loss(cfg):
    loss_name, loss_cfg = cfg.items()[0]
    if hasattr(loss_func, loss_name):
        loss = getattr(loss_func, loss_name)(*loss_cfg)
    else:
        raise Exception(f'{loss_name} not support yet')
    return loss
