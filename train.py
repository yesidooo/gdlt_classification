import os.path
import argparse
from tqdm import tqdm

import torch
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from dataset import BaseDataset
from model import build_classifier, build_loss


def train(cfg, train_loader, val_loader, model, loss_fun, optimizer, lr_scheduler):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    best_acc, best_epoch = 0.0, 0
    for epoch in range(1, cfg.max_epoch+1):
        cur_lr = lr_scheduler.get_lr()
        model.train()
        train_loss, train_acc = 0.0, 0.0
        with tqdm(total=22500, unit='img') as bar1:
            bar1.set_description('train_epoch {}'.format(epoch))
            for idx, data in enumerate(train_loader):
                for key in data.keys():
                    data[key] = data[key].to(device)
                optimizer.zero_grad()
                pred = model(data['img'])
                loss = loss_fun(pred, data['label'])
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * cfg.batch_size
                _, cls = torch.max(pred, dim=1)
                correct_count = cls.eq(data['label'].data.view_as(cls))
                acc = torch.mean(correct_count.type(torch.FloatTensor)).item()
                train_acc += acc * cfg.batch_size
                bar1.update(cfg.batch_size)
                bar1.set_postfix({'lr': cur_lr, 'loss': loss.item(), 'acc': acc})
        avg_train_loss = train_loss / (len(train_loader) * cfg.batch_size)
        avg_train_acc = train_acc / (len(train_loader) * cfg.batch_size)

        # val
        val_loss, val_acc = 0.0, 0.0
        y_true, y_pred = torch.tensor([]), torch.tensor([])
        model.eval()
        with torch.no_grad():
            with tqdm(total=2500, unit='img') as bar2:
                bar2.set_description('val')
                for idx, data in enumerate(val_loader):
                    for key in data.keys():
                        data[key] = data[key].to(device)
                    pred = model(data['img'])
                    loss = loss_fun(pred, data['label'])
                    val_loss += loss.item() * cfg.batch_size
                    _, cls = torch.max(pred, dim=1)
                    correct_count = cls.eq(data['label'].data.view_as(cls))
                    acc = torch.mean(correct_count.type(torch.FloatTensor)).item()
                    val_acc += acc * cfg.batch_size
                    bar2.update(cfg.batch_size)
                    bar2.set_postfix({'loss': loss.item(), 'acc': acc})

                    y_true = torch.cat((y_true, data['label']), dim=0)
                    y_pred = torch.cat((y_pred, cls), dim=0)
        y_true, y_pred = y_true.data.numpy(), y_pred.data.numpy()
        f1 = f1_score(y_true, y_pred, average='macro')
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')

        avg_val_loss = val_loss / (len(val_loader) * cfg.batch_size)
        avg_val_acc = val_acc / (len(val_loader) * cfg.batch_size)

        print(f'avg_train_loss: {avg_train_loss}, avg_train_acc: {avg_train_acc}')
        print(f'avg_val_loss: {avg_val_loss}, avg_val_acc: {avg_val_acc}')

        lr_scheduler.step()

        state_dict = {'model': model.state_dict(),
                      'f1_score': f1,
                      'accuracy_score': accuracy,
                      'precision_score': precision,
                      'recall_score': recall,
                      'avg_val_loss': avg_val_loss,
                      'avg_val_acc': avg_val_acc}
        if not os.path.exists(cfg.checkpoint_path):
            os.mkdir(cfg.checkpoint_path)
        torch.save(state_dict, f'{cfg.checkpoint_path}/checkpoint_ep_{str(epoch).zfill(3)}')


def train_one_epoch(epoch, model, loader, optimizer, loss_fn, args, device=torch.device('cuda'), lr_scheduler=None,
        saver=None, output_dir=None, amp_autocast=suppress, loss_scaler=None, model_ema=None, mixup_fn=None,):
    raise NotImplementedError()


def validate(model, loader, loss_fn, args, device=torch.device('cuda'), amp_autocast=suppress, log_suffix=''):
    raise NotImplementedError()


def main(exp_name):
    cfg = Config()
    cfg.update_exp(exp_name)
    train_dataset = Cat_Dog(dataset_path=cfg.dataset_path, mode='train')
    val_dataset = Cat_Dog(dataset_path=cfg.dataset_path, mode='val')
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_works)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_works)
    model = build_classifier(cfg)
    loss_fun = build_loss(loss_mode=cfg.loss_mode)
    params_dict = [{'params': model.backbone.parameters(), 'lr': cfg.backbone_lr},
                   {'params': model.head.parameters()}]
    optimizer = torch.optim.AdamW(params_dict, lr=cfg.lr, weight_decay=cfg.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [5, ])
    kwargs = dict(cfg=cfg,
                  train_loader=train_loader,
                  val_loader=val_loader,
                  model=model,
                  loss_fun=loss_fun,
                  optimizer=optimizer,
                  lr_scheduler=lr_scheduler)
    train(**kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default='exp', help='experiment name')
    args = parser.parse_args()
    main(args.exp_name)