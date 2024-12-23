import yaml
import torch
from argparse import Namespace
from model_zoo import get_model
from train_model import Train_model
from loss_model import OhemCrossEntropyLoss, BootstrappedCELoss, DetailAggregateLoss
from lr_scheduler import get_lr_fun
from dataset import data
import numpy as np
import os

torch.cuda.empty_cache()
os.environ['PYTHONHASHSEED'] = str(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main_method(train_config):
    torch.cuda.empty_cache()
    model = get_model(train_config)
    model.cuda()
    if train_config.dataset == "CamVid":
        data_loader = data.get_camvid(
            data_path=train_config.data_path,
            batch_size=train_config.batch_size,
            num_workers=train_config.num_workers)
    elif train_config.dataset == "MF" and not train_config.six_channels:
        data_loader = data.get_MF(
            data_path=train_config.data_path,
            batch_size=train_config.batch_size,
            num_workers=train_config.num_workers,
            train_config=train_config)
    elif train_config.dataset == "MF" and train_config.six_channels:
        from dataset2.dataloader import get_dataloader
        data_loader = get_dataloader(train_config)
    elif train_config.dataset == "PST" and train_config.six_channels:
        from dataset2.dataloader import get_dataloader
        data_loader = get_dataloader(train_config)
    else:
        raise NotImplementedError
    if train_config.loss_type == "OhemCrossEntropyLoss":
        loss_fun = OhemCrossEntropyLoss(
            ignore_label=train_config.ignore_label,
            thresh=train_config.thresh,
            min_kept=train_config.min_kept)
    elif train_config.loss_type == "BootstrappedCELoss":
        loss_fun = BootstrappedCELoss(
            train_config)
    elif train_config.loss_type == "CrossEntropyLoss":
        loss_fun = torch.nn.CrossEntropyLoss(ignore_index=train_config.ignore_label)
    else:
        raise NotImplementedError
    if train_config.multi_loss:
        detail_loss = DetailAggregateLoss()
    else:
        detail_loss = None
    if train_config.optimizer == "SGD":
        if train_config.multi_loss:
            optimizer = torch.optim.SGD([{'params': model.parameters()},
                                         {'params': detail_loss.parameters()}],
                lr=train_config.lr,
                momentum=train_config.momentum,
                weight_decay=train_config.weight_decay)
        else:
              optimizer = torch.optim.SGD(model.parameters(),
                lr=train_config.lr,
                momentum=train_config.momentum,
                weight_decay=train_config.weight_decay)
    elif train_config.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay)
    else:
        raise NotImplementedError
    lr_func = get_lr_fun(train_config)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    train_model = Train_model(
        model,
        loss_fun,
        optimizer,
        lr_scheduler,
        data_loader,
        detail_loss,
        train_config)
    train_model.train()


if __name__ == '__main__':
    with open("configs/train_MF.yaml", 'r') as f:
        args = yaml.safe_load(f)
    train_config = Namespace(**args)
    main_method(train_config)
