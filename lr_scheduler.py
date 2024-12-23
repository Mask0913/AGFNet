import torch
import math


def poly_lr_scheduler(current_iter, total_iters, warmup_iters, warmup_factor, p=0.9):
    lr=(1 - current_iter / total_iters) ** p
    if current_iter < warmup_iters:
        alpha=warmup_factor+(1-warmup_factor)*(current_iter/warmup_iters)
        lr*=alpha
    return lr

def get_lr_fun(config):
    if 'poly' == config.lr_scheduler:
        return lambda x : poly_lr_scheduler(x, config.iter, config.warmup_iters, config.warmup_factor)