from utils import *
import torch.optim as optim
import numpy as np
import torch


def get_finetune_optimizer(opts, model, dataloader, max_epoch, freeze_backbone=False):
    params = []

    if not freeze_backbone:
        params.append(
            {
                "params": filter(lambda p: p.requires_grad,
                                 model.module.body.parameters()
                                 if opts.parallel else model.body.parameters()),
                'weight_decay': opts.weight_decay
            }
        )

    params.append(
        {
            "params": filter(lambda p: p.requires_grad,
                             model.module.head.parameters()
                             if opts.parallel else model.head.parameters()),
            'weight_decay': opts.weight_decay
        }
    )

    optimizer = torch.optim.SGD(params, lr=opts.lr, momentum=0.9, nesterov=True)

    if opts.lr_policy == 'poly':
        scheduler = PolyLR(
            optimizer, max_iters=max_epoch * len(dataloader), power=0.9
        )
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=5000, gamma=0.1
        )
        # lambda_epoch = lambda e: 1.0 if e < 60 else (0.1 if e < 80 else 0.01 if e < 90 else (0.001))
        # lr_scheduler = torch.optsim.lr_scheduler.LambdaLR(optsimizer, lr_lambda=lambda_epoch, last_epoch=-1)
    else:
        raise NotImplementedError

    return optimizer, scheduler


