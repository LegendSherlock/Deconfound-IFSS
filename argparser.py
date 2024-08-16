import argparse
from utils import tasks
import json
import os


def modify_command_options(opts):
    if opts.dataset == 'voc':
        opts.num_classes = 21
        if opts.task == '19-1':
            opts.label_base_num = 18
            opts.label_novel_num = 1
        elif opts.task == '15-5':
            opts.label_base_num = 10
            opts.label_novel_num = 5
        elif opts.task == '15-5s':
            opts.label_base_num = 14
            opts.label_novel_num = 1
        elif opts.task == '10-1':
            opts.label_base_num = 9
            opts.label_novel_num = 1
    elif opts.dataset == 'ade':
        opts.num_classes = 151
    elif opts.dataset == "cityscapes":
        opts.num_classes = 17
    elif opts.dataset == "cityscapes_domain":
        opts.num_classes = 19
    elif opts.dataset == "cityscapes_classdomain":
        opts.num_classes = 19
    else:
        raise NotImplementedError(f"Unknown dataset: {opts.dataset}")


    opts.loss_kd = 10  # Knowlesge Distillation (Soft_CrossEntropy)
    opts.unce = True  # Unbiased Cross Entropy instead of CrossEntropy
    opts.unkd = True  # Unbiased Knowledge Distillation instead of Knowledge Distillation
    opts.init_balanced = True # Weight Imprinting

    return opts


def parse_args():
    parser = argparse.ArgumentParser()
    # -----------------------------------------------------------------------------
    # normal
    # -----------------------------------------------------------------------------

    parser.add_argument("--parallel", action='store_true', default=False)
    parser.add_argument('--world_size', default=4, type=int, help='number of distributed processes')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument('--seed', type=int, default=42,
                        help='number of episodes per batch')
    parser.add_argument('--num_workers', type=int, default=0)

    # -----------------------------------------------------------------------------
    # dataset
    # -----------------------------------------------------------------------------

    parser.add_argument('--dataset', type=str, default='voc',
                            choices=['voc', 'ade', 'cityscapes_domain'])
    parser.add_argument("--data_root", type=str, default='/data1/lihao/VOC12', help="path to Dataset")  # '/data1/lihao/VOC12' 'D:/data/VOC12'

    # -----------------------------------------------------------------------------
    # train
    # -----------------------------------------------------------------------------
    parser.add_argument('--learn_base_epoch', type=int, default=30,
                            help='number of learn_base epochs')
    parser.add_argument('--meta_train_epoch', type=int, default=10,
                            help='number of meta_train epochs')
    parser.add_argument('--learn_novel_epoch', type=int, default=30,
                            help='number of learn_novel epochs')
    parser.add_argument(
        "--val_interval", type=int, default=1, help="epoch interval for eval (default: 1)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="batch size of trainloader"
    )

    parser.add_argument('--train_episode', type=int, default=1000,
                            help='number of episodes per validation')
    parser.add_argument('--val_episode', type=int, default=500,
                            help='number of episodes per validation')

    parser.add_argument('--save_root', default='./experiments/')
    parser.add_argument(
        "--amp",
        type=bool,
        default=True,
        help="Use this to use Automatic mixed precision"
    )
    parser.add_argument(
        "--lr_policy",
        type=str,
        default='poly',
        choices=['poly', 'step'],
        help="lr schedule policy (default: poly)"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help='weight decay (default: 1e-4) or 5e-4'
    )
    parser.add_argument(
        "--lr", type=float, nargs="+", default=[0.01], help="learning rate (default: 0.007)"
    )

    # -----------------------------------------------------------------------------
    # Incremental
    # -----------------------------------------------------------------------------
    parser.add_argument(
        "--task",
        type=str,
        default="15-5s",
        choices=tasks.get_task_list(),
        help="Task to be executed (default: 19-1)"
    )
    parser.add_argument(
        "--step",
        type=int,
        nargs="+",
        default=[0],
        help="The incremental step in execution (default: 0)"
    )
    parser.add_argument("--nshot", type=int, default=-1,
                        help="If step>0, the shot to use for FSL (Def=5), -1 means many shot learning")
    parser.add_argument("--ishot", type=int, default=1,
                        help="exp number")
    parser.add_argument(
        "--data_masking",
        type=str,
        default="current",
        choices=["current", "current+old", "all", "new"]
    )
    parser.add_argument(
        "--overlap",
        action='store_true',
        default=True,
        help="Use this to not use the new classes in the old training set"
    )

    # -----------------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------------
    parser.add_argument(
        "--backbone",
        type=str,
        default='resnet101',
        choices=['resnet50', 'resnet101'],
        help='backbone for the body (def: resnet50)'
    )
    parser.add_argument(
        "--output_stride",
        type=int,
        default=16,
        choices=[8, 16],
        help='stride for the backbone (def: 16)'
    )

    parser.add_argument(
        "--pooling",
        type=int,
        default=32,
        help='pooling in ASPP for the validation phase (def: 32)'
    )



    args = parser.parse_args()
    args = modify_command_options(args)
    return args