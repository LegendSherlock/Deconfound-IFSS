# -*- coding: utf-8 -*-

from utils import *
from argparser import parse_args
from utils import tasks
import copy
from modules.deeplab_with_proto import deeplab_with_proto
from modules.segmentation_module import IncrementalSegmentationModule
from modules import resnet_dialated as resnet
from modules.meta_learner import *
import torch.nn as nn
from trainer import *
import os
from dataset import *
import wandb
import pandas as pd

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# os.environ["WANDB_API_KEY"] = 'd4fb616ac694e5e2729693648e225c5c4e85130f'
# os.environ["WANDB_MODE"] = "offline"

def get_dataset(opts):
    labels, labels_old, path_base = tasks.get_task_labels(opts.dataset, opts.task, opts.step)
    opts.labels_old = labels_old
    opts.labels = labels
    labels_cum = list(range(21))  # labels_old + labels
    if not os.path.exists(path_base):
        os.makedirs(path_base, exist_ok=True)

    train_transform = transform.Compose(
        [
            transform.RandomResizedCrop(512, (0.5, 2.0)),
            transform.RandomHorizontalFlip(),
            transform.ToTensor(),
            transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transform.Compose(
        [
            transform.RandomResizedCrop(512, (0.5, 2.0)),
            transform.ToTensor(),
            transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    if opts.dataset == 'voc':
        dataset = VOCSegmentationIncremental
    elif opts.dataset == 'ade':
        dataset = AdeSegmentationIncremental
    elif opts.dataset == 'cityscapes_domain':
        dataset = CityscapesSegmentationIncrementalDomain
    else:
        raise NotImplementedError

    train_dst = dataset(
        root=opts.data_root,
        train=True,
        transform=train_transform,
        labels=list(labels),
        labels_old=list(labels_old),
        idxs_path=path_base + f"/train-{opts.step}.npy",
        overlap=opts.overlap,
        data_masking=opts.data_masking,
        nshot=opts.nshot if opts.step != 0 else -1,
        ishot=opts.ishot,
        opts=opts
    )

    val_dst = dataset(
        root=opts.data_root,
        train=False,
        transform=val_transform,
        labels=list(labels),
        labels_old=list(labels_old),
        idxs_path=path_base + f"/val-{opts.step}.npy",
        data_masking=opts.data_masking,
        opts=opts
    )

    test_dst = dataset(
        root=opts.data_root,
        train=False,
        transform=val_transform,
        labels=list(labels_cum),
        idxs_path=path_base + f"/test-{opts.step}.npy",
        opts=opts
    )
    opts.logger.log(f"Dataset: {opts.dataset}, Train set: {len(train_dst)}, Val set: {len(val_dst)},"
                    f" Test set: {len(test_dst)}, new labels: {train_dst.labels}, old labels: {train_dst.labels_old}")

    return (train_dst, val_dst, test_dst)


def get_model(opts, classes=None):
    body = resnet.Res101_Deeplab(pretrained=True, stop_layer='fc')

    head_channels = 256

    head_seg = deeplab_with_proto(
        body.out_channels,
        head_channels,
        256,
        norm=nn.BatchNorm2d,
        out_stride=opts.output_stride,
        pooling_size=opts.pooling,
        classes=classes
    )

    model_seg = IncrementalSegmentationModule(
        body,
        head_seg,
        head_channels,
        classes=classes,
        opts=opts
    )

    head_meta = ProtoComNet(in_dim=body.out_channels, opts=opts)

    model_meta = MetaLearner(body, head_meta, opts=opts)

    return model_seg, model_meta


if __name__ == '__main__':
    opts = parse_args()
    opts = pre_process(opts)

    for i, (step, lr) in enumerate(zip(copy.deepcopy(opts.step), copy.deepcopy(opts.lr))):
        # 多重实验
        opts.step = step
        opts.lr = lr
        opts.classes = tasks.get_per_task_classes(opts.dataset, opts.task, opts.step)    # 目前类数
        # opts.inital_nb_classes = tasks.get_per_task_classes(opts.dataset, opts.task, opts.step)[0]

        datasets = get_dataset(opts)

        model_seg, model_meta = get_model(opts, classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step))
        opts.logger.log('Number of Parameters: seg:%d  meta: %d' % (get_model_para_number(model_seg), get_model_para_number(model_meta)))
        if opts.parallel:
            model_seg = parallel_model(opts, model_seg)
            model_meta = parallel_model(opts, model_meta)
            # USE OMP_NUM_THREADS=12 python -m torch.distributed.run --nproc_per_node=4 main.py --parallel
        else:
            model_seg = model_seg.to(opts.gpu)
            model_meta = model_meta.to(opts.gpu)

        if opts.rank == 0:
            # 记录模型的维度,梯度,参数信息
            wandb.watch(model_seg, log="all")

        if opts.step == 0:
            val_score_seg = learn_base(opts, model_seg, datasets)
            opts.logger.info(f'val score seg: {val_score_seg}')
            if not opts.task == 'offline':
                val_loss_meta = meta_train(opts, model_meta, datasets)
                opts.logger.info(f'val score meta: {val_loss_meta}')
        else:
            model_seg_old, _ = get_model(
                opts, classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step - 1)
            )
            if opts.parallel:
                model_seg_old = parallel_model(opts, model_seg_old)
            else:
                model_seg_old = model_seg_old.to(opts.gpu)

            # repeat ishot exps for few-shot learning
            # all_ishot_val_scores = []
            # if opts.step == 0:
            #     opts.ishot = 1
            for exp in range(opts.ishot):
                opts.exp = exp
                opts.logger.info(f'now exp {exp + 1}')
                val_score_novel = learn_novel(opts, model_seg, model_seg_old, model_meta, datasets)
                opts.logger.info(f'val score novel: {val_score_novel}')

        # test_score = test_step(opts, model_seg, datasets)
        # all_ishot_val_scores.append(test_score)


            # df = pd.DataFrame(all_ishot_val_scores)
            # df = df.drop(['Class IoU', 'Class Acc'], axis=1)
            # df['Total samples'] = df['Total samples'].astype(float)
            # df = df.T
            # df['mean'] = df.mean(axis=1)
            # try:
            #     opts.logger.log(df)
            # except:
            #     print(df)



