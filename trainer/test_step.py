from utils import *
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils.loss import (KnowledgeDistillationLoss,
                        UnbiasedCrossEntropy,
                        UnbiasedKnowledgeDistillationLoss)
import torch.nn.functional as F
from tqdm import tqdm
from functools import reduce

def test_step(opts, model, datasets):
    # 测试模型在所有类上的性能
    opts.logger.info("*** Test the model on all seen classes...")
    (_, _, dataset_test) = datasets

    test_loader = DataLoader(
        dataset=dataset_test,
        sampler=(DistributedSampler(dataset_test,
                                    num_replicas=opts.world_size,
                                    rank=opts.rank)
                 if opts.parallel else None),
        batch_size=1,
        shuffle=False,
        num_workers=opts.num_workers,
        pin_memory=True,
        collate_fn=my_collate,
        drop_last=True
    )

    # instance trainer (model must have already the previous step weights)
    trainer = Test_Trainer(
        model,
        opts=opts,
    )
    if opts.step == 0:
        opts.ckpt = f"{opts.save_path}/step_{opts.step}_base.pth"
    else:
        opts.ckpt = f"{opts.save_path}/step_{opts.step}_novel_{opts.nshot}shot_{opts.exp}.pth"
    if os.path.isfile(opts.ckpt):
        # 断点续练
        checkpoint = torch.load(opts.ckpt, map_location="cpu")
        if opts.parallel:
            model.module.body.load_state_dict(checkpoint["backbone_state"])
            model.module.head.load_state_dict(checkpoint["head_state"])
        else:
            model.body.load_state_dict(checkpoint["backbone_state"])
            model.head.load_state_dict(checkpoint["head_state"])
        opts.logger.log("[!] Model restored from %s" % opts.ckpt)
        del checkpoint
    # if opts.step == 0:
    #     step_proto_path = f"{opts.save_path}/step_{opts.step}_prototypes.pth"
    # else:
    step_proto_path = f"{opts.save_path}/step_1_prototypes_{opts.exp}.pth"
    # load proto
    if os.path.isfile(step_proto_path):
        proto_ckpt = torch.load(step_proto_path, map_location="cpu")
        trainer.proto_dict = proto_ckpt['prototypes']
        opts.logger.info(f"load prototypes from {step_proto_path}")
        del proto_ckpt


    sample_ids = None
    label2color = utils.Label2Color(cmap=utils.color_map(opts.dataset))  # convert labels to images
    denorm = utils.Denormalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )  # de-normalization for original images

    val_metrics = StreamSegMetrics(opts.num_classes)

    test_score = trainer.test(
        loader=test_loader,
        metrics=val_metrics,
        ret_samples_ids=sample_ids,
        logger=opts.logger,
        label2color=label2color,
        denorm=denorm
    )
    opts.logger.info('test done!')
    return test_score


class Test_Trainer:
    def __init__(self, model, opts):
        self.model = model
        self.step = opts.step
        self.classes = opts.classes
        self.parallel = opts.parallel
        self.rank = opts.rank
        self.opts = opts

        self.proto_dict = dict([(k, []) for k in range(1, opts.num_classes)])  # 背景不提取原型
        self.proto_path = f"{opts.save_path}/step_{opts.step}_prototypes.pth"

    def test(self, loader, metrics, ret_samples_ids=None,
             logger=None, end_task=False, label2color=None,
             denorm=None):
        logger.info('testing')
        metrics.reset()
        model = self.model
        model.eval()

        count = 0
        with torch.no_grad():
            for i, data in enumerate(tqdm(loader)):
                image_path, data_query, labels_query, classes_query = data
                data_query = data_query.cuda().to(torch.float32)
                labels_query = labels_query.cuda().to(torch.long)
                # vis
                if len(classes_query[0]) == 1:
                    continue

                outputs = self.model(data_query, labels_query, classes_query, prototypes=self.proto_dict, mode='exist')
                _, prediction = outputs.max(dim=1)

                data_query = data_query.cpu().numpy()
                labels_query = labels_query.cpu().numpy()
                prediction = prediction.cpu().numpy()
                metrics.update(labels_query, prediction)

                # old_pred = prediction.copy()
                # old_pred[old_pred>=self.classes[0]] = 0
                # new_pred = prediction.copy()
                # new_pred[new_pred<self.classes[0]] = 0
                # new_pred[new_pred==255] = 0
                #
                # old_true = labels_query.copy()
                # old_true[old_true>=self.classes[0]] = 0
                # new_true = labels_query.copy()
                # new_true[new_true<self.classes[0]] = 0
                # new_true[new_true == 255] = 0
                # bg_true = labels_query.copy()
                # bg_true[labels_query==0] = 1
                # bg_true[labels_query!=0] = 0

                if self.rank == 0:
                    count += 1
                    logger.log(f"num: {count}, image_path: {image_path[0]}, cls: {classes_query[0]}")
                    wandb.log(
                        {'images': wandb.Image((denorm(data_query[0]).transpose(1, 2, 0) * 255).astype(np.uint8)),
                         'masks': {
                             'true': wandb.Image(label2color(labels_query[0]).astype(np.uint8)),
                             'pred': wandb.Image(label2color(prediction[0]).astype(np.uint8)),
                             # 'old_pred': wandb.Image(label2color(old_pred[0]).astype(np.uint8)),
                             # 'new_pred': wandb.Image(label2color(new_pred[0]).astype(np.uint8)),
                             # 'old_true': wandb.Image(label2color(old_true[0]).astype(np.uint8)),
                             # 'new_true': wandb.Image(label2color(new_true[0]).astype(np.uint8)),
                             # 'bg_true': wandb.Image(label2color(bg_true[0]).astype(np.uint8)),
                         }}
                        )

            score = metrics.get_results()
            return score
