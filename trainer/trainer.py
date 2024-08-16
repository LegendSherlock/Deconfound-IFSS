import collections
import math
import statistics
from functools import reduce
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import distributed
from torch.nn import functional as F
import wandb
import numpy as np
import os

from utils.loss import (KnowledgeDistillationLoss,
                        UnbiasedCrossEntropy,
                        UnbiasedKnowledgeDistillationLoss)


class Trainer:
    def __init__(self, model, model_old, model_meta, opts):
        self.model_old = model_old
        self.model = model
        self.model_meta = model_meta
        self.step = opts.step
        self.classes = opts.classes
        self.use_amp = opts.amp
        self.parallel = opts.parallel
        self.rank = opts.rank
        self.opts = opts

        self.featurePool = list()
        self.featurePoolid = list()
        self.proto_dict = dict([(k, []) for k in range(1, opts.num_classes)])  # 背景不提取原型
        self.proto_path = f"{opts.save_path}/step_{opts.step}_prototypes_{opts.exp}.pth"
        self.last_matrix_path = f"{opts.save_path}/step_{opts.step-1}_matrix.pth"
        self.matrix_path = f"{opts.save_path}/step_{opts.step}_matrix.pth"


        if opts.dataset == "cityscapes_domain":
            self.old_classes = opts.num_classes
            self.nb_classes = opts.num_classes
            self.nb_current_classes = opts.num_classes
            self.nb_new_classes = opts.num_classes
        elif self.classes is not None:
            new_classes = self.classes[-1]
            tot_classes = reduce(lambda a, b: a + b, self.classes)
            self.old_classes = tot_classes - new_classes
            self.nb_classes = opts.num_classes
            self.nb_current_classes = tot_classes
            self.nb_new_classes = new_classes
        else:
            self.old_classes = 0
            self.nb_classes = None

        # Select the Loss Type
        reduction = 'none'

        if opts.unce and self.old_classes != 0:
            self.criterion = UnbiasedCrossEntropy(
                old_cl=self.old_classes, ignore_index=255, reduction=reduction
            )
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction=reduction)

        self.lkd = opts.loss_kd
        self.lkd_flag = self.lkd > 0. and model_old is not None
        if opts.unkd:
            self.lkd_loss = UnbiasedKnowledgeDistillationLoss(reduction="none", alpha=1.)
        else:
            self.lkd_loss = KnowledgeDistillationLoss(alpha=1.)

        self.dataset = opts.dataset

    def after(self, train_loader, logger, proto_path=None, matrix_path=None):
        # CIM中的𝑃(𝑃_𝑡=𝑝|𝐶_𝑡=𝑐 )
        # c和p的一个权重，旧类c的数据和旧类p原型的softmax(-平均距离)，
        # 权重矩阵维度为|C_{0:t}|*|C_{0:t}|

        self.model.eval()
        logger.info('preparing for CIM')
        # load weight_matrix
        if os.path.isfile(self.last_matrix_path):
            CIM_ckpt = torch.load(self.last_matrix_path, map_location="cpu")
            weight_matrix = CIM_ckpt['matrix'].cuda()



        # 原型向量
        prototypes_classes_list = []
        prototypes_true_list = []
        for k, v in self.proto_dict.items():
            prototypes_true_list.append(v)
            prototypes_classes_list.append(k)

        prototypes_true = torch.cat(prototypes_true_list, dim=0)

        # 类别的平均向量
        proto_cls_list = []
        cls_list = []
        with torch.no_grad():
            for i, data in enumerate(tqdm(train_loader)):
                # if i == 10:
                #     break
                data_query, labels_query, classes_query = data
                data_query = data_query.cuda().to(torch.float32)
                labels_query = labels_query.cuda().to(torch.long)

                for cls in classes_query[0]:
                    x, _ = self.model.module.body(data_query) if self.parallel else self.model.body(
                        data_query)
                    cls_mask = labels_query.clone()
                    cls_mask[cls_mask != cls] = 0
                    feature_size = x.shape[2:]
                    cls_mask = cls_mask.unsqueeze(1).to(torch.float32)
                    cls_mask = F.interpolate(cls_mask, feature_size, mode='bilinear', align_corners=True)
                    feature1 = x * cls_mask
                    b, c, h, w = feature1.size()
                    feature1 = feature1.view(b, c, h * w)  # b * c * n
                    proto_cls = torch.mean(feature1, dim=2)
                    proto_cls_list.append(proto_cls)
                    cls_list.append(cls)

        img_feat = torch.cat(proto_cls_list, dim=0).cpu()

        prototypes_cls_list = []
        for old_class in prototypes_classes_list:
            indices = [ind for ind, ele in enumerate(cls_list) if ele == old_class]
            cls_feat = torch.mean(img_feat[indices], dim=0)
            prototypes_cls_list.append(cls_feat)

        prototypes_cls = torch.cat(prototypes_true_list, dim=0)

        # compute score = softmax(-distance) from prototypes_true and prototypes_cls
        weight_matrix_list = []
        for p in prototypes_true:
            p = p.unsqueeze(0)
            dist = (prototypes_cls - p).pow(2).sum(1).sqrt()
            weight_p = F.softmax(-dist)
            weight_matrix_list.append(weight_p.unsqueeze(0))

        weight_matrix = torch.cat(weight_matrix_list, dim=0)  # 这种计算方式15*15，每一行为一个原型和15个旧类的权重

        state = {
            "matrix": weight_matrix,
            "prototypes_cls": prototypes_cls  # 为了便于后续更新matrix我们需要保留计算出的每个类的原型
        }
        torch.save(state, matrix_path)
        logger.info('weight matrix saved!')


    def before(self, train_loader, logger):
        # 保存各新类boost后的原型
        self.model_meta.eval()
        with torch.no_grad():
            data_query, labels_query = [], []
            for data in train_loader:
                _, data_query_, labels_query_, classes_query_ = data
                b, c, h, w = data_query_.size()
                data_query.append(data_query_.cuda())
                labels_query.append(labels_query_.cuda())
            data_query = torch.cat(data_query, dim=0).reshape(len(self.opts.labels), self.opts.nshot, c, h, w)
            labels_query = torch.cat(labels_query, dim=0).reshape(len(self.opts.labels), self.opts.nshot, h, w)

            # get boost prototypes
            boost_prototypes, bad_prototypes, _ = self.model_meta(data_query, labels_query, self.opts.labels_old,
                                                                  self.opts.labels,
                                                                  proto_dict=self.proto_dict, meta_test=True)

            # save boost prototypes
            for cls in range(len(boost_prototypes)):
                cls_proto = boost_prototypes[cls].unsqueeze(0)
                self.proto_dict[self.old_classes + cls] = cls_proto.cpu().detach()

                state = {
                    "prototypes": self.proto_dict
                }
                torch.save(state, self.proto_path)
            logger.info(f'step {self.step} prototypes saved!')

    def before_full(self, train_loader, logger):
        self.model.eval()
        logger.info('saving prototypes')
        with torch.no_grad():
            # 计算每个类的原型向量
            for _, data in enumerate(tqdm(train_loader)):
                img_path, images, masks, classes = data
                images = images.cuda().to(torch.float32)
                masks = masks.cuda().to(torch.long)

                embs, _ = self.model.body(images) if not self.parallel else self.model.module.body(images)

                # to cpu
                embs = embs.cpu()
                masks = masks.cpu()

                for cls in classes:
                    self.featurePoolid.append(cls[0])
                    class_mask = masks.clone()
                    class_mask[class_mask != cls] = 0
                    class_mask[class_mask == cls] = 1

                    # get proto
                    feature_size = embs.shape[2:]
                    class_mask = class_mask.unsqueeze(0).to(torch.float32)
                    class_mask = F.interpolate(class_mask, feature_size, mode='bilinear', align_corners=True)
                    feature1 = embs * class_mask

                    b, c, h, w = feature1.size()
                    feature1 = feature1.view(b, c, h * w)  # b * c * n
                    proto_img = torch.mean(feature1, dim=2)

                    self.featurePool.append(proto_img)

        # 存储到字典中
        cls_dict = dict([(self.old_classes + k, []) for k in range(self.classes[1])])
        for img in range(len(self.featurePoolid)):
            cls = self.featurePoolid[img]
            proto = self.featurePool[img]

            cls_dict[cls].append(proto)

        # 取平均并保存
        for cls in range(self.classes[1]):
            cls_proto = torch.stack(cls_dict[self.old_classes + cls], dim=0).mean(dim=0)
            self.proto_dict[self.old_classes + cls] = cls_proto

        # save boost prototypes
        state = {
            "prototypes": self.proto_dict
        }
        torch.save(state, self.proto_path)
        logger.info(f'step {self.step} prototypes saved!')

    def train(self, cur_epoch, optim, train_loader, scheduler=None, print_int=10, logger=None):
        """Train and return epoch loss"""
        logger.info("Epoch %d, lr = %f" % (cur_epoch, optim.param_groups[0]['lr']))

        criterion = self.criterion

        epoch_loss = 0.0
        reg_loss = 0.0
        interval_loss = 0.0
        if self.parallel:
            train_loader.sampler.set_epoch(cur_epoch)  # shuffle

        self.model.train()
        for cur_step, data in enumerate(tqdm(train_loader)):
            img_path, data_query, labels_query, classes_query = data
            data_query = data_query.cuda().to(torch.float32)
            labels_query = labels_query.cuda().to(torch.long)

            with torch.no_grad():
                # 不能输入未来类的原型
                classes_query_old = []
                for i in classes_query:
                    for cls in i:
                        if cls >= self.old_classes:
                            # 例如15-5，step1，oldcls=16，cls>=16的消除
                            i.remove(cls)

                    classes_query_old.append(i)

                # load weight_matrix
                if os.path.isfile(self.last_matrix_path):
                    CIM_ckpt = torch.load(self.last_matrix_path, map_location="cpu")
                    weight_matrix = CIM_ckpt['matrix'].cuda()

                outputs_old = self.model_old(
                    data_query, labels_query, classes_query_old, prototypes=self.proto_dict, mode='CIM', weight_matrix=weight_matrix
                )


            optim.zero_grad()
            if self.use_amp:
                scaler = torch.cuda.amp.GradScaler()
                with torch.cuda.amp.autocast():
                    outputs = self.model(data_query, labels_query, classes_query, prototypes=self.proto_dict, mode='exist')
                    loss_tot, loss, lkd = self.compute_losses(outputs, labels_query, criterion, outputs_old=outputs_old)

                scaler.scale(loss_tot).backward()
                scaler.step(optim)
                scaler.update()
            else:
                outputs = self.model(data_query, labels_query, classes_query, prototypes=self.proto_dict, mode='exist')
                loss_tot, loss, lkd = self.compute_losses(outputs, labels_query, criterion, outputs_old=outputs_old)
                loss_tot.backward()
                optim.step()

            if scheduler is not None:
                scheduler.step()

            epoch_loss += loss.item()
            interval_loss += loss.item() + lkd.item()

            if (cur_step + 1) % print_int == 0:
                interval_loss = interval_loss / print_int
                logger.info(
                    f"{print_int} interval loss: Epoch {cur_epoch}, Batch {cur_step + 1}/{len(train_loader)},"
                    f" Loss={interval_loss}"
                )
                logger.info(
                    f"batch Loss made of: CE {loss}, LKD {lkd}"
                )
                if self.rank == 0:
                    # wandb record
                    wandb.log({
                        "Train Loss": interval_loss})

                interval_loss = 0.0

        # collect statistics from multiple processes
        epoch_loss = torch.tensor(epoch_loss).cuda()
        reg_loss = torch.tensor(reg_loss).cuda()

        epoch_loss = epoch_loss / len(train_loader)
        reg_loss = reg_loss / len(train_loader)

        logger.info(f"Epoch {cur_epoch}, Class Loss={epoch_loss}, Reg Loss={reg_loss}")

        return (epoch_loss, reg_loss)

    def compute_losses(self, outputs, labels, criterion, outputs_old=None):
        lkd = torch.tensor(0.)

        # xxx BCE / Cross Entropy Loss
        loss = criterion(outputs, labels)  # B x H x W
        loss = loss.mean()  # scalar

        if self.lkd_flag:
            lkd = self.lkd * self.lkd_loss(
                outputs, outputs_old
            )

            lkd = torch.mean(lkd)

        loss_tot = loss + lkd
        return loss_tot, loss, lkd

    def validate(self, loader, metrics, ret_samples_ids=None,
                 logger=None, label2color=None,
                 denorm=None):
        """Do validation and return specified samples"""
        metrics.reset()
        model = self.model
        criterion = self.criterion
        model.eval()

        tot_loss = 0.0

        with torch.no_grad():
            for i, data in enumerate(tqdm(loader)):
                img_path, data_query, labels_query, classes_query = data
                data_query = data_query.cuda().to(torch.float32)
                labels_query = labels_query.cuda().to(torch.long)

                # 不能输入未来类的原型
                classes_query_old = []
                for i in classes_query:
                    for cls in i:
                        if cls >= self.old_classes:
                            # 例如15-5，step1，oldcls=16，cls>=16的消除
                            i.remove(cls)

                    classes_query_old.append(i)

                # load weight_matrix
                if os.path.isfile(self.last_matrix_path):
                    CIM_ckpt = torch.load(self.last_matrix_path, map_location="cpu")
                    weight_matrix = CIM_ckpt['matrix'].cuda()

                outputs_old = self.model_old(
                    data_query, labels_query, classes_query_old, prototypes=self.proto_dict, mode='CIM',
                    weight_matrix=weight_matrix
                )

                outputs = self.model(data_query, labels_query, classes_query, prototypes=self.proto_dict, mode='exist')

                loss_tot, loss, lkd = self.compute_losses(outputs, labels_query, criterion, outputs_old=outputs_old)

                if self.rank == 0:
                    wandb.log({"Val Loss": loss_tot, "batch": i})

                tot_loss += loss_tot.item()

                _, prediction = outputs.max(dim=1)

                data_query = data_query.cpu().numpy()
                labels_query = labels_query.cpu().numpy()
                prediction = prediction.cpu().numpy()
                metrics.update(labels_query, prediction)

                if ret_samples_ids is not None and i in ret_samples_ids:  # get samples
                    if self.rank == 0:
                        wandb.log(
                            {'images': wandb.Image((denorm(data_query[0]).transpose(1, 2, 0) * 255).astype(np.uint8)),
                             'masks': {
                                 'true': wandb.Image(label2color(labels_query[0]).astype(np.uint8)),
                                 'pred': wandb.Image(label2color(prediction[0]).astype(np.uint8)),
                             }}
                        )

            score = metrics.get_results()

            tot_loss = torch.tensor(tot_loss).cuda()
            tot_loss = tot_loss / len(loader)

        return tot_loss, score


