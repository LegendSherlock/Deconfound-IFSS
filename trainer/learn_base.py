from utils import *
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from functools import reduce
from tqdm import tqdm
from torch.nn import functional as F



def learn_base(opts, model, datasets):
    # 训练验证模型在基类上的性能，并在最后保存原型
    (dataset_train, dataset_val, _) = datasets

    train_loader = DataLoader(
        dataset=dataset_train,
        sampler=(DistributedSampler(dataset_train,
                                        num_replicas=opts.world_size,
                                        rank=opts.rank)
                     if opts.parallel else None),
        batch_size=4,  # 128
        shuffle=False if opts.parallel else True,
        num_workers=opts.num_workers,
        pin_memory=True,
        collate_fn=my_collate,
        drop_last=True
    )
    val_loader = DataLoader(
        dataset=dataset_val,
        sampler=(DistributedSampler(dataset_val,
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



    optimizer, scheduler = my_optim.get_finetune_optimizer(opts, model, train_loader, opts.learn_base_epoch, freeze_backbone=False)


    # instance trainer (model must have already the previous step weights)
    trainer = Base_Trainer(
        model,
        opts=opts,
    )

    best_score = 0.0
    last_epoch = 0

    proto_path = f"{opts.save_path}/step_{opts.step}_prototypes.pth"
    matrix_path = f"{opts.save_path}/step_{opts.step}_matrix.pth"
    opts.ckpt = f"{opts.save_path}/step_{opts.step}_base.pth"


    # load model
    if os.path.isfile(opts.ckpt):
        # 断点续练
        checkpoint = torch.load(opts.ckpt, map_location="cpu")
        if opts.parallel:
            model.module.body.load_state_dict(checkpoint["backbone_state"])
            model.module.head.load_state_dict(checkpoint["head_state"])
        else:
            model.body.load_state_dict(checkpoint["backbone_state"])
            model.head.load_state_dict(checkpoint["head_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        last_epoch = checkpoint["epoch"]
        best_score = checkpoint['best_score']
        opts.logger.log("[!] Model restored from %s" % opts.ckpt)

        del checkpoint


    opts.logger.log("Optimizer:\n%s" % optimizer)

    # tSNE = False
    # if tSNE:
    #
    #     proto_path = f"{opts.save_path}/step_{opts.step}_prototypes.pth"
    #     # load proto
    #     if os.path.isfile(proto_path):
    #         proto_ckpt = torch.load(proto_path, map_location="cpu")
    #         trainer.proto_dict = proto_ckpt['prototypes']
    #
    #     # tSNE出嵌入图
    #     import matplotlib.pyplot as plt
    #     # import matplotlib
    #     # matplotlib.use('TkAgg')
    #     from sklearn.manifold import TSNE
    #     from PIL import Image
    #
    #     proto_cls_list = []
    #     cls_list = []
    #     with torch.no_grad():
    #         for i, data in enumerate(tqdm(train_loader)):
    #             # if i == 10:
    #             #     break
    #             data_query, labels_query, classes_query = data
    #             data_query = data_query.cuda().to(torch.float32)
    #             labels_query = labels_query.cuda().to(torch.long)
    #
    #             x, attentions = model.module.body(data_query) if trainer.parallel else trainer.model.body(
    #                 data_query)
    #             for cls in classes_query[0]:
    #                 cls_mask = labels_query.clone()
    #                 cls_mask[cls_mask != cls] = 0
    #                 feature_size = x.shape[2:]
    #                 cls_mask = cls_mask.unsqueeze(1).to(torch.float32)
    #                 cls_mask = F.interpolate(cls_mask, feature_size, mode='bilinear', align_corners=True)
    #                 feature1 = x * cls_mask
    #                 b, c, h, w = feature1.size()
    #                 feature1 = feature1.view(b, c, h * w)  # b * c * n
    #                 proto_cls = torch.mean(feature1, dim=2)
    #                 proto_cls_list.append(proto_cls)
    #                 cls_list.append(cls)

        # img_feat = torch.cat(proto_cls_list, dim=0).cpu()
        #
        # prototypes_true_list = []
        # for v in trainer.proto_dict.values():
        #     prototypes_true_list.append(v)
        #
        # prototypes_true = torch.cat(prototypes_true_list, dim=0)
        # prototypes_classes = list(trainer.proto_dict.keys())
        #
        # tsne = TSNE(n_components=2, random_state=0, perplexity=3, learning_rate=100, init='pca')
        #
        # X_tsne = tsne.fit_transform(img_feat)
        # X_prototypes = tsne.fit_transform(prototypes_true)
        #
        # tx_, ty_ = X_tsne[:, 0], X_tsne[:, 1]
        # tx = (tx_ - np.min(tx_)) / (np.max(tx_) - np.min(tx_))
        # ty = (ty_ - np.min(ty_)) / (np.max(ty_) - np.min(ty_))
        #
        # proto_x, proto_y = X_prototypes[:, 0], X_prototypes[:, 1]
        # proto_x = (proto_x - np.min(tx_)) / (np.max(tx_) - np.min(tx_))
        # proto_y = (proto_y - np.min(tx_)) / (np.max(tx_) - np.min(tx_))
        #
        # plt.figure(figsize=(6, 5))
        # # plt.title('TSNE')
        # # 按类别绘图
        # for cls in prototypes_classes:
        #     if cls == 6:
        #         break
        #     indices = [ind for ind, ele in enumerate(cls_list) if ele == cls]
        #
        #     x_input, y_input = tx[indices], ty[indices]
        #     color = plt.cm.Set1(cls)  # 定义颜色的种类
        #     plt.scatter(x_input, y_input, color=color, s=12)
        #
        #     indices = [ind for ind, ele in enumerate(prototypes_classes) if ele == cls]
        #     x_input, y_input = proto_x[indices], proto_y[indices]
        #     plt.scatter(x_input, y_input, color=color, marker='^', s=16)
        #
        # plt.savefig('./wandb/tsne.png')
        # print('tsne finished')



    if opts.rank == 0:
        sample_ids = np.random.choice(
            len(val_loader), size=310, replace=False
        )  # sample idxs for visualization
        opts.logger.log(f"The samples id are {sample_ids}")  # todo better show
    else:
        sample_ids = None
    label2color = utils.Label2Color(cmap=utils.color_map(opts.dataset))  # convert labels to images
    denorm = utils.Denormalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )  # de-normalization for original images

    val_metrics = StreamSegMetrics(opts.num_classes)

    if last_epoch >= opts.learn_base_epoch:
        opts.logger.info(f'learn base skipped, now epoch is {last_epoch}')
        opts.logger.info("validate on val set...")
        model.eval()
        val_loss, val_score = trainer.validate(
            loader=val_loader,
            metrics=val_metrics,
            ret_samples_ids=sample_ids,
            logger=opts.logger,
            label2color=label2color,
            denorm=denorm
        )

        opts.logger.info("Done validation")
        opts.logger.info(
            f"End of Validation {last_epoch}/{opts.learn_base_epoch}, Validation Loss={val_loss}"
        )

        opts.logger.info(val_metrics.to_str(val_score))

        # save proto and prepare for CIM
        if not os.path.isfile(proto_path) or not os.path.isfile(matrix_path):
            model.eval()
            trainer.after(train_loader=train_loader, logger=opts.logger, proto_path=proto_path, matrix_path=matrix_path)

        return val_score

    # train/val here
    opts.logger.info('learn base start!')
    for cur_epoch in range(last_epoch+1, opts.learn_base_epoch+1):
        # =====  Train  =====
        model.train()

        epoch_loss = trainer.train(
            cur_epoch=cur_epoch,
            optim=optimizer,
            train_loader=train_loader,
            scheduler=scheduler,
            logger=opts.logger
        )

        opts.logger.info(
            f"End of Epoch {cur_epoch}/{opts.learn_base_epoch}, Average Loss={epoch_loss}"
        )

        # =====  Validation  =====
        if (cur_epoch + 1) % opts.val_interval == 0:
            opts.logger.info("validate on val set...")
            model.eval()
            val_loss, val_score = trainer.validate(
                loader=val_loader,
                metrics=val_metrics,
                ret_samples_ids=sample_ids,
                logger=opts.logger,
                label2color=label2color,
                denorm=denorm,
            )

            opts.logger.info("Done validation")
            opts.logger.info(
                f"End of Validation {cur_epoch}/{opts.learn_base_epoch}, Validation Loss={val_loss}"
            )

            opts.logger.info(val_metrics.to_str(val_score))

            # =====  Save Best Model  =====
            score = val_score['Mean IoU']
            if score > best_score:
                best_score = score

                # best model to build incremental steps
                state = {
                    "epoch": cur_epoch,
                    "backbone_state": model.body.state_dict() if not opts.parallel else model.module.body.state_dict(),
                    "head_state": model.head.state_dict() if not opts.parallel else model.module.head.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_score": best_score
                }
                torch.save(state, opts.ckpt)
                opts.logger.info("[!] Checkpoint saved.")
            else:
                if cur_epoch == opts.learn_base_epoch:
                    # load last model and save with last epoch
                    if os.path.isfile(opts.ckpt):
                        checkpoint = torch.load(opts.ckpt, map_location="cpu")
                        if opts.parallel:
                            model.module.body.load_state_dict(checkpoint["backbone_state"])
                            model.module.head.load_state_dict(checkpoint["head_state"])
                        else:
                            model.body.load_state_dict(checkpoint["backbone_state"])
                            model.head.load_state_dict(checkpoint["head_state"])
                        optimizer.load_state_dict(checkpoint["optimizer_state"])
                        scheduler.load_state_dict(checkpoint["scheduler_state"])
                        best_score = checkpoint['best_score']
                        opts.logger.log("[!] load best model from %s" % opts.ckpt)
                        del checkpoint

                    state = {
                        "epoch": cur_epoch,
                        "backbone_state": model.body.state_dict() if not opts.parallel else model.module.body.state_dict(),
                        "head_state": model.head.state_dict() if not opts.parallel else model.module.head.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_score": best_score
                    }
                    torch.save(state, opts.ckpt)
                    opts.logger.info("[!] Final checkpoint saved!!!")
    # save proto

    # temp
    # if os.path.isfile(proto_path):
    #     proto_ckpt = torch.load(proto_path, map_location="cpu")
    #     trainer.proto_dict = proto_ckpt['prototypes']

    trainer.after(train_loader=train_loader, logger=opts.logger, proto_path=proto_path, matrix_path=matrix_path)

    opts.logger.info('learn base done!')
    return best_score


class Base_Trainer:
    def __init__(self, model, opts):
        self.model = model
        self.step = opts.step
        self.classes = opts.classes
        self.use_amp = opts.amp
        self.parallel = opts.parallel
        self.rank = opts.rank
        self.ret_intermediate = False

        self.featurePool = list()
        self.featurePoolid = list()
        self.proto_dict = dict([(k, []) for k in range(1, self.classes[0])])  # 背景不提取原型


        # Select the Loss Type
        reduction = 'none'
        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction=reduction)


    def after(self, train_loader, logger, proto_path=None, matrix_path=None):
        self.model.eval()
        logger.info('saving prototypes')
        with torch.no_grad():
            # 计算每个类的原型向量
            for _, data in enumerate(tqdm(train_loader)):
                images, masks, classes = data
                images = images.cuda().to(torch.float32)
                masks = masks.cuda().to(torch.long)

                embs, _ = self.model.body(images) if not self.parallel else self.model.module.body(images)

                # to cpu
                embs = embs.cpu()
                masks = masks.cpu()

                for img in range(len(classes)):
                    for cls in classes[img]:
                        self.featurePoolid.append(cls)
                        class_mask = masks[img].clone()
                        class_mask[class_mask != cls] = 0
                        class_mask[class_mask == cls] = 1

                        # get proto
                        feature_size = embs[img].shape[1:]
                        class_mask = class_mask.unsqueeze(0).unsqueeze(1).to(torch.float32)
                        class_mask = F.interpolate(class_mask, feature_size, mode='bilinear', align_corners=True)
                        feature1 = embs[img].unsqueeze(0) * class_mask

                        b, c, h, w = feature1.size()
                        feature1 = feature1.view(b, c, h * w)  # b * c * n
                        proto_img = torch.mean(feature1, dim=2)

                        self.featurePool.append(proto_img)  # .clone().detach()

        # 存储到字典中
        cls_dict = dict([(k, []) for k in range(1, self.classes[0])])  # 背景不提取原型
        for img in range(len(self.featurePoolid)):
            cls = self.featurePoolid[img]
            proto = self.featurePool[img]

            cls_dict[cls].append(proto)

        # 取平均并保存
        for cls in range(1, self.classes[0]):
            cls_proto = torch.stack(cls_dict[cls], dim=0).mean(dim=0)
            self.proto_dict[cls] = cls_proto

        state = {
            "prototypes": self.proto_dict
        }
        torch.save(state, proto_path)
        logger.info('prototypes saved!')


        logger.info('preparing for CIM')
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
            "prototypes_old_cls": prototypes_cls  # 为了便于后续更新matrix我们需要保留计算出的每个类的原型
        }
        torch.save(state, matrix_path)
        logger.info('weight matrix saved!')

    def train(self, cur_epoch, optim, train_loader, scheduler=None, print_int=10, logger=None):
        """Train and return epoch loss"""
        logger.info("Epoch %d, lr = %f" % (cur_epoch, optim.param_groups[0]['lr']))

        model = self.model
        criterion = self.criterion

        epoch_loss = 0.0
        interval_loss = 0.0

        if self.parallel:
            train_loader.sampler.set_epoch(cur_epoch)  # shuffle

        model.train()
        for cur_step, data in enumerate(tqdm(train_loader)):
            img_path, images, labels, classes = data
            images = images.cuda().to(torch.float32)
            labels = labels.cuda().to(torch.long)

            optim.zero_grad()
            if self.use_amp:
                scaler = torch.cuda.amp.GradScaler()
                with torch.cuda.amp.autocast():
                    outputs = model(images, labels, classes, prototypes=self.proto_dict, mode='get')
                    loss = criterion(outputs, labels)
                    loss = loss.mean()

                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                outputs = model(images, labels, classes, prototypes=self.proto_dict, mode='get')
                loss = criterion(outputs, labels)
                loss = loss.mean()

                loss.backward()
                optim.step()

            if scheduler is not None:
                scheduler.step()

            epoch_loss += loss.item()
            interval_loss += loss.item()

            if (cur_step + 1) % print_int == 0:
                interval_loss = interval_loss / print_int
                logger.info(
                    f"{print_int} interval loss: Epoch {cur_epoch}, Batch {cur_step + 1}/{len(train_loader)},"
                    f" Loss={interval_loss}"
                )
                logger.info(
                    f"batch Loss made of: CE {loss}"
                )
                if self.rank == 0:
                    # wandb record
                    wandb.log({
                        "Train Loss": interval_loss})

                interval_loss = 0.0


        epoch_loss = epoch_loss / len(train_loader)

        logger.info(f"Epoch {cur_epoch}, Class Loss={epoch_loss},")

        return epoch_loss


    def validate(self, loader, metrics, ret_samples_ids=None,
                 logger=None, label2color=None,
                 denorm=None):
        """Do validation and return specified samples"""
        metrics.reset()
        model = self.model
        criterion = self.criterion
        model.eval()

        val_loss = 0.0
        count = 0
        with torch.no_grad():
            for i, data in enumerate(tqdm(loader)):
                image_path, images, labels, classes = data

                # vis
                if len(classes[0]) == 1:
                    continue

                images = images.cuda().to(torch.float32)
                labels = labels.cuda().to(torch.long)

                outputs = model(images, labels, classes, prototypes=self.proto_dict, mode='get')

                loss = criterion(outputs, labels)

                loss = loss.mean()

                if self.rank == 0:
                    wandb.log({"Val Loss": loss, "batch": i})


                val_loss += loss.item()

                _, prediction = outputs.max(dim=1)

                images = images.cpu().numpy()
                labels = labels.cpu().numpy()
                prediction = prediction.cpu().numpy()
                metrics.update(labels, prediction)

                old_true = labels.copy()
                old_true[old_true>=self.classes[0]] = 0
                new_true = labels.copy()
                new_true[new_true<self.classes[0]] = 0
                new_true[new_true == 255] = 0

                # if ret_samples_ids is not None and i in ret_samples_ids:  # get samples
                if self.rank == 0:
                    count += 1
                    # logger.log(f"num: {count}, image_path: {image_path[0]}, cls: {classes[0]}")
                    wandb.log({'images': wandb.Image((denorm(images[0]).transpose(1, 2, 0) * 255).astype(np.uint8)),
                               'masks': {
                                   'true': wandb.Image(label2color(labels[0]).astype(np.uint8)),
                                   'pred': wandb.Image(label2color(prediction[0]).astype(np.uint8)),
                                   'old_true': wandb.Image(label2color(old_true[0]).astype(np.uint8)),
                                   'new_true': wandb.Image(label2color(new_true[0]).astype(np.uint8)),
                               }}
                              )

            score = metrics.get_results()

            val_loss = val_loss / len(loader)

        return val_loss, score
