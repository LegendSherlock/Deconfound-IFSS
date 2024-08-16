from utils import *
from dataset.utils import FewShotDataset
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader


def meta_train(opts, model, datasets):
    # 在基类上元学习，最终得到一个训练过的PRM模型
    (dataset_train, dataset_val, _) = datasets

    FS_dataset_train = FewShotDataset(
        dataset=dataset_train,
        nKnovel=opts.label_novel_num,
        nKbase=opts.label_base_num,
        nShotNovel=opts.nshot,
        epoch_size=opts.train_episode,  # num of batches per epoch
        cls_info_path=opts.data_root + f'/cls_info_train.json'
    )

    train_loader = DataLoader(
        dataset=FS_dataset_train,
        sampler=(DistributedSampler(FS_dataset_train,
                                    num_replicas=opts.world_size,
                                    rank=opts.rank)
                 if opts.parallel else None),
        batch_size=1,
        shuffle=False if opts.parallel else True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    FS_dataset_val = FewShotDataset(
        dataset=dataset_val,
        nKnovel=opts.label_novel_num,
        nKbase=opts.label_base_num,
        nShotNovel=opts.nshot,
        epoch_size=opts.val_episode,  # num of batches per epoch
        cls_info_path=opts.data_root + f'/cls_info_val.json'
    )
    val_loader = DataLoader(
        dataset=FS_dataset_val,
        sampler=(DistributedSampler(FS_dataset_val,
                                    num_replicas=opts.world_size,
                                    rank=opts.rank)
                 if opts.parallel else None),
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    optimizer, scheduler = my_optim.get_finetune_optimizer(opts, model, train_loader, opts.meta_train_epoch,
                                                           freeze_backbone=True)

    # xxx Set up Trainer
    trainer = Meta_Trainer(
        model,
        opts=opts,
    )

    best_score = 1000.0
    last_epoch = 0

    proto_path = f"{opts.save_path}/step_{opts.step}_prototypes.pth"
    learnbase_ckpt = f"{opts.save_path}/step_{opts.step}_base.pth"
    opts.ckpt = f"{opts.save_path}/step_{opts.step}_meta_{opts.nshot}shot.pth"

    # load proto
    if os.path.isfile(proto_path):
        proto_ckpt = torch.load(proto_path, map_location="cpu")
        trainer.proto_dict = proto_ckpt['prototypes']

    # load model
    if os.path.isfile(learnbase_ckpt):
        # 加载backbone的模型权重\
        checkpoint = torch.load(learnbase_ckpt, map_location="cpu")
        if opts.parallel:
            model.module.body.load_state_dict(checkpoint["backbone_state"])
        else:
            model.body.load_state_dict(checkpoint["backbone_state"])
        opts.logger.log("[!] Model restored from %s" % learnbase_ckpt)
        del checkpoint
    if os.path.isfile(opts.ckpt):
        # 断点续练
        checkpoint = torch.load(opts.ckpt, map_location="cpu")
        if opts.parallel:
            model.module.head.load_state_dict(checkpoint["head_state"])
        else:
            model.head.load_state_dict(checkpoint["head_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        last_epoch = checkpoint["epoch"]
        best_score = checkpoint['best_score']
        opts.logger.log("[!] Model restored from %s" % opts.ckpt)

        del checkpoint

    opts.logger.log("Optimizer:\n%s" % optimizer)

    if last_epoch >= opts.meta_train_epoch:
        print(f'meta train skipped, now epoch is {last_epoch}')
        opts.logger.info("validate on val set...")
        model.eval()
        val_loss = trainer.validate(
            cur_epoch=last_epoch,
            val_loader=val_loader,
            logger=opts.logger
        )

        opts.logger.info("Done validation")
        opts.logger.info(
            f"End of Validation {last_epoch}/{opts.meta_train_epoch}, Validation Loss={val_loss}"
        )

        return val_loss
    # train/val here
    opts.logger.info('meta train start!')
    for cur_epoch in range(last_epoch + 1, opts.meta_train_epoch + 1):
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
            f"End of Epoch {cur_epoch}/{opts.meta_train_epoch}, Average Loss={epoch_loss}"
        )

        # =====  Validation  =====
        if (cur_epoch + 1) % opts.val_interval == 0:
            opts.logger.info("validate on val set...")
            model.eval()
            val_loss = trainer.validate(
                cur_epoch=cur_epoch,
                val_loader=val_loader,
                logger=opts.logger
            )

            opts.logger.info("Done validation")
            opts.logger.info(
                f"End of Validation {cur_epoch}/{opts.meta_train_epoch}, Validation Loss={val_loss}"
            )

            # =====  Save Best Model  =====
            score = val_loss
            if score < best_score:
                best_score = score

                # best model to build incremental steps
                state = {
                    "epoch": cur_epoch,
                    "head_state": model.head.state_dict() if not opts.parallel else model.module.head.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_score": best_score
                }
                torch.save(state, opts.ckpt)
                opts.logger.info("[!] Checkpoint saved.")
            else:
                if cur_epoch == opts.meta_train_epoch:
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
    opts.logger.info('meta train done!')
    return best_score


class Meta_Trainer:
    def __init__(self, model, opts):
        self.model = model
        self.step = opts.step
        self.classes = opts.classes
        self.proto_dict = dict([(k, []) for k in range(1, self.classes[0])])  # 背景不提取原型
        self.parallel = opts.parallel

    def train(self, cur_epoch, optim, train_loader, scheduler=None, print_int=10, logger=None):
        model = self.model

        epoch_loss = 0.0
        interval_loss = 0.0

        model.module.body.eval() if self.parallel else model.body.eval()
        model.module.head.eval() if self.parallel else model.head.train()

        if self.parallel:
            train_loader.sampler.set_epoch(cur_epoch)  # shuffle
        for cur_step, data in enumerate(tqdm(train_loader)):
            data_query, labels_query, labels_base, labels_novel = data
            data_query = data_query.squeeze(0).cuda()
            labels_query = labels_query.squeeze(0).cuda().to(torch.long)
            labels_base = [label_base_.numpy()[0] for label_base_ in labels_base]
            labels_novel = [label_novel_.numpy()[0] for label_novel_ in labels_novel]

            boost_prototypes, bad_prototypes, proto_novel = model(data_query, labels_query, labels_base, labels_novel,
                                                  self.proto_dict)
            criterion = nn.MSELoss(reduction='none')
            loss = criterion(boost_prototypes, proto_novel)

            loss = loss.mean()

            optim.zero_grad()
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

                interval_loss = 0.0

        epoch_loss = epoch_loss / len(train_loader)

        logger.info(f"Epoch {cur_epoch}, Class Loss={epoch_loss}")

        return epoch_loss

    def validate(self, cur_epoch, val_loader, logger=None):
        model = self.model
        model.eval()

        val_loss = 0.0

        with torch.no_grad():
            for cur_step, data in enumerate(tqdm(val_loader)):
                data_query, labels_query, labels_base, labels_novel = data
                data_query = data_query.squeeze(0).cuda()
                labels_query = labels_query.squeeze(0).cuda().to(torch.long)
                labels_base = [label_base_.numpy()[0] for label_base_ in labels_base]
                labels_novel = [label_novel_.numpy()[0] for label_novel_ in labels_novel]

                boost_prototypes, bad_prototypes, proto_novel = model(data_query, labels_query, labels_base,
                                                                      labels_novel,
                                                                      self.proto_dict)
                criterion = nn.MSELoss(reduction='none')
                loss = criterion(boost_prototypes, proto_novel)

                loss = loss.mean()
                val_loss += loss.item()

            val_loss = val_loss / len(val_loader)

        return val_loss
