from utils import *
from dataset.utils import FewShotDataset
from torch.utils.data import DataLoader
from trainer.trainer import Trainer
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from torch.nn import functional as F


def learn_novel(opts, model_seg, model_seg_old, model_meta, datasets):
    # 按STEP在新类上训练和验证
    (dataset_train, dataset_val, dataset_test) = datasets

    train_loader = DataLoader(
        dataset=dataset_train,
        sampler=(DistributedSampler(dataset_train,
                                    num_replicas=opts.world_size,
                                    rank=opts.rank)
                 if opts.parallel else None),
        batch_size=4,
        shuffle=False if opts.parallel else True,
        num_workers=0,
        pin_memory=True,
        collate_fn=my_collate,
        drop_last=True
    )

    train_loader_not_shuffle = DataLoader(
        dataset=dataset_train,
        batch_size=1,
        shuffle=False,
        num_workers=0,
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


    optimizer, scheduler = my_optim.get_finetune_optimizer(opts, model_seg, train_loader, opts.learn_novel_epoch, freeze_backbone=False)

    # instance trainer (model must have already the previous step weights)
    trainer = Trainer(
        model_seg,
        model_seg_old,
        model_meta,
        opts=opts
    )

    best_score = 0.0
    last_epoch = 0

    base_proto_path = f"{opts.save_path}/step_0_prototypes.pth"
    learnbase_ckpt = f"{opts.save_path}/step_0_base.pth"
    metatrain_ckpt = f"{opts.save_path}/step_0_meta_{opts.nshot}shot.pth"
    if opts.step-1 != 0:
        step_ckpt = f"{opts.save_path}/step_{opts.step-1}_novel_{opts.nshot}shot.pth"
    else:
        step_ckpt = learnbase_ckpt
    opts.ckpt = f"{opts.save_path}/step_{opts.step}_novel_{opts.nshot}shot_{opts.exp}.pth"

    proto_path = f"{opts.save_path}/step_{opts.step}_prototypes.pth"
    matrix_path = f"{opts.save_path}/step_{opts.step}_matrix.pth"

    # load proto
    if os.path.isfile(base_proto_path):
        proto_ckpt = torch.load(base_proto_path, map_location="cpu")
        trainer.proto_dict = proto_ckpt['prototypes']

    # load model
    if os.path.isfile(step_ckpt):
        # 新模型读取旧模型权重
        step_checkpoint = torch.load(step_ckpt, map_location="cpu")

        if opts.parallel:
            model_seg.module.body.load_state_dict(step_checkpoint["backbone_state"])
            model_seg.module.head.load_state_dict(step_checkpoint["head_state"], strict=False)
            model_seg_old.module.body.load_state_dict(step_checkpoint["backbone_state"])
            model_seg_old.module.head.load_state_dict(step_checkpoint["head_state"])
            model_meta.module.body.load_state_dict(step_checkpoint["backbone_state"])
            if opts.init_balanced:
                # implement the balanced initialization (new cls has weight of background and bias = bias_bkg - log(N+1)
                model_seg.module.init_new_classifier()
        else:
            model_seg.body.load_state_dict(step_checkpoint["backbone_state"])
            model_seg.head.load_state_dict(step_checkpoint["head_state"], strict=False)
            model_seg_old.body.load_state_dict(step_checkpoint["backbone_state"])
            model_seg_old.head.load_state_dict(step_checkpoint["head_state"])
            model_meta.body.load_state_dict(step_checkpoint["backbone_state"])
            if opts.init_balanced:
                # implement the balanced initialization (new cls has weight of background and bias = bias_bkg - log(N+1)
                model_seg.init_new_classifier()

        # put the old model into distributed memory and freeze it
        for par in model_seg_old.parameters():
            par.requires_grad = False
        model_seg_old.eval()

        opts.logger.log(f"[!] Model_old loaded from {step_ckpt}")
        opts.logger.log(f"[!] Model_new initialized from {step_ckpt}")
        # clean memory
        del step_checkpoint


    if os.path.isfile(metatrain_ckpt):
        # 读取metatrain模型的head
        checkpoint = torch.load(metatrain_ckpt, map_location="cpu")
        if opts.parallel:
            model_meta.module.head.load_state_dict(checkpoint["head_state"])
        else:
            model_meta.head.load_state_dict(checkpoint["head_state"])
        opts.logger.log("[!] Model_meta restored from %s" % metatrain_ckpt)

        del checkpoint

    if os.path.isfile(opts.ckpt):
        # 断点续练
        checkpoint = torch.load(opts.ckpt, map_location="cpu")
        if opts.parallel:
            model_seg.module.body.load_state_dict(checkpoint["backbone_state"])
            model_seg.module.head.load_state_dict(checkpoint["head_state"])
        else:
            model_seg.body.load_state_dict(checkpoint["backbone_state"])
            model_seg.head.load_state_dict(checkpoint["head_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        last_epoch = checkpoint["epoch"]
        best_score = checkpoint['best_score']
        opts.logger.log("[!] Model_new restored from %s" % opts.ckpt)
        del checkpoint

    opts.logger.log("Optimizer:\n%s" % optimizer)

    if opts.rank == 0:
        sample_ids = np.random.choice(
            len(val_loader), size=1, replace=False
        )  # sample idxs for visualization
        opts.logger.log(f"The samples id are {sample_ids}")  # todo better show
    else:
        sample_ids = None
    label2color = utils.Label2Color(cmap=utils.color_map(opts.dataset))  # convert labels to images
    denorm = utils.Denormalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )  # de-normalization for original images

    val_metrics = StreamSegMetrics(opts.num_classes)

    if not os.path.isfile(f"{opts.save_path}/step_{opts.step}_prototypes_{opts.exp}.pth"):
        if opts.nshot != -1:
            trainer.before(
                train_loader=train_loader_not_shuffle,
                logger=opts.logger,
            )
        else:
            trainer.before_full(
                train_loader=train_loader_not_shuffle,
                logger=opts.logger,
            )
    else:
        proto_ckpt = torch.load(f"{opts.save_path}/step_{opts.step}_prototypes_{opts.exp}.pth", map_location="cpu")
        trainer.proto_dict = proto_ckpt['prototypes']
        opts.logger.info(f"load prototypes from {opts.save_path}/step_{opts.step}_prototypes.pth")
        del proto_ckpt


    if last_epoch >= opts.learn_novel_epoch:
        print(f'learn novel skipped, now epoch is {last_epoch}')
        opts.logger.info("validate on val set...")
        model_seg.eval()
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
            f"End of Validation {last_epoch}/{opts.learn_novel_epoch}, Validation Loss={val_loss},"
        )

        opts.logger.info(val_metrics.to_str(val_score))
        # save proto and prepare for CIM
        # if not os.path.isfile(proto_path) or not os.path.isfile(matrix_path):
        #     model_seg.eval()
        #     trainer.after(train_loader=train_loader, logger=opts.logger, proto_path=proto_path, matrix_path=matrix_path)
        return val_score

    # train/val here
    opts.logger.info('learn novel start!')
    for cur_epoch in range(last_epoch + 1, opts.learn_novel_epoch + 1):
        # =====  Train  =====
        model_seg.train()

        epoch_loss = trainer.train(
            cur_epoch=cur_epoch,
            optim=optimizer,
            train_loader=train_loader,
            scheduler=scheduler,
            logger=opts.logger,
        )

        opts.logger.info(
            f"End of Epoch {cur_epoch}/{opts.learn_novel_epoch}, Average Loss={epoch_loss[0] + epoch_loss[1]},"
            f" Class Loss={epoch_loss[0]}, Reg Loss={epoch_loss[1]}"
        )

        # =====  Validation  =====
        if (cur_epoch + 1) % opts.val_interval == 0:
            opts.logger.info("validate on val set...")
            model_seg.eval()
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
                f"End of Validation {cur_epoch}/{opts.learn_novel_epoch}, Validation Loss={val_loss}"
            )

            opts.logger.info(val_metrics.to_str(val_score))

            # =====  Save Best Model  =====
            score = val_score['Mean IoU']
            if score > best_score:
                best_score = score

                # best model to build incremental steps
                state = {
                    "epoch": cur_epoch,
                    "backbone_state": model_seg.body.state_dict() if not opts.parallel else model_seg.module.body.state_dict(),
                    "head_state": model_seg.head.state_dict() if not opts.parallel else model_seg.module.head.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_score": best_score
                }
                torch.save(state, opts.ckpt)
                opts.logger.info("[!] Checkpoint saved.")
            else:
                if cur_epoch == opts.learn_novel_epoch:
                    # load last model and save with last epoch
                    if os.path.isfile(opts.ckpt):
                        checkpoint = torch.load(opts.ckpt, map_location="cpu")
                        if opts.parallel:
                            model_seg.module.body.load_state_dict(checkpoint["backbone_state"])
                            model_seg.module.head.load_state_dict(checkpoint["head_state"])
                        else:
                            model_seg.body.load_state_dict(checkpoint["backbone_state"])
                            model_seg.head.load_state_dict(checkpoint["head_state"])
                        optimizer.load_state_dict(checkpoint["optimizer_state"])
                        scheduler.load_state_dict(checkpoint["scheduler_state"])
                        best_score = checkpoint['best_score']
                        opts.logger.log("[!] load best model from %s" % opts.ckpt)
                        del checkpoint

                    state = {
                        "epoch": cur_epoch,
                        "backbone_state": model_seg.body.state_dict() if not opts.parallel else model_seg.module.body.state_dict(),
                        "head_state": model_seg.head.state_dict() if not opts.parallel else model_seg.module.head.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_score": best_score
                    }
                    torch.save(state, opts.ckpt)
                    opts.logger.info("[!] Final checkpoint saved!!!")

    # trainer.after(train_loader=train_loader, logger=opts.logger, proto_path=proto_path, matrix_path=matrix_path)
    opts.logger.info(f'learn novel: {opts.step} done!')



    return val_score
