from pathlib import Path


class WandbLogger(object):
    def __init__(self,
                 wandb=None,
                 tb_writer=None,
                 hyp=None,
                 config=None,
                 save_dir: Path = None,
                 resume: str = 'allow',
                 project: str = '',
                 name: str = '',
                 id=None) -> None:
        # is run wandb
        self.wandb = wandb
        self.wandb_run = None
        # is run Tensorboard
        self.tb_writer = tb_writer
        # wandb init params
        self.config = config
        self.resume = resume
        self.project = project
        self.name = name
        self.id = id
        # dir
        self.save_dir = save_dir
        # log
        self.tags = [
            # train loss
            'train/box_loss', 'train/obj_loss', 'train/cls_loss',
            # val loss
            'metrics/precision', 'metrics/recall',
            'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
            'val/box_loss', 'val/obj_loss', 'val/cls_loss',
            # params
            'x/lr0', 'x/lr1', 'x/lr2']

        self.init(hyp)

    def is_init_wandb_until(self) -> bool:
        is_run: bool = self.wandb and self.wandb.run is None
        return is_run

    def init(self, hyp) -> None:
        if self.is_init_wandb_until():
            self.config.hyp = hyp  # add hyperparameters
            self.wandb_run = self.wandb.init(
                config=self.config,
                resume=self.resume,
                project=self.project,
                name=self.name,
                id=self.id)
        else:
            print('Not defined wandb.')

    def update_train_log(self, log_vals, epoch) -> None:
        for tag, x in zip(self.tags, log_vals):
            if self.tb_writer:
                self.tb_writer.add_scalar(tag, x, epoch)  # tensorboard
            if self.wandb:
                self.wandb.log({tag: x})  # W&B

    def update_mosaic_log(self) -> None:
        if self.wandb:
            self.wandb.log({"Mosaics": [self.wandb.Image(
                str(x), caption=x.name) for x in self.save_dir.glob('train*.jpg')]})

    def update_val_log(self) -> None:
        if self.wandb and self.wandb.run:
            self.wandb.log({"Validation": [self.wandb.Image(
                str(f), caption=f.name) for f in sorted(self.save_dir.glob('test*.jpg'))]})
        else:
            print('Not defined wandb and wandb.run.')

    def update_result_log(self, final) -> None:
        if self.wandb:
            self.wandb.log({"Results": [self.wandb.Image(str(self.save_dir / x), caption=x)
                                        for x in ['precision_recall_curve.png']]})
            if self.config.log_artifacts:
                self.wandb.log_artifact(artifact_or_path=str(final),
                                        type='model',
                                        name=self.save_dir.stem)
        else:
            print('Not defined wandb.')

    def finish(self) -> None:
        if self.wandb and self.wandb.run:
            self.wandb.run.finish()
        else:
            print('Not defined wandb and wandb.run.')
