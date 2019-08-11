import torch
import argparse

import numpy as np
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torchvision.models as models

from pytorch_lightning import Trainer
from test_tube import Experiment
from test_tube import HyperOptArgumentParser

from voc_loader import load_voc
from metrics import mean_average_precision
from utils import available_gpu
from utils import LightningConfig
from utils import AverageMeter


class Basic_Trainer(pl.LightningModule):
    def __init__(self, config: LightningConfig):
        # do some initialization
        super(Basic_Trainer, self).__init__()
        self.model = models.alexnet(pretrained=config.pretrained)
        # self.criterion = nn.BCEWithLogitsLoss().cuda()
        # dunno why, but this one is better
        self.criterion = nn.BCELoss().cuda()

        # Reconfigure last classifier for fine-tuning
        self.model.classifier[6] = nn.Linear(4096, 20)

        # xavier_uniform init decrease performance
        # nn.init.xavier_uniform_(self.model.classifier[6].weight)
        self.model.cuda()

    def forward(self, x):
        # forward propagation
        return self.model(x)

    def training_step(self, batch, batch_nb):
        # one training step for one batch, return type in dict
        x, y = batch
        x_val = x.cuda()
        y_val = y.cuda().float()

        output = self.forward(x_val)
        output = nn.Sigmoid()(output)

        return {"loss": self.criterion(output, y_val)}

    def validation_step(self, batch, batch_nb):
        # one validation step for one batch, return type in dict
        x, y = batch
        x_val = x.cuda()
        y_val = y.cuda().float()

        output = self.forward(x_val)
        output = nn.Sigmoid()(output)
        loss = self.criterion(output, y_val)

        return {"val_loss": loss.item(), "pred": output, "label": y_val}

    def validation_end(self, outputs):
        # integrate outputs for one validation epoch, return type in dict
        validation_loss = AverageMeter()
        for loss_dict in outputs:
            validation_loss.update(loss_dict["val_loss"], 1)

            batch_pred = loss_dict["pred"].detach().cpu().numpy()
            batch_label = loss_dict["label"].detach().cpu().numpy()

            if "pred_whole" in locals():
                pred_whole = np.concatenate((pred_whole, batch_pred))
                label_whole = np.concatenate((label_whole, batch_label))
            else:
                pred_whole = batch_pred
                label_whole = batch_label

        mAP = mean_average_precision(pred_whole, label_whole)
        return {"avg_val_loss": validation_loss.avg, "meanAP": mAP}

    def configure_optimizers(self):
        # return list of optimizers
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20)
        return [optimizer], [scheduler]

    @pl.data_loader
    def tng_dataloader(self):
        return load_voc("train")

    @pl.data_loader
    def val_dataloader(self):
        return load_voc("val")

    @pl.data_loader
    def test_dataloader(self):
        return load_voc("test")


def main(args):
    print(f"Running on GPU{available_gpu()}")
    model = Basic_Trainer(
        LightningConfig(
            imagepath="",
            train_labelpath="",
            valid_labelpath="",
            pretrained=args.pretrained,
        )
    )
    exp = Experiment(save_dir=f"runs/{args.desc}", name="Baseline")
    exp.argparse(args)

    trainer = Trainer(experiment=exp, max_nb_epochs=args.epochs)
    trainer.fit(model)
    exp.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("fai-ssl-challenge-parser")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr_decay", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--desc", type=str, default="no description specified")
    parser.add_argument("--pretrained", action="store_true")

    parser.set_defaults(pretrained=False)
    args = parser.parse_args()
    main(args)
