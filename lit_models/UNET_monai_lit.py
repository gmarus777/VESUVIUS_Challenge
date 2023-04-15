from collections import defaultdict
from io import StringIO
from pathlib import Path
from typing import Tuple, List


import monai
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL.Image as Image
import pytorch_lightning as pl
import seaborn as sns
import torch
from monai.data import CSVDataset
from monai.data import DataLoader
from monai.inferers import sliding_window_inference
from monai.visualize import matshow3d
from torchmetrics import Dice
from torchmetrics import MetricCollection
from tqdm.auto import tqdm
try:
    import wandb
except ModuleNotFoundError:
    pass

class UNET_lit(pl.LightningModule):
    def __init__(
        self,

        z_dim: int,
        weight_decay: float = 0.0005,
        learning_rate: float = 0.0004,
        gamma: float = 0.85,
        milestones: List[int] = [2, 4, 5, 6, 7, 9, 10, 12, 15, 17, 20, 25],
    ):
        super().__init__()

        self.save_hyperparameters()
        wandb.init()
        self.z_dim = z_dim
        self.metrics = self._init_metrics()
        self.lr = learning_rate
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.milestones = milestones

        self.model = self._init_model()
        self.loss = self._init_loss()



    def _init_model(self):
        return monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels= self.z_dim,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            dropout=.3,
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels, masks = batch
        labels = labels.long()
        outputs = self.model(images.squeeze(1))
        loss = self.loss(outputs, labels, masks)

        self.log("train/loss", loss, on_step=True,on_epoch=True, prog_bar=True)
        self.metrics["train_metrics"](outputs, labels)
        wandb.log({"train/loss": loss})

        outputs = {"loss": loss}

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels, masks = batch
        labels = labels.long()
        outputs = self.model(images.squeeze(1))
        loss = self.loss(outputs, labels, masks)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.metrics["val_metrics"](outputs, labels)
        wandb.log({"train/loss": loss})

        outputs = {"loss": loss}

        return loss


    def test_step(self, batch, batch_idx):
        images, labels, masks = batch
        outputs = self.model(images.squeeze(1))
        loss = self.loss(outputs, labels, masks)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.metrics["val_metrics"](outputs, labels)
        wandb.log({"train/loss": loss})

        outputs = {"loss": loss}

        return loss





    def _init_loss(self):

        loss = monai.losses.DiceLoss(sigmoid=True)
        return monai.losses.MaskedLoss(loss)

    def _init_metrics(self):
        metric_collection = MetricCollection(
            {
                "dice": Dice(),
            }
        )

        return torch.nn.ModuleDict(
            {
                "train_metrics": metric_collection.clone(prefix="train_"),
                "val_metrics": metric_collection.clone(prefix="val_"),
            }
        )


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma)
        return [optimizer], [scheduler]