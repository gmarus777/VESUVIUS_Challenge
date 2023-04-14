from typing import List

import torch
import torch.nn as nn
import pytorch_lightning as pl





class ResNet3d_lit(pl.LightningModule):
    def __init__(
        self,
        model,
        WandB = True,
        lr: float = 0.0003,
        weight_decay: float = 0.0005,
        milestones: List[int] = [2,4,5,6,7,9,10,12,15,17,20,25],
        gamma: float = 0.85,
    ):
        super().__init__()



        # TODO: implement saving parameters
        # self.save_hyperparameters()  # save parameters

        self.lr = lr
        self.learning_rate = lr
        self.weight_decay = weight_decay
        self.milestones = milestones
        self.gamma = gamma



        self.model = model

        self.loss_fn = nn.CrossEntropyLoss() # use num_classes =2
        #self.loss_fn = nn.BCELoss()     # use num_classes =1



    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch

        logits = self.model(images)

        loss = self.loss_fn(logits,labels.squeeze(1).long())
        self.log("train/loss", loss, prog_bar=True)


        outputs = {"loss": loss}

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch

        logits = self.model(images)

        loss = self.loss_fn(logits, labels)
        self.log("train/loss", loss, prog_bar=True)

        outputs = {"loss": loss}

    def test_step(self, batch, batch_idx):
        imgs, targets = batch

        # targets = targets.squeeze(1)
        preds = self.model.predict(imgs)
        test_cer = self.test_cer(preds, targets)
        self.log("test/cer", test_cer)
        return preds



    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma)
        return [optimizer], [scheduler]

    def configure_optimizers_new(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=0.001, total_steps=100)
        return [optimizer], [scheduler]