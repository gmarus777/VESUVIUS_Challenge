
from typing import Tuple, List
import monai
from monai.inferers import sliding_window_inference
import pytorch_lightning as pl
import torch.nn as nn
import math
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from warmup_scheduler import GradualWarmupScheduler
from torchmetrics import Dice, FBetaScore
from torchmetrics import MetricCollection
from tqdm.auto import tqdm
import segmentation_models_pytorch as smp

try:
    import wandb
except ModuleNotFoundError:
    pass

# ssl solution
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


'''
monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels= self.z_dim,
            out_channels=1,
            channels=( 64, 128, 256, 512, 1024),
            strides=(2, 2, 2, 2, ),
            num_res_units=4,
            dropout=0,
        )



monai.networks.nets.FlexibleUNet(in_channels = self.z_dim,
                              out_channels =1 ,
                              backbone = 'efficientnet-b0',
                              pretrained=True,
                              decoder_channels=(1024,512, 256, 128, 64, 32,),
                              spatial_dims=2,
                              norm=('batch', {'eps': 0.001, 'momentum': 0.1}),
                              act=('relu', {'inplace': True}),
                              dropout=0.0,
                              decoder_bias=True,
                              upsample='deconv',
                              interp_mode='nearest',
                              is_pad=False)
                              
smp.Unet(
            encoder_name='se_resnext50_32x4d',
            encoder_weights='imagenet',
            in_channels=self.z_dim,
            classes=1,
            activation=None,
        )                              
                              
                              
                              
'''

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")
class UNET_lit(pl.LightningModule):
    def __init__(
        self,
        use_wandb = True,
        z_dim= 32,
        patch_size = (512,512),
        sw_batch_size=16 ,
        eta_min = 1e-7,
        t_max = 50,
        max_epochs = 700,
        weight_decay: float = 0.0001,
        learning_rate: float = 0.0005,
        gamma: float = 0.85,
        milestones: List[int] = [  100, 150, 200, 250, 300, 350, 400, 450, 500],
    ):
        super().__init__()

        self.save_hyperparameters()

        if use_wandb:
            wandb.init()
        self.z_dim = z_dim
        self.metrics = self._init_metrics()
        self.lr = learning_rate
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.milestones = milestones

        self.model = self._init_model()
        self.loss_dice_masked = self._init_loss()
        #self.loss_dice =self._init_loss_DiceCE()
        #self.BCE_loss = torch.nn.BCEWithLogitsLoss()

        self.loss_dice = smp.losses.DiceLoss(mode='binary')
        self.loss_bce = smp.losses.SoftBCEWithLogitsLoss()
        self.loss_focal = smp.losses.FocalLoss(
                                mode = 'binary',
                                  alpha=.25,
                                  gamma=2.0,
                                  ignore_index=None,
                                  reduction='mean',
                                  normalized=False,
                                  reduced_threshold=None)

        self.loss = self.criterion



    def _init_model(self):
        return  monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels= self.z_dim,
            out_channels=1,
            channels=( 64, 128, 256, 512, 1024,2048),
            strides=(2, 2, 2, 2, 2),
            num_res_units=6,
            dropout=0,
        )

    def criterion(self, y_pred, y_true, mask):
        return  0.1* self.loss_bce(y_pred, y_true) + 0.5 * self.loss_dice_masked(y_pred, y_true, mask) + 10*self.loss_focal(y_pred, y_true)
        #return self.loss_bce(y_pred, y_true) +  self.loss_dice(y_pred, y_true,) +  self.loss_focal(y_pred, y_true)
        #return self.loss_bce(y_pred, y_true)
        #return self.loss_focal(y_pred, y_true)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images = batch["volume_npy"].as_tensor().to(DEVICE)
        labels = batch["label_npy"].long().to(DEVICE)
        masks = batch["mask_npy"].to(DEVICE)
        outputs = self.model(images)

        #loss = self.loss(outputs, labels, masks)
        loss = self.criterion(outputs, labels.float(), masks)


        self.log("train/loss", loss.as_tensor(), on_step=True,on_epoch=True, prog_bar=True)
        #self.log("loss Dice", loss_2.as_tensor(), on_step=False, on_epoch=True, prog_bar=True)
        self.metrics["train_metrics"](outputs, labels)
        wandb.log({"train/loss": loss.as_tensor()})
        #wandb.log({"loss BCE": loss_2.as_tensor()})

        outputs = {"loss": loss}

        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["volume_npy"].as_tensor().to(DEVICE)
        labels = batch["label_npy"].long().to(DEVICE)
        masks = batch["mask_npy"].to(DEVICE)
        outputs = self.model(images)

        #loss = self.loss(outputs, labels, masks)
        #loss_2 = self.loss_dice(outputs, labels, masks)

        loss = self.criterion(outputs, labels.float(), masks)
        preds = torch.sigmoid(outputs.detach()).gt(.5).int()

        tp, fp, fn, tn = smp.metrics.get_stats(outputs, labels.long(), mode='binary', threshold=0.5)
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
        recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
        fbeta = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=.5, reduction='micro-imagewise')
        precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro-imagewise")


        accuracy_simple = (preds == labels).sum().float().div(labels.size(0) * labels.size(2) ** 2)


        # FBETas
        fbeta_score_1 = FBetaScore(task="binary", beta=.5, threshold=.1, ).to(DEVICE)
        fbeta_score_4 = FBetaScore(task="binary", beta=.5, threshold=.4, ).to(DEVICE)
        fbeta_score_6 = FBetaScore(task="binary", beta=.5, threshold=.6, ).to(DEVICE)
        fbeta_score_75 = FBetaScore(task="binary", beta=.5, threshold=.75, ).to(DEVICE)
        fbeta_score_83 = FBetaScore(task="binary", beta=.5, threshold=.83, ).to(DEVICE)
        fbeta_score_90 = FBetaScore(task="binary", beta=.5, threshold=.9, ).to(DEVICE)
        fbeta_score_95 = FBetaScore(task="binary", beta=.5, threshold=.95, ).to(DEVICE)
        fbeta_1 = fbeta_score_1(torch.sigmoid(outputs), labels)
        fbeta_4 = fbeta_score_4(torch.sigmoid(outputs), labels)
        fbeta_6 = fbeta_score_6(torch.sigmoid(outputs), labels)
        fbeta_75 = fbeta_score_75(torch.sigmoid(outputs), labels)
        fbeta_83 = fbeta_score_83(torch.sigmoid(outputs), labels)
        fbeta_90 = fbeta_score_90(torch.sigmoid(outputs), labels)
        fbeta_95 = fbeta_score_95(torch.sigmoid(outputs), labels)


        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("accuracy", accuracy.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("recall", recall.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("precision", precision.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("FBETA", fbeta.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("accuracy_simple", accuracy_simple, on_step=False, on_epoch=True, prog_bar=True)






        self.log("fbeta_1", fbeta_1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("fbeta_4", fbeta_4, on_step=False, on_epoch=True, prog_bar=True)
        self.log("fbeta_6", fbeta_6, on_step=False, on_epoch=True, prog_bar=True)
        self.log("fbeta_75", fbeta_75, on_step=False, on_epoch=True, prog_bar=True)
        self.log("fbeta_83", fbeta_83, on_step=False, on_epoch=True, prog_bar=True)
        self.log("fbeta_90", fbeta_90, on_step=False, on_epoch=True, prog_bar=True)
        self.log("fbeta_95", fbeta_95, on_step=False, on_epoch=True, prog_bar=True)


        self.metrics["val_metrics"](outputs, labels)

        wandb.log({"val/loss": loss.as_tensor()})
        wandb.log({"accuracy": accuracy.item()})
        wandb.log({"recall": recall.item()})
        wandb.log({"precision": precision.item()})
        wandb.log({"FBETA": fbeta.item()})
        wandb.log({"accuracy_simple": accuracy_simple.as_tensor()})


        wandb.log({"fbeta_1": fbeta_1.as_tensor()})
        wandb.log({"fbeta_4": fbeta_4.as_tensor()})
        wandb.log({"fbeta_6": fbeta_6.as_tensor()})
        wandb.log({"fbeta_75": fbeta_75.as_tensor()})
        wandb.log({"fbeta_83": fbeta_83.as_tensor()})
        wandb.log({"fbeta_90": fbeta_90.as_tensor()})
        wandb.log({"fbeta_95": fbeta_95.as_tensor()})



        outputs = {"loss": loss}

        return loss


    def predict_step(self, batch, batch_idx):
        images = batch["volume_npy"].as_tensor()
        masks = batch["mask_npy"]
        h, w = images.shape[2], images.shape[3]
        h_mod = h % 512
        w_mod = w % 512
        h -= h_mod
        w -= w_mod
        outputs = sliding_window_inference(
            inputs=images,
            roi_size= (h,w),#self.hparams.patch_size,
            sw_batch_size=self.hparams.sw_batch_size,
            predictor=self,
            overlap=0.1,
            mode='gaussian',
        )
        return outputs.sigmoid().squeeze()



    def _init_loss(self):

        loss = monai.losses.DiceLoss(sigmoid=True)
        return monai.losses.MaskedLoss(loss)

    def _init_loss_DiceCE(self):
        loss =monai.losses.DiceCELoss(sigmoid=True)
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
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma)
        scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.t_max,  eta_min=self.hparams.eta_min,verbose =True )
        return [optimizer], [scheduler]

    def configure_optimizers_alternative(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = self.get_scheduler(optimizer)
        return [optimizer], [scheduler]

        #torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs,  eta_min=self.hparams.eta_min, )


    def get_scheduler(self, optimizer):
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.t_max,
                                                                      eta_min=self.hparams.eta_min, )
        scheduler = GradualWarmupSchedulerV2(
            optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)

        return scheduler



class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    """
    https://www.kaggle.com/code/underwearfitting/single-fold-training-of-resnet200d-lb0-965
    """
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(
            optimizer, multiplier, total_epoch, after_scheduler)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]


