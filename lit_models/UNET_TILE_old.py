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
import torch.nn.functional as F

try:
    import wandb
except ModuleNotFoundError:
    pass

# ssl solution
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

THRESHOLD = .4

'''
GCP:
monai.networks.nets.FlexibleUNet(in_channels = self.z_dim,
                              out_channels =1 ,
                              backbone = 'efficientnet-b3',
                              pretrained=True,
                              decoder_channels=( 1024, 768, 512, 256, 128, 64, ),
                              spatial_dims=2,
                              norm=('batch', {'eps': 0.001, 'momentum': 0.1}),
                              #act=('relu', {'inplace': True}),
                              act = None,
                              dropout=0.0,
                              decoder_bias=False,
                              upsample='deconv',
                              interp_mode='nearest',
                              is_pad=False)



monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels=self.z_dim,
            out_channels=1,
            channels=( 32, 64, 128, 256,512,512, 1204 ),
            strides=(2, 2, 2, 2,2, 2),
            num_res_units=4,
            dropout=0,
            norm='batch',
            bias =False,

        )

 monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels= self.z_dim,
            out_channels=1,
            channels=(  32, 64, 128, 256, 512, ),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            dropout=0,
            norm = 'batch',
            bias =False,

        )




monai.networks.nets.FlexibleUNet(in_channels = self.z_dim,
                              out_channels =1 ,
                              backbone = 'efficientnet-b0',
                              pretrained=True,
                              decoder_channels=(512, 256, 128, 64, 32,),
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

### MONAI ###
        self.diceloss = monai.losses.DiceLoss(include_background=True,
                                              sigmoid=True,
                                              squared_pred=False,
                                              jaccard=False,
                                              batch=True
                                              )

        self.monai_tverskyLoss = monai.losses.TverskyLoss(include_background=True,
                                                          sigmoid=True,
                                                          softmax=False,
                                                          other_act=None,
                                                          alpha=0.5,
                                                          beta=0.5,
                                                          # reduction=LossReduction.MEAN,
                                                          smooth_nr=0,
                                                          smooth_dr=1e-06,
                                                          batch=True
                                                          )

        self.focalloss = monai.losses.FocalLoss(include_background=True,
                                                gamma=2.0,
                                                # weight=.25,
                                                # focal_weight=.25,
                                                )


 self.loss_focal = smp.losses.FocalLoss(
            mode='binary',
            # alpha=.1,
            gamma=2.0,
            ignore_index=None,

            normalized=False,
            reduced_threshold=None)

'''

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")


class UNET_TILE_lit(pl.LightningModule):
    def __init__(
            self,
            use_wandb=True,
            z_dim=32,
            patch_size=(512, 512),
            sw_batch_size=16,
            eta_min=1e-7,
            t_max=50,
            max_epochs=700,
            weight_decay: float = 0.0001,
            learning_rate: float = 0.0005,
            gamma: float = 0.85,
            milestones: List[int] = [100, 150, 200, 250, 300, 350, 400, 450, 500],
    ):
        super().__init__()

        self.save_hyperparameters()

        if use_wandb:
            wandb.init()

        self.use_wandb = use_wandb
        self.z_dim = z_dim
        self.metrics = self._init_metrics()
        self.lr = learning_rate
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.milestones = milestones

        self.model = self._init_model()

        self.loss_old = self.criterion  # MixedLoss(10.0, 2.0) #self.criterion

        #self.loss = self._init_new_loss()

        ## LOSSES######

        # MY LOSS FUNCITONS

        #self.mine_focal = FocalLoss(2)

        # Image one has ratio 8
        # Image two has ratio 7
        # Image 3 has ratio 12
        #self.weighted_bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2))

        ## SMP ##
        self.loss_dice = smp.losses.DiceLoss(mode='binary',
                                             log_loss=False,
                                             # smooth=0.1,

                                             )

        self.loss_tversky = smp.losses.TverskyLoss(mode='binary',
                                                   classes=None,
                                                   log_loss=False,
                                                   from_logits=True,
                                                   alpha=0.5,
                                                   beta=0.5,
                                                   gamma=2.0)

        self.loss_focal = smp.losses.FocalLoss(mode='binary',
                                               alpha=None,
                                               gamma=2.0,
                                               ignore_index=None,
                                               reduction='mean',
                                               normalized=False,
                                               reduced_threshold=None)

        self.loss_monai_focal_dice =monai.losses.DiceFocalLoss(include_background=True,
                                                               to_onehot_y=False,
                                                               sigmoid=True,
                                                               softmax=False,
                                                               other_act=None,
                                                               squared_pred=False,
                                                               jaccard=False,
                                                               reduction='mean',
                                                               smooth_nr=1e-05,
                                                               smooth_dr=1e-05,
                                                               batch=True,
                                                               gamma=2.0,
                                                               focal_weight=None,
                                                               lambda_dice=1.0,
                                                               lambda_focal=1.0
                                                               )


        self.loss_bce = smp.losses.SoftBCEWithLogitsLoss(pos_weight=torch.tensor(0.5)) #pos_weight=torch.tensor(1)





    def criterion(self, y_pred, y_true):
        # return  0.5*self.loss_bce(y_pred, y_true) +  self.loss_dice(y_pred, y_true) #+ 2*self.loss_focal(y_pred, y_true)
        # return self.loss_bce(y_pred, y_true) +  self.loss_dice(y_pred, y_true,) +  self.loss_focal(y_pred, y_true)
        # return self.loss_focal(y_pred*mask, y_true) + .8*self.loss_dice(y_pred*mask, y_true)
        # return self.loss_focal(y_pred * mask, y_true) + self.loss_tversky(y_pred * mask, y_true)
        # return self.monai_masked_tversky(y_pred, y_true, mask) +  self.masked_focal(y_pred, y_true, mask)

        # return 0.2*self.monai_masked_tversky(y_pred, y_true, mask) +  0.5*self.loss_bce(y_pred*mask, y_true.float())
        # return  self.monai_masked_tversky(y_pred, y_true, mask) +  self.mine_focal(y_pred*mask, y_true.float())

        return self.loss_bce(y_pred , y_true.float()) + 0.5*self.loss_monai_focal_dice(y_pred , y_true.float() )
        #return self.loss_bce(y_pred , y_true.float())  #+ 0.5*self.loss_tversky(y_pred , y_true.float())



    def _init_model(self):
        return monai.networks.nets.FlexibleUNet(in_channels=self.z_dim,
                                         out_channels=1,
                                         backbone='efficientnet-b3',
                                         pretrained=True,
                                         decoder_channels=(512, 256, 128, 64, 32),
                                         spatial_dims=2,
                                         norm=('batch', {'eps': 0.001, 'momentum': 0.1}),
                                         # act=('relu', {'inplace': True}),
                                         act=None,
                                         dropout=0.0,
                                         decoder_bias=False,
                                         upsample='deconv',
                                         interp_mode='nearest',
                                         is_pad=False)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        #images = batch["volume_npy"].as_tensor().to(DEVICE)
        #labels = batch["label_npy"].long().to(DEVICE)
        #masks = batch["mask_npy"].to(DEVICE)
        images, labels = batch
        labels = labels.long()
        outputs = self.model(images)

        # if not using masked multiple outputs by masks
        loss = self.loss_old(outputs, labels)
        # loss = self.loss(outputs, labels, masks)
        # loss = self.combined_loss(outputs, labels, masks)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        # self.log("loss Dice", loss_2.as_tensor(), on_step=False, on_epoch=True, prog_bar=True)

        self.metrics["train_metrics"](outputs, labels)

        if self.use_wandb:
            wandb.log({"train/loss": loss})
            # wandb.log({"loss BCE": loss_2.as_tensor()})

        outputs = {"loss": loss}

        return loss

    def validation_step(self, batch, batch_idx):
        #images = batch["volume_npy"].as_tensor().to(DEVICE)
        #labels = batch["label_npy"].long().to(DEVICE)
        #masks = batch["mask_npy"].to(DEVICE)
        images, labels = batch
        labels = labels.long()
        outputs = self.model(images)

        # loss = self.loss(outputs, labels, masks)
        # loss_2 = self.loss_dice(outputs, labels, masks)

        # loss = self.loss(outputs, labels.float(), masks)
        loss = self.loss_old(outputs, labels.long())
        # loss = self.combined_loss(outputs, labels, masks)

        preds = torch.sigmoid(outputs.detach()).gt(.5).int()

        bce = self.loss_bce(outputs , labels.float())
        dice = self.loss_dice(outputs, labels.float())

        # MY FBETA
        #y_preds = torch.sigmoid(outputs).to('cpu').numpy()
        #fbeta_sm, precision_sm, recall_sm = fbeta_numpy(y_preds, labels)


        # SMP METRICS
        smooth = 1e-5
        tp, fp, fn, tn = smp.metrics.get_stats(torch.sigmoid(outputs), labels.long(), mode='binary', threshold=THRESHOLD)
        tp, fp, fn, tn = tp.to(DEVICE), fp.to(DEVICE), fn.to(DEVICE), tn.to(DEVICE)
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")
        recall = smp.metrics.recall(tp+smooth, fp, fn, tn, reduction="micro")
        fbeta = smp.metrics.fbeta_score(tp+smooth, fp, fn, tn, beta=.5, reduction='micro', )
        precision = smp.metrics.precision(tp+smooth, fp, fn, tn, reduction="micro")

        accuracy_simple = (preds == labels).sum().float().div(labels.size(0) * labels.size(2) ** 2)

        # FBETas
        fbeta_score_1 = FBetaScore(task="binary", beta=.5, threshold=.1, ).to(DEVICE)
        fbeta_score_4 = FBetaScore(task="binary", beta=.5, threshold=.4, ).to(DEVICE)
        fbeta_score_6 = FBetaScore(task="binary", beta=.5, threshold=.6, ).to(DEVICE)
        fbeta_score_75 = FBetaScore(task="binary", beta=.5, threshold=.75, ).to(DEVICE)
        fbeta_score_83 = FBetaScore(task="binary", beta=.5, threshold=.83, ).to(DEVICE)
        fbeta_score_90 = FBetaScore(task="binary", beta=.5, threshold=.9, ).to(DEVICE)
        fbeta_score_95 = FBetaScore(task="binary", beta=.5, threshold=.95, ).to(DEVICE)
        fbeta_1 = fbeta_score_1(torch.sigmoid(outputs ), labels)
        fbeta_4 = fbeta_score_4(torch.sigmoid(outputs ), labels)
        fbeta_6 = fbeta_score_6(torch.sigmoid(outputs ), labels)
        fbeta_75 = fbeta_score_75(torch.sigmoid(outputs ), labels)
        fbeta_83 = fbeta_score_83(torch.sigmoid(outputs ), labels)
        fbeta_90 = fbeta_score_90(torch.sigmoid(outputs ), labels)
        fbeta_95 = fbeta_score_95(torch.sigmoid(outputs ), labels)

        self.log("val_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        #self.log("tverky", tversky.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("accuracy", accuracy.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("recall", recall.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("precision", precision.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("FBETA", fbeta.item(), on_step=False, on_epoch=True, prog_bar=True)
        #self.log("fbeta_sm", fbeta_sm.item(), on_step=False, on_epoch=True, prog_bar=True)
        #self.log(" precision_sm", precision_sm.item(), on_step=False, on_epoch=True, prog_bar=True)
        #self.log("recall_sm", recall_sm.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("BCE", bce, on_step=False, on_epoch=True, prog_bar=True)
        self.log("DICE", dice, on_step=False, on_epoch=True, prog_bar=True)
        #self.log("FOCAL", focal, on_step=False, on_epoch=True, prog_bar=True)
        self.log("accuracy_simple", accuracy_simple, on_step=False, on_epoch=True, prog_bar=True)

        self.log("fbeta_1", fbeta_1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("fbeta_4", fbeta_4, on_step=False, on_epoch=True, prog_bar=True)
        self.log("fbeta_6", fbeta_6, on_step=False, on_epoch=True, prog_bar=True)
        self.log("fbeta_75", fbeta_75, on_step=False, on_epoch=True, prog_bar=True)
        self.log("fbeta_83", fbeta_83, on_step=False, on_epoch=True, prog_bar=True)
        self.log("fbeta_90", fbeta_90, on_step=False, on_epoch=True, prog_bar=True)
        self.log("fbeta_95", fbeta_95, on_step=False, on_epoch=True, prog_bar=True)

        self.metrics["val_metrics"](outputs, labels)
        if self.use_wandb:
            wandb.log({"val_loss": loss})
            wandb.log({"accuracy": accuracy.item()})
            wandb.log({"recall": recall.item()})
            wandb.log({"precision": precision.item()})
            wandb.log({"FBETA": fbeta.item()})
            #wandb.log({"fbeta_sm": fbeta_sm.item()})
            #wandb.log({"precision_sm": precision_sm.item()})
            #wandb.log({"recall_sm": recall_sm.item()})
            wandb.log({"BCE": bce})
            wandb.log({"DICE": dice.item()})
            #wandb.log({"Focal": focal.item()})
            wandb.log({"accuracy_simple": accuracy_simple})

            wandb.log({"fbeta_1": fbeta_1})
            wandb.log({"fbeta_4": fbeta_4})
            wandb.log({"fbeta_6": fbeta_6})
            wandb.log({"fbeta_75": fbeta_75})
            wandb.log({"fbeta_83": fbeta_83})
            wandb.log({"fbeta_90": fbeta_90})
            wandb.log({"fbeta_95": fbeta_95})

        outputs = {"loss": loss}

        return loss

    def predict_step(self, batch, batch_idx):
        #images = batch["volume_npy"].as_tensor()
        #masks = batch["mask_npy"]
        images = batch
        h, w = images.shape[2], images.shape[3]
        h_mod = h % 512
        w_mod = w % 512
        h -= h_mod
        w -= w_mod
        outputs = sliding_window_inference(
            inputs=images,
            roi_size=(h, w),  # self.hparams.patch_size,
            sw_batch_size=self.hparams.sw_batch_size,
            predictor=self,
            overlap=0.1,
            mode='gaussian',
        )
        return outputs.sigmoid().squeeze()

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
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.t_max,
                                                               eta_min=self.hparams.eta_min, verbose=True)
        return [optimizer], [scheduler]

    def configure_optimizers_alternative(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = self.get_scheduler(optimizer)
        return [optimizer], [scheduler]

        # torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs,  eta_min=self.hparams.eta_min, )

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
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                    self.base_lrs]





def fbeta_numpy(targets, preds, beta=0.5, smooth=1e-5):
    """
    https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/397288
    """
    y_true_count = targets.sum()
    ctp = preds[targets==1].sum()
    cfp = preds[targets==0].sum()
    beta_squared = beta * beta

    c_precision = ctp / (ctp + cfp + smooth)
    c_recall = ctp / (y_true_count + smooth)
    dice = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall + smooth)

    return dice, c_precision, c_recall



class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
               ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.mean()


def dice_loss(input, target):
    input = torch.sigmoid(input)
    smooth = 1.0

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)

    def forward(self, input, target, mask):
        loss = self.alpha * self.focal(input * mask, target) - torch.log(dice_loss(input * mask, target))
        return loss.mean()


