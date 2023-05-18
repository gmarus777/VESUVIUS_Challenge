from typing import Tuple, List
import monai
from monai.inferers import sliding_window_inference
import pytorch_lightning as pl
import torch.nn as nn
import math
import time
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from torchmetrics import Dice, FBetaScore
from torchmetrics import MetricCollection
from tqdm.auto import tqdm
import segmentation_models_pytorch as smp
import torch.nn.functional as F
from lit_models.Loss_functions import ComboBCEDiceLoss, TverskyLoss

try:
    import wandb
except ModuleNotFoundError:
    pass

# ssl solution
import ssl

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")


ssl._create_default_https_context = ssl._create_unverified_context


# Image one has ratio 8
# Image two has ratio 7
# Image 3 has ratio 12


class Lit_Model(pl.LightningModule):
    def __init__(
            self,
            cfg,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.cfg = cfg

        if self.cfg.use_wandb:
            wandb.init()



        self.metrics = self._init_metrics()
        self.model = self._init_model()

        self.loss_function = self._init_loss



        #### LOSS Functions ###


        # Torch loss functions
        #self.weighted_bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2))

        # SMP loss functions
        #self.loss_dice = smp.losses.DiceLoss(mode='binary',
        #                                     log_loss=False,
         #                                    # smooth=0.1, )

        self.loss_tversky = smp.losses.TverskyLoss(mode='binary',
                                                   classes=None,
                                                   log_loss=False,
                                                   from_logits=True,
                                                   alpha=0.5,
                                                   beta=0.5,
                                                   gamma=2.0,
                                                   smooth=1e-05,
                                                   ignore_index=None,
                                                   eps=1e-05,
                                                   )

        self.loss_focal = smp.losses.FocalLoss(mode='binary',
                                               alpha=None,
                                               gamma=2.0,
                                               ignore_index=None,
                                               reduction='mean',
                                               normalized=False,
                                               reduced_threshold=None)

        self.loss_bce = smp.losses.SoftBCEWithLogitsLoss(pos_weight=torch.tensor(0.5))  # pos_weight=torch.tensor(1), smooth_factor=0.1


        # MONAI loss functions

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


        #self.loss_tversky_monai = monai.losses.TverskyLoss(include_background=True,
        #                                                   to_onehot_y=False,
         #                                                  sigmoid=True,
           #                                                softmax=False,
            #                                               other_act=None,
             #                                              alpha=0.5,
              #                                             beta=0.5,
               #                                            #reduction=LossReduction.MEAN,
                #                                           smooth_nr=1e-04,
                 #                                          smooth_dr=1e-04,
                  #                                         batch=True)


        #self.loss_tversky_custom = TverskyLoss( alpha=0.5, beta=0.5, eps=1e-7,).to(DEVICE)



    def _init_loss(self, y_pred, y_true):
        #return self.loss_bce(y_pred , y_true.float()) + 0.5*self.loss_monai_focal_dice(y_pred , y_true.float() )
        return self.loss_bce(y_pred , y_true.float())  +  0.5*self.loss_tversky(y_pred , y_true.float()) #+ 0.5*self.loss_focal(y_pred , y_true.float())
        #return self.loss_monai_focal_dice(y_pred , y_true)
        #return self.loss_bce(y_pred , y_true.float()) + self.loss_tversky_monai(y_pred , y_true.float())


    def _init_model(self):
        return self.cfg.model



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


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # get images and labels
        images, labels = batch
        labels = labels.long()
        #images = images.unsqueeze(1)

        # run images through the model
        outputs = self.model(images)

        # apply binary mask
        #outputs = outputs*binary_mask

        # apply loss functions
        loss = self.loss_function(outputs, labels)


        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.metrics["train_metrics"](outputs, labels)

        if self.cfg.use_wandb:
            wandb.log({"train/loss": loss})

        outputs = {"loss": loss}

        return loss

    def validation_step(self, batch, batch_idx):
        # get images and labels
        images, labels = batch
        labels = labels.long()
        #images = images.unsqueeze(1)

        # run images through the model
        outputs = self.model(images)

        # apply loss functions
        loss = self.loss_function(outputs, labels)

        # apply binary mask
        #outputs = outputs * binary_mask

        # Get predicitons with 0.5 TH to compute accuracy
        preds = torch.sigmoid(outputs.detach()).gt(.5).int()


        # Monitor BCE and Dice loss
        bce = self.loss_bce(outputs , labels.float())
        dice = self.loss_tversky(outputs, labels.float())


        # SMP METRICS
        smooth = 1e-5
        tp, fp, fn, tn = smp.metrics.get_stats(torch.sigmoid(outputs), labels.long(), mode='binary', threshold=self.cfg.THRESHOLD)
        tp, fp, fn, tn = tp.to(self.cfg.device), fp.to(self.cfg.device), fn.to(self.cfg.device), tn.to(self.cfg.device)
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")
        recall = smp.metrics.recall(tp+smooth, fp, fn, tn, reduction="micro")
        fbeta = smp.metrics.fbeta_score(tp+smooth, fp, fn, tn, beta=.5, reduction='micro', )
        precision = smp.metrics.precision(tp+smooth, fp, fn, tn, reduction="micro")

        accuracy_simple = (preds == labels).sum().float().div(labels.size(0) * labels.size(2) ** 2)

        # Alternative FBETas
        fbeta_score_1 = FBetaScore(task="binary", beta=.5, threshold=.1, ).to(self.cfg.device)
        fbeta_score_4 = FBetaScore(task="binary", beta=.5, threshold=.4, ).to(self.cfg.device)
        fbeta_score_6 = FBetaScore(task="binary", beta=.5, threshold=.6, ).to(self.cfg.device)
        fbeta_score_75 = FBetaScore(task="binary", beta=.5, threshold=.75, ).to(self.cfg.device)
        fbeta_score_83 = FBetaScore(task="binary", beta=.5, threshold=.83, ).to(self.cfg.device)
        fbeta_score_90 = FBetaScore(task="binary", beta=.5, threshold=.9, ).to(self.cfg.device)
        fbeta_score_95 = FBetaScore(task="binary", beta=.5, threshold=.95, ).to(self.cfg.device)
        fbeta_1 = fbeta_score_1(torch.sigmoid(outputs ), labels)
        fbeta_4 = fbeta_score_4(torch.sigmoid(outputs ), labels)
        fbeta_6 = fbeta_score_6(torch.sigmoid(outputs ), labels)
        fbeta_75 = fbeta_score_75(torch.sigmoid(outputs ), labels)
        fbeta_83 = fbeta_score_83(torch.sigmoid(outputs ), labels)
        fbeta_90 = fbeta_score_90(torch.sigmoid(outputs ), labels)
        fbeta_95 = fbeta_score_95(torch.sigmoid(outputs ), labels)

        self.log("val_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("accuracy", accuracy.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("recall", recall.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("precision", precision.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("FBETA", fbeta.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("BCE loss", bce, on_step=False, on_epoch=True, prog_bar=True)
        self.log("DICE loss", dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log("accuracy with 0.5", accuracy_simple, on_step=False, on_epoch=True, prog_bar=True)

        self.log("fbeta_1", fbeta_1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("fbeta_4", fbeta_4, on_step=False, on_epoch=True, prog_bar=True)
        self.log("fbeta_6", fbeta_6, on_step=False, on_epoch=True, prog_bar=True)
        self.log("fbeta_75", fbeta_75, on_step=False, on_epoch=True, prog_bar=True)
        self.log("fbeta_83", fbeta_83, on_step=False, on_epoch=True, prog_bar=True)
        self.log("fbeta_90", fbeta_90, on_step=False, on_epoch=True, prog_bar=True)
        self.log("fbeta_95", fbeta_95, on_step=False, on_epoch=True, prog_bar=True)

        self.metrics["val_metrics"](outputs, labels)

        if self.cfg.use_wandb:
            wandb.log({"val_loss": loss})
            wandb.log({"accuracy": accuracy.item()})
            wandb.log({"recall": recall.item()})
            wandb.log({"precision": precision.item()})
            wandb.log({"FBETA": fbeta.item()})
            wandb.log({"BCE": bce})
            wandb.log({"DICE": dice.item()})
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

    # TODO: not implemented properly
    def predict_step(self, batch, batch_idx):
        pass





    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.t_max,
                                                               eta_min=self.cfg.eta_min, verbose=True)
        return [optimizer], [scheduler]

















