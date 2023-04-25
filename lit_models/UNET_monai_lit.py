
from typing import Tuple, List


import monai
from monai.inferers import sliding_window_inference
import pytorch_lightning as pl

import torch

from torchmetrics import Dice, FBetaScore
from torchmetrics import MetricCollection
from tqdm.auto import tqdm
try:
    import wandb
except ModuleNotFoundError:
    pass


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

'''

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")
class UNET_lit(pl.LightningModule):
    def __init__(
        self,
        use_wandb = True,
        z_dim= 32,
        patch_size = (512,512),
        sw_batch_size=16 ,
        eta_min = 1e-6,
        t_max = 200,
        max_epochs = 700,
        weight_decay: float = 0.00005,
        learning_rate: float = 0.0003,
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
        self.loss = self._init_loss()



    def _init_model(self):
        return monai.networks.nets.FlexibleUNet(in_channels = self.z_dim,
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

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images = batch["volume_npy"].as_tensor().to(DEVICE)
        labels = batch["label_npy"].long().to(DEVICE)
        masks = batch["mask_npy"].to(DEVICE)
        outputs = self.model(images)

        loss = self.loss(outputs, labels, masks)

        self.log("train/loss", loss.as_tensor(), on_step=True,on_epoch=True, prog_bar=True)
        self.metrics["train_metrics"](outputs, labels)
        wandb.log({"train/loss": loss.as_tensor()})

        outputs = {"loss": loss}

        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["volume_npy"].as_tensor().to(DEVICE)
        labels = batch["label_npy"].long().to(DEVICE)
        masks = batch["mask_npy"].to(DEVICE)
        outputs = self.model(images)

        loss = self.loss(outputs, labels, masks)
        preds = torch.sigmoid(outputs.detach()).gt(.5).int()

        accuracy = (preds == labels).sum().float().div(labels.size(0) * labels.size(2) ** 2)
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
        #fbeta_score_vesuvio = self.fbeta_score_vesuvio(torch.sigmoid(outputs).to(dtype=torch.long, device=DEVICE),labels, 0.4 )


        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("fbeta_1", fbeta_1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("fbeta_4", fbeta_4, on_step=False, on_epoch=True, prog_bar=True)
        self.log("fbeta_6", fbeta_6, on_step=False, on_epoch=True, prog_bar=True)
        self.log("fbeta_75", fbeta_75, on_step=False, on_epoch=True, prog_bar=True)
        self.log("fbeta_83", fbeta_83, on_step=False, on_epoch=True, prog_bar=True)
        self.log("fbeta_90", fbeta_90, on_step=False, on_epoch=True, prog_bar=True)
        self.log("fbeta_95", fbeta_95, on_step=False, on_epoch=True, prog_bar=True)
        self.metrics["val_metrics"](outputs, labels)

        wandb.log({"val/loss": loss.as_tensor()})
        wandb.log({"accuracy": accuracy.as_tensor()})
        wandb.log({"fbeta_1": fbeta_1.as_tensor()})
        wandb.log({"fbeta_4": fbeta_4.as_tensor()})
        wandb.log({"fbeta_6": fbeta_6.as_tensor()})
        wandb.log({"fbeta_75": fbeta_75.as_tensor()})
        wandb.log({"fbeta_83": fbeta_83.as_tensor()})
        wandb.log({"fbeta_90": fbeta_90.as_tensor()})
        wandb.log({"fbeta_95": fbeta_95.as_tensor()})
        #wandb.log({"fbeta_vesuvio": fbeta_score_vesuvio.as_tensor()})


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
        scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.t_max,  eta_min=self.hparams.eta_min, )
        return [optimizer], [scheduler]

    #torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs,  eta_min=self.hparams.eta_min, )



    def fbeta_score_vesuvio(self, preds, targets, threshold, beta=0.5, smooth=1e-5):
        preds_t = torch.where(preds > threshold, 1.0, 0.0).float()
        y_true_count = targets.sum()

        ctp = preds_t[targets == 1].sum()
        cfp = preds_t[targets == 0].sum()
        beta_squared = beta * beta

        c_precision = ctp / (ctp + cfp + smooth)
        c_recall = ctp / (y_true_count + smooth)
        res = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall + smooth)

        return res