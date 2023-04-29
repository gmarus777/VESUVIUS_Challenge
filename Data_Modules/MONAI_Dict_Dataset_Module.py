
from collections import defaultdict
from io import StringIO
from pathlib import Path
from typing import Tuple

import lovely_numpy as ln
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
from monai.data.utils import decollate_batch, no_collation, pad_list_data_collate


PATH = Path().resolve().parents[0]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

KAGGLE_DIR = PATH / "kaggle"

INPUT_DIR = KAGGLE_DIR / "input"

COMPETITION_DATA_DIR = INPUT_DIR / "vesuvius-challenge-ink-detection"

TRAIN_DATA_CSV_PATH = COMPETITION_DATA_DIR / "data_train_1.csv"
TEST_DATA_CSV_PATH = COMPETITION_DATA_DIR / "data_test_1.csv"


'''
  monai.transforms.RandGridDistortiond(
                    keys=("volume_npy", "mask_npy", "label_npy"),
                    num_cells=5,
                    prob=0.5,
                    distort_limit=(-0.3, 0.3),
                    mode=monai.utils.enums.GridSampleMode.BILINEAR,
                    padding_mode='zeros',
                    device='gpu',
                    allow_missing_keys=False
                ),

'''

# scroll_1 size = 8181, 6330
# scroll_2 size = 14830, 9506
# scroll_3 size = 7606, 5249


class MONAI_CSV_Scrolls_Dataset(pl.LightningDataModule):

    def __init__(self,
                 patch_size=512,
                 z_start=0,
                 z_dim=64,
                 shared_height=None,
                 downsampling=None,
                 train__fragment_id=[1,2],
                 val_fragment_id=[3],
                 stage='train',
                 batch_size=1,
                 num_samples=1,
                 num_workers=0,
                 on_gpu=False,
                 data_csv_path=None,

                 ):

        super().__init__()
        self.save_hyperparameters()

        self.df = pd.read_csv(data_csv_path)
        self.keys = ("volume_npy", "mask_npy", "label_npy")
        self.train_transform = self.train_transforms()
        self.val_transform = self.val_transforms()
        self.predict_transform = self.predict_transforms()

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_val_df = self.df[self.df.stage == "train"].reset_index(drop=True)

            train_df = train_val_df[
                train_val_df.fragment_id != self.hparams.val_fragment_id
                ].reset_index(drop=True)

            val_df = train_val_df[
                train_val_df.fragment_id == self.hparams.val_fragment_id
                ].reset_index(drop=True)

            self.train_dataset = self._dataset(train_df, self.train_transform)
            self.val_dataset = self._dataset(val_df, self.val_transform)

            print(f"# train: {len(self.train_dataset)}")
            print(f"# val: {len(self.val_dataset)}")

        if stage == "predict" or stage is None:
            predict_df = self.df[self.df.stage == "test"].reset_index(drop=True)
            self.predict_dataset = self._dataset(predict_df, self.predict_transform)

    def _dataset(self, df, transform):
        return CSVDataset(
            src=df,
            transform=transform,
        )

    def train_transforms_old(self):
        return monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(
                    keys="volume_npy",
                ),
                monai.transforms.LoadImaged(
                    keys=("mask_npy", "label_npy"),
                    ensure_channel_first=True,
                ),

            ]
        )

    def train_transforms(self):

        return monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(
                    keys="volume_npy",
                ),

                monai.transforms.LoadImaged(
                    keys=("mask_npy", "label_npy"),
                    ensure_channel_first=True,
                ),



                monai.transforms.RandWeightedCropd(
                    keys=("volume_npy", "mask_npy", "label_npy"),
                    spatial_size=self.hparams.patch_size,
                    num_samples=self.hparams.num_samples,
                    w_key="mask_npy",
                ),

                monai.transforms.RandAxisFlipd(
                    keys=self.keys,
                    prob=0.4,

                ),

                monai.transforms.RandFlipd(
                    keys=self.keys,
                    prob=0.5,
                    spatial_axis=0,
                ),
                monai.transforms.RandFlipd(
                    keys=self.keys,
                    prob=0.5,
                    spatial_axis=1,
                ),

                monai.transforms.RandZoomd(
                    keys=("volume_npy", "mask_npy", "label_npy"),
                    prob=.5,
                    min_zoom=0.8,
                    max_zoom=1.2,
                    # padding_mode=NumpyPadMode.EDGE,
                    # align_corners=None,
                    keep_size=True,

                ),

                monai.transforms.RandRotated(
                    keys=("volume_npy", "mask_npy", "label_npy"),
                    range_x=1.0,
                    prob=0.4,
                    keep_size=True,
                    padding_mode='zeros',

                ),

                monai.transforms.Rand2DElasticd(
                    keys=("volume_npy", "mask_npy", "label_npy"),
                    spacing =(20,20),
                    magnitude_range = (1,2),
                    spatial_size=None,
                    prob=.4,
                    rotate_range=(0,2),
                    shear_range=(2,2),
                    translate_range=None,
                    scale_range=(.2,.2),
                    padding_mode='zeros',

                ),


                monai.transforms.RandScaleIntensityd(
                    keys="volume_npy",
                    factors=.2,
                    prob=0.4,

                ),

                monai.transforms.RandAdjustContrastd(
                    keys="volume_npy",
                    prob=0.4,

                ),

                monai.transforms.RandGaussianNoised(
                    keys="volume_npy",
                    prob=0.5,
                    mean=0.0,
                    std=0.4,
                ),



                monai.transforms.RandCoarseDropoutd(
                    keys=("volume_npy", "mask_npy", "label_npy"),  #keys=("volume_npy", "mask_npy", "label_npy"), #keys="volume_npy",
                    holes= 1,
                    spatial_size = (8,8),
                    max_holes =3,
                    max_spatial_size=(64, 64),
                    fill_value=0.0,
                    prob=0.4,
                ),

                monai.transforms.RandCoarseDropoutd(
                    keys=("volume_npy", "mask_npy", "label_npy"),
                    # keys=("volume_npy", "mask_npy", "label_npy"), #keys="volume_npy",
                    holes=1,
                    spatial_size=(32, 32),
                    max_holes=2,
                    max_spatial_size=(96, 96),
                    fill_value=0.0,
                    prob=0.4,
                ),




                #monai.transforms.Resized(
                   # keys= self.keys,
                    #spatial_size =(512,512),
                    #size_mode='all'
                #),

                monai.transforms.NormalizeIntensityd(
                    keys="volume_npy",
                    nonzero=True,
                    channel_wise=True,
                ),

                monai.transforms.ToTensord(keys=self.keys),
            ]
        )

    def val_transforms(self):
        return monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(
                    keys="volume_npy",
                ),
                monai.transforms.LoadImaged(
                    keys=("mask_npy", "label_npy"),
                    ensure_channel_first=True,
                ),



                monai.transforms.RandWeightedCropd(
                    keys=("volume_npy", "mask_npy", "label_npy"),
                    spatial_size=self.hparams.patch_size,
                    num_samples=self.hparams.num_samples,
                    w_key="mask_npy",
                ),

                monai.transforms.NormalizeIntensityd(
                    keys="volume_npy",
                    nonzero=True,
                    channel_wise=True,
                ),

                monai.transforms.ToTensord(keys = self.keys),

            ]
        )

    def predict_transforms(self):
        return monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(
                    keys="volume_npy",
                ),
                monai.transforms.LoadImaged(
                    keys="mask_npy",
                    ensure_channel_first=True,
                ),

                monai.transforms.NormalizeIntensityd(
                    keys="volume_npy",
                    nonzero=True,
                    channel_wise=True,
                ),

            ]
        )

    def train_dataloader(self):
        return self._dataloader(self.train_dataset, train=True)

    def val_dataloader(self):
        return self._dataloader(self.val_dataset)

    def predict_dataloader(self):
        return self._dataloader(self.predict_dataset)

    def _dataloader(self, dataset, train=False):
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=train,
            num_workers=self.hparams.num_workers,
            collate_fn=pad_list_data_collate,
        )

