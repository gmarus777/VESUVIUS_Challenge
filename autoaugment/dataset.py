import torch.utils.data

import torch.utils.data
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader, Dataset
import cv2
import torch
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform
import pytorch_lightning as pl
from tqdm.auto import tqdm

PATCH_SIZE = 224
Z_DIM = 16
STRIDE = PATCH_SIZE // 2
TRAIN_FRAGMENT_ID=[2,3]


PATH = Path().resolve().parents[0]
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

KAGGLE_DIR = PATH / "kaggle"

INPUT_DIR = KAGGLE_DIR / "input"

COMPETITION_DATA_DIR = INPUT_DIR / "vesuvius-challenge-ink-detection"
COMPETITION_DATA_DIR_str = "kaggle/input/vesuvius-challenge-ink-detection/"

TRAIN_DATA_CSV_PATH = COMPETITION_DATA_DIR / "data_train_1.csv"
TEST_DATA_CSV_PATH = COMPETITION_DATA_DIR / "data_test_1.csv"





class Vesuvius_Tile_Datset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.images, self.labels = self.get_train_dataset()
        self.transform = transform


    def __len__(self):
        # return len(self.df)
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            data = self.transform(image=image, mask=label)
            image = data['image']
            label = data['mask']

        return image, label




    # METHODS TO LOAD DATA
    def get_train_dataset(self):
        train_images = []
        train_masks = []

        for fragment_id in TRAIN_FRAGMENT_ID:
            image, mask = self.read_image_mask(fragment_id)

            x1_list = list(range(0, image.shape[1] - PATCH_SIZE + 1, STRIDE))
            y1_list = list(range(0, image.shape[0] - PATCH_SIZE + 1,STRIDE))

            for y1 in y1_list:
                for x1 in x1_list:
                    y2 = y1 + PATCH_SIZE
                    x2 = x1 + PATCH_SIZE
                    # xyxys.append((x1, y1, x2, y2))

                    train_images.append(image[y1:y2, x1:x2])
                    train_masks.append(mask[y1:y2, x1:x2, None])

        return train_images, train_masks

    def get_val_dataset(self):
        valid_images = []
        valid_masks = []
        valid_xyxys = []

        for fragment_id in self.cfg.val_fragment_id:
            image, mask = self.read_image_mask(fragment_id)

            x1_list = list(range(0, image.shape[1] - self.cfg.patch_size + 1, self.cfg.stride))
            y1_list = list(range(0, image.shape[0] - self.cfg.patch_size + 1, self.cfg.stride))

            for y1 in y1_list:
                for x1 in x1_list:
                    y2 = y1 + self.cfg.patch_size
                    x2 = x1 + self.cfg.patch_size
                    # xyxys.append((x1, y1, x2, y2))

                    valid_images.append(image[y1:y2, x1:x2])
                    valid_masks.append(mask[y1:y2, x1:x2, None])

                    valid_xyxys.append([x1, y1, x2, y2])

        return valid_images, valid_masks, valid_xyxys

    def get_test_dataset(self):
        for fragment_id in self.cfg.val_fragment_id:
            test_images = self.read_image_test(fragment_id)

    def read_image_test(self, fragment_id):
        images = []

        # idxs = range(65)
        mid = 65 // 2
        start = mid - Z_DIM // 2
        end = mid + Z_DIM // 2
        idxs = range(start, end)

        for i in tqdm(idxs):
            image = cv2.imread(COMPETITION_DATA_DIR_str + f"test/{fragment_id}/surface_volume/{i:02}.tif", 0)

            pad0 = (PATCH_SIZE - image.shape[0] % PATCH_SIZE)
            pad1 = (PATCH_SIZE - image.shape[1] % PATCH_SIZE)

            image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)

            images.append(image)
        images = np.stack(images, axis=2)

        return images

    def read_image_mask(self, fragment_id):

        images = []

        # idxs = range(65)
        mid = 65 // 2
        start = mid - Z_DIM // 2
        end = mid + Z_DIM // 2
        idxs = range(start, end)

        for i in tqdm(idxs):
            image = cv2.imread(COMPETITION_DATA_DIR_str + f"train/{fragment_id}/surface_volume/{i:02}.tif", 0)

            pad0 = (PATCH_SIZE - image.shape[0] % PATCH_SIZE)
            pad1 = (PATCH_SIZE - image.shape[1] % PATCH_SIZE)

            image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)

            images.append(image)

        images = np.stack(images, axis=2)

        mask = cv2.imread(COMPETITION_DATA_DIR_str + f"train/{fragment_id}/inklabels.png", 0)
        mask = np.pad(mask, [(0, pad0), (0, pad1)], constant_values=0)

        mask = mask.astype('float32')
        mask /= 255.0

        return images, mask






