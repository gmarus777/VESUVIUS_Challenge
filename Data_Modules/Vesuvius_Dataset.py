from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader, Dataset
import cv2
import torch
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
from tqdm.auto import tqdm




#COMPETITION_DATA_DIR = INPUT_DIR / "vesuvius-challenge-ink-detection"
#COMPETITION_DATA_DIR_str = "kaggle/input/vesuvius-challenge-ink-detection/"



'''
FOR CFG Template look at CFG_TEMPLATE.py

'''

# TODO: FINISH TEST DATALOADER, USE INFERENCE NOTEBOOK procedures for submission and predictions

class Vesuvius_Tile_Datamodule(pl.LightningDataModule):
    def __init__(self,
                 cfg,
                 ):

        super().__init__()

        self.save_hyperparameters()
        self.cfg = cfg

        # Get transforms
        self.train_transform = self.get_transforms('train')
        self.val_transform = self.get_transforms('val')
        self.test_transform = self.get_transforms('test')

        if self.cfg.stage.lower() == 'train':
            self.train_data, self.train_labels, self.train_binary_masks = self.get_train_dataset()
            self.val_data, self.val_labels, self.val_pos, self.val_binary_masks = self.get_val_dataset()


            self.train_dataset = Vesuvius_Tile_Datset(images=self.train_data,
                                                      labels=self.train_labels,
                                                      transform=self.train_transform)

            self.val_dataset = Vesuvius_Tile_Datset(images=self.val_data,
                                                    labels=self.val_labels,


                                                    transform=self.val_transform)

        # TEST IS NOT FINISHED: DO IT
        elif self.cfg.stage.lower() == 'test':
            print('Not implemented, use INFERENCE NOTEBOOK procedures for submission and predictions')
            #self.test_data, self.test_pos, self.binary_mask = self.get_test_dataset()




    # Data processing is based on https://www.kaggle.com/code/tanakar/2-5d-segmentaion-baseline-training
    def get_train_dataset(self):
        train_images = []
        train_masks = []
        train_binary_masks = []

        for fragment_id in self.cfg.train_fragment_id:
            image, mask, binary_mask = self.read_image_mask('train', fragment_id)

            x1_list = list(range(0, image.shape[1] - self.cfg.patch_size + 1, self.cfg.stride))
            y1_list = list(range(0, image.shape[0] - self.cfg.patch_size + 1, self.cfg.stride))

            for y1 in y1_list:
                for x1 in x1_list:
                    y2 = y1 + self.cfg.patch_size
                    x2 = x1 + self.cfg.patch_size
                    # xyxys.append((x1, y1, x2, y2))

                    train_images.append(image[y1:y2, x1:x2])
                    train_masks.append(mask[y1:y2, x1:x2, None])
                    train_binary_masks.append(binary_mask[y1:y2, x1:x2, None])

        return train_images, train_masks, train_binary_masks

    def get_val_dataset(self):
        valid_images = []
        valid_masks = []
        valid_binary_masks = []
        valid_xyxys = []

        for fragment_id in self.cfg.val_fragment_id:
            image, mask, binary_mask = self.read_image_mask('train', fragment_id)

            x1_list = list(range(0, image.shape[1] - self.cfg.patch_size + 1, self.cfg.stride))
            y1_list = list(range(0, image.shape[0] - self.cfg.patch_size + 1, self.cfg.stride))

            for y1 in y1_list:
                for x1 in x1_list:
                    y2 = y1 + self.cfg.patch_size
                    x2 = x1 + self.cfg.patch_size
                    # xyxys.append((x1, y1, x2, y2))

                    valid_images.append(image[y1:y2, x1:x2])
                    valid_masks.append(mask[y1:y2, x1:x2, None])
                    valid_binary_masks.append(binary_mask[y1:y2, x1:x2, None])

                    valid_xyxys.append([x1, y1, x2, y2])

        return valid_images, valid_masks,  valid_xyxys, valid_binary_masks



    def get_test_dataset(self):
        test_images = []
        test_xyxys = []
        binary_masks = []

        for fragment_id in self.cfg.val_fragment_id:
            test_images, binary_mask = self.read_image_mask_test('test',fragment_id)
            binary_masks.append(binary_mask)

            x1_list = list(range(0, test_images.shape[1] - self.cfg.patch_size + 1, self.cfg.stride))
            y1_list = list(range(0, test_images.shape[0] - self.cfg.patch_size + 1, self.cfg.stride))



            for y1 in y1_list:
                for x1 in x1_list:
                    y2 = y1 + self.cfg.patch_size
                    x2 = x1 + self.cfg.patch_size

                    test_images.append(test_images[y1:y2, x1:x2])
                    test_xyxys.append((x1, y1, x2, y2))

        return test_images, test_xyxys, binary_masks









    def read_image_mask(self, stage, fragment_id):

        images = []

        # idxs = range(65)
        mid = 65 // 2
        start = mid - self.cfg.z_dim // 2
        end = mid + self.cfg.z_dim // 2
        idxs = range(start, end)

        for i in tqdm(idxs):
            image = cv2.imread(self.cfg.competition_data_dir + f"{stage}/{fragment_id}/surface_volume/{i:02}.tif", 0)

            pad0 = (self.cfg.patch_size - image.shape[0] % self.cfg.patch_size)
            pad1 = (self.cfg.patch_size - image.shape[1] % self.cfg.patch_size)

            image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)

            images.append(image)

        images = np.stack(images, axis=2)

        mask = cv2.imread(self.cfg.competition_data_dir + f"train/{fragment_id}/inklabels.png", 0)
        mask = np.pad(mask, [(0, pad0), (0, pad1)], constant_values=0)

        mask = mask.astype('float32')
        mask /= 255.0

        binary_mask = cv2.imread(self.cfg.competition_data_dir + f"train/{fragment_id}/mask.png", 0)
        binary_mask = (binary_mask / 255).astype(int)

        return images, mask, binary_mask






    def get_transforms(self, stage):
        if stage.lower() == 'train':
            transforms = A.Compose(self.cfg.train_transforms)
        elif stage.lower() == 'val':
            transforms = A.Compose(self.cfg.val_transforms)
        elif stage.lower() == 'test':
            transforms = A.Compose(self.cfg.test_transforms)
        return transforms

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        """
        construct a dataloader for training data
        data is shuffled !
        :param args:
        :param kwargs:
        :return:
        """
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.on_gpu,
            # multiprocessing_context="spawn"
            # multiprocessing_context="fork",
            # collate_fn=self.collate_function,
        )

    def val_dataloader(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.on_gpu,
            # multiprocessing_context="spawn"
            # multiprocessing_context="fork",
            # collate_fn=self.collate_function
        )


class Vesuvius_Tile_Datset(Dataset):
    def __init__(self, images, labels=None, binary_masks=None,  transform=None):
        self.images = images
        self.labels = labels
        self.bianary_masks =binary_masks
        self.transform = transform

    def __len__(self):
        # return len(self.df)
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        bianary_mask = self.bianary_masks[idx]

        if self.transform:
            data = self.transform(image=image, mask=label)
            image = data['image']
            label = data['mask']


        return image, label


class Vesuvius_Tile_Datset_TEST(Dataset):
    def __init__(self, images, test_pos=None, binary_masks =None,  transform=None):
        self.images = images
        self.test_pos = test_pos
        self.binary_masks = binary_masks
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




