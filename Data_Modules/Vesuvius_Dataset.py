import segmentation_models_pytorch as smp

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
Z_DIM = 24

PATH = Path().resolve().parents[0]
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

KAGGLE_DIR = PATH / "kaggle"

INPUT_DIR = KAGGLE_DIR / "input"

COMPETITION_DATA_DIR = INPUT_DIR / "vesuvius-challenge-ink-detection"
COMPETITION_DATA_DIR_str = "kaggle/input/vesuvius-challenge-ink-detection/"



'''
class CFG:
    train_fragment_id=[2,3],
    val_fragment_id=[1],
    batch_size = 32,
    patch_size = 224,
    z_dim = 16,
    stride = patch_size // 2,
    comp_dataset_path = COMPETITION_DATA_DIR
    num_workers = 8
    on_gpu = True
    test_fragment_id = ['a','b']




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
            self.train_data, self.train_labels = self.get_train_dataset()
            self.val_data, self.val_labels, self.val_pos = self.get_val_dataset()


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

        for fragment_id in self.cfg.train_fragment_id:
            image, mask = self.read_image_mask('train', fragment_id)

            x1_list = list(range(0, image.shape[1] - self.cfg.patch_size + 1, self.cfg.stride))
            y1_list = list(range(0, image.shape[0] - self.cfg.patch_size + 1, self.cfg.stride))

            for y1 in y1_list:
                for x1 in x1_list:
                    y2 = y1 + self.cfg.patch_size
                    x2 = x1 + self.cfg.patch_size
                    # xyxys.append((x1, y1, x2, y2))

                    train_images.append(image[y1:y2, x1:x2])
                    train_masks.append(mask[y1:y2, x1:x2, None])

        return train_images, train_masks

    def get_val_dataset(self):
        valid_images = []
        valid_masks = []
        valid_xyxys = []

        for fragment_id in self.cfg.val_fragment_id:
            image, mask = self.read_image_mask('train', fragment_id)

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

        return images, mask


    def read_image_mask_test(self, stage, fragment_id):

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

        binary_mask  = cv2.imread(COMPETITION_DATA_DIR_str + f"test/{fragment_id}/mask.png", 0)
        binary_mask = (binary_mask / 255).astype(int)
        pad0 = (self.cfg.patch_size - binary_mask.shape[0] % self.cfg.patch_size)
        pad1 = (self.cfg.patch_size - binary_mask.shape[1] % self.cfg.patch_size)
        binary_mask = np.pad(binary_mask, [(0, pad0), (0, pad1)], constant_values=0)

        return images, binary_mask



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
    def __init__(self, images, labels=None,  transform=None):
        self.images = images
        self.labels = labels
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





'''
Additional 

A..augmentations.geometric.transforms.Perspective(scale=(0.05, 0.1),
                                                    keep_size=True,
                                                     pad_mode=0, 
                                                     pad_val=0, 
                                                     mask_pad_val=0, 
                                                     fit_output=False, 
                                                     interpolation=1, 
                                                     always_apply=False, 
                                                     p=0.5)


A.augmentations.geometric.resize.RandomScale(scale_limit=0.1, 
                                                interpolation=1, 
                                                always_apply=False, 
                                                p=0.5)



A.augmentations.geometric.transforms.OpticalDistortion(distort_limit=0.05, 
                                                        shift_limit=0.05, 
                                                        interpolation=1, 
                                                        border_mode=4, 
                                                        value=None, 
                                                        mask_value=None, 
                                                        always_apply=False, 
                                                        p=0.5)    




A.augmentations.geometric.transforms.ElasticTransform(alpha=1, 
                                                        sigma=50, 
                                                        alpha_affine=50, 
                                                        interpolation=1, 
                                                        border_mode=4, 
                                                        value=None, 
                                                        mask_value=None, 
                                                        always_apply=False, 
                                                        approximate=False, 
                                                        same_dxdy=False, 
                                                        p=0.5)                                           






Original Transforms:


    class Image_Transforms:

    train_transforms = A.Compose(
        [
            # A.RandomResizedCrop(
            #     size, size, scale=(0.85, 1.0)),
            A.Resize(PATCH_SIZE, PATCH_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.75),
            A.ShiftScaleRotate(p=0.75),
            A.OneOf([
                A.GaussNoise(var_limit=[10, 50]),
                A.GaussianBlur(),
                A.MotionBlur(),
            ], p=0.4),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
            A.CoarseDropout(max_holes=1, max_width=int(PATCH_SIZE * 0.3), max_height=int(PATCH_SIZE * 0.3),
                            mask_fill_value=0, p=0.5),
            # A.Cutout(max_h_size=int(size * 0.6),
            #          max_w_size=int(size * 0.6), num_holes=1, p=1.0),
            A.Normalize(
                mean=[0] * Z_DIM,
                std=[1] * Z_DIM,
            ),
            ToTensorV2(transpose_mask=True),
        ]
    )

    val_transforms = A.Compose(
        [
        A.Resize(PATCH_SIZE, PATCH_SIZE),
        A.Normalize(
            mean=[0] * Z_DIM,
            std=[1] * Z_DIM
        ),

        ToTensorV2(transpose_mask=True),
    ]
    )







    Updated:


    class Image_Transforms:

    train_transforms = A.Compose(
        [
            # A.RandomResizedCrop(
            #     size, size, scale=(0.85, 1.0)),
            #A.Resize(PATCH_SIZE, PATCH_SIZE),
            A.augmentations.geometric.resize.RandomScale(scale_limit=0.1,
                                                         interpolation=1,
                                                         always_apply=False,
                                                         p=0.3),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(.25, (-.3, .3), p=0.75),
            A.ShiftScaleRotate(p=0.75),
            A.GaussianBlur(blur_limit=(3, 7), p=0.4),
            A.GaussNoise(var_limit=[10, 50], p=0.4),

    A.OneOf([
                A.GaussNoise(var_limit=[10, 60]),
                A.GaussianBlur(blur_limit=(1, 9)),
                A.MotionBlur(blur_limit=9),
            ], p=0.3),

            A.augmentations.geometric.transforms.ElasticTransform(alpha=120,
                                                                  sigma=120*0.05,
                                                                  alpha_affine=120 * 0.03,
                                                                  interpolation=1,
                                                                  border_mode=cv2.BORDER_CONSTANT,
                                                                  value=0,
                                                                  mask_value=0,
                                                                  always_apply=False,
                                                                  approximate=False,
                                                                  same_dxdy=False,
                                                                  p=0.4),

            A.augmentations.geometric.transforms.OpticalDistortion(distort_limit=0.1,
                                                                   shift_limit=0.02,
                                                                   interpolation=1,
                                                                   border_mode=cv2.BORDER_CONSTANT,
                                                                   value=0,
                                                                   mask_value=0,
                                                                   always_apply=False,
                                                                   p=0.3),

            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
            A.CoarseDropout(max_holes=1, max_width=64, max_height=64,
                            mask_fill_value=0, p=0.5),
            A.CoarseDropout(max_holes=1, max_width=38, max_height=32,
                            mask_fill_value=0, p=0.5),
            # A.Cutout(max_h_size=int(size * 0.6),
            #          max_w_size=int(size * 0.6), num_holes=1, p=1.0),
            A.Resize(PATCH_SIZE, PATCH_SIZE),
            A.Normalize(
                mean=[0] * Z_DIM,
                std=[1] * Z_DIM,
            ),
            ToTensorV2(transpose_mask=True),
        ]
    )

    val_transforms = A.Compose(
        [
        A.Resize(PATCH_SIZE, PATCH_SIZE),
        A.Normalize(
            mean=[0] * Z_DIM,
            std=[1] * Z_DIM
        ),

        ToTensorV2(transpose_mask=True),
    ]
    )



    '''