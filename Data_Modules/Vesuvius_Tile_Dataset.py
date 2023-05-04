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


PATCH_SIZE =224
Z_DIM = 10


PATH = Path().resolve().parents[0]
#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

KAGGLE_DIR = PATH / "kaggle"

INPUT_DIR = KAGGLE_DIR / "input"

COMPETITION_DATA_DIR = INPUT_DIR / "vesuvius-challenge-ink-detection"
COMPETITION_DATA_DIR_str =  "kaggle/input/vesuvius-challenge-ink-detection/"

TRAIN_DATA_CSV_PATH = COMPETITION_DATA_DIR / "data_train_1.csv"
TEST_DATA_CSV_PATH = COMPETITION_DATA_DIR / "data_test_1.csv"

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
class Vesuvius_Tile_Datamodule(pl.LightningDataModule):
    def __init__(self,
                 cfg,
                 ):

        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.train_transform = Image_Transforms.train_transforms
        self.val_transform  = Image_Transforms.val_transforms
        self.train_data, self.train_labels = self.get_train_dataset()
        self.val_data, self.val_labels, self.val_pos = self.get_val_dataset()

        self.train_dataset = Vesuvius_Tile_Datset(images=self.train_data,
                                                  labels=self.train_labels,
                                                  transform=self.train_transform)

        self.val_dataset = Vesuvius_Tile_Datset(images=self.val_data,
                                                labels= self.val_labels,
                                                transform = self.val_transform)


    def get_train_dataset(self):
        train_images = []
        train_masks = []

        for fragment_id in self.cfg.train_fragment_id:
            image, mask = self.read_image_mask(fragment_id)

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

    def read_image_test(self,fragment_id):
        images = []

        # idxs = range(65)
        mid = 65 // 2
        start = mid - self.cfg.z_dim // 2
        end = mid + self.cfg.z_dim // 2
        idxs = range(start, end)

        for i in tqdm(idxs):
            image = cv2.imread(COMPETITION_DATA_DIR_str + f"test/{fragment_id}/surface_volume/{i:02}.tif", 0)

            pad0 = (self.cfg.patch_size - image.shape[0] % self.cfg.tile_size)
            pad1 = (self.cfg.patch_size - image.shape[1] % self.cfg.tile_size)

            image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)

            images.append(image)
        images = np.stack(images, axis=2)

        return images


    def read_image_mask(self, fragment_id):

        images = []

        # idxs = range(65)
        mid = 65 // 2
        start = mid - self.cfg.z_dim // 2
        end = mid + self.cfg.z_dim // 2
        idxs = range(start, end)

        for i in tqdm(idxs):
            image = cv2.imread(COMPETITION_DATA_DIR_str + f"train/{fragment_id}/surface_volume/{i:02}.tif", 0)

            pad0 = (self.cfg.patch_size - image.shape[0] % self.cfg.patch_size)
            pad1 = (self.cfg.patch_size - image.shape[1] % self.cfg.patch_size)

            image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)

            images.append(image)

        images = np.stack(images, axis=2)

        mask = cv2.imread(COMPETITION_DATA_DIR_str +f"train/{fragment_id}/inklabels.png", 0)
        mask = np.pad(mask, [(0, pad0), (0, pad1)], constant_values=0)

        mask = mask.astype('float32')
        mask /= 255.0

        return images, mask

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
            #multiprocessing_context="spawn"
            #multiprocessing_context="fork",
            #collate_fn=self.collate_function,
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
            #multiprocessing_context="spawn"
            #multiprocessing_context="fork",
            #collate_fn=self.collate_function
        )







class Vesuvius_Tile_Datset(Dataset):
    def __init__(self, images,  labels=None, transform=None):
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
                                                
                                                
                                                
A..augmentations.geometric.transforms.OpticalDistortion(distort_limit=0.05, 
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
'''



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