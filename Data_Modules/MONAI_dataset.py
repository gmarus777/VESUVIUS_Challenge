import torch
#from torch.utils.data import ConcatDataset, DataLoader
import pytorch_lightning as pl
import glob
import PIL.Image as Image
import numpy as np
from tqdm import tqdm
import monai
from monai.data import ArrayDataset, GridPatchDataset, create_test_image_3d, PatchIter, DataLoader
from monai.transforms import Compose, ScaleIntensity



from Data_Modules.Base_Dataset import Base_Dataset
from Data_Modules.Monai_Base_Dataset import Monai_Base_Dataset


PATH = 'kaggle/input/vesuvius-challenge/'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# scroll_1 size = 8181, 6330
# scroll_2 size = 14830, 9506
# scroll_3 size = 7606, 5249



class MONAI_Scrolls_Dataset(pl.LightningDataModule):

    def __init__(self,
                 patch_size = 512,
                 z_start = 27,
                 z_dim = 10,
                 validation_rect = (1100, 3500, 700, 950),
                shared_height = None,
                 downsampling =None,
                 train_scroll_fragments = [1,2],
                 val_scroll_fragments = [3],
                 stage = 'train',
                 batch_size=8,
                 num_workers =4 ,
                 on_gpu= False,


                 ):
        super().__init__()


        self.patch_size = patch_size
        self.z_start = z_start
        self.z_dim = z_dim
        self.validation_rect = validation_rect
        self.shared_height = shared_height
        self.downsampling = downsampling
        self.train_scroll_fragments = train_scroll_fragments
        self.val_scroll_fragments =val_scroll_fragments
        self.stage = stage
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.on_gpu = on_gpu


    def prepare_data(self, *args, **kwargs):
        if self.stage.lower() == 'train':


            z_slices = [[] for _ in range(len(self.train_scroll_fragments))]
            labels =  [[] for _ in range(len(self.train_scroll_fragments))]
            masks = [[] for _ in range(len(self.train_scroll_fragments))]

            z_slices_val =  [[] for _ in range(len( self.val_scroll_fragments))]
            labels_val =  [[] for _ in range(len( self.val_scroll_fragments))]
            masks_val =  [[] for _ in range(len( self.val_scroll_fragments))]

            for i in self.train_scroll_fragments:
                # get z_slices .tiffs paths
                z_slices[i-1] += sorted(glob.glob(f"{PATH}/{'train'}/{i}/surface_volume/*.tif"))[self.z_start:self.z_start + self.z_dim]
                # get labels
                labels[i-1] = self.load_labels('train', i)
                # get masks
                masks[i-1] = self.load_mask('train', i)

            # validation part
            for i in self.val_scroll_fragments:
                # get z_slices .tiffs paths
                z_slices_val[0] += sorted(glob.glob(f"{PATH}/{'train'}/{i}/surface_volume/*.tif"))[self.z_start:self.z_start + self.z_dim]
                # get labels
                labels_val[0] = self.load_labels('train', i)
                # get masks
                masks_val[0] = self.load_mask('train', i)

            # get images of z-slices and convert them to tensors for train
            images = [[] for _ in range(len(self.train_scroll_fragments))]
            for i in range(len(self.train_scroll_fragments)):
                images[i] = self.load_slices(z_slices[i])

            # same for validation images
            images_val = [[]]
            images_val[0] = self.load_slices( z_slices_val[0])


            # concat images, labels and masks of different scrolls
            images_tensors = torch.cat([image for image in images], axis=-1)
            label_tensors =  torch.cat([label for label in labels], axis=-1)
            mask_tensors =  np.concatenate([mask for mask in masks], axis=-1)
            mask_tensors = torch.from_numpy(mask_tensors)

            images_tensors_val = torch.cat([image for image in images_val], axis=-1)
            label_tensors_val = torch.cat([label for label in labels_val], axis=-1)
            mask_tensors_val = np.concatenate([mask for mask in masks_val], axis=-1)
            mask_tensors_val = torch.from_numpy(mask_tensors_val)

            del images
            del z_slices
            del labels
            del masks


            #self.mask =  torch.from_numpy(mask_tensors)
            #self.image_tensors = images_tensors
            #self.label_tensors = label_tensors
            #del mask_tensors

            array_ds = monai.data.ArrayDataset(img=images_tensors.unsqueeze(0),
                                               # img_transform = transform,
                                               seg=mask_tensors.unsqueeze(0).unsqueeze(0),
                                               # seg_transform = transform,
                                               # label_transform =transform,
                                               labels=label_tensors.unsqueeze(0).unsqueeze(0)
                                               )
            del mask_tensors
            del images_tensors
            del label_tensors

            patch_iter = monai.data.PatchIter(patch_size=(self.patch_size, self.patch_size))

            def img_seg_iter(x):
                for im, seg, label in zip(patch_iter(x[0]), patch_iter(x[1]), patch_iter(x[2])):
                    # uncomment this to confirm the coordinates
                    # print("coord img:", im[1].flatten(), "coord seg:", seg[1].flatten())
                    yield ((im[0], seg[0], label[0]),)

            self.data_train = monai.data.GridPatchDataset(array_ds, patch_iter=img_seg_iter, with_coordinates=False)
            del array_ds


            array_ds_val =  monai.data.ArrayDataset(img=images_tensors_val.unsqueeze(0),
                                               # img_transform = transform,
                                               seg=mask_tensors_val.unsqueeze(0).unsqueeze(0),
                                               # seg_transform = transform,
                                               # label_transform =transform,
                                               labels=label_tensors_val.unsqueeze(0).unsqueeze(0)
                                               )
            del mask_tensors_val
            del images_tensors_val
            del label_tensors_val

            self.data_val = monai.data.GridPatchDataset(array_ds_val, patch_iter=img_seg_iter, with_coordinates=False)
            del array_ds_val



        # TODO: finish the same for test, note paths are different
        elif self.stage.lower() == 'test':

            # get z_slices paths
            z_slices = [[], []]
            for i, l in enumerate(['a','b']):
                z_slices[i] = sorted(glob.glob(f"{PATH}/{'test'}/{l}/surface_volume/*.tif"))[self.z_star:self.z_star + self.z_dim]



    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        """
        construct a dataloader for training data
        data is shuffled !
        :param args:
        :param kwargs:
        :return:
        """
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            #shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            # shuffle=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        return DataLoader(
            self.data_test,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
            collate_fn=self.collate_function,
        )






    # image_stack = torch.stack([torch.from_numpy(image) for image in images], dim=0).to(DEVICE)
    def load_slices(self, z_slices_fnames):
        images = []
        for z, filename in tqdm(enumerate(z_slices_fnames)):
            img = Image.open(filename)
            img = self.resize(img)
            z_slice = np.array(img, dtype="float32")/65535.0
            images.append(z_slice)
        return torch.stack([torch.from_numpy(image) for image in images], dim=0)#.to(DEVICE)



    def load_mask(self, split, index):
        img = Image.open(f"{PATH}/{split}/{index}/mask.png").convert('1')
        img = self.resize(img)
        return np.array(img)



    def load_labels(self, split, index):
        img = Image.open(f"{PATH}/{split}/{index}/inklabels.png")
        img = self.resize(img)
        return torch.from_numpy(np.array(img)).gt(0).float()#.to(DEVICE)


    def resize(self, img):
        current_width, current_height = img.size
        aspect_ratio = current_width / current_height
        new_width = int(self.shared_height * aspect_ratio)
        new_size = (new_width, self.shared_height)
        img = img.resize(new_size)
        return img



    def split_train_val(self,mask):
        rect = self.validation_rect
        not_border = np.zeros(mask.shape, dtype=bool)
        not_border[self.buffer:mask.shape[0] - self.buffer, self.buffer:mask.shape[1] - self.buffer] = True
        arr_mask = np.array(mask) * not_border
        inside_rect = np.zeros(mask.shape, dtype=bool) * arr_mask
        inside_rect[rect[1]:rect[1] + rect[3] + 1, rect[0]:rect[0] + rect[2] + 1] = True
        outside_rect = np.ones(mask.shape, dtype=bool) * arr_mask
        outside_rect[rect[1]:rect[1] + rect[3] + 1, rect[0]:rect[0] + rect[2] + 1] = False
        pixels_inside_rect = np.argwhere(inside_rect)
        pixels_outside_rect = np.argwhere(outside_rect)
        return pixels_outside_rect, pixels_inside_rect





