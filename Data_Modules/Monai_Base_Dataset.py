import torch
from torch.utils.data import Dataset, ConcatDataset



class Monai_Base_Dataset(Dataset):

    def __init__(self,
                image_stack,
                label,
                pixels,
                 buffer,
                 z_dim,
                 mask

                 ):

        self.image_stack = image_stack
        self.label = label
        self.pixels = pixels
        self.buffer = buffer
        self.z_dim = z_dim
        self.mask = mask



    def __len__(self):
        return len(self.pixels)



    def __getitem__(self, index: int):
        y,x = self.pixels[index]
        subvolume = self.image_stack[:, y - self.buffer:y + self.buffer , x - self.buffer:x + self.buffer ].view(1, self.z_dim, self.buffer * 2 ,self.buffer * 2 )
        inklabel = self.label[ y - self.buffer:y + self.buffer  , x - self.buffer:x + self.buffer  ].view(1, self.buffer * 2 ,self.buffer * 2 )
        mask = self.mask[ y - self.buffer:y + self.buffer  , x - self.buffer:x + self.buffer  ].view(1, self.buffer * 2 ,self.buffer * 2 )
        return subvolume, inklabel, mask

#






