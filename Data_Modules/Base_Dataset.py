import torch
from torch.utils.data import Dataset, ConcatDataset



class Base_Dataset(Dataset):

    def __init__(self,
                image_stack,
                label,
                pixels,
                 buffer,
                 z_dim,

                 ):

        self.image_stack = image_stack
        self.label = label
        self.pixels = pixels
        self.buffer = buffer
        self.z_dim = z_dim



    def __len__(self):
        return len(self.pixels)



    def __getitem__(self, index: int):
        y,x = self.pixels[index]
        subvolume = self.image_stack[:, y - self.buffer:y + self.buffer + 1, x - self.buffer:x + self.buffer + 1].view(1, self.z_dim, self.buffer * 2 + 1,self.buffer * 2 + 1)
        inklabel = self.label[y, x].view(1)
        return subvolume, inklabel

#






