import torch
import torch.nn as nn


class PreBackbone_3D(nn.Module):
    def __init__(self, batch_norm = False ):



        super(PreBackbone_3D, self).__init__()

        self.leaky_relu = nn.LeakyReLU( inplace=True)
        self.batch_norm = batch_norm

        self.conv = nn.Conv3d(in_channels=1,
                              out_channels=1,
                              kernel_size = (3, 1, 1),
                              stride=(1, 1, 1),
                              padding= (1, 0, 0)
                              )

        torch.nn.init.xavier_uniform_(self.conv.weight)
        torch.nn.init.zeros_(self.conv.bias)

        self.pool = nn.AvgPool3d(kernel_size = (2 ,1 ,1), stride=(2 ,1 ,1))
        self.batch_norm = torch.nn.BatchNorm3d( num_features=1, momentum=0.9)






    def forward(self, x):
        x = x.unsqueeze(1) # (B,C,H,W) -> (B, 1 C, H, W)


        y = self.conv(x)
        y = self.pool(y)
        y = self.leaky_relu(y)
        if self.batch_norm:
            y = self.batch_norm(y)  # (B, 1 C, H, W) -> (B, 1 C/2, H, W)


        y = self.conv(y)
        y = self.pool(y)
        y = self.leaky_relu(y)
        if self.batch_norm:
            y = self.batch_norm(y)  # (B, 1 C, H, W) -> (B, 1 C/2, H, W)


        y = self.conv(y)
        y = self.pool(y)
        y = self.leaky_relu(y)
        if self.batch_norm:
            y = self.batch_norm(y)  # (B, 1 C, H, W) -> (B, 1 C/2, H, W)

        y = self.conv(y)
        y = self.pool(y)
        y = self.leaky_relu(y)
        if self.batch_norm:
            y = self.batch_norm(y)  # (B, 1 C, H, W) -> (B, 1 C/2, H, W)

        return y.squeeze(1)

