import torch
import torch.nn as nn


# TODO: implement attnetion between slices. currently only local info


# TODO: try using Depth-wise convolution https://medium.com/@zurister/depth-wise-convolution-and-depth-wise-separable-convolution-37346565d4ec
# TODO:
class PreBackbone_3D(nn.Module):
    def __init__(self, filter_sizes = [6,12,24,48], batch_norm=False):


        super(PreBackbone_3D, self).__init__()

        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.batch_norm = batch_norm
        self.pool = nn.AvgPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))
        self.global_pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.batch_norm = torch.nn.BatchNorm3d(num_features=1, momentum=0.9)

        # layer 1
        self.conv1 = nn.Conv3d(in_channels=1,
                               out_channels=filter_sizes[0],
                               kernel_size=(3, 1, 1),
                               stride=(2, 1, 1),
                               padding=(1, 0, 0)
                               )

        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.zeros_(self.conv1.bias)

        # layer 2
        self.conv2 = nn.Conv3d(in_channels=filter_sizes[0]//2,
                               out_channels=filter_sizes[1],
                               kernel_size=(3, 1, 1),
                               stride=(2, 1, 1),
                               padding=(1, 0, 0)
                               )
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.zeros_(self.conv2.bias)

        # layer 3
        self.conv3 = nn.Conv3d(in_channels=filter_sizes[1]//2,
                               out_channels=filter_sizes[2],
                               kernel_size=(3, 1, 1),
                               stride=(2, 1, 1),
                               padding=(1, 0, 0)
                               )

        torch.nn.init.xavier_uniform_(self.conv3.weight)
        torch.nn.init.zeros_(self.conv3.bias)

        # layer 4
        self.conv4 = nn.Conv3d(in_channels=filter_sizes[2]//2,
                               out_channels=filter_sizes[3],
                               kernel_size=(3, 1, 1),
                               stride=(2, 1, 1),
                               padding=(1, 0, 0)
                               )

        torch.nn.init.xavier_uniform_(self.conv3.weight)
        torch.nn.init.zeros_(self.conv3.bias)






    def forward(self, x):
        x = x.unsqueeze(1)  # (B,C,H,W) -> (B, 1,  C, H, W)

        # Layer 1
        y = self.conv1(x)   # (B, 1,  C, H, W) -> (B, 4,  C/2, H, W)

        y = y.permute(0, 2, 1, 3, 4)
        y = self.pool(y)                # (B, 4,  C/2, H, W) ->  (B, 2,  C/2, H, W)
        y = y.permute(0, 2, 1, 3, 4)
        y = self.leaky_relu(y)
        # if self.batch_norm:
        #   y = self.batch_norm(y)

        # Layer 2
        y = self.conv2(y)
        y = y.permute(0, 2, 1, 3, 4)
        y = self.pool(y)
        y = y.permute(0, 2, 1, 3, 4)
        y = self.leaky_relu(y)

        # Layer 3
        y = self.conv3(y)
        y = y.permute(0, 2, 1, 3, 4)
        y = self.pool(y)
        y = y.permute(0, 2, 1, 3, 4)
        y = self.leaky_relu(y)

        # Layer 4
        y = self.conv4(y)  #(B, 48,  3, H, W)  where filter_sizes[-1]=48

       # Final convolution
        y = y.permute(0, 2, 1, 3, 4)
        y = self.global_pool(y)
        y = y.permute(0, 2, 1, 3, 4)

        y = self.leaky_relu(y)


        y = self.batch_norm(y)
        return y.squeeze(1)

