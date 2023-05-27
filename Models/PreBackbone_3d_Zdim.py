import torch
import torch.nn as nn
from einops import rearrange


# TODO: implement attnetion between slices. currently only local info


# TODO: try using Depth-wise convolution https://medium.com/@zurister/depth-wise-convolution-and-depth-wise-separable-convolution-37346565d4ec
# TODO:


class PreBackbone_3D_ZDIM(nn.Module):
    def __init__(self, out_channels = 3, z_dim= 24, emdedding_dims=[4], filter_sizes=[8, 16, 32,], batch_norm=False):

        super(PreBackbone_3D_ZDIM, self).__init__()

        self.z_dim = z_dim//2

        self.embed_layer = Embed(emdedding_dims=emdedding_dims)
        self.attention = EfficientMultiHeadAttention(channels=self.z_dim, att_dim =256)

        self.pool = nn.AvgPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))
        self.global_pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.global_pool_final = nn.AdaptiveAvgPool3d((out_channels, None, None))
        self.batch_norm = torch.nn.BatchNorm3d(num_features=1, momentum=0.1)
        self.leaky_relu = nn.LeakyReLU(inplace=False)

        # layer 1
        self.conv1 = nn.Conv3d(in_channels=5,
                               out_channels=filter_sizes[0],
                               kernel_size=(3, 3, 3),
                               stride=(2, 1, 1),
                               padding=(1, 1, 1)
                               )

        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.zeros_(self.conv1.bias)

        # layer 2
        self.conv2 = nn.Conv3d(in_channels=filter_sizes[0] // 2,
                               out_channels=filter_sizes[1],
                               kernel_size=(3, 3, 3),
                               stride=(2, 1, 1),
                               padding=(1, 1, 1)
                               )
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.zeros_(self.conv2.bias)

        # layer 3
        self.conv3 = nn.Conv3d(in_channels=filter_sizes[1] // 2,
                               out_channels=filter_sizes[2],
                               kernel_size=(3, 3, 3),
                               stride=(2, 1, 1),
                               padding=(1, 1, 1)
                               )

        torch.nn.init.xavier_uniform_(self.conv3.weight)
        torch.nn.init.zeros_(self.conv3.bias)

        # layer 3
        self.conv4 = nn.Conv3d(in_channels=1,
                               out_channels=1,
                               kernel_size=(3, 3, 3),
                               stride=(2, 1, 1),
                               padding=(1, 1, 1)
                               )

        torch.nn.init.xavier_uniform_(self.conv3.weight)
        torch.nn.init.zeros_(self.conv3.bias)

    def forward(self, x):
        x = x.unsqueeze(1)



        # Stage 1 Embedding and Z_Dim attention

        # embed layer produces tensors:
        # x_orig = (B, emdedding_dims[0], C/2, H, W) for residual connection
        # x_att = (B, 1, C/2, H/4, W/4)
        x_orig, x_att = self.embed_layer(x)
        x_orig = self.leaky_relu(x_orig)
        x_att = self.leaky_relu(x_att)


        # attention layer for z_dim
        x_after_att = self.attention(x_att)

        x = torch.cat((x_orig, x_after_att), dim=1)

        # Stage 2 Convolutions -- 3 layers

        # Layer 1
        y = self.conv1(x)  # (B, 1,  C, H, W) -> (B, 4,  C/2, H, W)

        y = y.permute(0, 2, 1, 3, 4)
        y = self.pool(y)  # (B, 4,  C/2, H, W) ->  (B, 2,  C/2, H, W)

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
        y = self.global_pool(y)
        y = self.leaky_relu(y)
        y = y.permute(0, 2, 1, 3, 4) # (B,1,64, H, W)

        # Final pooling to out_channles channels

        y = self.conv4(y)
        y = self.global_pool_final(y)
        #y = self.leaky_relu(y)

        #y = self.batch_norm(y)
        return y.squeeze(1)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = rearrange(x, "b c d h w -> b d h w c")
        x = super().forward(x)
        x = rearrange(x, "b d h w c -> b c d h w")
        return x


class Embed(nn.Module):
    def __init__(self, in_channels=1, emdedding_dims=[4], ):
        super().__init__()

        self.conv_3d = nn.Conv3d(in_channels=1,
                                 out_channels=emdedding_dims[0],
                                 kernel_size=(3, 1, 1),
                                 stride=(2, 1, 1),
                                 padding=(1, 0, 0)
                                 )
            # add a laeyr where bothz and x,y go down by half and change the next one by 2
        self.conv_3d_embed = nn.Conv3d(in_channels=emdedding_dims[0],
                                       out_channels=1,
                                       kernel_size=(1, 4, 4),
                                       stride=(1, 4, 4),
                                       padding=(0, 1, 1)
                                       )

        self.norm = LayerNorm2d(emdedding_dims[0])
        self.norm_embed = LayerNorm2d(1)

    def forward(self, x):
        x = self.conv_3d(x)
        x = self.norm(x)
        x_embed = self.conv_3d_embed(x)
        x_embed = self.norm_embed(x_embed)
        return x, x_embed


class EfficientMultiHeadAttention(nn.Module):
    def __init__(self, channels, att_dim =256, reduction_ratio: int = 1, num_heads: int = 8):
        super().__init__()

        self.reducer = nn.Sequential(
            nn.Conv2d(in_channels=channels,
                      out_channels=channels,
                      kernel_size=(4, 4),
                      stride=(4, 4),
                      padding=(0, 0)),

            LayerNorm_att(channels, ) )

        self.att = nn.MultiheadAttention(att_dim, num_heads=8, batch_first=True)

    def forward(self, x, ):
        x = x.squeeze(1)  # (b z h w)

        reduced_x = self.reducer(x)

        _, c, h, w = reduced_x.shape

        # attention needs tensor of shape (batch, sequence_length, channels)
        reduced_x = rearrange(reduced_x, "b c h w -> b  c ( h w )")
        x = rearrange(x, "b c  h w -> b  c ( h w)")
        out = self.att(reduced_x, reduced_x, reduced_x)[0]
        # reshape it back to (batch, channels, height, width)
        out = rearrange(out, "b  c (h w) -> b c h w", h=h, w=w, )
        out = nn.functional.interpolate(out, size=(256, 256), mode="bilinear", align_corners=False)
        out = out.unsqueeze(1)

        return out


class LayerNorm_att(nn.LayerNorm):
    def forward(self, x):
        x = rearrange(x, "b c h w -> b  h w c")
        x = super().forward(x)
        x = rearrange(x, "b  h w c -> b c  h w")
        return x


class ClassifierHead(nn.Module):
    def __init__(self):
        super(ClassifierHead, self).__init__()

        # Apply a 1x1x1 convolution to reduce the channel dimension from 3 to 1
        self.conv = nn.Conv2d(in_channels=3,
                              out_channels=1,
                              kernel_size=1,
                              stride=1,
                              padding=0)

    def forward(self, x):
        x = self.conv(x)
        return x