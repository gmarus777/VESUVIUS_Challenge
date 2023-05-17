from PVT2 import PyramidVisionTransformerV2, Up, OutConv
import torch.nn as nn
from functools import partial
import torch





class PVT_w_UNet(nn.Module):
    def __init__(self, in_channels, embed_dims=[64, 128, 256, 512], n_classes=1, ):
        super().__init__()

        self.embed_dims = embed_dims

        self.pvt = PyramidVisionTransformerV2(img_size=PATCH_SIZE,
                                              patch_size=4,
                                              in_chans=Z_DIM,
                                              num_classes=1,
                                              embed_dims=embed_dims,
                                              num_heads=[1, 2, 4, 8],
                                              mlp_ratios=[8, 8, 4, 4],
                                              qkv_bias=True,
                                              qk_scale=None,
                                              drop_rate=0.,
                                              attn_drop_rate=0.,
                                              drop_path_rate=0.1,
                                              norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                              depths=[3, 4, 6, 3],
                                              sr_ratios=[8, 4, 2, 1]
                                              )

        self.up1 = Up(self.embed_dims[-1], self.embed_dims[-2])
        self.up2 = Up(self.embed_dims[-2], self.embed_dims[-3])
        self.up3 = Up(self.embed_dims[-3], self.embed_dims[-4])
        self.up4 = Up(self.embed_dims[-4], in_channels, last_layer=True)

        self.out_conv = OutConv(in_channels, n_classes)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.pvt(x)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.out_conv(x)

        return logits










