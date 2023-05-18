import torch
import torch.nn as nn
import torch.nn.functional as F


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False, downsample =False):
        super().__init__()
        self.upsample = upsample
        self.downsample = downsample
        self.downsample_4x = nn.MaxPool2d(kernel_size=4, stride=4)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), stride=1, padding=1, bias=False),
            #nn.GroupNorm(32, out_channels, eps=1e-03),
            torch.nn.SyncBatchNorm(out_channels, eps=1e-03, momentum=0.1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        if self.downsample:
            x = self.downsample_4x(x)

        return x


class FPNBlock(nn.Module):
    def __init__(self, pyramid_channels, skip_channels):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        skip = self.skip_conv(skip)
        x = x + skip
        return x


class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()

        blocks = [Conv3x3GNReLU(in_channels, out_channels, upsample=bool(n_upsamples))]

        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(Conv3x3GNReLU(out_channels, out_channels, upsample=True))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)


class MergeBlock(nn.Module):
    def __init__(self, policy):
        super().__init__()
        if policy not in ["add", "cat"]:
            raise ValueError("`merge_policy` must be one of: ['add', 'cat'], got {}".format(policy))
        self.policy = policy

    def forward(self, x):
        if self.policy == "add":
            return sum(x)
        elif self.policy == "cat":
            return torch.cat(x, dim=1)
        else:
            raise ValueError("`merge_policy` must be one of: ['add', 'cat'], got {}".format(self.policy))


class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class FPNDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        encoder_channels,
        encoder_depth=5,
        pyramid_channels=256,
        segmentation_channels=128,
        dropout=0.2,
        merge_policy="add",
    ):
        super().__init__()

        self.out_channels = segmentation_channels if merge_policy == "add" else segmentation_channels * 4
        if encoder_depth < 3:
            raise ValueError("Encoder depth for FPN decoder cannot be less than 3, got {}.".format(encoder_depth))

        encoder_channels = encoder_channels[::-1]
        encoder_channels = encoder_channels[: encoder_depth + 1]

        self.p5 = nn.Conv2d(encoder_channels[0], pyramid_channels, kernel_size=1).to(DEVICE)
        self.p4 = FPNBlock(pyramid_channels, encoder_channels[1]).to(DEVICE)
        self.p3 = FPNBlock(pyramid_channels, encoder_channels[2]).to(DEVICE)
        self.p2 = FPNBlock(pyramid_channels, encoder_channels[3]).to(DEVICE)
        self.p1 = nn.Conv2d(in_channels, pyramid_channels, kernel_size=1).to(DEVICE)

        self.seg_blocks = nn.ModuleList(
            [ SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=n_upsamples) for n_upsamples in [5, 4, 3, 2] ]
                 )

        self.merge = MergeBlock(merge_policy)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)
        self.block = Conv3x3GNReLU(pyramid_channels, segmentation_channels, upsample=False, downsample=False).to(DEVICE)

        self.conv_fuse = nn.Sequential(
                    nn.ConvTranspose2d(
                        segmentation_channels*5, segmentation_channels, kernel_size=1, stride=1),
                    torch.nn.SyncBatchNorm(segmentation_channels, eps=1e-03, momentum=0.1),
                    #nn.GroupNorm(32, segmentation_channels, eps=1e-03),
                    nn.GELU(),
                    nn.ConvTranspose2d(
                        segmentation_channels, segmentation_channels, kernel_size=1, stride=1),
                )

        self.linear_pred = nn.Conv2d(segmentation_channels, 1, kernel_size=1)


    def forward(self, *features):
        c1, c2, c3, c4, c5 = features

        p5 = self.p5(c5)
        p4 = self.p4(p5, c4)
        p3 = self.p3(p4, c3)
        p2 = self.p2(p3, c2)
        p1 = self.p1(c1)

        #print('before seg', p5.shape, p4.shape, p3.shape, p2.shape, p1.shape )


        feature_pyramid = [seg_block(p) for seg_block, p in zip(self.seg_blocks, [p5, p4, p3, p2])]
        #for i, f in enumerate(feature_pyramid):
            #print(i, f.shape)
        f1 =  self.block(p1)

        feature_pyramid += [f1]

        x = self.merge(feature_pyramid)
        x = self.conv_fuse(x)
        x = self.dropout(x)
        x =  self.linear_pred(x)


        return x


class Feature2Pyramid(nn.Module):
    """Feature2Pyramid.

    A neck structure connect ViT backbone and decoder_heads.

    Args:
        embed_dims (int): Embedding dimension.
        rescales (list[float]): Different sampling multiples were
            used to obtain pyramid features. Default: [4, 2, 1, 0.5].
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='SyncBN', requires_grad=True).
    """

    def __init__(self,
                 embed_dim,
                 rescales=[4, 2, 1, 0.5],
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        self.rescales = rescales
        self.upsample_4x = None
        for k in self.rescales:
            if k == 4:
                self.upsample_4x = nn.Sequential(
                    nn.ConvTranspose2d(
                        embed_dim, embed_dim, kernel_size=2, stride=2),
                    torch.nn.SyncBatchNorm(embed_dim, eps=1e-05, momentum=0.1),
                    nn.GELU(),
                    nn.ConvTranspose2d(
                        embed_dim, embed_dim, kernel_size=2, stride=2),
                )
            elif k == 2:
                self.upsample_2x = nn.Sequential(
                    nn.ConvTranspose2d(
                        embed_dim, embed_dim, kernel_size=2, stride=2))
            elif k == 1:
                self.identity = nn.Identity()
            elif k == 0.5:
                self.downsample_2x = nn.MaxPool2d(kernel_size=2, stride=2)
            elif k == 0.25:
                self.downsample_4x = nn.MaxPool2d(kernel_size=4, stride=4)
            else:
                raise KeyError(f'invalid {k} for feature2pyramid')

    def forward(self, inputs):
        assert len(inputs) == len(self.rescales)
        outputs = []
        if self.upsample_4x is not None:
            ops = [
                self.upsample_4x, self.upsample_2x, self.identity,
                self.downsample_2x
            ]
        else:
            ops = [
                self.upsample_2x, self.identity, self.downsample_2x,
                self.downsample_4x
            ]
        for i in range(len(inputs)):
            outputs.append(ops[i](inputs[i]))
        return tuple(outputs)