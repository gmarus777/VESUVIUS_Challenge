import math
import json
import torch
import torch.nn as nn
import torchvision.models

RESNET_DIM = 512

class ResNet3d(nn.Module):

    def __init__(
            self,
            num_classes=2,
            embedding_dim=128,
            z_dim=None,
            buffer=None,

    ):

        super().__init__()

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        ### Encoder ###
        r3d_18 = torchvision.models.video.r3d_18(pretrained=False)
        self.backbone = nn.Sequential(
            r3d_18.stem,
            r3d_18.layer1,
            r3d_18.layer2,
            r3d_18.layer3,
            r3d_18.layer4,
        )

        self.output_dim = self.embedding_dim * math.ceil(z_dim / 8) * (math.ceil((buffer * 2 + 1) / 16)**2)
        self.bottleneck = nn.Conv3d(RESNET_DIM, self.embedding_dim, 1)  # in channels, out channels, stride
        self.fc = nn.Linear(self.output_dim, self.num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        init_range = 0.1
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-init_range, init_range)

        nn.init.kaiming_normal_(
            self.bottleneck.weight.data,
            a=0,
            mode="fan_out",
            nonlinearity="relu",
        )
        if self.bottleneck.bias is not None:
            _, fan_out = nn.init._calculate_fan_in_and_fan_out(self.bottleneck.weight.data)
            bound = 1 / math.sqrt(fan_out)
            nn.init.normal_(self.bottleneck.bias, -bound, bound)

    def forward(self, x):
        if x.shape[1] == 1:  # (B, 1, z_dim, r, r) where r = Buffer*2+1
            x = x.repeat(1, 3, 1, 1, 1)

        x = self.backbone(x.float())  # (B, RESNET_DIM, Z_DIM, R, R); Z_DIM = z_dim // 8, R = r // 16
        x = self.bottleneck(x)  # (B, E,  Z_DIM, R); E:= embedding dim

        x = x.flatten(start_dim=1)  # (B, E *Z_DIM * R*R)
        output = self.fc(x)  # ( B, num_classes)
        return output