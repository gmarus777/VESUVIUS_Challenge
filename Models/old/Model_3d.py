import inspect
import math
import sys
from typing import Union

import numpy as np
import torch




class Encoder_3D(torch.nn.Module):
    def __init__(
        self, subvolume_shape, batch_norm_momentum, batch_norm, filters, in_channels
    ):
        super().__init__()

        input_shape = subvolume_shape

        self._batch_norm = batch_norm
        self._in_channels = in_channels

        self.pool = torch.nn.AvgPool3d(kernel_size=2, stride=2)
        self.global_pool = torch.nn.AdaptiveAvgPool3d((1, None, None))

        paddings = [1, 1, 1, 1]
        kernel_sizes = [3, 3, 3, 3]
        strides = [1, 1, 1, 1]

        self.conv1 = torch.nn.Conv3d(
            in_channels=in_channels,
            out_channels=filters[0],
            kernel_size=kernel_sizes[0],
            stride=strides[0],
            padding=paddings[0],
        )
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.zeros_(self.conv1.bias)
        self.batch_norm1 = torch.nn.BatchNorm3d(
            num_features=filters[0], momentum=batch_norm_momentum
        )


        self.conv2 = torch.nn.Conv3d(
            in_channels=filters[0],
            out_channels=filters[1],
            kernel_size=kernel_sizes[1],
            stride=strides[1],
            padding=paddings[1],
        )
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.zeros_(self.conv2.bias)
        self.batch_norm2 = torch.nn.BatchNorm3d(
            num_features=filters[1], momentum=batch_norm_momentum
        )


        self.conv3 = torch.nn.Conv3d(
            in_channels=filters[1],
            out_channels=filters[2],
            kernel_size=kernel_sizes[2],
            stride=strides[2],
            padding=paddings[2],
        )
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        torch.nn.init.zeros_(self.conv3.bias)
        self.batch_norm3 = torch.nn.BatchNorm3d(
            num_features=filters[2], momentum=batch_norm_momentum
        )


        self.conv4 = torch.nn.Conv3d(
            in_channels=filters[2],
            out_channels=filters[3],
            kernel_size=kernel_sizes[3],
            stride=strides[3],
            padding=paddings[3],
        )
        torch.nn.init.xavier_uniform_(self.conv4.weight)
        torch.nn.init.zeros_(self.conv4.bias)
        self.batch_norm4 = torch.nn.BatchNorm3d(
            num_features=filters[3], momentum=batch_norm_momentum
        )



    def forward(self, x):
        x = x.unsqueeze(1)

        y = self.conv1(x)
        y = torch.nn.functional.relu(y)
        if self._batch_norm:
            y = self.batch_norm1(y)
        y = self.pool(y)

        y = self.conv2(y)
        y = torch.nn.functional.relu(y)
        if self._batch_norm:
            y = self.batch_norm2(y)
        y = self.pool(y)

        y = self.conv3(y)
        y = torch.nn.functional.relu(y)
        if self._batch_norm:
            y = self.batch_norm3(y)
        y = self.pool(y)

        y = self.conv4(y)
        y = torch.nn.functional.relu(y)
        if self._batch_norm:
            y = self.batch_norm4(y)

        y = y.permute(0, 2, 1, 3, 4)
        y = self.global_pool(y)

        return y.squeeze(1)


class Encoder_Head

