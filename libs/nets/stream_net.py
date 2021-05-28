#!/usr/bin/env python
# encoding: utf-8
# author: fan.mo
# email: fmo@voxelcloud.net.cn

import torch
import torch.nn as nn
import torch.nn.functional as F


class StreamNet(nn.Module):
    arg1 = 0.2
    arg2 = 0.1

    def __init__(self, inp_dim, out_dim, depth=2, width=512, activation="relu"):
        super(StreamNet, self).__init__()

        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.depth = depth
        self.width = width
        self.activation = activation

        self.layers = self.make_layers()


    def make_layers(self):
        if self.activation == "relu":
            act = nn.ReLU()
        elif self.activation == "leaky_relu":
            act = nn.LeakyReLU(self.arg1)
        else:
            raise ValueError("activation method must be in ['relu', 'leaky_relu']")

        layers = list()
        layers.append(nn.Linear(self.inp_dim, self.width))
        layers.append(act)

        for _ in range(self.depth-1):
            layers.append(nn.Linear(self.width, self.width))
            layers.append(act)

        layers.append(nn.Linear(self.width, self.out_dim))
        # layers.append(nn.Softmax(dim=1))

        return nn.Sequential(*layers)


    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":

    StreamNet.arg1 = 0.3
    net = StreamNet(50, 10)
    print(net)
    x = torch.randn(8, 50)
    y = net(x)
    print(y[0])

