#!/usr/bin/env python
# encoding: utf-8
# author: fan.mo
# email: fmo@voxelcloud.net.cn

import torch
import torch.nn as nn
import torch.nn.functional as F


class LossFunction(nn.Module):

    def __init__(self, loss_type, loss_weights):
        self.loss_type = loss_type
        self.loss_weights = loss_weights


    def forward(self, inp, label):
        if loss_type == "focal_loss":
            self.loss = self.focal_loss(loss_weights)

        elif loss_type == "ce_loss":
            self.loss = self.cross_entropy(loss_weights)



    def focal_loss(self)
