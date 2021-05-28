#!/usr/bin/env python
# encoding: utf-8
# author: fan.mo
# email: fmo@voxelcloud.net.cn

import os
import json
import timm

import torch

from tqdm import tqdm
from libs.nets.efficientnet_pytorch import EfficientNet
from libs.nets.stream_net import StreamNet
from libs.layers.loss import LossFunction

from libs.data.data_loader import train_loader, basic_loader
from utils.utils import load_pretrain_model


def build_stream_net(config, input_shape):

    out_dim = config['out_dim']
    active = config['activation']
    arg1 = config['arg1']
    arg2 = config['arg2']
    depth = config['depth']
    width = config['width']

    StreamNet.arg1 = arg1
    StreamNet.arg2 = arg2

    net = StreamNet(inp_dim=input_shape, out_dim=out_dim, depth=depth)

    return net


def configure_loss(cfg):

    loss_type = cfg['loss']
    lw = cfg['loss_weights']

    loss = LossFunction(loss_type, lw)

    return loss


def train(config):

    merge_method = config["merge_method"]
    training_stage = config["training_stage"]

    model_list = list()

    def hook(module, inp, outp):
        features.(outp.clone().detach())

    for idx, model_config in enumerate(config["models"]):

        stream = dict()

        # load model architect
        model_name = model_config["model_name"]
        num_classes = model_config["num_classes"]
        if model_name.startswith("efficientnet-b"):
            model = EfficientNet.from_pretrained(model_name,
                num_classes=num_classes)
        else:
            model = timm.create_model(model_name, num_classes=num_classes, pretrained=True)

        # load trained model
        ckpt_path = model_config['checkpoint_path']
        key_ckpt = model_config['key_checkpoint']
        model = load_pretrain_model(model, ckpt_path, key_ckpt)

        # add hook to extract feature(s)
        feature_names = model_config['stream_feature_names']
        feature_dims = model_config['stream_feature_dims']

        features

        for fn, fd in zip(feature_names, feature_dims):
            ftr = eval(f"model.{fn}.register_forward_hook(hook)")

            handles.append(ftr)
            print(ftr)
            handle_dims.append(fd)

        # transform for image preprocess
        image_size = model_config['image_size']
        resize_mode = model_config['resize_mode']

        # TODO

        # dataloader
        data_config = config['data_loader'][idx]
        # TODO

        stream['model'] = model
        stream['hook'] = handles
        stream['dataloader'] = data_loader
        stream['feature_dim'] = handle_dims

        model_list.append(stream)

    sw_config = config['stream_weights']

    input_tensor_shape = 0
    for model in model_list:
        ft_dim = model['feature_dim']
        input_tensor_shape += ft_dim

    # stream network
    stream_net = build_stream_net(sw_config, input_tensor_shape)


    # optimizer & loss
    tr_config = config['train_op']
    optimizer = configure_optimizer(tr_config)
    loss_function = configure_loss(tr_config)

    global_step = 0

    # training
    for epoch in epochs:

        for data, label in data_loader:

            inp = []

            for i in range(len(data)):
                y = model[i](data[i]) # execute forward propagation

                mid = features[i][-1]
                inp.append(mid)

            if merge_method == "concat":
                bla bla
            elif merge_method == "add":
                bla bla
            else:
                pass

            out = stream_net(inp)

            optimizer.zero_grad()

            loss = loss_function(out, label)

            loss.backward()

            optimizer.step()

            global_step += 1


if __name__ == "__main__":

    config = json.load(open('configs/exp_train.cfg'))
    train(config)
