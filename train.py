#!/usr/bin/env python
# encoding: utf-8
# author: fan.mo
# email: fmo@voxelcloud.net.cn

import os
import json
import timm

from tqdm import tqdm
from libs.nets.efficientnet_pytorch import EfficientNet
from libs.nets.stream_net import StreamNet

from data_loader import train_loader, basic_loader
from utils import load_pretrain_model


def hook(module, inp, outp):
    return outp.clone().detach()


def train(config):

    merge_method = config["merge_method"]
    training_stage = config["training_stage"]

    model_list = list()

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
        feature_name = model_config['stream_feature_names']
        handles = list()
        for fn in feature_name:
            ftr = eval(f"model.{fn}.register_forward_hook(hook)")
            handles.append(ftr)
            print(ftr)

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

        model_list.append(stream)

    sw_config = config['stream_weights']
    if sw_config['pretrained_model']:
        sw = torch.load(sw_conig['pretrained_model'])
    else:
        model = # TODO
        sw = torch.load()
        pass


if __name__ == "__main__":

    config = json.load(open('configs/exp_train.cfg'))
    train(config)
