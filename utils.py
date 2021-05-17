#!/usr/bin/env python
# encoding: utf-8
# author: fan.mo
# email: fmo@voxelcloud.net.cn

import json
import torch
import yaml
from collections import OrderedDict

with open('path_infor.yaml', 'r') as f:
    all_path_infor = yaml.load(f, Loader=yaml.FullLoader)


def load_pretrain_model(model, ckpt, key):
    """Load pretrained model from checkpoint path
    Args:
        model: (Torch Model)
        ckpt: checkpoint path
        key: saved key in checkpoint data
    """
    model_dict = model.state_dict()
    state_dict = torch.load(ckpt, map_loacation='cpu')
    if key:
        if key in state_dict:
            state_dict = state_dict[key]
        else:
            key_list = list( filter(lambda x: 'state_dict' in x, state_dict.keys()))
            assert len(key_list) > 0
            state_dict = state_dict[key_list[0]]

    ckpt_dict = OrderedDict()
    first_layer_name = list(state_dict.keys())[0]

    if first_layer_name.startswith('module.'):
        start_idx = 7
    elif first_layer_name.startswith('model.'):
        start_idx = 6
    else:
        start_idx = 0

    for k, v in state_dict.items():
        name = k[start_idx:]
        ckpt_dict[name] = v

    if model_dict[list(model_dict.keys())[-1]].shape == ckpt_dict[list(ckpt_dict.keys())[-1]].shape:
        model.load_state_dict(ckpt_dict)
    else:
        ckpt_dict.pop(list(ckpt_dict)[-1])
        ckpt_dict.pop(list(ckpt_dict)[-1])
        msg = model.load_state_dict(ckpt_dict, strict=False)
        print('load msg ', msg)

    print(f'load model from {ckpt}')

    return model


def one_data_infor(data_name, scale, label_name, all_path_infor=all_path_infor):
    item = all_path_infor[data_name]
    one_csv_infor = dict()
    one_csv_infor['csv_path'], one_csv_infor['img_dir'] = \
        item['csv_path'], item['img_dir']

    if label_name in item:
        one_csv_infor['label_col'] = item[label_name]
    else:
        one_csv_infor['label_col'] = label_name
    if 'img_path_col' in item:
        one_csv_infor['path_col'] = item['img_path_col']
    else:
        one_csv_infor['path_col'] = 'path'

    one_csv_infor['scale'] = scale
    label_map_key = f'{label_name}_map'
    if label_map_key in item and item[label_map_key]:
        one_csv_infor['label_map'] = item[label_map_key]

    return one_csv_infor


def get_data_infor(data_names, label_name):
    if isinstance(data_names, str):
        csv_infor = one_data_infor(data_names, 1.0, label_name)
    elif isinstance(data_names, dict):
        csv_infor = []
        for name in data_names:
            tmp_csv_info = one_data_infor(name, scale=data_names[name], label_name=label_name)
            csv_infor.append(tmp_csv_info)
    else:
        raise TypeError

    return csv_infor

