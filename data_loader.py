#!/usr/bin/env python
# encoding: utf-8
# author: fan.mo
# email: fmo@voxelcloud.net.cn

import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

import cv2
from skimage.io import imread, imsave

import albumentations
import albumentations.augmentations.transforms as transforms
import albumentations.core.composition as composition

from black_cut import RandomCenterCut, process_fundus, pad_square


def augmentation(mode, target_size, prob = 0.5, aug_m=2):
    '''
    description: augmentation
    mode: 'train' 'test'
    target_size: int or list, the shape of image ,
    aug_m: Strength of transform
    '''
    high_p = prob
    low_p = high_p / 2.0
    M = aug_m
    first_size = [int(x/0.7) for x in target_size]

    if mode == 'train':
        return composition.Compose([

            transforms.Resize(first_size[0], first_size[1], interpolation=3),
            transforms.Flip(p=0.5),
            composition.OneOf([
                RandomCenterCut(scale=0.1 * M),
                transforms.ShiftScaleRotate(shift_limit=0.05*M, scale_limit=0.1*M, rotate_limit=180,
                                border_mode=cv2.BORDER_CONSTANT, value=0),
                albumentations.imgaug.transforms.IAAAffine(shear=(-10*M, 10*M), mode='constant')
                ], p=high_p),

            transforms.RandomBrightnessContrast(brightness_limit=0.1*M, contrast_limit=0.03*M, p=high_p),
            transforms.HueSaturationValue(hue_shift_limit=5*M, sat_shift_limit=15*M, val_shift_limit=10*M, p=high_p),
            transforms.OpticalDistortion(distort_limit=0.03 *M, shift_limit=0, border_mode=cv2.BORDER_CONSTANT, value=0, p=low_p),

            composition.OneOf([
                transforms.Blur(blur_limit=7),
                albumentations.imgaug.transforms.IAASharpen(),
                transforms.GaussNoise(var_limit=(2.0, 10.0), mean=0),
                transforms.ISONoise()
                ], p=low_p),

            transforms.Resize(target_size[0], target_size[1], interpolation=3),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0)
        ], p=1)

    else:
        return composition.Compose([
            transforms.Resize(target_size[0], target_size[1], interpolation=3),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0)
        ], p=1)


class DistributedProxySampler(DistributedSampler):

    def __init__(self, sampler, num_replicas=None, rank=None):
        '''Use DistributedSampler to package random sampler
        Args:
            sampler: WeightedRandomSampler
            num_replicas: number of using gpu
            rank: current gpu id
        '''
        super(DistributedProxySampler, self).__init__(sampler, num_replicas=num_replicas, rank=rank, shuffle=False)
        self.sampler = sampler


    def __iter__(self):
        torch.manual_seed(self.epoch)
        indices = list(self.sampler)
        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        if len(indices) != self.total_size:
            raise RuntimeError(f"{len(indices)} vs {self.total_size}")
        indices = indices[self.rank: self.total_size: self.num_replicas]
        if len(indices) != self.num_samples:
            raise RuntimeError(f"{len(indices)} vs {self.num_replicas}")

        return iter(indices)


class BasicDataset(Dataset):

    def __init__(self,
                 img_paths,
                 labels,
                 tg_size,
                 mode,
                 resize_mode=1,
                 prob=0.5,
                 aug_m=2,
                 need_pre_process=False,
                 aug_out_dir=None):
        self.img_paths = img_paths
        self.labels = labels
        self.tg_size = tg_size
        self.num_images = len(self.img_paths)
        self.pre_process = need_pre_process
        self.aug_out_dir = aug_out_dir
        self.resize_mode = resize_mode

        if isinstance(self.tg_size, int):
            self.tg_size = [self.tg_size, self.tg_size]

        self.aug = augmentation(mode, self.tg_size, prob, aug_m)
        print(f'Creating dataset with {self.num_images} examples)')


    def __len__(self):
        return self.num_images


    def __getitem__(self, i):
        img_path =  self.img_paths[i]
        label = self.labels[i]
        try :
            img = imread(img_path).astype(np.uint8)
            if self.resize_mode > 1:
                raise ValueError("resize_mode should be 0 or 1")
            elif self.resize_mode == 1:
                img = pad_square(img)
            assert not img is None
            assert img.shape[0] > 10 and img.shape[1] > 10 and img.shape[2] == 3 and img.ndim == 3
        except  Exception as err_infor:
            print(img_path)
            print(err_infor)
            return self.__getitem__(np.random.randint(self.num_images))

        if self.pre_process:
            try:
                img = process_fundus(img, out_radius=500, square_pad=False)
            except Exception as err_infor:
                print( f'process err. image path{img_path}, {err_infor}')
                img = img

        augmented = self.aug(image=img)
        image = augmented['image']
        ## for debug, view data augmentation images
        if self.aug_out_dir:
            os.makedirs( self.aug_out_dir , exist_ok=True)
            basename = os.path.basename(img_path)
            new_name = os.path.join(self.aug_out_dir, basename)
            img_id , suffix   = os.path.splitext(new_name)
            original_name = img_id+'_original_'+ suffix
            image_aug =  (image.copy() + 1 ) / 2 * 255
            imsave( new_name,image_aug.astype(np.uint8))
            imsave( original_name,img)
        image = np.transpose(image, (2, 0, 1))
        assert image.shape == (3, self.tg_size[0], self.tg_size[1])
        return {'image': torch.Tensor(image), 'label': label, 'path':img_path}


def get_sample_weight(label, alpha=1.0):
    """
    get weight of every sample,
    alpha == 0, natural distribution
    alpha == 1, uniform distribution

    Args:
        label ([np.ndarray or list]): [description]
        alpha (float, optional): range [0, 1].
                When `alpha` == 0, the  class distribution will be natural distribution
                When `alpha` == 1, the  class distribution will be uniform distribution
                Defaults to 1.0.
    Returns:
        [np.ndarray]: weight of every sample
    """
    if isinstance(label, list):
        label = np.array(label)

    n_samples = label.shape[0]
    count_pre_class = np.bincount(label)
    n_class = count_pre_class.shape[0] # class_number

    original_weights = np.ones(n_class)
    uniform_weights = n_samples / count_pre_class / n_class
    final_weights = original_weights * (1-alpha) + uniform_weights * alpha
    weight_pre_sample = final_weights[label]

    return weight_pre_sample


def sample_df_fun(df, label_name, scale):
    """
    According to scale and label_name ,sample dataframe
    Args:
        df (pd.dataframe):
        label_name (str): the normalized label name
        scale (int ,float,list): sampling quantity, sampling ratio, stratified sampling
    Returns:
        dataframe after sample
    """
    def sample_int_float(df, number, seed=11):
        if number == 1.0: # avoid be shuffled
            return df
        if isinstance(number, int ) :
            if number > df.shape[0]:
                print(f'sample number:{number} > actual number :{df.shape[0]}. #set replace = True' )
                replace = True
            else:
                replace = False
            return df.sample(n=number, replace=replace, random_state=seed)

        if isinstance(number, float):
            replace = True if number > 1 else False
            return df.sample(frac=number, replace=replace, random_state=seed)

    def group_sample_fun(group, scale):
        name = group.name
        number = scale[name]
        return sample_int_float(group, number)

    if isinstance (scale, list):
        sample_df = df.groupby(by=label_name, group_keys=False).apply(group_sample_fun, scale)
    else:
        sample_df = sample_int_float(df, scale)

    return sample_df


def basic_read_csv(csv_path, root_dir='', path_col='path', label_col='dr'):
    """
    description:  read csv
    return: image_path: absolute path,shape (N,)
            label: shape (N,)
    """

    csv = pd.read_csv(csv_path)
    image_path = root_dir + csv[path_col] ## Broadcasting
    image_path = image_path.to_numpy()
    if label_col == 'fake':
        label = np.zeros(image_path.shape[0], dtype=np.int32 )
    else:
        label = csv[label_col].to_numpy()

    return image_path, label


def relabel_merge_multiple_dataset(csv_infor):
    """
    according csv_infor ,read ,relabel ,sample, merge csv.
    """
    if isinstance(csv_infor,dict):
        csv_infor = [csv_infor]

    final_df = pd.DataFrame(columns=['path', 'label'])

    for item in csv_infor:
        paths, labels = basic_read_csv(item['csv_path'],
            item['img_dir'], item['path_col'], item['label_col']) # read csv

        data = pd.concat([pd.Series(paths), pd.Series(labels)], axis=1)
        data.columns = ['path', 'label']
        data = data.dropna(how='any')
        data['label'] = data['label'].astype(np.int16)

        if 'label_map' in item:
            map_table = item['label_map']
            for val in data['label'].unique():
                if  val not in map_table:
                    map_table[val] = val
            print('{} use label map \n map is #{}#'.format (item['csv_path'],map_table ) )
            data['label'] = data['label'].map(map_table)    # relabel
            data = data.dropna(how = 'any')  # if we set {-1: None} in label_map ,we drop can drop -1
            data['label'] = data['label'].astype(np.int16)

        data = sample_df_fun(data, 'label', item['scale']) # sample

        final_df = pd.concat([final_df, data], axis=0) # merge

    final_paths = final_df['path'].to_numpy()
    final_labels = final_df['label'].astype(np.int16).to_numpy()
    count_val, count = np.unique(final_labels, return_counts=True)
    print(f'unique val {count_val}, count: {count}.')
    return final_paths, final_labels


def train_loader(csv_infor,
                 tg_size,
                 resize_mode=1,
                 num_samples=100,
                 replacement=False,
                 prob=0.5,
                 aug_m=2,
                 batch_size=10,
                 num_workers=16,
                 num_replicas=1,
                 rank=0,
                 sample_distribution=1.0):
    """
    Args:
        csv_infor (dict or list):  key: 'csv_path' 'img_dir''path_col''label_col' 'scale' 'label_map'
        tg_size (int or list): shape of input image
        resize_mode: (int): the way image will be resized, 0 for directly resize, 1 for fill-with-black to square
        num_samples (int, optional): the data number of one epoch. Defaults to 100.
        replacement (bool, optional): . Defaults to False.
        prob (float, optional):data augmentation probability .range [0,1] Defaults to 0.5.
        aug_m (int, optional): data augmentation magnitude . range [0,3], Defaults to 2.
        batch_size (int, optional):  Defaults to 10.
        num_workers (int, optional):  Defaults to 16.
        num_replicas (int, optional): number of using gpu. Defaults to 1.
        rank (int, optional):  gpu id. Defaults to 0.
        sample_distribution (float, optional): [description]. Defaults to 1.0,uniform distribution.

    Returns:
        training data loader
    """

    final_path, final_label = relabel_merge_multiple_dataset(csv_infor)

    final_weight = get_sample_weight(final_label, sample_distribution)
    sampler = WeightedRandomSampler(weights=final_weight, num_samples=num_samples, replacement=replacement)
    if num_replicas > 1:
        sampler = DistributedProxySampler(sampler, num_replicas=num_replicas, rank=rank)
        print(f'# GPU rank ={rank}, num_dataset={len(sampler)}')

    dataset = BasicDataset(final_path, final_label, tg_size, mode ='train',
        resize_mode=resize_mode, prob=prob, aug_m=aug_m)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
        shuffle=False, pin_memory=True, sampler=sampler, num_workers=num_workers)

    return loader


def basic_loader(csv_infor, tg_size, mode, resize_mode=1, batch_size=10, num_workers=16, num_replicas=1, rank=0):

    paths, labels = relabel_merge_multiple_dataset(csv_infor)

    dataset = BasicDataset(paths, labels, tg_size, mode=mode, resize_mode=resize_mode)
    if num_replicas > 1 :
        sampler = range(labels.shape[0])
        sampler = DistributedProxySampler(sampler, num_replicas=num_replicas, rank=rank)
        print(f'# GPU rank ={rank}, num_dataset={len(sampler)}')
    else:
        sampler = None
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                            sampler=sampler, num_workers=num_workers)
    return loader


if __name__ == "__main__":

    from utils import get_data_infor, all_path_infor
    csv_infor = get_data_infor('cataract_train', 'cataract')
    print(csv_infor)
    a = train_loader(csv_infor=csv_infor, tg_size=512, resize_mode=1,
            num_samples=100, replacement=True, prob=0.5, aug_m=2, batch_size=16,
            num_workers=1, num_replicas=1, rank=0,
            sample_distribution=1.0)
    print(a)
