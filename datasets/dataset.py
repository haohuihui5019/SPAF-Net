import itertools

import cv2
from PIL import Image
import torch.utils.data as data
import os
from glob import glob
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
import random
import numpy as np
import h5py
import math
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomGrayscale


def modify_matrices(seg_cls_target):
    # ...（同上，转换为numpy数组和复制矩阵的步骤）

    # 创建一个与输入矩阵形状相同的零矩阵
    if len(seg_cls_target) != 3:
        raise ValueError("seg_cls_target must contain exactly 3 matrices")

        # 获取第一个矩阵的形状（假设所有矩阵形状相同）
    rows, cols = seg_cls_target[0].shape

    result_matrix = np.zeros_like(seg_cls_target[0])

    # 遍历矩阵的每个位置

    for i in range(rows):

        for j in range(cols):

            # 找到非零值（如果有的话）

            values = [seg_cls_target[0][i, j], seg_cls_target[1][i, j], seg_cls_target[2][i, j]]

            non_zero_values = [v for v in values if v != 0]

            # 如果有非零值，则取第一个非零值（或根据需要处理）

            if non_zero_values:
                result_matrix[i, j] = non_zero_values[0]

    return result_matrix

def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w


class Crowd(data.Dataset):
    def __init__(self, root, crop_size, downsample_ratio, method='train', info=None):
        png_files = glob(os.path.join(root, 'img', '*.png'))
        img_files = glob(os.path.join(root, 'img', '*.jpg'))
        all_files = png_files + img_files
        self.im_list = sorted(all_files)

        if method not in ['train', 'val']:
            raise Exception('Method is not implemented!')
        self.label_list = []
        if method == 'train':
            try:
                with open(info) as f:
                    for i in f:
                        image_name = i[:-1]
                        self.label_list.append(image_name)
            except:
                raise Exception("please give right info")

            labeled = []
            for i in self.im_list:

                if os.path.basename(i) in self.label_list:

                    labeled.append(1)

                else:

                    labeled.append(0)

            labeled = np.array(labeled)

            self.labeled_idx = np.where(labeled == 1)[0]

            self.unlabeled_idx = np.where(labeled == 0)[0]

        self.c_size = crop_size
        self.d_ratio = downsample_ratio
        self.root = root
        self.method = method
        assert self.c_size % self.d_ratio == 0
        self.w_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.s_transform = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):

        im_path = self.im_list[item]
        name = os.path.basename(im_path).split('.')[0]
        # gd_path = os.path.join(self.root, 'gt_points', '{}.npy'.format(name))
        img = Image.open(im_path).convert('RGB')
        # keypoints = np.load(gd_path)

        if self.method == 'train':
            den_map = []
            if os.path.exists(os.path.join(self.root, 'gt_den', '{}_bud.npy'.format(name))):
                den_map_bud_path = os.path.join(self.root, 'gt_den', '{}_bud.npy'.format(name))
            else :
                width, height = img.size
                zero_array = np.zeros((width, height))
                # 将数组保存为.npy文件
                np.save(os.path.join(self.root, 'gt_den', '{}_bud.npy'.format(name)), zero_array)
                den_map_bud_path = os.path.join(self.root, 'gt_den', '{}_bud.npy'.format(name))

            if os.path.exists(os.path.join(self.root, 'gt_den', '{}_bloom.npy'.format(name))):

                den_map_bloom_path = os.path.join(self.root, 'gt_den', '{}_bloom.npy'.format(name))
            else :
                width, height = img.size
                zero_array = np.zeros((width, height))

                # 将数组保存为.npy文件
                np.save(os.path.join(self.root, 'gt_den', '{}_bloom.npy'.format(name)), zero_array)
                den_map_bloom_path = os.path.join(self.root, 'gt_den', '{}_bloom.npy'.format(name))
            if os.path.exists(os.path.join(self.root, 'gt_den', '{}_faded.npy'.format(name))):

                den_map_faded_path = os.path.join(self.root, 'gt_den', '{}_faded.npy'.format(name))
            else :
                width, height = img.size
                zero_array = np.zeros((width, height))

                # 将数组保存为.npy文件
                np.save(os.path.join(self.root, 'gt_den', '{}_faded.npy'.format(name)), zero_array)
                den_map_faded_path = os.path.join(self.root, 'gt_den', '{}_faded.npy'.format(name))

            den_map.append(np.load(den_map_bud_path))
            den_map.append(np.load(den_map_bloom_path))
            den_map.append(np.load(den_map_faded_path))

            label = (os.path.basename(im_path) in self.label_list)
            return self.train_transform_density_map(img, den_map, label ,name)

        elif self.method == 'val':
            w, h = img.size
            new_w = math.ceil(w / 32) * 32
            new_h = math.ceil(h / 32) * 32
            img = img.resize((new_w, new_h), Image.BICUBIC)
            if os.path.exists(os.path.join(self.root, 'gt_den', '{}_bud.npy'.format(name))):
                den_map_bud_path = os.path.join(self.root, 'gt_den', '{}_bud.npy'.format(name))
            else:
                width, height = img.size
                zero_array = np.zeros((width, height))
                # 将数组保存为.npy文件
                np.save(os.path.join(self.root, 'gt_den', '{}_bud.npy'.format(name)), zero_array)
                den_map_bud_path = os.path.join(self.root, 'gt_den', '{}_bud.npy'.format(name))
            if os.path.exists(os.path.join(self.root, 'gt_den', '{}_bloom.npy'.format(name))):
                den_map_bloom_path = os.path.join(self.root, 'gt_den', '{}_bloom.npy'.format(name))
            else:
                width, height = img.size
                zero_array = np.zeros((width, height))

                # 将数组保存为.npy文件
                np.save(os.path.join(self.root, 'gt_den', '{}_bloom.npy'.format(name)), zero_array)
                den_map_bloom_path = os.path.join(self.root, 'gt_den', '{}_bloom.npy'.format(name))
            if os.path.exists(os.path.join(self.root, 'gt_den', '{}_faded.npy'.format(name))):
                den_map_faded_path = os.path.join(self.root, 'gt_den', '{}_faded.npy'.format(name))
            else:
                width, height = img.size
                zero_array = np.zeros((width, height))

                # 将数组保存为.npy文件
                np.save(os.path.join(self.root, 'gt_den', '{}_faded.npy'.format(name)), zero_array)
                den_map_faded_path = os.path.join(self.root, 'gt_den', '{}_faded.npy'.format(name))
            den_map_bud = np.load(den_map_bud_path)
            den_map_bloom = np.load(den_map_bloom_path)
            den_map_faded = np.load(den_map_faded_path)

            den_map_bud = den_map_bud.astype(np.float32, copy=False)
            den_map_bud = Image.fromarray(den_map_bud)
            damp_transform = ToTensor()
            den_map_bud = damp_transform(den_map_bud)

            den_map_bloom = den_map_bloom.astype(np.float32, copy=False)
            den_map_bloom = Image.fromarray(den_map_bloom)
            damp_transform = ToTensor()
            den_map_bloom = damp_transform(den_map_bloom)

            den_map_faded = den_map_faded.astype(np.float32, copy=False)
            den_map_faded = Image.fromarray(den_map_faded)
            damp_transform = ToTensor()
            den_map_faded = damp_transform(den_map_faded)
            den_map = []
            den_map.append(den_map_bud.sum())
            den_map.append(den_map_bloom.sum())
            den_map.append(den_map_faded.sum())
            return self.w_transform(img), den_map, name
            # return {'image': self.w_transform(img),
            #         'gt_counts': den_map.sum(),
            #         'imagename': name}

    def train_transform_density_map(self, img, den_map, label, name):

        wd, ht = img.size
        if random.random() > 0.88:
            img = img.convert('L').convert('RGB')
        # re_size = random.random() * 0.5 + 0.75
        # re_size = random.random() * 0.5 + 0.75
        # wdd = (int)(wd * re_size)
        # htt = (int)(ht * re_size)
        # if min(wdd, htt) >= self.c_size:
        #     wd = wdd
        #     ht = htt
        #     img = img.resize((wd, ht))
        #     den_map[0] = cv2.resize(den_map[0][:, :], (wd, ht), interpolation=cv2.INTER_CUBIC) / (re_size ** 2)
        #     den_map[1] = cv2.resize(den_map[1][:, :], (wd, ht), interpolation=cv2.INTER_CUBIC) / (re_size ** 2)
        #     den_map[2] = cv2.resize(den_map[2][:, :], (wd, ht), interpolation=cv2.INTER_CUBIC) / (re_size ** 2)
        # st_size = min(wd, ht)
        # assert st_size >= self.c_size
        # i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        # img = F.crop(img, i, j, h, w)
        #
        # den_map[0] = den_map[0][i: (i + h), j: (j + w)]
        h, w = den_map[0].shape
        den_map[0] = den_map[0].reshape([h // self.d_ratio, self.d_ratio, w // self.d_ratio, self.d_ratio]).sum(axis=(1, 3))
        # den_map[1] = den_map[1][i: (i + h), j: (j + w)]
        den_map[1] = den_map[1].reshape([h // self.d_ratio, self.d_ratio, w // self.d_ratio, self.d_ratio]).sum(axis=(1, 3))
        # den_map[2] = den_map[2][i: (i + h), j: (j + w)]
        den_map[2] = den_map[2].reshape([h // self.d_ratio, self.d_ratio, w // self.d_ratio, self.d_ratio]).sum(axis=(1, 3))

        if random.random() > 0.5:
            img = F.hflip(img)
            den_map[0] = np.fliplr(den_map[0])
            den_map[1] = np.fliplr(den_map[1])
            den_map[2] = np.fliplr(den_map[2])
        seg_cls_target = (den_map[0] > 0) * 1
        # seg_cls_target.append((den_map[0] > 0) * 1)
        # seg_cls_target.append((den_map[1] > 0) * 2)
        # seg_cls_target.append((den_map[2] > 0) * 3)
        # seg_cls_target = modify_matrices(seg_cls_target)
        return self.w_transform(img), self.s_transform(img), torch.from_numpy(den_map[0].copy()).float().unsqueeze(
            0), torch.from_numpy(den_map[1].copy()).float().unsqueeze(0), torch.from_numpy(den_map[2].copy()).float().unsqueeze(
            0), torch.from_numpy(seg_cls_target.copy()).to(torch.int64).unsqueeze(0), label, name

class TwoStreamBatchSampler(data.Sampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size
        # print(len(self.primary_indices),self.primary_batch_size)

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
