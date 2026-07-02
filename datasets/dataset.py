from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image

import random
import h5py
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from scipy import ndimage
from PIL import Image


class NPY_datasets(Dataset):
    def __init__(self, path_Data, config, train=True):
        super(NPY_datasets, self)
        if train:
            images_list = sorted(os.listdir(path_Data + 'train/images/'))
            masks_list = sorted(os.listdir(path_Data + 'train/masks/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data + 'train/images/' + images_list[i]
                mask_path = path_Data + 'train/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = config.train_transformer
        else:
            images_list = sorted(os.listdir(path_Data + 'val/images/'))
            masks_list = sorted(os.listdir(path_Data + 'val/masks/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data + 'val/images/' + images_list[i]
                mask_path = path_Data + 'val/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = config.test_transformer

    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        img = np.array(Image.open(img_path).convert('RGB'))
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255
        img, msk = self.transformer((img, msk))
        return img, msk

    def __len__(self):
        return len(self.data)


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def random_rotate_lidc(image, label):
    angle = np.random.uniform(-15.0, 15.0)
    image = ndimage.rotate(image, angle, order=1, reshape=False, mode='nearest')
    label = ndimage.rotate(label, angle, order=0, reshape=False, mode='nearest')
    return image, label


def random_scale_lidc(image, label, scale_range=(0.9, 1.1)):
    scale = np.random.uniform(scale_range[0], scale_range[1])
    h, w = image.shape
    scaled_h = max(2, int(round(h * scale)))
    scaled_w = max(2, int(round(w * scale)))

    image_scaled = zoom(image, (scaled_h / h, scaled_w / w), order=1)
    label_scaled = zoom(label, (scaled_h / h, scaled_w / w), order=0)

    if scaled_h >= h and scaled_w >= w:
        start_y = (scaled_h - h) // 2
        start_x = (scaled_w - w) // 2
        image_scaled = image_scaled[start_y:start_y + h, start_x:start_x + w]
        label_scaled = label_scaled[start_y:start_y + h, start_x:start_x + w]
    else:
        pad_top = (h - scaled_h) // 2
        pad_bottom = h - scaled_h - pad_top
        pad_left = (w - scaled_w) // 2
        pad_right = w - scaled_w - pad_left
        image_scaled = np.pad(
            image_scaled,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=0,
        )
        label_scaled = np.pad(
            label_scaled,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=0,
        )

    return image_scaled, label_scaled


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class LIDCCropGenerator(object):
    def __init__(self, output_size, rotation_prob=0.4, scale_prob=0.4):
        self.output_size = output_size
        self.rotation_prob = rotation_prob
        self.scale_prob = scale_prob

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() < self.rotation_prob:
            image, label = random_rotate_lidc(image, label)
        if random.random() < self.scale_prob:
            image, label = random_scale_lidc(image, label)

        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=1)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        image = torch.from_numpy(np.ascontiguousarray(image.astype(np.float32))).unsqueeze(0)
        label = torch.from_numpy(np.ascontiguousarray((label > 0).astype(np.int64)))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name + '.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        # label = sample['label']
        # print(f"2222222222 Target min: {label.min()}, Target max: {label.max()}")
        return sample


class ACDC_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train" or self.split == "valid":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, self.split, slice_name)
            data = np.load(data_path)
            image, label = data['img'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}".format(vol_name)
            data = np.load(filepath)
            image, label = data['img'], data['label']

        sample = {'image': image, 'label': label}
        if self.transform and self.split == "train":
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample


class LIDCROIDataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
        self.data_dir = os.path.join(base_dir, self.split)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample_name = self.sample_list[idx].strip('\n')
        data_path = os.path.join(self.data_dir, sample_name)
        data = np.load(data_path)
        image, label = data['img'], data['label']

        sample = {'image': image, 'label': label}
        if self.transform and self.split == "train":
            sample = self.transform(sample)
        else:
            image = torch.from_numpy(np.ascontiguousarray(image.astype(np.float32))).unsqueeze(0)
            label = torch.from_numpy(np.ascontiguousarray((label > 0).astype(np.int64)))
            sample = {'image': image, 'label': label}
        sample['case_name'] = sample_name
        return sample

