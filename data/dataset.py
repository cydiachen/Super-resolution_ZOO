# -*- coding: utf-8 -*-
import torch
import torch.utils.data as data
from os import listdir
import os
from os.path import join
from PIL import Image, ImageOps
import random
import torchvision.transforms as transforms
import cv2
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt


def loadFromFile(path):
    if path is None:
        return None, None

    gts_dir = path
    data = []
    for dirpath, dirnames, filenames in os.walk(gts_dir):
        filenames.sort()

        for file in filenames:
            sr_fullpath = os.path.join(dirpath, file)
            data.append(sr_fullpath)
    return data


def load_lr_sr(file_path, is_mirror=False, is_gray=True, scale=2.0, is_scale_back=False):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)

    if is_mirror and random.randint(0, 1) is 0:
        img = ImageOps.mirror(img)

    # height = 512
    # width = 512
    img = img[0:512, 0:512]

    height, width, channel = img.shape

    img_lr = cv2.resize(img, (int(height / scale), int(width / scale)), interpolation=cv2.INTER_AREA)

    # hym
    if is_gray is False:
        b, g, r = cv2.split(img)
        b_lr, g_lr, r_lr = cv2.split(img_lr)
        img = cv2.merge([r, g, b])
        img_lr = cv2.merge([r_lr, g_lr, b_lr])
    if is_gray is True:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2GRAY)

    if is_scale_back:
        img_lr = cv2.resize(img_lr, (height, width), interpolation=cv2.INTER_CUBIC)
        return img_lr, img
    else:
        return img_lr, img


def totalFile(dir):
    count = 0
    for file in os.listdir(dir):
        if os.path.isfile(os.path.join(dir, file)):
            count += 1
        elif os.path.isdir(os.path.join(dir, file)):
            count += totalFile(os.path.join(dir, file))

    return count


# hym
def random_crop(hr, lr, size, scale):
    h, w = lr.shape[:-1]
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)

    hsize = size * scale
    hx, hy = x * scale, y * scale

    # have to copy before be called by transform function
    crop_lr = lr[y:y + size, x:x + size].copy()
    crop_hr = hr[hy:hy + hsize, hx:hx + hsize].copy()

    return crop_hr, crop_lr


# Data Augmentation BY HYM
def random_flip_and_rotate(im1, im2):
    if random.random() < 0.5:
        im1 = np.flipud(im1)
        im2 = np.flipud(im2)

    if random.random() < 0.5:
        im1 = np.fliplr(im1)
        im2 = np.fliplr(im2)

    angle = random.choice([0, 1, 2, 3])
    im1 = np.rot90(im1, angle)
    im2 = np.rot90(im2, angle)

    # have to copy before be called by transform function
    return im1.copy(), im2.copy()


class ImageDatasetFromFile(data.Dataset):
    def __init__(self, path, is_mirror=False, is_gray=False, upscale=2.0, is_scale_back=False, size=64):
        super(ImageDatasetFromFile, self).__init__()
        self.size = size
        self.path = path
        # hym
        self.upscale = upscale
        self.is_mirror = is_mirror
        self.is_scale_back = is_scale_back
        self.is_gray = is_gray

        self.input_transform = transforms.Compose([
            transforms.ToTensor()])

    def __getitem__(self, idx):

        if self.is_mirror:
            is_mirror = random.randint(0, 1) is 0
        else:
            is_mirror = False

        image_filenames = loadFromFile(self.path)
        fullpath = join(self.path, image_filenames[idx])

        img_lr, img = load_lr_sr(fullpath, self.is_mirror, self.is_gray, self.upscale, self.is_scale_back)
        # hym
        # img_lr, img = random_crop(img_lr, img, self.size, self.upscale)
        # img_lr,img = random_flip_and_rotate(img_lr,img)

        input = self.input_transform(img_lr)
        target = self.input_transform(img)

        return input, target

    def __len__(self):
        return totalFile(self.path)


if __name__ == '__main__':
    for titer, batch in enumerate(train_data_loader):
        input, target = Variable(batch[0]), Variable(batch[1])

        Input = input.permute(0, 2, 3, 1).cpu().data.numpy()
        Target = target.permute(0, 2, 3, 1).cpu().data.numpy()

        plt.figure("Input Image")
        plt.imshow(Input[0, :, :, :])
        plt.axis('on')
        plt.title('image')
        plt.show()

        plt.figure("Target Image")
        plt.imshow(Target[0, :, :, :])
        plt.axis('on')
        plt.title('Target')
        plt.show()
