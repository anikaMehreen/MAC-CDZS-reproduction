import math
import numbers
import random
import warnings
from collections.abc import Sequence
from typing import Tuple, List, Optional
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

class RandomResizedCrop(torch.nn.Module):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)):
        super().__init__()
        self.size = size
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale: List[float], ratio: List[float]) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (array): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        width, height, bands = img.shape
        area = height * width
        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def resized_crop(self, img, i, j, h, w, size):
        img = img[i:i+h, j:j+w, :]
        size_all = (size, size, img.shape[2])

        import cv2
        res = np.zeros(size_all)
        for i in range(img.shape[2]):
            gray_i = img[:, :, i]
            data = cv2.resize(gray_i, dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            res[:, :, i] = data
        return res

    def forward(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        res_img = self.resized_crop(img, i, j, h, w, self.size)
        return res_img

class RandomHorizontalFlip(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def hflip(self, img):
        res = np.zeros_like(img)
        wide = img.shape[1]
        for i in range(wide):
            res[:, wide-1-i, :] = img[:, i, :]
        return res

    def forward(self, img):
        if torch.rand(1) < self.p:
            return self.hflip(img)
        return img

class RandomVerticalFlip(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def vflip(self, img):
        res = np.zeros_like(img)
        wide = img.shape[0]
        for i in range(wide):
            res[wide-1-i, :, :] = img[i, :, :]
        return res

    def forward(self, img):
        if torch.rand(1) < self.p:
            return self.vflip(img)
        return img

class RandomRotation(torch.nn.Module):
    def __init__(self, degres, fill=0):
        super().__init__()
        self.degrees = [float(d) for d in degres]
        self.fill = fill

    @staticmethod
    def get_params(degrees: List[float]) -> float:
        angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
        return angle

    def rotate(self, img, angle, fill):
        res = np.zeros_like(img)
        h, w = img.shape[:2]
        rotate_center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(rotate_center, angle, 1.0)
        new_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
        new_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))
        index_h = (new_h - h) // 2
        index_w = (new_w - w) // 2

        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2
        for b in range(img.shape[2]):
            img_b = img[:, :, b]
            rotated_img = cv2.warpAffine(img_b, M, (new_w, new_h))
            res[:, :, b] = rotated_img[index_h:index_h+h, index_w:index_w+w]

        return res

    def forward(self, img):
        fill = self.fill

        if isinstance(fill, (int, float)):
            fill = [float(fill)] * img.shape[2]
        else:
            fill = [float(f) for f in fill]

        angle = self.get_params(self.degrees)

        res = self.rotate(img, angle, fill)
        return res

def Aug(data):
    b, h, w, c = data.shape
    randomresizedcrop = RandomResizedCrop(h)
    randomhorizontalflip = RandomHorizontalFlip(p=0.5)
    randomVerticalflip = RandomVerticalFlip(p=0.5)
    randomRotation = RandomRotation((0, 90))
    for i in range(b):
        data_i = data[i, :, :, :]
        data_i = randomresizedcrop(data_i)
        data_i = randomhorizontalflip(data_i)
        data_i = randomVerticalflip(data_i)
        data_i = randomRotation(data_i)
        data[i, :, :, :] = data_i
    return data
