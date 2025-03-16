import random

import SimpleITK
import cv2
import numpy
import numpy as np
import torch
from scipy import ndimage
from torchvision import transforms


class RandomFlip_LR:  # x方向上翻转
    def __init__(self, prob=0.5):
        self.prob = prob

    def _flip(self, image, prob):  # [1, 96, 128]  [z, y, x]
        if prob <= self.prob:
            image = image.flip(2)
        return image

    def __call__(self, image, labelp, labelv):
        prob = random.uniform(0, 1)
        return self._flip(image, prob), self._flip(labelp, prob), self._flip(labelv, prob)


class RandomFlip_UD:  # y方向上翻转
    def __init__(self, prob=0.5):
        self.prob = prob

    def _flip(self, image, prob):  # [1, 96, 128]  [z, y, x]
        if prob <= self.prob:
            image = image.flip(1)
        return image

    def __call__(self, image, labelp, labelv):
        prob = random.uniform(0, 1)
        return self._flip(image, prob), self._flip(labelp, prob), self._flip(labelv, prob)


class RandomRotate:
    def __init__(self, count=0):  # count = 0, 1, 2, 3  ==>  0°, 90°, 180°, 270°
        self.count = count

    # 将矩阵img逆时针旋转（90×k）°以后返回rot_img，k取负数时表示顺时针旋转。
    # rot_img = np.rot90(image, k=2)
    def _rotate(self, image, count):  # [1, 96, 128]
        image = torch.rot90(image, count, [1, 2])
        return image

    def __call__(self, image, labelp, labelv):
        # count = self.count
        # count = np.random.randint(0, 4)
        prob = random.uniform(0, 1)
        if prob <= 0.5:
            count = 2
        else:
            count = 0
        return self._rotate(image, count), self._rotate(labelp, count), self._rotate(labelv, count)


class RandomRotateAngle:  # 逆时针旋转-10°~10°
    def __init__(self, prob=0.5):
        self.prob = prob

    def _rotate(self, image, prob, angle):  # [1, 96, 128]
        if prob <= self.prob:
            image = image.numpy()
            c, h, w = image.shape[0], image.shape[1], image.shape[2]
            # 旋转angle角度  -30表示逆时针旋转30度
            new_image = ndimage.rotate(image[0], angle, order=0, reshape=False)
            # 放射变化的方法进行旋转
            # m = cv2.getRotationMatrix2D(center=(w // 2, h // 2), angle=-angle, scale=1.0)
            # new_image = cv2.warpAffine(src=image[0], M=m, dsize=(w, h), borderValue=(0, 0, 0))
            new_image_array = new_image[np.newaxis, :, :]
            new_image_array = torch.FloatTensor(new_image_array)
            image = new_image_array
            # SimpleITK.WriteImage(SimpleITK.GetImageFromArray(image), s+str(int(prob*1000))+'.nii.gz')
        return image

    def __call__(self, image, labelp, labelv):
        # angle = self.angle
        prob = random.uniform(0, 1)
        angle = np.random.randint(-30, 30)
        return self._rotate(image, prob, angle), self._rotate(labelp, prob, angle), self._rotate(labelv, prob, angle)


class LeftDownMove:  # 右下角平移
    def __init__(self, left=30, down=30):
        self.left = left
        self.down = down

    def _rotate(self, image, prob, left, down):  # [1, 96, 128]
        if prob <= 0.5:
            image = image.numpy()
            c, h, w = image.shape[0], image.shape[1], image.shape[2]
            # 使用numpy构建移动矩阵
            m = np.float32([[1, 0, left], [0, 1, down]])
            # 第一个参数为原图像，第二个参数为移动矩阵，可以自定义，第三个参数为输出图像大小
            new_image = cv2.warpAffine(src=image[0], M=m, dsize=(w, h), borderValue=(0, 0, 0))
            new_image_array = new_image[np.newaxis, :, :]
            new_image_array = torch.FloatTensor(new_image_array)
            image = new_image_array
        return image

    def __call__(self, image, labelp, labelv):
        prob = random.uniform(0, 1)
        left = np.random.randint(0, self.left)
        down = np.random.randint(0, self.down)
        return self._rotate(image, prob, left, down), self._rotate(labelp, prob, left, down), self._rotate(labelv, prob, left, down)

class ToTensor:
    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image, labelp, labelv):
        image = self.to_tensor(image)
        labelp = self.to_tensor(labelp)
        labelv = self.to_tensor(labelv)
        return image, labelp, labelv


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, labelp, labelv):
        for t in self.transforms:
            image, labelp, labelv = t(image, labelp, labelv)
        return image, labelp, labelv
