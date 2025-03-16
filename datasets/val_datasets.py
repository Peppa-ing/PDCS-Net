import numpy as np
import os

import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
from utils.transforms import Compose, RandomFlip_LR, RandomFlip_UD, RandomRotate, RandomRotateAngle, LeftDownMove


class Val_Dataset(Dataset):
    def __init__(self, args, index=''):
        self.args = args
        self.file_path_list = self.get_file_path(os.path.join(args.root_path, 'val_path_list.txt'), index)
        # 预处理
        self.transforms = Compose([
            RandomRotate(count=2),  # 随机旋转0° 90° 180° 270°
            RandomFlip_LR(prob=0.5),  # 左右翻转
            RandomFlip_UD(prob=0.5),  # 上下旋转
            RandomRotateAngle(prob=0.5),  # 逆时针旋转30°
            # LeftDownMove(left=30, down=30)  # 右下角平移
        ])

    def __getitem__(self, index):
        # 'D:\MyCodes\pytorch\Dual_UNet_plaque_segmentation\processed_datasets\image\001-image001.nii.gz'
        image_path = self.file_path_list[index][0]
        labelp_path = self.file_path_list[index][1]
        labelv_path = self.file_path_list[index][2]

        image = sitk.ReadImage(image_path, sitk.sitkFloat32)
        labelp = sitk.ReadImage(labelp_path, sitk.sitkFloat32)
        labelv = sitk.ReadImage(labelv_path, sitk.sitkFloat32)

        image_array = sitk.GetArrayFromImage(image)
        labelp_array = sitk.GetArrayFromImage(labelp)
        labelv_array = sitk.GetArrayFromImage(labelv)

        # 将array转换为tensor 增加一个维度
        image_array = torch.FloatTensor(image_array)  # [1, 768, 1024]
        labelp_array = torch.FloatTensor(labelp_array)  # [1, 768, 1024]
        labelv_array = torch.FloatTensor(labelv_array)  # [1, 768, 1024]
        # image_array, labelp_array, labelv_array = self.transforms(image_array, labelp_array, labelv_array)

        return image_array, labelp_array.squeeze(0), labelv_array.squeeze(0)  # [1, 768, 1024] [768, 1024] [768, 1024]

        # # 将array转换为tensor 增加一个维度
        # image_array = torch.FloatTensor(image_array)  # [1, 96, 128, 128]
        # label_array = torch.FloatTensor(label_array).squeeze(0)  # [96, 128, 128]
        #
        # return image_array, label_array

    def __len__(self):
        return len(self.file_path_list)

    def get_file_path(self, file_path, index):  # D:\MyCodes\pytorch\Dual_UNet_plaque_segmentation\val_path_list.txt
        file_path_list = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:  # '001-image001.nii.gz\n'
                image_name = line.strip()  # '001-image001.nii.gz'
                # 'D:\MyCodes\pytorch\Dual_UNet_plaque_segmentation\processed_datasets\image\001-image001.nii.gz'
                # 'D:\MyCodes\pytorch\Dual_UNet_plaque_segmentation\processed_datasets\label\001-label001-p.nii.gz'
                # 'D:\MyCodes\pytorch\Dual_UNet_plaque_segmentation\processed_datasets\label\001-label001-v.nii.gz'
                if index != '':
                    image_path = os.path.join(self.args.precessed_datasets_path + index, 'image', image_name)
                    labelp_path = os.path.join(self.args.precessed_datasets_path + index, 'label', image_name.replace('image', 'label').replace('.nii.gz', '-p.nii.gz'))
                    labelv_path = os.path.join(self.args.precessed_datasets_path + index, 'label', image_name.replace('image', 'label').replace('.nii.gz', '-v.nii.gz'))
                else:
                    image_path = os.path.join(self.args.precessed_datasets_path, 'image', image_name)
                    labelp_path = os.path.join(self.args.precessed_datasets_path, 'label', image_name.replace('image', 'label').replace('.nii.gz', '-p.nii.gz'))
                    labelv_path = os.path.join(self.args.precessed_datasets_path, 'label', image_name.replace('image', 'label').replace('.nii.gz', '-v.nii.gz'))
                file_path_list.append([image_path, labelp_path, labelv_path])
        return file_path_list