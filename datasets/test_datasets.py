import numpy as np
import os

import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, DataLoader

import configs


"""
    Linux系统需要修改以下参数
        name中的分割符号 将'\\'改为'/'
"""


class Test_Dataset(Dataset):
    def __init__(self, args, index=''):
        self.args = args
        self.file_path_list = self.get_file_path(os.path.join(args.root_path, 'test_path_list.txt'), index)

    def __getitem__(self, index):
        # 'D:\MyCodes\pytorch\Dual_UNet_plaque_segmentation\processed_datasets\image\001-image001.nii.gz'
        image_path = self.file_path_list[index][0]
        labelp_path = self.file_path_list[index][1]
        labelv_path = self.file_path_list[index][2]

        # 'image-9273362.nii.gz'
        name = image_path.split('/')[-1]

        image = sitk.ReadImage(image_path, sitk.sitkFloat32)
        labelp = sitk.ReadImage(labelp_path, sitk.sitkFloat32)
        labelv = sitk.ReadImage(labelv_path, sitk.sitkFloat32)

        image_array = sitk.GetArrayFromImage(image)
        labelp_array = sitk.GetArrayFromImage(labelp)
        labelv_array = sitk.GetArrayFromImage(labelv)

        # 将array转换为tensor 增加一个维度
        image_array = torch.FloatTensor(image_array)  # [1, 768, 1024]
        labelp_array = torch.FloatTensor(labelp_array).squeeze(0)  # [768, 1024]
        labelv_array = torch.FloatTensor(labelv_array).squeeze(0)  # [768, 1024]

        return image_array, labelp_array, labelv_array, name

    def __len__(self):
        return len(self.file_path_list)

    def get_file_path(self, file_path, index):  # D:\MyCodes\pytorch\Dual_UNet_plaque_segmentation\test_path_list.txt
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

# if __name__ == "__main__":
#     args = configs.args
#     test_datasets = Test_Dataset(args)
#     # 定义数据加载
#     test_dl = DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=0)
#     test_data_iter = iter(test_dl)
#     images, labels_p, labels_v, name = test_data_iter.next()
#     print(1)


