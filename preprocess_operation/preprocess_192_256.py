import os
import numpy as np
import SimpleITK as sitk
import configs

# 归一化 Z-score标准化方法
# 这种方法给予原始数据的均值（mean）和标准差（standard deviation）进行数据的标准化
# 经过处理的数据符合标准正态分布，即均值为0，标准差为1
def normalize_data(data):
    # data = (data - data.min()) / (data.max() - data.min())
    # data[data < 0] = 0
    # data[data > 1] = 1
    # data = (data * 255).astype('uint8')
    data = np.array(data, dtype=np.float32)
    means = data.mean()
    stds = data.std()
    data -= means
    data /= stds
    return data

def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()
    originSpacing = itkimage.GetSpacing()
    newSize = np.array(newSize, float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int32)
    resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())  # 设置输出的大小
    resampler.SetOutputSpacing(newSpacing.tolist())  # 设置输出图像间距
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)
    return itkimgResampled

def fix_data(raw_dataset_path, fixed_dataset_path):
    if not os.path.exists(fixed_dataset_path):
        os.makedirs(os.path.join(fixed_dataset_path, 'image'))
        os.makedirs(os.path.join(fixed_dataset_path, 'label'))
    file_list = os.listdir(os.path.join(raw_dataset_path, 'image'))
    Numbers = len(file_list)
    print('Total numbers of samples is :', Numbers)
    for i, filename in enumerate(file_list):
        print("==== {} | {}/{} ====".format(filename, i + 1, Numbers))
        # 001-image001.nii.gz   001-label001-p.nii.gz、001-label001-v.nii.gz
        image_path = os.path.join(raw_dataset_path, 'image', filename)
        labelp_path = os.path.join(raw_dataset_path, 'label', filename.replace('image', 'label').replace('.nii.gz', '-p.nii.gz'))
        labelv_path = os.path.join(raw_dataset_path, 'label', filename.replace('image', 'label').replace('.nii.gz', '-v.nii.gz'))
        new_image, new_labelp, new_labelv = process(image_path, labelp_path, labelv_path)
        if new_image != None and new_labelp != None and new_labelv != None:
            sitk.WriteImage(new_image, os.path.join(fixed_dataset_path, 'image', filename))
            sitk.WriteImage(new_labelp, os.path.join(fixed_dataset_path, 'label', filename.replace('image', 'label').replace('.nii.gz', '-p.nii.gz')))
            sitk.WriteImage(new_labelv, os.path.join(fixed_dataset_path, 'label', filename.replace('image', 'label').replace('.nii.gz', '-v.nii.gz')))

#             原图路径     label路径
def process(image_path, labelp_path, labelv_path):
    image = sitk.ReadImage(image_path, sitk.sitkFloat32)
    image_array = sitk.GetArrayFromImage(image)  # (1, 768, 1024) or (1, 768, 1024, 3)
    labelp = sitk.ReadImage(labelp_path, sitk.sitkFloat32)
    labelp_array = sitk.GetArrayFromImage(labelp)
    labelv = sitk.ReadImage(labelv_path, sitk.sitkFloat32)
    labelv_array = sitk.GetArrayFromImage(labelv)
    # label_array = label_array[0, :, :]  # (768, 1024)
    print("Ori shape:", image_array.shape, labelp_array.shape, labelv_array.shape)  # 1 * 768 * 1024

    # 保存为对应的格式
    new_image = sitk.GetImageFromArray(image_array)
    new_image.SetSpacing(image.GetSpacing())
    new_image.SetOrigin(image.GetOrigin())
    new_image.SetDirection(image.GetDirection())
    new_labelp = sitk.GetImageFromArray(labelp_array)
    new_labelp.SetSpacing(image.GetSpacing())
    new_labelp.SetOrigin(image.GetOrigin())
    new_labelp.SetDirection(image.GetDirection())
    new_labelv = sitk.GetImageFromArray(labelv_array)
    new_labelv.SetSpacing(image.GetSpacing())
    new_labelv.SetOrigin(image.GetOrigin())
    new_labelv.SetDirection(image.GetDirection())
    # 图像与标签resize 1 768 1024 重采样resample   mask用最近邻插值，CT/MRI图像用线性插值  (512, 384, 1)  (1024, 768, 1)  (256, 192, 1)  (128, 96, 1)
    resize_image = resize_image_itk(new_image, (256, 192, 1), resamplemethod=sitk.sitkLinear)
    resize_labelp = resize_image_itk(new_labelp, (256, 192, 1), resamplemethod=sitk.sitkNearestNeighbor)
    resize_labelv = resize_image_itk(new_labelv, (256, 192, 1), resamplemethod=sitk.sitkNearestNeighbor)
    # 输出当前shape
    new_image = sitk.GetArrayFromImage(resize_image)
    new_labelp = sitk.GetArrayFromImage(resize_labelp)
    new_labelv = sitk.GetArrayFromImage(resize_labelv)
    print("Preprocessed shape:", new_image.shape, new_labelp.shape, new_labelv.shape)  # (1, 768, 1024)
    # 图像标准化
    resize_image_array = sitk.GetArrayFromImage(resize_image)  # (1, 768, 1024)
    nor_resize_image_array = normalize_data(resize_image_array)
    nor_resize_image = sitk.GetImageFromArray(nor_resize_image_array)
    nor_resize_image.SetSpacing(resize_image.GetSpacing())
    nor_resize_image.SetOrigin(resize_image.GetOrigin())
    nor_resize_image.SetDirection(resize_image.GetDirection())

    return nor_resize_image, resize_labelp, resize_labelv

if __name__ == '__main__':
    args = configs.args
    # 输入数据集路径
    raw_dataset_path = args.original_datasets_path
    # 预处理后的数据集的输出路径
    fixed_dataset_path = args.precessed_datasets_path
    # fixed_dataset_path = args.precessed_datasets_path + '_192_256'

    # 对原始数据集进行初步处理
    fix_data(raw_dataset_path, fixed_dataset_path)  # 对原始图像进行修剪并保存