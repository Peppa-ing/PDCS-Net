import os.path

import SimpleITK as sitk

if __name__ == '__main__':
    # plaque_label_path = '../plaque datasets/label/063-label200-p.nii.gz'
    # vessel_label_path = '../plaque datasets/label/063-label200-v.nii.gz'
    # merge_label_path = '../img'
    # plaque_label = sitk.ReadImage(plaque_label_path, sitk.sitkFloat32)
    # vessel_label = sitk.ReadImage(vessel_label_path, sitk.sitkFloat32)
    # plaque_label_array = sitk.GetArrayFromImage(plaque_label)
    # vessel_label_array = sitk.GetArrayFromImage(vessel_label)
    # merge_label_array = plaque_label_array + vessel_label_array
    # merge_label = sitk.GetImageFromArray(merge_label_array)
    # merge_label.SetSpacing(plaque_label.GetSpacing())
    # merge_label.SetDirection(plaque_label.GetDirection())
    # merge_label.SetOrigin(plaque_label.GetOrigin())
    # sitk.WriteImage(merge_label, os.path.join(merge_label_path, '063-label200-v.nii.gz'))

    path = r'D:\MyCodes\pytorch\Dual_UNet_plaque_segmentation\plaque datasets\image'
    path2 = r'D:\MyCodes\pytorch\Dual_UNet_plaque_segmentation\plaque datasets\label'
    merge_label_path = r'D:\MyCodes\pytorch\Dual_UNet_plaque_segmentation\plaque datasets\labelpv'
    image_lists = os.listdir(path)
    for name in image_lists:
        plaque_label_path = os.path.join(path2, name.replace('image', 'label').replace('.nii.gz', '-p.nii.gz'))
        vessel_label_path = os.path.join(path2, name.replace('image', 'label').replace('.nii.gz', '-v.nii.gz'))
        plaque_label = sitk.ReadImage(plaque_label_path, sitk.sitkFloat32)
        vessel_label = sitk.ReadImage(vessel_label_path, sitk.sitkFloat32)
        plaque_label_array = sitk.GetArrayFromImage(plaque_label)
        vessel_label_array = sitk.GetArrayFromImage(vessel_label)
        merge_label_array = plaque_label_array + 2 * vessel_label_array
        merge_label_array[merge_label_array == 3] = 1
        merge_label = sitk.GetImageFromArray(merge_label_array)
        merge_label.SetSpacing(plaque_label.GetSpacing())
        merge_label.SetDirection(plaque_label.GetDirection())
        merge_label.SetOrigin(plaque_label.GetOrigin())
        sitk.WriteImage(merge_label, os.path.join(merge_label_path, name.replace('image', 'label')))