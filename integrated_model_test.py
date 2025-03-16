import logging
import os
import sys
import time

import cv2
import numpy as np

import SimpleITK as sitk
import torch
import xlwt
from medpy import metric
from torch.utils.data import DataLoader
from tqdm import tqdm

import configs
from datasets.test_datasets import Test_Dataset
from model.DAUNet import DAUNet
from model.DSAUNet import DSAUNet
from model.Dual_Attention_UNet import Dual_Attention_UNet
from model.Dual_UNet import Dual_UNet
from model.IUNet import IUNet, IUNet2, IUNet3
from model.MDAUNet import MDAUNet
from model.MSFFUNet import MSFFUNet
from model.PDASNet import PDASNet

# 归一化 Z-score标准化方法
# 这种方法给予原始数据的均值（mean）和标准差（standard deviation）进行数据的标准化
# 经过处理的数据符合标准正态分布，即均值为0，标准差为1
from model.SDNet import SDNet


def normalize_data(data):
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
    newSize = newSize.astype(np.int32)  # spacing肯定不能是整数
    resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())  # 设置输出的大小
    resampler.SetOutputSpacing(newSpacing.tolist())  # 设置输出图像间距
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)
    return itkimgResampled


def test(image_array, net):
    # 测试模式 让网络进入一个测试状态
    net.eval()
    images = torch.FloatTensor(image_array).unsqueeze(0)  # (1, 1, 96, 128)
    with torch.no_grad():
        # 将数据拷贝到device中
        images = images.to(device)
        time_start = time.time()
        preds_p1, preds_v1, preds_p2, preds_v2 = net(images)
        time_end = time.time()
        time_sum = time_end - time_start
        print(name + ": " + str(time_sum))

        # # sigmoid将一个real value映射到（0, 1）的区间，用来做二分类
        preds_p1 = torch.sigmoid(preds_p1)  # 值域0-1之间
        preds_v1 = torch.sigmoid(preds_v1)  # 值域0-1之间
        preds_p1 = (preds_p1 >= 0.5).float()  # (1, 1, 768, 1024)
        preds_v1 = (preds_v1 >= 0.5).float()  # (1, 1, 768, 1024)
        preds_p2 = torch.sigmoid(preds_p2)  # 值域0-1之间
        preds_v2 = torch.sigmoid(preds_v2)  # 值域0-1之间
        preds_p2 = (preds_p2 >= 0.5).float()  # (1, 1, 768, 1024)
        preds_v2 = (preds_v2 >= 0.5).float()  # (1, 1, 768, 1024)

        preds_p = preds_p1 + preds_p2
        preds_p[preds_p == 2] = 1
        preds_v = preds_v1 + preds_v2
        preds_v[preds_v == 2] = 1

        preds_p = preds_p.squeeze(dim=0)  # (1, 192, 256)
        preds_p = np.asarray(preds_p.cpu().numpy(), dtype='uint8')

        preds_v = preds_v.squeeze(dim=0)  # (1, 192, 256)
        preds_v = np.asarray(preds_v.cpu().numpy(), dtype='uint8')
    return preds_p, preds_v, time_sum


if __name__ == '__main__':
    args = configs.args
    coarse_weights = 'experiments/PDCSNet/best_model.pth'  # 权重地址
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 加载网络
    net = IUNet3(in_channels=1, out_channels=1)
    # 将网络拷贝到device中
    net.to(device=device)
    net.load_state_dict(torch.load(coarse_weights))
    # 测试步骤开始 保证不会对这部分进行调优 没有梯度
    # p_total_accuracy = 0.0
    # v_total_accuracy = 0.0
    p_total_precision = 0.0
    v_total_precision = 0.0
    p_total_recall = 0.0
    v_total_recall = 0.0
    # p_total_acc_pixels = 0.0
    # v_total_acc_pixels = 0.0
    p_total_pre_pixels = 0.0
    v_total_pre_pixels = 0.0
    p_total_rec_pixels = 0.0
    v_total_rec_pixels = 0.0
    # dice_loss = 0.0
    # miou = 0.0

    test_image_path = 'test_path_list.txt'
    test_loader = []
    with open(test_image_path, 'r', encoding='utf-8') as f:
        for line in f:  # '001-image001.nii.gz\n'
            image_name = line.strip()  # '001-image001.nii.gz'
            test_loader.append(image_name)

    p_dice_dict = {}
    v_dice_dict = {}
    p_iou_dict = {}
    v_iou_dict = {}
    p_hd_dict = {}
    v_hd_dict = {}
    p_sen_dict = {}
    v_sen_dict = {}
    p_pre_dict = {}
    v_pre_dict = {}
    num_dict = {}
    avg_time = []

    logging.basicConfig(filename='./test_log.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("{} test iterations per epoch".format(len(test_loader)))

    # 使用xlwt生成xls的excel文件
    workbook = xlwt.Workbook(encoding='utf-8')
    p_sheet = workbook.add_sheet('测试集斑块指标')
    v_sheet = workbook.add_sheet('测试集血管指标')

    p_columns = ['image_name', 'plaque_dice', 'plaque_iou', 'plaque_hd95', 'plaque_sen', 'plaque_pre']
    v_columns = ['image_name', 'vessel_dice', 'vessel_iou', 'vessel_hd95', 'vessel_sen', 'vessel_pre']

    for col, column in enumerate(p_columns):
        p_sheet.write(0, col, column)

    for col, column in enumerate(v_columns):
        v_sheet.write(0, col, column)

    for i, name in tqdm(enumerate(test_loader), total=len(test_loader)):
        index = name.split('-image')[0]
        if index not in num_dict:
            num_dict[index] = 0
            p_dice_dict[index] = 0.0
            v_dice_dict[index] = 0.0
            p_iou_dict[index] = 0.0
            v_iou_dict[index] = 0.0
            p_hd_dict[index] = 0.0
            v_hd_dict[index] = 0.0
            p_sen_dict[index] = 0.0
            v_sen_dict[index] = 0.0
            p_pre_dict[index] = 0.0
            v_pre_dict[index] = 0.0

        # 读取测试集原图和对应标签图 001-image001-p.nii.gz   001-label001-p.nii.gz
        image_path = os.path.join(args.original_datasets_path, 'image', name)
        labelp_path = os.path.join(args.original_datasets_path, 'label', name.replace('image', 'label').replace('.nii.gz', '-p.nii.gz'))
        labelv_path = os.path.join(args.original_datasets_path, 'label', name.replace('image', 'label').replace('.nii.gz', '-v.nii.gz'))
        image = sitk.ReadImage(image_path, sitk.sitkFloat32)
        labelp = sitk.ReadImage(labelp_path, sitk.sitkFloat32)
        labelv = sitk.ReadImage(labelv_path, sitk.sitkFloat32)
        image_array = sitk.GetArrayFromImage(image)
        labelp_array = sitk.GetArrayFromImage(labelp)
        labelv_array = sitk.GetArrayFromImage(labelv)

        # resize到粗分割网络所需的输入尺寸 (x, y, z) x为矢状面 y为冠状面 z为横截面
        resize_image = resize_image_itk(image, (256, 192, 1), resamplemethod=sitk.sitkLinear)  # UNet
        resize_image_arr = sitk.GetArrayFromImage(resize_image)
        resize_image_arr = normalize_data(resize_image_arr)

        # 预测斑块和血管
        plaque_predict_array, vessel_predict_array, time_sum = test(resize_image_arr, net)  # (1, 192, 256)
        if i != 0:
            avg_time.append(time_sum)

        plaque_predict = sitk.GetImageFromArray(plaque_predict_array)
        plaque_predict.SetSpacing(resize_image.GetSpacing())
        plaque_predict.SetOrigin(resize_image.GetOrigin())
        plaque_predict.SetDirection(resize_image.GetDirection())
        vessel_predict = sitk.GetImageFromArray(vessel_predict_array)
        vessel_predict.SetSpacing(resize_image.GetSpacing())
        vessel_predict.SetOrigin(resize_image.GetOrigin())
        vessel_predict.SetDirection(resize_image.GetDirection())
        # 粗分割预测标签恢复至原始图像尺寸
        plaque_label = resize_image_itk(plaque_predict, image.GetSize(), resamplemethod=sitk.sitkNearestNeighbor)
        vessel_label = resize_image_itk(vessel_predict, image.GetSize(), resamplemethod=sitk.sitkNearestNeighbor)
        plaque_label_array = sitk.GetArrayFromImage(plaque_label)
        vessel_label_array = sitk.GetArrayFromImage(vessel_label)
        plaque_vessel_label_array = plaque_label_array + 2 * vessel_label_array
        plaque_vessel_label_array[plaque_vessel_label_array == 3] = 1
        plaque_vessel_label = sitk.GetImageFromArray(plaque_vessel_label_array)
        plaque_vessel_label.SetSpacing(image.GetSpacing())
        plaque_vessel_label.SetDirection(image.GetDirection())
        plaque_vessel_label.SetOrigin(image.GetOrigin())

        # 保存预测的标签图和真实的标签图
        preds_p, preds_v = plaque_label_array, vessel_label_array
        labels_p, labels_v = labelp_array, labelv_array

        # 计算准确率 分割对了多少
        # total_accuracy += (preds == labels).sum()
        # total_acc_pixels += preds.size(0) * preds.size(1) * preds.size(2) * preds.size(3)
        # 计算精确率 分割的区域中正确的有多少
        p_total_precision += (preds_p * labels_p).sum()
        v_total_precision += (preds_v * labels_v).sum()
        p_total_pre_pixels += (preds_p == 1).sum()
        v_total_pre_pixels += (preds_v == 1).sum()
        # 计算召回率 正确的分割区域有多少正确的被分割出来了
        # Sensitivity(Recall, true positive rate) 真阳性率（true positive rate，TPR）也称为敏感度和召回率
        p_total_recall += (preds_p * labels_p).sum()
        v_total_recall += (preds_v * labels_v).sum()
        p_total_rec_pixels += (labels_p == 1).sum()
        v_total_rec_pixels += (labels_v == 1).sum()


        # dice = 2 * TP / (2 * TP + FP + FN)
        # miou = TP / (TP + FP + FN)
        # 用于防止分母为0
        smooth = 1.
        # 计算Dice系数，Dice系数是一种集合相似度度量函数，通常用于计算两个样本的相似度，取值范围在[0,1]，类似于IoU
        num_dict[index] += 1
        p_dice_dict[index] += (2. * (preds_p * labels_p).sum() + smooth) / (preds_p.sum() + labels_p.sum() + smooth)
        tp_dice = (2. * (preds_p * labels_p).sum() + smooth) / (preds_p.sum() + labels_p.sum() + smooth)
        v_dice_dict[index] += (2. * (preds_v * labels_v).sum() + smooth) / (preds_v.sum() + labels_v.sum() + smooth)
        tv_dice = (2. * (preds_v * labels_v).sum() + smooth) / (preds_v.sum() + labels_v.sum() + smooth)
        p_iou_dict[index] += ((preds_p * labels_p).sum() + smooth) / (preds_p.sum() + labels_p.sum() - (preds_p * labels_p).sum() + smooth)
        tp_iou = ((preds_p * labels_p).sum() + smooth) / (preds_p.sum() + labels_p.sum() - (preds_p * labels_p).sum() + smooth)
        v_iou_dict[index] += ((preds_v * labels_v).sum() + smooth) / (preds_v.sum() + labels_v.sum() - (preds_v * labels_v).sum() + smooth)
        tv_iou = ((preds_v * labels_v).sum() + smooth) / (preds_v.sum() + labels_v.sum() - (preds_v * labels_v).sum() + smooth)
        # plaque
        if preds_p.sum() > 0 and labels_p.sum() > 0:
            p_hd_dict[index] += metric.binary.hd95(preds_p, labels_p, image.GetSpacing()[0])
            tp_hd = metric.binary.hd95(preds_p, labels_p, image.GetSpacing()[0])
        elif preds_p.sum() == 0 and labels_p.sum() > 0:
            gt = labels_p
            gt = gt[0].astype(np.uint8)
            gt = gt * 255
            gt = cv2.resize(gt, (512, 512))
            contours, hierarchy = cv2.findContours(gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[0]
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            p_hd_dict[index] += radius
            tp_hd = radius
        elif preds_p.sum() > 0 and labels_p.sum() == 0:
            pred = preds_p
            pred = pred[0].astype(np.uint8)
            pred = pred * 255
            pred = cv2.resize(pred, (512, 512))
            contours, hierarchy = cv2.findContours(pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[0]
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            p_hd_dict[index] += radius
            tp_hd = radius
        else:  # preds.sum() == 0 and labels.sum() == 0
            p_hd_dict[index] += 0
            tp_hd = 0
        # vessel
        if preds_v.sum() > 0 and labels_v.sum() > 0:
            v_hd_dict[index] += metric.binary.hd95(preds_v, labels_v, image.GetSpacing()[0])
            tv_hd = metric.binary.hd95(preds_v, labels_v, image.GetSpacing()[0])
        elif preds_v.sum() == 0 and labels_v.sum() > 0:
            gt = labels_v
            gt = gt[0].astype(np.uint8)
            gt = gt * 255
            gt = cv2.resize(gt, (512, 512))
            contours, hierarchy = cv2.findContours(gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[0]
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            v_hd_dict[index] += radius
            tv_hd = radius
        elif preds_v.sum() > 0 and labels_v.sum() == 0:
            pred = preds_v
            pred = pred[0].astype(np.uint8)
            pred = pred * 255
            pred = cv2.resize(pred, (512, 512))
            contours, hierarchy = cv2.findContours(pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[0]
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            v_hd_dict[index] += radius
            tv_hd = radius
        else:  # preds.sum() == 0 and labels.sum() == 0
            v_hd_dict[index] += 0
            tv_hd = 0
        p_sen_dict[index] += ((preds_p * labels_p).sum() + smooth) / (labels_p.sum() + smooth)
        tp_sen = ((preds_p * labels_p).sum() + smooth) / (labels_p.sum() + smooth)
        v_sen_dict[index] += ((preds_v * labels_v).sum() + smooth) / (labels_v.sum() + smooth)
        tv_sen = ((preds_v * labels_v).sum() + smooth) / (labels_v.sum() + smooth)
        p_pre_dict[index] += ((preds_p * labels_p).sum() + smooth) / (preds_p.sum() + smooth)
        tp_pre = ((preds_p * labels_p).sum() + smooth) / (preds_p.sum() + smooth)
        v_pre_dict[index] += ((preds_v * labels_v).sum() + smooth) / (preds_v.sum() + smooth)
        tv_pre = ((preds_v * labels_v).sum() + smooth) / (preds_v.sum() + smooth)

        plaque_preds_array = preds_p
        vessel_preds_array = preds_v
        # 转化成.nii.gz的图
        final_plaque_label, final_vessel_label = plaque_label, vessel_label
        if not os.path.exists('./results'):
            os.makedirs('./results/image')
            os.makedirs('./results/label')
            os.makedirs('./results/labelpv')
        # 001-image001.nii.gz   001-label001.nii.gz
        sitk.WriteImage(image, os.path.join('./results/image', name))
        sitk.WriteImage(final_plaque_label, os.path.join('./results/label', name.replace('image', 'label').replace('.nii.gz', '-p.nii.gz')))
        sitk.WriteImage(final_vessel_label, os.path.join('./results/label', name.replace('image', 'label').replace('.nii.gz', '-v.nii.gz')))
        sitk.WriteImage(plaque_vessel_label, os.path.join('./results/labelpv', name.replace('image', 'label')))

        # time_end = time.time()
        # time_sum = time_end - time_start
        # print(name + ": " + str(time_sum))

        p_data = [str(name), '{:.4f}'.format(tp_dice), '{:.4f}'.format(tp_iou), '{:.4f}'.format(tp_hd),
                '{:.4f}'.format(tp_sen), '{:.4f}'.format(tp_pre)]
        for col, column_data in enumerate(p_data):
            p_sheet.write(i + 1, col, column_data)

        v_data = [str(name), '{:.4f}'.format(tv_dice), '{:.4f}'.format(tv_iou), '{:.4f}'.format(tv_hd),
                  '{:.4f}'.format(tv_sen), '{:.4f}'.format(tv_pre)]
        for col, column_data in enumerate(v_data):
            v_sheet.write(i + 1, col, column_data)

    workbook.save('test_PDCSNet_pv.xls')

    """
        像素准确率（Pixel Accuracy，PA）   类别像素准确率（Class Pixel Accuray，CPA）  类别平均像素准确率（Mean Pixel Accuracy，MPA）
        交并比（Intersection over Union，IoU）  平均交并比（Mean Intersection over Union，MIoU）
    """
    # print("Got {}/{} with PA {:.2%}".format(total_accuracy, total_acc_pixels, total_accuracy / total_acc_pixels))
    print("Plaque Got {}/{} with CPA {:.2%}".format(p_total_precision, p_total_pre_pixels, p_total_precision / p_total_pre_pixels))
    print("Vessel Got {}/{} with CPA {:.2%}".format(v_total_precision, v_total_pre_pixels, v_total_precision / v_total_pre_pixels))
    print("Plaque Got {}/{} with TPR {:.2%}".format(p_total_recall, p_total_rec_pixels, p_total_recall / p_total_rec_pixels))
    print("Vessel Got {}/{} with TPR {:.2%}".format(v_total_recall, v_total_rec_pixels, v_total_recall / v_total_rec_pixels))
    # print("Test_MIoU：{:.2%}".format(miou / len(test_loader)))
    p_dice = 0.0
    v_dice = 0.0
    p_iou = 0.0
    v_iou = 0.0
    p_hd = 0.0
    v_hd = 0.0
    p_sen = 0.0
    v_sen = 0.0
    p_pre = 0.0
    v_pre = 0.0
    for key, item in p_dice_dict.items():
        p_dice += item / num_dict[key]
        v_dice += v_dice_dict[key] / num_dict[key]
        p_iou += p_iou_dict[key] / num_dict[key]
        v_iou += v_iou_dict[key] / num_dict[key]
        p_hd += p_hd_dict[key] / num_dict[key]
        v_hd += v_hd_dict[key] / num_dict[key]
        p_sen += p_sen_dict[key] / num_dict[key]
        v_sen += v_sen_dict[key] / num_dict[key]
        p_pre += p_pre_dict[key] / num_dict[key]
        v_pre += v_pre_dict[key] / num_dict[key]
        print(f'Plaque : patient_id: {key}\t'
              f'plaque_dice: {item / num_dict[key]}\t'
              f'plaque_iou: {p_iou_dict[key] / num_dict[key]}\t'
              f'plaque_hd95: {p_hd_dict[key] / num_dict[key]}\t'
              f'plaque_sen: {p_sen_dict[key] / num_dict[key]}\t'
              f'plaque_sen: {p_pre_dict[key] / num_dict[key]}')
        print(f'Vessel: patient_id: {key}\t'
              f'vessel_dice: {v_dice_dict[key] / num_dict[key]}\t'
              f'vessel_iou: {v_iou_dict[key] / num_dict[key]}\t'
              f'vessel_hd95: {v_hd_dict[key] / num_dict[key]}\t'
              f'vessel_sen: {v_sen_dict[key] / num_dict[key]}\t'
              f'vessel_pre: {v_pre_dict[key] / num_dict[key]}')
        logging.info('Plaque——patient idx %s: plaque_dice %f plaque_iou %f plaque_hd95 %f plaque_sen %f plaque_pre %f'
                     % (key, item / num_dict[key],
                        p_iou_dict[key] / num_dict[key],
                        p_hd_dict[key] / num_dict[key],
                        p_sen_dict[key] / num_dict[key],
                        p_pre_dict[key] / num_dict[key]))
        # logging.info('Vessel——patient idx %s: vessel_dice %f vessel_diou %f vessel_sen %f' % (key, v_dice_dict[key] / num_dict[key],
        #              v_iou_dict[key] / num_dict[key], v_sen_dict[key] / num_dict[key]))
    print("Plaque Test Patient Avg Dice：{:.2%}  IoU：{:.2%} hd95：{:.2f} sen：{:.2%} pre：{:.2%}".
          format(p_dice / len(p_dice_dict), p_iou / len(p_iou_dict), p_hd / len(p_hd_dict), p_sen / len(p_sen_dict), p_pre / len(p_pre_dict)))
    print("Vessel Test Patient Avg Dice：{:.2%}  IoU：{:.2%} hd95：{:.2f}  sen：{:.2%} pre：{:.2%}".
          format(v_dice / len(v_dice_dict), v_iou / len(v_iou_dict), v_hd / len(v_hd_dict), v_sen / len(v_sen_dict), v_pre / len(v_pre_dict)))
    logging.info('Plaque——Testing performance in best val model: mean_dice : %f mean_iou : %f mean_hd95 : %f mean_sen : %f mean_pre : %f'
        % (p_dice / len(p_dice_dict), p_iou / len(p_iou_dict), p_hd / len(p_hd_dict), p_sen / len(p_sen_dict), p_pre / len(p_pre_dict)))
    logging.info('Vessel——Testing performance in best val model: mean_dice : %f mean_iou : %f mean_hd95 : %f mean_sen : %f mean_pre : %f'
        % (v_dice / len(v_dice_dict), v_iou / len(v_iou_dict), v_hd / len(v_hd_dict), v_sen / len(v_sen_dict), v_pre / len(v_pre_dict)))
    print("Avg Model Predict Time: {:.4f} Std:{:.4f}".format(np.mean(avg_time), np.std(avg_time)))
