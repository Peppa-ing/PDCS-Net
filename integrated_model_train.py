import datetime
import os
import logging
import random
import sys

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import configs
from datasets.train_datasets import Train_Dataset
from datasets.val_datasets import Val_Dataset
from model.DAUNet import DAUNet
from model.DSAUNet import DSAUNet
from model.Dual_Attention_UNet import Dual_Attention_UNet

from model.Dual_UNet import Dual_UNet
from model.IUNet import IUNet, IUNet2, IUNet3
from model.MDAUNet import MDAUNet
from model.MSFFUNet import MSFFUNet
from model.PDASNet import PDASNet
from utils import loss

if __name__ == '__main__':
    # seed = 1234  # 设置一个种子，确保可以复现
    # random.seed(seed)  # Python random module.
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    save_path = './experiments/PDCSNet'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    """
        使用gpu需要：
        网络模型   数据(输入，标注)   损失函数   .cuda() 或者 .to(device)
    """
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 选择数据集
    index = ''  # 原图resize 256 * 192
    args = configs.args
    # 加载训练数据
    train_loader = DataLoader(dataset=Train_Dataset(args=args, index=index), batch_size=8, shuffle=True, num_workers=0)
    # 加载验证集数据
    val_loader = DataLoader(dataset=Val_Dataset(args=args, index=index), batch_size=8, shuffle=True, num_workers=0)
    # 加载网络
    net = IUNet3(in_channels=1, out_channels=1)
    # 将网络拷贝到device中
    net.to(device=device)
    # 定义损失函数 This loss combines a `Sigmoid` layer and the `BCELoss` in one single class  目标targets应为0到1之间的数字
    loss_bce = nn.BCEWithLogitsLoss()
    loss_dice = loss.DiceLoss()
    loss_func = loss.BCE_Dice_Loss()
    loss_bce.to(device)
    loss_dice.to(device)
    loss_func.to(device)
    # 设置学习率
    learning_rate = 0.0001  # 0.0001
    # scheduler = get_linear_schedule_with_warmup(learning_rate=0.0001, num_warmup_steps=1150, num_training_steps=232000)
    # 优化器      Adam优化器   所需要训练的参数     学习率=0.0001最好    weight_decay=0.001
    optimizer = optim.Adam(params=net.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    best_dice = 0.0
    best_epoch = 0

    now_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    # 添加tensorboard
    # 将train_dice 保存到"train_logs/train_logs" 文件夹
    writer = SummaryWriter("train_logs_" + now_time + "/train")
    # 将 val_dice 保存到 "train_logs/val_logs" 文件夹
    val_writer = SummaryWriter("train_logs_" + now_time + "/val")
    # logging.basicConfig(filename="train_logs_" + now_time + "/log.txt", level=logging.INFO,
    #                     format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # 训练50轮
    epochs = 150
    trigger = 0  # early stop计数器
    index = 0

    for e in range(epochs):
        # 更新学习率
        lr = learning_rate * (0.1 ** (e // 50))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # train训练模型
        print("-----------第{}/{}轮训练开始---------(learning rate:{})".format(e + 1, epochs, lr))
        # 让网络进入训练模式   Dropout  BatchNorm层
        net.train()
        # 开始训练，数据集的数据全部用过一遍
        # 按照batch_size开始训练  tqdm显示进度条
        running_loss = 0.0
        plaque_loss = 0.0
        vessel_loss = 0.0
        val_loss = 0.0
        val_plaque_loss = 0.0
        val_vessel_loss = 0.0
        plaque_dice = 0.0
        vessel_dice = 0.0
        val_plaque_dice = 0.0
        val_vessel_dice = 0.0
        plaque_miou = 0.0
        vessel_miou = 0.0
        plaque_sensitivity = 0.0
        vessel_sensitivity = 0.0
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            images, labels_p, labels_v = data  # (1, 1, 768, 1024) (1, 768, 1024) (1, 768, 1024)
            # (1, 768, 1024)->(1, 1, 768, 1024)
            labels_p = labels_p.float().unsqueeze(1)
            labels_v = labels_v.float().unsqueeze(1)
            # 将数据拷贝到device中
            images = images.to(device)
            labels_p = labels_p.to(device)
            labels_v = labels_v.to(device)
            # 使用网络参数，输出预测结果
            outputs_p1, outputs_v1, outputs_p2, outputs_v2 = net(images)

            # sigmoid将一个real value映射到（0, 1）的区间，用来做二分类
            preds_p1 = torch.sigmoid(outputs_p1)  # 值域0-1之间
            preds_v1 = torch.sigmoid(outputs_v1)  # 值域0-1之间
            preds_p1 = (preds_p1 >= 0.5).float()  # (1, 1, 768, 1024)
            preds_v1 = (preds_v1 >= 0.5).float()  # (1, 1, 768, 1024)
            preds_p2 = torch.sigmoid(outputs_p2)  # 值域0-1之间
            preds_v2 = torch.sigmoid(outputs_v2)  # 值域0-1之间
            preds_p2 = (preds_p2 >= 0.5).float()  # (1, 1, 768, 1024)
            preds_v2 = (preds_v2 >= 0.5).float()  # (1, 1, 768, 1024)

            preds_p = preds_p1 + preds_p2
            preds_p[preds_p == 2] = 1
            preds_v = preds_v1 + preds_v2
            preds_v[preds_v == 2] = 1

            # 计算loss
            # loss_p = loss_func(outputs_p, labels_p)
            # loss_v = loss_func(outputs_v, labels_v)

            # plaque loss = plaque_bce_loss + plaque_dice_loss
            p_bce_loss1 = loss_bce(outputs_p1, labels_p)
            p_bce_loss2 = loss_bce(outputs_p2, labels_p)
            p_dice_loss1 = loss_dice(outputs_p1, labels_p)
            p_dice_loss2 = loss_dice(outputs_p2, labels_p)
            # vessel loss = vessel_bce_loss + vessel_dice_loss
            v_bce_loss1 = loss_bce(outputs_v1, labels_v)
            v_bce_loss2 = loss_bce(outputs_v2, labels_v)
            v_dice_loss1 = loss_dice(outputs_v1, labels_v)
            v_dice_loss2 = loss_dice(outputs_v2, labels_v)

            loss_p1 = 0.6 * p_bce_loss1 + 0.4 * p_dice_loss1
            loss_p2 = 0.6 * p_bce_loss2 + 0.4 * p_dice_loss2
            loss_v1 = 0.6 * v_bce_loss1 + 0.4 * v_dice_loss1
            loss_v2 = 0.6 * v_bce_loss2 + 0.4 * v_dice_loss2
            loss1 = 0.6 * loss_p1 + 0.4 * loss_v1
            loss2 = 0.6 * loss_p2 + 0.4 * loss_v2
            loss = 0.4 * loss1 + 0.6 * loss2

            # zero the parameter gradients
            optimizer.zero_grad()  # 对上一步求出来的梯度进行清零
            # forward + backward + optimize
            loss.backward()  # 反向传播，得到梯度
            optimizer.step()  # 对得到的梯度进行优化 对卷积核的参数进行调整

            # 更新学习率
            # lr_ = scheduler[index]
            # index += 1
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr_

            # print statistics
            running_loss += loss.item()
            plaque_loss += loss_p1.item() + loss_p2.item()
            vessel_loss += loss_v1.item() + loss_v2.item()

            # 用于防止分母为0
            smooth = 1.
            # dice = 2 * TP / (2 * TP + FP + FN)
            # miou = TP / (TP + FP + FN)
            # sensitivity = TP / (TP + FN)
            # 计算Dice系数，Dice系数是一种集合相似度度量函数，通常用于计算两个样本的相似度，取值范围在[0,1]，类似于IoU
            plaque_dice += (2 * (preds_p * labels_p).sum() + smooth) / (preds_p.sum() + labels_p.sum() + smooth)
            vessel_dice += (2 * (preds_v * labels_v).sum() + smooth) / (preds_v.sum() + labels_v.sum() + smooth)
            plaque_miou += ((preds_p * labels_p).sum() + smooth) / (preds_p.sum() + labels_p.sum() - (preds_p * labels_p).sum() + smooth)
            vessel_miou += ((preds_v * labels_v).sum() + smooth) / (preds_v.sum() + labels_v.sum() - (preds_v * labels_v).sum() + smooth)
            plaque_sensitivity += ((preds_p * labels_p).sum() + smooth) / (labels_p.sum() + smooth)
            vessel_sensitivity += ((preds_v * labels_v).sum() + smooth) / (labels_v.sum() + smooth)

            if i % 100 == 0:
                print("训练次数：{}，Plaque_Loss：{}，Vessel_Loss：{}，"
                      .format(i + 1, loss_p1.item() + loss_p2.item(), loss_v1.item() + loss_v2.item()))
                print("训练次数：{}，Loss：{}".format(i + 1, loss.item()))

        # val验证模型 测试模式 让网络进入一个测试状态
        net.eval()
        with torch.no_grad():
            for data in tqdm(val_loader, total=len(val_loader)):
                images, labels_p, labels_v = data  # image:(1, 1, 768, 1024)  labels:(1, 768, 1024)
                labels_p = labels_p.float().unsqueeze(1)  # (1, 1, 768, 1024)
                labels_v = labels_v.float().unsqueeze(1)  # (1, 1, 768, 1024)
                # 将数据拷贝到device中
                images = images.to(device)
                labels_p = labels_p.to(device)
                labels_v = labels_v.to(device)
                # 使用网络参数，输出预测结果
                outputs_p1, outputs_v1, outputs_p2, outputs_v2 = net(images)

                # sigmoid将一个real value映射到（0, 1）的区间，用来做二分类
                preds_p1 = torch.sigmoid(outputs_p1)  # 值域0-1之间
                preds_v1 = torch.sigmoid(outputs_v1)  # 值域0-1之间
                preds_p1 = (preds_p1 >= 0.5).float()  # (1, 1, 768, 1024)
                preds_v1 = (preds_v1 >= 0.5).float()  # (1, 1, 768, 1024)
                preds_p2 = torch.sigmoid(outputs_p2)  # 值域0-1之间
                preds_v2 = torch.sigmoid(outputs_v2)  # 值域0-1之间
                preds_p2 = (preds_p2 >= 0.5).float()  # (1, 1, 768, 1024)
                preds_v2 = (preds_v2 >= 0.5).float()  # (1, 1, 768, 1024)

                preds_p = preds_p1 + preds_p2
                preds_p[preds_p == 2] = 1
                preds_v = preds_v1 + preds_v2
                preds_v[preds_v == 2] = 1

                # 计算loss
                # loss_p = loss_func(outputs_p, labels_p)
                # loss_v = loss_func(outputs_v, labels_v)

                # plaque loss = plaque_bce_loss + plaque_dice_loss
                p_bce_loss1 = loss_bce(outputs_p1, labels_p)
                p_bce_loss2 = loss_bce(outputs_p2, labels_p)
                p_dice_loss1 = loss_dice(outputs_p1, labels_p)
                p_dice_loss2 = loss_dice(outputs_p2, labels_p)
                # vessel loss = vessel_bce_loss + vessel_dice_loss
                v_bce_loss1 = loss_bce(outputs_v1, labels_v)
                v_bce_loss2 = loss_bce(outputs_v2, labels_v)
                v_dice_loss1 = loss_dice(outputs_v1, labels_v)
                v_dice_loss2 = loss_dice(outputs_v2, labels_v)

                loss_p1 = 0.6 * p_bce_loss1 + 0.4 * p_dice_loss1
                loss_p2 = 0.6 * p_bce_loss2 + 0.4 * p_dice_loss2
                loss_v1 = 0.6 * v_bce_loss1 + 0.4 * v_dice_loss1
                loss_v2 = 0.6 * v_bce_loss2 + 0.4 * v_dice_loss2
                loss1 = 0.6 * loss_p1 + 0.4 * loss_v1
                loss2 = 0.6 * loss_p2 + 0.4 * loss_v2
                loss = 0.4 * loss1 + 0.6 * loss2

                # print statistics
                val_loss += loss.item()
                val_plaque_loss += loss_p1.item() + loss_p2.item()
                val_vessel_loss += loss_v1.item() + loss_v2.item()

                # 用于防止分母为0
                smooth = 1.
                val_plaque_dice += (2. * (preds_p * labels_p).sum() + smooth) / (preds_p.sum() + labels_p.sum() + smooth)
                val_vessel_dice += (2. * (preds_v * labels_v).sum() + smooth) / (preds_v.sum() + labels_v.sum() + smooth)

        trigger += 1
        # # 保存dice值最大的网络参数
        # if val_dice > best_dice:
        #     trigger = 0
        #     best_dice = val_dice
        #     best_epoch = e + 1
        #     torch.save(net.state_dict(), os.path.join(save_path, 'best_model.pth'))

        # # 保存loss值最小的网络参数
        if val_loss < best_loss:
            trigger = 0
            best_loss = val_loss
            best_epoch = e + 1
            torch.save(net.state_dict(), os.path.join(save_path, 'best_model.pth'))

        if trigger >= 10:
            print("=> early stopping")
            break

        # 这一轮结束
        print('第{}轮 Train_Loss: {}  Train_Plaque_Loss: {}  Train_Vessel_Loss: {}'.format(e + 1, running_loss / len(train_loader), plaque_loss / len(train_loader), vessel_loss / len(train_loader)))
        print('第{}轮 Val_Loss: {}  Val_Plaque_Loss: {}  Val_Vessel_Loss: {}'.format(e + 1, val_loss / len(val_loader), val_plaque_loss / len(val_loader), val_vessel_loss / len(val_loader)))
        print('第{}轮 Train_Plaque_Dice: {} Train_Vessel_Dice: {}'.format(e + 1, plaque_dice / len(train_loader), vessel_dice / len(train_loader)))
        print('第{}轮 Val_Plaque_Dice: {} Val_Vessel_Dice: {}'.format(e + 1, val_plaque_dice / len(val_loader), val_vessel_dice / len(val_loader)))
        print('第{}轮 Train_Plaque_MIoU: {} Train_Vessel_MIoU: {}'.format(e + 1, plaque_miou / len(train_loader), vessel_miou / len(train_loader)))
        print('第{}轮 Train_Plaque_Sensitivity: {} Train_Vessel_Sensitivity: {}'.format(e + 1, plaque_sensitivity / len(train_loader), vessel_sensitivity / len(train_loader)))
        # print("Got {}/{} with Accuracy {:.2%}".format(total_accuracy, total_acc_pixels, total_accuracy / total_acc_pixels))
        print('Best performance at Epoch: {}'.format(best_epoch))
        writer.add_scalars("Train_Loss", {"train_loss": running_loss / len(train_loader),
                                          "plaque_loss": plaque_loss / len(train_loader),
                                          "vessel_loss": vessel_loss / len(train_loader)}, e + 1)
        writer.add_scalars("Val_Loss", {"val_loss": val_loss / len(val_loader),
                                        "plaque_loss": val_plaque_loss / len(val_loader),
                                        "vessel_loss": val_vessel_loss / len(val_loader)}, e + 1)
        writer.add_scalars("Plaque", {"train_plaque_dice": plaque_dice / len(train_loader),
                                      "train_plaque_MIoU": plaque_miou / len(train_loader),
                                      "train_plaque_sensitivity": plaque_sensitivity / len(train_loader),
                                      "val_plaque_dice": val_plaque_dice / len(val_loader)}, e + 1)
        writer.add_scalars("Vessel", {"train_vessel_dice": vessel_dice / len(train_loader),
                                      "train_vessel_MIoU": vessel_miou / len(train_loader),
                                      "train_vessel_sensitivity": vessel_sensitivity / len(train_loader),
                                      "val_vessel_dice": val_vessel_dice / len(val_loader)}, e + 1)
        # writer.add_scalar("Loss/train_loss", running_loss / len(train_loader), e + 1)
        # writer.add_scalar("Loss/train_plaque_loss", plaque_loss / len(train_loader), e + 1)
        # writer.add_scalar("Loss/train_vessel_loss", vessel_loss / len(train_loader), e + 1)
        # val_writer.add_scalar("Loss/val_loss", val_loss / len(val_loader), e + 1)
        # val_writer.add_scalar("Loss/val_plaque_loss", val_plaque_loss / len(val_loader), e + 1)
        # val_writer.add_scalar("Loss/val_vessel_loss", val_vessel_loss / len(val_loader), e + 1)
        # writer.add_scalar("Dice/train_plaque_dice", plaque_dice / len(train_loader), e + 1)
        # writer.add_scalar("Dice/train_vessel_dice", vessel_dice / len(train_loader), e + 1)
        # val_writer.add_scalar("Dice/val_plaque_dice", val_plaque_dice / len(val_loader), e + 1)
        # val_writer.add_scalar("Dice/val_vessel_dice", val_vessel_dice / len(val_loader), e + 1)
        # writer.add_scalar("Train_MIoU/Plaque_MIoU", plaque_miou / len(train_loader), e + 1)
        # writer.add_scalar("Train_MIoU/vessel_MIoU", vessel_miou / len(train_loader), e + 1)
        # writer.add_scalar("Train_Sensitivity/plaque_sensitivity", plaque_sensitivity / len(train_loader), e + 1)
        # writer.add_scalar("Train_Sensitivity/vessel_sensitivity", vessel_sensitivity / len(train_loader), e + 1)
    writer.close()
    val_writer.close()


