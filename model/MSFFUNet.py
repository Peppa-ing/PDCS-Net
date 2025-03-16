import torch
import torch.nn as nn

from model.Multi_Scale_Feature_Fusion import MSFF, MSFF_Dilation, MSFF_2


class Double_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Double_Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),  # 加入Bn层提高网络泛化能力（防止过拟合），加收敛速度
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(UpSample, self).__init__()
        if bilinear:  # bilinear表示在二维度进行插值
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.upsample(x)


class MSFFUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MSFFUNet, self).__init__()
        self.conv1 = Double_Block(in_channels, 64)
        self.conv2 = Double_Block(64, 128)
        self.conv3 = Double_Block(128, 256)
        self.conv4 = Double_Block(256, 512)
        self.conv5 = Double_Block(512, 1024)

        self.maxpool = nn.MaxPool2d(2)

        self.MSFF = MSFF(1024)
        # self.MSFF = MSFF_2(1024)
        # self.MSFF = MSFF_Dilation(1024)

        # 新增dropout层
        self.dropout = nn.Dropout2d(0.3)

        # plaque
        self.upsample1_p = UpSample(1024, 512)
        self.upsample2_p = UpSample(512, 256)
        self.upsample3_p = UpSample(256, 128)
        self.upsample4_p = UpSample(128, 64)

        self.up_conv4_p = Double_Block(512 + 512, 512)
        self.up_conv3_p = Double_Block(256 + 256, 256)
        self.up_conv2_p = Double_Block(128 + 128, 128)
        self.up_conv1_p = Double_Block(64 + 64, 64)

        self.last_conv_p = nn.Conv2d(64, out_channels, kernel_size=1)

        # vessel
        self.upsample1_v = UpSample(1024, 512)
        self.upsample2_v = UpSample(512, 256)
        self.upsample3_v = UpSample(256, 128)
        self.upsample4_v = UpSample(128, 64)

        self.up_conv4_v = Double_Block(512 + 512, 512)
        self.up_conv3_v = Double_Block(256 + 256, 256)
        self.up_conv2_v = Double_Block(128 + 128, 128)
        self.up_conv1_v = Double_Block(64 + 64, 64)

        self.last_conv_v = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):  # [2, 1, 768, 1024]
        # encoder
        conv1 = self.conv1(x)  # [2, 64, 768, 1024]
        x = self.maxpool(conv1)  # [2, 64, 384, 512]

        conv2 = self.conv2(x)  # [2, 128, 384, 512]
        x = self.maxpool(conv2)  # [2, 128, 192, 256]

        conv3 = self.conv3(x)  # [2, 256, 192, 256]
        x = self.maxpool(conv3)  # [2, 256, 96, 128]

        conv4 = self.conv4(x)  # [2, 512, 96, 128]
        x = self.maxpool(conv4)  # [2, 512, 48, 64]

        # 新增dropout层
        x = self.dropout(x)

        x = self.conv5(x)  # [2, 1024, 48, 64]
        x = self.MSFF(x)  # [2, 1024, 48, 64]

        # 新增dropout层
        x = self.dropout(x)

        feature_x = x

        # decoder1——carotid plaque
        x1 = self.upsample1_p(feature_x)  # [2, 512, 96, 128]
        # 因为使用了3*3卷积核和 padding=1 的组合，所以卷积过程图像尺寸不发生改变，所以省去了crop操作！
        x1 = torch.cat([x1, conv4], dim=1)  # [2, 512 + 512, 96, 128]

        x1 = self.up_conv4_p(x1)  # [2, 512, 96, 128]
        x1 = self.upsample2_p(x1)  # [2, 256, 192, 256]
        x1 = torch.cat([x1, conv3], dim=1)  # [2, 256 + 256, 192, 256]

        x1 = self.up_conv3_p(x1)  # [2, 256, 192, 256]
        x1 = self.upsample3_p(x1)  # [2, 128, 384, 512]
        x1 = torch.cat([x1, conv2], dim=1)  # [2, 128 + 128, 384, 512]

        x1 = self.up_conv2_p(x1)  # [2, 128, 384, 512]
        x1 = self.upsample4_p(x1)  # [2, 64, 768, 1024]
        x1 = torch.cat([x1, conv1], dim=1)  # [2, 64 + 64, 768, 1024]

        x1 = self.up_conv1_p(x1)  # [2, 64, 768, 1024]

        plaque_out = self.last_conv_p(x1)  # [2, 1, 768, 1024]

        # decoder2——blood vessel
        x2 = self.upsample1_v(feature_x)  # [2, 512, 96, 128]
        # 因为使用了3*3卷积核和 padding=1 的组合，所以卷积过程图像尺寸不发生改变，所以省去了crop操作！
        x2 = torch.cat([x2, conv4], dim=1)  # [2, 512 + 512, 96, 128]

        x2 = self.up_conv4_v(x2)  # [2, 512, 96, 128]
        x2 = self.upsample2_v(x2)  # [2, 256, 192, 256]
        x2 = torch.cat([x2, conv3], dim=1)  # [2, 256 + 256, 192, 256]

        x2 = self.up_conv3_v(x2)  # [2, 256, 192, 256]
        x2 = self.upsample3_v(x2)  # [2, 128, 384, 512]
        x2 = torch.cat([x2, conv2], dim=1)  # [2, 128 + 128, 384, 512]

        x2 = self.up_conv2_v(x2)  # [2, 128, 384, 512]
        x2 = self.upsample4_v(x2)  # [2, 64, 768, 1024]
        x2 = torch.cat([x2, conv1], dim=1)  # [2, 64 + 64, 768, 1024]

        x2 = self.up_conv1_v(x2)  # [2, 64, 768, 1024]

        vessel_out = self.last_conv_v(x2)  # [2, 1, 768, 1024]

        return plaque_out, vessel_out


# class MSFFUNet(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(MSFFUNet, self).__init__()
#         self.conv1 = Double_Block(in_channels, 64)
#         self.conv2 = Double_Block(64, 128)
#         self.conv3 = Double_Block(128, 256)
#         self.conv4 = Double_Block(256, 512)
#         self.conv5 = Double_Block(512, 1024)
#
#         self.maxpool = nn.MaxPool2d(2)
#
#         self.MSFF = MSFF(1024)
#         # self.MSFF = MSFF_Dilation(1024)
#
#         # 新增dropout层
#         self.dropout = nn.Dropout2d(0.3)
#
#         # plaque
#         self.upsample1_p = UpSample(1024, 512)
#         self.upsample2_p = UpSample(512, 256)
#         self.upsample3_p = UpSample(256, 128)
#         self.upsample4_p = UpSample(128, 64)
#
#         self.up_conv4_p = Double_Block(512 + 512, 512)
#         self.up_conv3_p = Double_Block(256 + 256, 256)
#         self.up_conv2_p = Double_Block(128 + 128, 128)
#         self.up_conv1_p = Double_Block(64 + 64, 64)
#
#         self.up5_p = nn.Upsample(scale_factor=16)  # 1024
#         self.up4_p = nn.Upsample(scale_factor=8)  # 512
#         self.up3_p = nn.Upsample(scale_factor=4)  # 256
#         self.up2_p = nn.Upsample(scale_factor=2)  # 128
#
#         self.fuse_1024_p = nn.Conv2d(1024, 64, 3, 1, 1)
#         self.fuse_512_p = nn.Conv2d(512, 64, 3, 1, 1)
#         self.fuse_256_p = nn.Conv2d(256, 64, 3, 1, 1)
#         self.fuse_128_p = nn.Conv2d(128, 64, 3, 1, 1)
#
#         self.fuse_conv_p = nn.Conv2d(64, out_channels, kernel_size=1)
#
#         # self.last_conv_p = nn.Conv2d(64, out_channels, kernel_size=1)
#
#         # vessel
#         self.upsample1_v = UpSample(1024, 512)
#         self.upsample2_v = UpSample(512, 256)
#         self.upsample3_v = UpSample(256, 128)
#         self.upsample4_v = UpSample(128, 64)
#
#         self.up_conv4_v = Double_Block(512 + 512, 512)
#         self.up_conv3_v = Double_Block(256 + 256, 256)
#         self.up_conv2_v = Double_Block(128 + 128, 128)
#         self.up_conv1_v = Double_Block(64 + 64, 64)
#
#         # self.up5_v = nn.Upsample(scale_factor=16)  # 1024
#         # self.up4_v = nn.Upsample(scale_factor=8)  # 512
#         # self.up3_v = nn.Upsample(scale_factor=4)  # 256
#         # self.up2_v = nn.Upsample(scale_factor=2)  # 128
#         #
#         # self.fuse_1024_v = nn.Conv2d(1024, 64, 3, 1, 1)
#         # self.fuse_512_v = nn.Conv2d(512, 64, 3, 1, 1)
#         # self.fuse_256_v = nn.Conv2d(256, 64, 3, 1, 1)
#         # self.fuse_128_v = nn.Conv2d(128, 64, 3, 1, 1)
#         #
#         # self.fuse_conv_v = nn.Conv2d(64 * 5, out_channels, kernel_size=1)
#
#         self.last_conv_v = nn.Conv2d(64, out_channels, kernel_size=1)
#
#     def forward(self, x):  # [2, 1, 768, 1024]
#         # encoder
#         conv1 = self.conv1(x)  # [2, 64, 768, 1024]
#         x = self.maxpool(conv1)  # [2, 64, 384, 512]
#
#         conv2 = self.conv2(x)  # [2, 128, 384, 512]
#         x = self.maxpool(conv2)  # [2, 128, 192, 256]
#
#         conv3 = self.conv3(x)  # [2, 256, 192, 256]
#         x = self.maxpool(conv3)  # [2, 256, 96, 128]
#
#         conv4 = self.conv4(x)  # [2, 512, 96, 128]
#         x = self.maxpool(conv4)  # [2, 512, 48, 64]
#
#         # 新增dropout层
#         x = self.dropout(x)
#
#         x = self.conv5(x)  # [2, 1024, 48, 64]
#         x = self.MSFF(x)  # [2, 1024, 48, 64]
#
#         # 新增dropout层
#         x = self.dropout(x)
#
#         feature_x = x
#         out5_p = x
#         # out5_v = x
#
#         # decoder1——carotid plaque
#         x1 = self.upsample1_p(feature_x)  # [2, 512, 96, 128]
#         # 因为使用了3*3卷积核和 padding=1 的组合，所以卷积过程图像尺寸不发生改变，所以省去了crop操作！
#         x1 = torch.cat([x1, conv4], dim=1)  # [2, 512 + 512, 96, 128]
#
#         x1 = self.up_conv4_p(x1)  # [2, 512, 96, 128]
#         out4_p = x1
#         x1 = self.upsample2_p(x1)  # [2, 256, 192, 256]
#         x1 = torch.cat([x1, conv3], dim=1)  # [2, 256 + 256, 192, 256]
#
#         x1 = self.up_conv3_p(x1)  # [2, 256, 192, 256]
#         out3_p = x1
#         x1 = self.upsample3_p(x1)  # [2, 128, 384, 512]
#         x1 = torch.cat([x1, conv2], dim=1)  # [2, 128 + 128, 384, 512]
#
#         x1 = self.up_conv2_p(x1)  # [2, 128, 384, 512]
#         out2_p = x1
#         x1 = self.upsample4_p(x1)  # [2, 64, 768, 1024]
#         x1 = torch.cat([x1, conv1], dim=1)  # [2, 64 + 64, 768, 1024]
#
#         x1 = self.up_conv1_p(x1)  # [2, 64, 768, 1024]
#         out1_p = x1
#
#         p2 = self.fuse_128_p(self.up2_p(out2_p))
#         p3 = self.fuse_256_p(self.up3_p(out3_p))
#         p4 = self.fuse_512_p(self.up4_p(out4_p))
#         p5 = self.fuse_1024_p(self.up5_p(out5_p))
#         # plaque_out = self.fuse_conv_p(torch.cat((out1_p, p2, p3, p4, p5), dim=1))  # [2, 64 * 5, 768, 1024]
#         plaque_out = self.fuse_conv_p(out1_p + p2 + p3 + p4 + p5)
#         # plaque_out = self.last_conv_p(x1)  # [2, 1, 768, 1024]
#
#         # decoder2——blood vessel
#         x2 = self.upsample1_v(feature_x)  # [2, 512, 96, 128]
#         # 因为使用了3*3卷积核和 padding=1 的组合，所以卷积过程图像尺寸不发生改变，所以省去了crop操作！
#         x2 = torch.cat([x2, conv4], dim=1)  # [2, 512 + 512, 96, 128]
#
#         x2 = self.up_conv4_v(x2)  # [2, 512, 96, 128]
#         # out4_v = x2
#         x2 = self.upsample2_v(x2)  # [2, 256, 192, 256]
#         x2 = torch.cat([x2, conv3], dim=1)  # [2, 256 + 256, 192, 256]
#
#         x2 = self.up_conv3_v(x2)  # [2, 256, 192, 256]
#         # out3_v = x2
#         x2 = self.upsample3_v(x2)  # [2, 128, 384, 512]
#         x2 = torch.cat([x2, conv2], dim=1)  # [2, 128 + 128, 384, 512]
#
#         x2 = self.up_conv2_v(x2)  # [2, 128, 384, 512]
#         # out2_v = x2
#         x2 = self.upsample4_v(x2)  # [2, 64, 768, 1024]
#         x2 = torch.cat([x2, conv1], dim=1)  # [2, 64 + 64, 768, 1024]
#
#         x2 = self.up_conv1_v(x2)  # [2, 64, 768, 1024]
#         # out1_v = x2
#
#         # vessel_out = self.fuse_conv_v(torch.cat((out1_v, self.fuse_128_v(self.up2_v(out2_v)),
#         #                                          self.fuse_256_v(self.up3_v(out3_v)),
#         #                                          self.fuse_512_v(self.up4_v(out4_v)),
#         #                                          self.fuse_1024_v(self.up5_v(out5_v))),
#         #                                          dim=1))  # [2, 64 * 5, 768, 1024]
#
#         vessel_out = self.last_conv_v(x2)  # [2, 1, 768, 1024]
#
#         return plaque_out, vessel_out

if __name__ == '__main__':
    input = torch.randn([2, 1, 192, 256])
    net = MSFFUNet(in_channels=1, out_channels=1)
    plaque_output, vessel_output = net(input)
    print(plaque_output.shape)  # torch.size([2, 1, 192, 256])
    print(vessel_output.shape)  # torch.size([2, 1, 192, 256])