import torch
import torch.nn as nn

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

class CAM(nn.Module):
    """
        MDAM encoding module 多方向注意机制(MDAM)
        Multi-direction Attention Module
    """
    def __init__(self, in_channels):
        super(CAM, self).__init__()
        self.channel_max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_channel = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
            inputs :
                x : input feature(N,C,H,W)
            returns :
                out : coordinate channel attention feature(N,C,H,W)
        """
        b, c, h, w = x.size()  # (b, c, h, w)->(8, 64, 128, 256)
        channel_maxpool_out = self.channel_max_pool(x)  # (b, c, 1, 1)
        channel_avgpool_out = self.channel_avg_pool(x)  # (b, c, 1, 1)
        channel_pool_out = torch.cat((channel_maxpool_out, channel_avgpool_out), dim=1)  # (b, 2c, 1, 1)
        channel_att = self.conv_channel(channel_pool_out)  # (b, c, 1, 1) !!!
        out = self.sigmoid(channel_att) * x
        return out

"""
    CFF:Cross Feature Fusion 交叉特征融合
    CAFF:Cross Attention Feature Fusion 交叉特征融合
"""
class CAFFUNet2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CAFFUNet2, self).__init__()
        self.conv1 = Double_Block(in_channels, 64)
        self.conv2 = Double_Block(64, 128)
        self.conv3 = Double_Block(128, 256)
        self.conv4 = Double_Block(256, 512)
        self.conv5 = Double_Block(512, 1024)

        self.maxpool = nn.MaxPool2d(2)

        # 新增dropout层
        self.dropout = nn.Dropout2d(0.3)

        # plaque
        self.upsample1_p = UpSample(1024, 512)
        self.upsample2_p = UpSample(512, 256)
        self.upsample3_p = UpSample(256, 128)
        self.upsample4_p = UpSample(128, 64)

        # 空间注意力机制
        # plaque
        # self.cam4_p = CAM(in_channels=1024)  # (512, 24, 32)
        self.cam3_p = CAM(in_channels=512)  # (256, 48, 64)
        self.cam2_p = CAM(in_channels=256)  # (128, 96, 128)
        self.cam1_p = CAM(in_channels=128)  # (64, 192, 256)

        # self.cross_conv4_p = nn.Conv2d(1024, 512, 3, 1, 1)
        self.cross_conv3_p = nn.Conv2d(1024, 512, 1)
        self.cross_conv2_p = nn.Conv2d(512, 256, 1)
        self.cross_conv1_p = nn.Conv2d(256, 128, 1)

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

        # self.cross_conv4_v = nn.Conv2d(1024, 512, 3, 1, 1)
        self.cross_conv3_v = nn.Conv2d(1024, 512, 1)
        self.cross_conv2_v = nn.Conv2d(512, 256, 1)
        self.cross_conv1_v = nn.Conv2d(256, 128, 1)

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

        # 新增dropout层
        x = self.dropout(x)

        feature_x = x

        # decoder1——carotid plaque
        x1 = self.upsample1_p(feature_x)  # [2, 512, 96, 128]
        # decoder2——blood vessel
        x2 = self.upsample1_v(feature_x)  # [2, 512, 96, 128]
        x1 = torch.cat([x1, conv4], dim=1)  # [2, 512 + 512, 96, 128]
        x2 = torch.cat([x2, conv4], dim=1)  # [2, 512 + 512, 96, 128]

        # decoder1——carotid plaque
        x1 = self.up_conv4_p(x1)  # [2, 512, 96, 128]
        # decoder2——blood vessel
        x2 = self.up_conv4_v(x2)  # [2, 512, 96, 128]
        x2 = self.cam3_p(x2)
        corss3_p = torch.cat([x1, x2], dim=1)  # [2, 512 + 512, 96, 128]
        corss3_v = torch.cat([x2, x1], dim=1)  # [2, 512 + 512, 96, 128]
        x1 = self.cross_conv3_p(corss3_p)  # [2, 512, 96, 128]
        x2 = self.cross_conv3_v(corss3_v)  # [2, 512, 96, 128]
        x1 = self.upsample2_p(x1)  # [2, 256, 192, 256]
        x2 = self.upsample2_v(x2)  # [2, 256, 192, 256]
        x1 = torch.cat([x1, conv3], dim=1)  # [2, 256 + 256, 192, 256]
        x2 = torch.cat([x2, conv3], dim=1)  # [2, 256 + 256, 192, 256]

        # decoder1——carotid plaque
        x1 = self.up_conv3_p(x1)  # [2, 256, 192, 256]
        # decoder2——blood vessel
        x2 = self.up_conv3_v(x2)  # [2, 256, 192, 256]
        x2 = self.cam2_p(x2)
        corss2_p = torch.cat([x1, x2], dim=1)  # [2, 256 + 256, 96, 128]
        corss2_v = torch.cat([x2, x1], dim=1)  # [2, 256 + 256, 96, 128]
        x1 = self.cross_conv2_p(corss2_p)  # [2, 256, 96, 128]
        x2 = self.cross_conv2_v(corss2_v)  # [2, 256, 96, 128]
        x1 = self.upsample3_p(x1)  # [2, 128, 384, 512]
        x2 = self.upsample3_v(x2)  # [2, 128, 384, 512]
        x1 = torch.cat([x1, conv2], dim=1)  # [2, 128 + 128, 384, 512]
        x2 = torch.cat([x2, conv2], dim=1)  # [2, 128 + 128, 384, 512]

        # decoder1——carotid plaque
        x1 = self.up_conv2_p(x1)  # [2, 128, 384, 512]
        # decoder2——blood vessel
        x2 = self.up_conv2_v(x2)  # [2, 128, 384, 512]
        x2 = self.cam1_p(x2)
        corss1_p = torch.cat([x1, x2], dim=1)  # [2, 128 + 128, 96, 128]
        corss1_v = torch.cat([x2, x1], dim=1)  # [2, 128 + 128, 96, 128]
        x1 = self.cross_conv1_p(corss1_p)  # [2, 128, 96, 128]
        x2 = self.cross_conv1_v(corss1_v)  # [2, 128, 96, 128]
        x1 = self.upsample4_p(x1)  # [2, 64, 768, 1024]
        x2 = self.upsample4_v(x2)  # [2, 64, 768, 1024]
        x1 = torch.cat([x1, conv1], dim=1)  # [2, 64 + 64, 768, 1024]
        x2 = torch.cat([x2, conv1], dim=1)  # [2, 64 + 64, 768, 1024]

        x1 = self.up_conv1_p(x1)  # [2, 64, 768, 1024]
        plaque_out = self.last_conv_p(x1)  # [2, 1, 768, 1024]

        x2 = self.up_conv1_v(x2)  # [2, 64, 768, 1024]
        vessel_out = self.last_conv_v(x2)  # [2, 1, 768, 1024]

        return plaque_out, vessel_out



if __name__ == '__main__':
    input = torch.randn([2, 1, 192, 256])
    net = CAFFUNet2(in_channels=1, out_channels=1)
    plaque_output, vessel_output = net(input)
    print(plaque_output.shape)  # torch.size([2, 1, 192, 256])
    print(vessel_output.shape)  # torch.size([2, 1, 192, 256])