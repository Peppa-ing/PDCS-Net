import torch
import torch.nn as nn

class MSFF_2(nn.Module):
    def __init__(self, in_channels):
        super(MSFF_2, self).__init__()
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_5x5 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_7x7 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_last = nn.Conv2d(4 * in_channels, in_channels, 1)
        # self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        out_1x1 = self.conv_1x1(x)
        out_3x3 = self.conv_3x3(x)
        out_5x5 = self.conv_5x5(x)
        out_7x7 = self.conv_7x7(x)
        out = self.conv_last(torch.cat((out_1x1, out_3x3, out_5x5, out_7x7), dim=1))
        # out = out * self.gamma + x * (1 - self.gamma)
        return out

class MSFF(nn.Module):
    """
        Multi-scale Feature Fusion 多尺度特征融合
    """

    def __init__(self, in_channels):
        super(MSFF, self).__init__()
        # 普通3x3卷积
        self.conv_common1x1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_common3x3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_common5x5 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        # 垂直方向3x1卷积
        self.conv_horizontal1x1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_horizontal3x1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_horizontal5x1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(5, 1), stride=1, padding=(2, 0)),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        # 水平方向1x3卷积
        self.conv_vertical1x1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_vertical1x3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_vertical1x5 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, 5), stride=1, padding=(0, 2)),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_common = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_horizontal = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_vertical = nn.Conv2d(in_channels, in_channels, 1)
        self.conv = nn.Conv2d(3 * in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        common1x1_out = self.conv_common1x1(x)
        common3x3_out = self.conv_common3x3(common1x1_out)
        common5x5_out = self.conv_common5x5(common3x3_out)
        common_out = self.conv_common(common1x1_out + common3x3_out + common5x5_out)
        horizontal1x1_out = self.conv_horizontal1x1(x)
        horizontal3x1_out = self.conv_horizontal3x1(horizontal1x1_out)
        horizontal5x1_out = self.conv_horizontal5x1(horizontal3x1_out)
        horizontal_out = self.conv_horizontal(horizontal1x1_out + horizontal3x1_out + horizontal5x1_out)
        vertical1x1_out = self.conv_vertical1x1(x)
        vertical1x3_out = self.conv_vertical1x3(vertical1x1_out)
        vertical1x5_out = self.conv_vertical1x5(vertical1x3_out)
        vertical_out = self.conv_vertical(vertical1x1_out + vertical1x3_out + vertical1x5_out)
        out = self.conv(torch.cat([common_out, horizontal_out, vertical_out], dim=1))
        out = out * self.gamma + x * (1 - self.gamma)
        return out


class MSFF_Dilation(nn.Module):
    """
        Multi-scale Feature Fusion 多尺度特征融合
    """

    def __init__(self, in_channels):
        super(MSFF_Dilation, self).__init__()
        # 普通3x3卷积——空洞卷积 dilation=1,2,3,5
        self.conv_common_d1 = nn.Sequential(  # Perception Field: k=3+(3-1)*(1-1)=3+0=3  (input+2*1-3)/1+1=input
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_common_d2 = nn.Sequential(  # Perception Field: k=3+(3-1)*(2-1)=3+2=5  (input+2*2-5)/1+1=input
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_common_d3 = nn.Sequential(  # Perception Field: k=3+(3-1)*(3-1)=3+4=7  (input+2*3-7)/1+1=input
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_common_d5 = nn.Sequential(  # Perception Field: k=3+(3-1)*(5-1)=3+8=11  (input+2*5-11)/1+1=input
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=5, dilation=5),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        # 垂直方向3x1卷积——空洞卷积 dilation=1,2,3,5
        self.conv_horizontal_d1 = nn.Sequential(  # Perception Field: k=3+(3-1)*(1-1)=3+0=3  (input+2*1-3)/1+1=input
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 1), padding=(1, 0), dilation=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_horizontal_d2 = nn.Sequential(  # Perception Field: k=3+(3-1)*(2-1)=3+2=5  (input+2*2-5)/1+1=input
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 1), padding=(2, 0), dilation=2),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_horizontal_d3 = nn.Sequential(  # Perception Field: k=3+(3-1)*(3-1)=3+4=7  (input+2*3-7)/1+1=input
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 1), padding=(3, 0), dilation=3),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_horizontal_d5 = nn.Sequential(  # Perception Field: k=3+(3-1)*(5-1)=3+8=11  (input+2*5-11)/1+1=input
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 1), padding=(5, 0), dilation=5),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        # 水平方向1x3卷积——空洞卷积 dilation=1,2,3
        self.conv_vertical_d1 = nn.Sequential(  # Perception Field: k=3+(3-1)*(1-1)=3+0=3  (input+2*1-3)/1+1=input
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, 3), padding=(0, 1), dilation=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_vertical_d2 = nn.Sequential(  # Perception Field: k=3+(3-1)*(2-1)=3+2=5  (input+2*2-5)/1+1=input
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, 3), padding=(0, 2), dilation=2),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_vertical_d3 = nn.Sequential(  # Perception Field: k=3+(3-1)*(3-1)=3+4=7  (input+2*3-7)/1+1=input
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, 3), padding=(0, 3), dilation=3),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_vertical_d5 = nn.Sequential(  # Perception Field: k=3+(3-1)*(5-1)=3+8=11  (input+2*5-11)/1+1=input
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, 3), padding=(0, 5), dilation=5),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_common = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_horizontal = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_vertical = nn.Conv2d(in_channels, in_channels, 1)
        self.conv = nn.Conv2d(3 * in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        common_out1 = self.conv_common_d1(x)
        common_out2 = self.conv_common_d2(common_out1)
        common_out3 = self.conv_common_d3(common_out2)
        common_out5 = self.conv_common_d5(common_out3)
        common_out = self.conv_common(common_out1 + common_out2 + common_out3 + common_out5)
        horizontal_out1 = self.conv_horizontal_d1(x)
        horizontal_out2 = self.conv_horizontal_d2(horizontal_out1)
        horizontal_out3 = self.conv_horizontal_d3(horizontal_out2)
        horizontal_out5 = self.conv_horizontal_d5(horizontal_out3)
        horizontal_out = self.conv_horizontal(horizontal_out1 + horizontal_out2 + horizontal_out3 + horizontal_out5)
        vertical_out1 = self.conv_vertical_d1(x)
        vertical_out2 = self.conv_vertical_d2(vertical_out1)
        vertical_out3 = self.conv_vertical_d3(vertical_out2)
        vertical_out5 = self.conv_vertical_d5(vertical_out3)
        vertical_out = self.conv_vertical(vertical_out1 + vertical_out2 + vertical_out3 + vertical_out5)
        out = self.conv(torch.cat([common_out, horizontal_out, vertical_out], dim=1))
        out = out * self.gamma + x * (1 - self.gamma)
        return out
