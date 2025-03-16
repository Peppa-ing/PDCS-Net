from torch import nn

from model.CAFFUNet import CAFFUNet2
from model.MSFFUNet import MSFFUNet


class IUNet3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(IUNet3, self).__init__()
        self.net1 = MSFFUNet(in_channels=1, out_channels=1)
        self.net2 = CAFFUNet2(in_channels=1, out_channels=1)


    def forward(self, x):
        outputs_p1, outputs_v1 = self.net1(x)
        outputs_p2, outputs_v2 = self.net2(x)

        return outputs_p1, outputs_v1, outputs_p2, outputs_v2
