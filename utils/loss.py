import torch.nn as nn
import torch

class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, preds, targets):  # (8, 1, 192, 256)
        # preds = torch.sigmoid(preds)
        m = nn.Sigmoid()
        preds = m(preds)
        batch_size = targets.size(0)
        loss = 0.
        smooth = 1.
        for i in range(batch_size):  # (8, 1, 192, 256)
            pred = preds[i]
            target = targets[i]
            # dice系数的定义
            dice_loss = (2. * (pred * target).sum() + smooth) / (pred.sum() + target.sum() + smooth)
            loss += (1 - dice_loss)
        return loss / batch_size

class BCE_Dice_Loss(nn.Module):

    def __init__(self):
        super(BCE_Dice_Loss, self).__init__()

    def forward(self, preds, targets):
        bce_fn = nn.BCEWithLogitsLoss()
        dice_fn = DiceLoss()
        bce = bce_fn(preds, targets)
        dice = dice_fn(preds, targets)
        return 0.5 * bce + 0.5 * dice