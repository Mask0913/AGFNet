import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms
from PIL import Image

'''
DetailAggregateLoss 包括dice loos和ce loss，用seghead的边界和GT输进去计算，
 然后多次度loss是你模型1/4,1/8,1/16,1/32都去计算损失，最后这些损失加一块backward。
 DetailAggregateLoss的params需要放进优化器里面去优化，这样才能更新参数。
 example：
   optimizer = torch.optim.SGD([{'params': model.parameters()},
                             {'params': detail_loss.parameters()}],
                                lr=train_config.lr,
                                momentum=train_config.momentum,
                                weight_decay=train_config.weight_decay)

多损失函数example：
y1, y2, y3, boundary = self.model(image)
loss1 = self.seg_loss(y1, label)
loss2 = self.seg_loss(y2, label)
loss3 = self.seg_loss(y3, label)
loss4, loss5 = self.detail_loss(boundary, label)
loss = loss1 + loss2 + loss3 + loss4 + loss5
self.optimizer.zero_grad()
loss.backward()

y1, y2, y3分别是 1/8， 1/16， 32/1的结果，需要经过分割头计算。
+ 
boundary使用seghead计算的，最终输出一个类别的分割结果
seg_loss可以是各种损失函数，OHEM，交叉熵，BCE等等
detail_loss就是DetailAggregateLoss

seghead example：
class SegHead(nn.Module):
    def __init__(self, inchan, mid_chan, n_class):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inchan, mid_chan, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True)
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(mid_chan, n_class, kernel_size=1, bias=False)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv_out(x)
        return x
记得在Seghead之前上采样到输入的分辨率。
             
'''



class dice_loss(nn.Module):
    def __init__(self):
        super(dice_loss, self).__init__()
        self.smooth = 1.
    def forward(self, input, target):
        n = input.size(0)
        iflat = input.view(n, -1)
        tflat = target.view(n, -1)
        intersection = (iflat * tflat).sum(1)
        loss = 1 - ((2. * intersection + self.smooth) /
                    (iflat.sum(1) + tflat.sum(1) + self.smooth))
        return loss.mean()

def get_boundary(gtmasks):
    laplacian_kernel = torch.tensor(
        [-1, -1, -1, -1, 8, -1, -1, -1, -1],
        dtype=torch.float32, device=gtmasks.device).reshape(1, 1, 3, 3).requires_grad_(False)
    # boundary_logits = boundary_logits.unsqueeze(1)
    boundary_targets = F.conv2d(gtmasks.unsqueeze(1), laplacian_kernel, padding=1)
    boundary_targets = boundary_targets.clamp(min=0)
    boundary_targets[boundary_targets > 0.1] = 1
    boundary_targets[boundary_targets <= 0.1] = 0
    return boundary_targets

class DetailAggregateLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super(DetailAggregateLoss, self).__init__()
        self.dice_loss = dice_loss()
        # laplacian_kernel for Detail
        self.laplacian_kernel = torch.tensor(
            [-1, -1, -1, -1, 8, -1, -1, -1, -1],
            dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(False).type(torch.cuda.FloatTensor)

        # i dont konw the fuse kernel from where? how to get the kernel number.
        self.fuse_kernel = torch.nn.Parameter(torch.tensor([[6. / 10], [3. / 10], [1. / 10]],
                                                           dtype=torch.float32).reshape(1, 3, 1, 1).type(
            torch.cuda.FloatTensor))

    def forward(self, boundary_logits, gtmasks):
        # boundary_logits mean net out. gtmask mean GT to GT boundary.
        # boundary_logits = boundary_logits.unsqueeze(1)
        boundary_targets = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, padding=1)
        boundary_targets = boundary_targets.clamp(min=0)
        # 0.1 is threshold  > 0.1 mean boundary.
        boundary_targets[boundary_targets > 0.1] = 1
        boundary_targets[boundary_targets <= 0.1] = 0

        boundary_targets_x2 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel,
                                       stride=2, padding=1)
        boundary_targets_x2 = boundary_targets_x2.clamp(min=0)

        boundary_targets_x4 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel,
                                       stride=4, padding=1)
        boundary_targets_x4 = boundary_targets_x4.clamp(min=0)

        boundary_targets_x8 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel,
                                       stride=8, padding=1)
        boundary_targets_x8 = boundary_targets_x8.clamp(min=0)

        boundary_targets_x8_up = F.interpolate(boundary_targets_x8, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x4_up = F.interpolate(boundary_targets_x4, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x2_up = F.interpolate(boundary_targets_x2, boundary_targets.shape[2:], mode='nearest')

        boundary_targets_x2_up[boundary_targets_x2_up > 0.1] = 1
        boundary_targets_x2_up[boundary_targets_x2_up <= 0.1] = 0

        boundary_targets_x4_up[boundary_targets_x4_up > 0.1] = 1
        boundary_targets_x4_up[boundary_targets_x4_up <= 0.1] = 0

        boundary_targets_x8_up[boundary_targets_x8_up > 0.1] = 1
        boundary_targets_x8_up[boundary_targets_x8_up <= 0.1] = 0
        # stack stride 1 2 4 .
        boudary_targets_pyramids = torch.stack((boundary_targets, boundary_targets_x2_up, boundary_targets_x4_up),
                                               dim=1)

        boudary_targets_pyramids = boudary_targets_pyramids.squeeze(2)
        # the last cov out the Detail of GT.
        boudary_targets_pyramid = F.conv2d(boudary_targets_pyramids, self.fuse_kernel)

        boudary_targets_pyramid[boudary_targets_pyramid > 0.1] = 1
        boudary_targets_pyramid[boudary_targets_pyramid <= 0.1] = 0

        if boundary_logits.shape[-1] != boundary_targets.shape[-1]:
            boundary_logits = F.interpolate(
                boundary_logits, boundary_targets.shape[2:], mode='bilinear', align_corners=True)

        # compute the bce and dice loss from Detail Guide.
        bce_loss = F.binary_cross_entropy_with_logits(boundary_logits, boudary_targets_pyramid)
        dice_loss = self.dice_loss(torch.sigmoid(boundary_logits), boudary_targets_pyramid)
        return bce_loss, dice_loss

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            nowd_params += list(module.parameters())
        return nowd_params

class BootstrappedCELoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        crop_h, crop_w = config.train_crop_size
        self.K = int(config.batch_size*crop_h*crop_w/16)
        self.threshold = config.thresh
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=config.ignore_label, reduction="none"
        )

    def forward(self, logits, labels):
        pixel_losses = self.criterion(logits, labels).contiguous().view(-1)
        mask=(pixel_losses>self.threshold)
        if torch.sum(mask).item()>self.K:
            pixel_losses=pixel_losses[mask]
        else:
            pixel_losses, _ = torch.topk(pixel_losses, self.K)
        return pixel_losses.mean()

class OhemCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_label=255, thresh=0.5, min_kept=10000, use_weight=True):
        super().__init__()
        self.ignore_index = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            weight = torch.FloatTensor(
                [0.094, 0.454, 0.862, 0.89, 1.288, 1.678, 7.497, 3.425, 2.337])
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)
        else:
             self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)


    def forward(self, pred, target):
        n, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()
        prob = F.softmax(pred, dim=1)
        prob = prob.transpose(0, 1).reshape(c, -1)
        if self.min_kept > num_valid:
            print("Lables: {}".format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
            kept_mask = mask_prob.le(threshold)
            valid_mask = valid_mask * kept_mask
            target = target * kept_mask.long()
        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(n, h, w)
        return self.criterion(pred, target)

if __name__ == '__main__':
    img = torch.randn(2, 1, 480, 640)
    label = torch.randint(0, 9, (2, 480, 640))
    img = img.cuda()
    label = label.cuda()
    #label = torchvision.transforms.ToTensor()(label)
    detail_loss = DetailAggregateLoss()
    ce_loss, dice_loss = detail_loss(img, label)
    print(ce_loss)
    print(dice_loss)













