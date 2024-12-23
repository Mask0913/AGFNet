import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from matplotlib import pyplot as plt


class SPPM(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(2),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        self.cov3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        self.covout = nn.Sequential(
            nn.Conv2d(inter_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.stage = nn.ModuleList([self.conv1, self.conv2, self.cov3])


    def forward(self, input):
        out = None
        for stage in self.stage:
            x = stage(input)
            x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=False)
            if out is None:
                out = x
            else:
                out += x
        out = self.covout(out)
        return out

class SPPM_T(nn.Module):
    def __init__(self, RGB_in_channels, T_in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(RGB_in_channels, RGB_in_channels//2, kernel_size=1, bias=False),
            nn.BatchNorm2d(RGB_in_channels//2),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(2),
            nn.Conv2d(RGB_in_channels, RGB_in_channels//2, kernel_size=1, bias=False),
            nn.BatchNorm2d(RGB_in_channels//2),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(RGB_in_channels, RGB_in_channels//2, kernel_size=1, bias=False),
            nn.BatchNorm2d(RGB_in_channels//2),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(T_in_channels, T_in_channels//2, kernel_size=1, bias=False),
            nn.BatchNorm2d(T_in_channels//2),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(2),
            nn.Conv2d(T_in_channels, T_in_channels//2, kernel_size=1, bias=False),
            nn.BatchNorm2d(T_in_channels//2),
            nn.ReLU(inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(T_in_channels, T_in_channels//2, kernel_size=1, bias=False),
            nn.BatchNorm2d(T_in_channels//2),
            nn.ReLU(inplace=True)
        )
        inter_channels = RGB_in_channels//2 + T_in_channels//2
        self.covout = nn.Sequential(
            nn.Conv2d(inter_channels, RGB_in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(RGB_in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(RGB_in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.stage_RGB = nn.ModuleList([self.conv1, self.conv2, self.conv3])
        self.stage_T = nn.ModuleList([self.conv4, self.conv5, self.conv6])


    def forward(self, rgb, t):
        rgb_out = None
        t_out = None
        for stage in self.stage_RGB:
            x = stage(rgb)
            x = F.interpolate(x, size=rgb.size()[2:], mode='bilinear', align_corners=False)
            if rgb_out is None:
                rgb_out = x
            else:
                rgb_out += x
        for stage in self.stage_T:
            x = stage(t)
            x = F.interpolate(x, size=t.size()[2:], mode='bilinear', align_corners=False)
            if t_out is None:
                t_out = x
            else:
                t_out += x
        # 拼接 rgb_out 和 t_out
        out = torch.cat((rgb_out, t_out), dim=1)
        out = self.covout(out)
        return out

class base_ppm(nn.Module):
    def __init__(self, RGB_in_channels, T_in_channels, out_channels):
        super().__init__()
        self.cov_out = nn.Sequential(
            nn.Conv2d(RGB_in_channels+T_in_channels, RGB_in_channels//2, kernel_size=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb, t):
        out = torch.cat((rgb, t), dim=1)
        out = self.cov_out(out)
        return out

class UAFM(nn.Module):
    """
        The UAFM with spatial attention, which uses mean and max values.
        Args:
            x_ch : The channel of x tensor, which is the low level feature.
            y_ch : The channel of y tensor, which is the high level feature.
            out_ch : The channel of output tensor.
            ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
            resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
        """
    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__()
        self.resize_mode = resize_mode
        self.convout = nn.Sequential(
            nn.Conv2d(y_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.convxy = nn.Sequential(
            nn.Conv2d(4, 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1)
        )
        self._scale = nn.Parameter(torch.ones(1), requires_grad=True)


    def avg_max_reduce_channel(self, x):
        # Reduce hw by avg and max
        # Return cat([avg_ch_0, max_ch_0, avg_ch_1, max_ch_1, ...])
        if not isinstance(x, (list, tuple)):
            return self.avg_max_reduce_channel_helper(x)
        elif len(x) == 1:
            return self.avg_max_reduce_channel_helper(x[0])
        else:
            res = []
            for xi in x:
                res.extend(self.avg_max_reduce_channel_helper(xi, use_concat=False))
            return torch.concat(res, dim=1)

    def avg_max_reduce_channel_helper(self, x, use_concat=True):
        # Reduce hw by avg and max, only support single input
        assert not isinstance(x, (list, tuple))
        mean_value = torch.mean(x, dim=1, keepdim=True)
        max_value = torch.max(x, dim=1, keepdim=True)
        max_value = max_value[0]
        if use_concat:
            res = torch.concat([mean_value, max_value], dim=1)
        else:
            res = [mean_value, max_value]
        return res

    def forward(self, x, y):
        x = F.interpolate(x, size=y.size()[2:], mode=self.resize_mode, align_corners=False)
        attention = self.avg_max_reduce_channel([x, y])
        attention = F.sigmoid(self.convxy(attention))
        out = x * attention + y * (self._scale - attention)
        out = self.convout(out)
        return out

class UAFM_T(nn.Module):

    def __init__(self, in_ch, t_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__()
        self.resize_mode = resize_mode
        self.convt = nn.Sequential(
            nn.Conv2d(t_ch, in_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )
        self.convout = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.convxy = nn.Sequential(
            nn.Conv2d(6, 3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1)
        )
        self.rgb_scale = nn.Parameter(torch.ones(1), requires_grad=True)
        self.t_scale = nn.Parameter(torch.ones(1), requires_grad=True)
        self.high_scale = nn.Parameter(torch.ones(1), requires_grad=True)
        self.vision_path = "vision_attention/"
        self.number = 0


    def mean_max_reduce_channel(self, x):
        # Reduce hw by avg and max
        # Return cat([avg_ch_0, max_ch_0, avg_ch_1, max_ch_1, ...])
        if not isinstance(x, (list, tuple)):
            return self.mean_max_reduce_channel_helper(x)
        elif len(x) == 1:
            return self.mean_max_reduce_channel_helper(x[0])
        else:
            res = []
            for xi in x:
                res.extend(self.mean_max_reduce_channel_helper(xi, use_concat=False))
            return torch.concat(res, dim=1)


    def mean_max_reduce_channel_helper(self, x, use_concat=True):
        # Reduce hw by avg and max, only support single input
        assert not isinstance(x, (list, tuple))
        mean_value = torch.mean(x, dim=1, keepdim=True)
        max_value = torch.max(x, dim=1, keepdim=True)
        max_value = max_value[0]
        if use_concat:
            res = torch.concat([mean_value, max_value], dim=1)
        else:
            res = [mean_value, max_value]
        return res

    def forward(self, high, rgb, t):
        high = F.interpolate(high, size=rgb.size()[2:], mode=self.resize_mode, align_corners=False)
        t = self.convt(t)
        #before_high = high.clone()
        #before_rgb = rgb.clone()
        #before_t = t.clone()
        attentions = self.mean_max_reduce_channel([high, rgb, t])
        attentions = F.sigmoid(self.convxy(attentions))
        out = self.rgb_scale * rgb * attentions + self.t_scale * t * attentions + self.high_scale * high * attentions
        out = self.convout(out)
        end_high = out.clone()
        # self.correlation(before_high, before_rgb, before_t, end_high)
        return out
    @torch.no_grad()
    def vision_attention(self, martix):
        # martix: [b, c, h, w]
        # mean max to 0 - 255
        pmin = np.min(martix)
        pmax = np.max(martix)
        martix = (martix - pmin) / (pmax - pmin + 1e-8) * 255
        martix = martix.astype(np.uint8)
        martix = cv2.applyColorMap(martix, cv2.COLORMAP_JET)
        martix = cv2.cvtColor(martix, cv2.COLOR_BGR2RGB)
        martix = cv2.resize(martix, (640, 480))
        return martix

    @torch.no_grad()
    def save_map_6(self, attentions):
        # 判断是否存在文件夹
        attentions2 = attentions.clone().cpu().numpy()
        vision_path = self.vision_path + str(self.number) + "/"
        if not os.path.exists(vision_path):
            os.makedirs(vision_path)
        attentions2 = attentions2.reshape(attentions2.shape[1], attentions2.shape[2], attentions2.shape[3])
        for i in range(attentions2.shape[0]):
            martix = attentions2[i]
            map = self.vision_attention(martix)
            path = vision_path + str(i) + ".png"
            cv2.imwrite(path, map)

    @torch.no_grad()
    def save_map_1(self, attentions):
        attentions1 = attentions.clone().cpu().numpy()
        vision_path = self.vision_path + str(self.number) + "/"
        attentions1 = attentions1.reshape(attentions1.shape[2], attentions1.shape[3])
        map = self.vision_attention(attentions1)
        path = vision_path + "mix" + ".png"
        cv2.imwrite(path, map)
        self.number += 1

    @torch.no_grad()
    def correlation(self, before_high, before_rgb, before_t, end_high):
        before_high = before_high.cpu().numpy().reshape(before_high.shape[1], before_high.shape[2], before_high.shape[3])
        before_rgb = before_rgb.cpu().numpy().reshape(before_rgb.shape[1], before_rgb.shape[2], before_rgb.shape[3])
        before_t = before_t.cpu().numpy().reshape(before_t.shape[1], before_t.shape[2], before_t.shape[3])
        end_high = end_high.cpu().numpy().reshape(end_high.shape[1], end_high.shape[2], end_high.shape[3])

        before_high = np.resize(before_high, (end_high.shape[0], before_high.shape[1], before_high.shape[2]))
        before_rgb = np.resize(before_rgb, (end_high.shape[0], before_rgb.shape[1], before_rgb.shape[2]))
        before_t = np.resize(before_t, (end_high.shape[0], before_t.shape[1], before_t.shape[2]))
        # 将 H x W x C 的矩阵转换成 C x (H x W) 的矩阵
        before_high = np.reshape(before_high, (before_high.shape[0], -1))
        before_rgb = np.reshape(before_rgb, (before_rgb.shape[0], -1))
        before_t = np.reshape(before_t, (before_t.shape[0], -1))
        end_high = np.reshape(end_high, (end_high.shape[0], -1))
        # 合并成一个2D数组
        combined_high_rgb = np.vstack((before_high, before_rgb))
        combined_high_t = np.vstack((before_high, before_t))
        combined_end_rgb = np.vstack((before_high, end_high))
        combined_end_t = np.vstack((before_t, end_high))
        # 计算每个对应位置的相关性
        correlation_matrix_high_rgb = np.corrcoef(combined_high_rgb)
        correlation_matrix_high_t = np.corrcoef(combined_high_t)
        correlation_matrix_end_rgb = np.corrcoef(combined_end_rgb)
        correlation_matrix_end_t = np.corrcoef(combined_end_t)
        # 保留正相关性，将负相关性设为零
        correlation_matrix_high_rgb = np.maximum(correlation_matrix_high_rgb, 0)
        correlation_matrix_high_t = np.maximum(correlation_matrix_high_t, 0)
        correlation_matrix_end_rgb = np.maximum(correlation_matrix_end_rgb, 0)
        correlation_matrix_end_t = np.maximum(correlation_matrix_end_t, 0)
        # 检查路径是否存在
        if not os.path.exists(self.vision_path + str(self.number)):
            os.makedirs(self.vision_path + str(self.number))
        # 设置相关性矩阵路径
        high_rgb_path = self.vision_path + str(self.number) + "/" + "high_rgb" + ".png"
        high_t_path = self.vision_path + str(self.number) + "/" + "high_t" + ".png"
        end_rgb_path = self.vision_path + str(self.number) + "/" + "end_rgb" + ".png"
        end_t_path = self.vision_path + str(self.number) + "/" + "end_t" + ".png"
        # 绘制相关性矩阵 保存
        plt.imshow(correlation_matrix_high_rgb, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.savefig(high_rgb_path)
        plt.imshow(correlation_matrix_high_t, cmap='viridis', interpolation='nearest')
        plt.savefig(high_t_path)
        plt.imshow(correlation_matrix_end_rgb, cmap='viridis', interpolation='nearest')
        plt.savefig(end_rgb_path)
        plt.imshow(correlation_matrix_end_t, cmap='viridis', interpolation='nearest')
        plt.savefig(end_t_path)
        plt.close()
        self.number += 1


class UAFM_T_1(nn.Module):
    def __init__(self, in_ch, t_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__()
        self.resize_mode = resize_mode
        self.convt = nn.Sequential(
            nn.Conv2d(t_ch, in_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )
        self.convout = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.convc = nn.Sequential(
            nn.Conv2d(in_ch*6, in_ch*3,kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch*3, in_ch,kernel_size=1, bias=False)
        )

        self.rgb_scale = nn.Parameter(torch.ones(1), requires_grad=True)
        self.t_scale = nn.Parameter(torch.ones(1), requires_grad=True)
        self.high_scale = nn.Parameter(torch.ones(1), requires_grad=True)

    def avg_max_reduce_spatial(self, x):
        # Reduce hw by avg and max
        # Return cat([avg_ch_0, max_ch_0, avg_ch_1, max_ch_1, ...])
        if not isinstance(x, (list, tuple)):
            return self.avg_max_reduce_spatial_helper(x)
        elif len(x) == 1:
            return self.avg_max_reduce_spatial_helper(x[0])
        else:
            res = []
            for xi in x:
                res.extend(self.avg_max_reduce_spatial_helper(xi, use_concat=False))
            return torch.concat(res, dim=1)

    def avg_max_reduce_spatial_helper(self, x, use_concat=True):
        assert not isinstance(x, (list, tuple))
        avg = F.adaptive_avg_pool2d(x, 1)
        max = F.adaptive_max_pool2d(x, 1)
        if use_concat:
            res = torch.concat([avg, max], dim=1)
        else:
            res = [avg, max]
        return res

    def forward(self, high, rgb, t):
        high = F.interpolate(high, size=rgb.size()[2:], mode=self.resize_mode, align_corners=False)
        t = self.convt(t)
        attentionc = self.avg_max_reduce_spatial([high, rgb, t])
        attentionc = self.convc(attentionc)
        attentionc = F.sigmoid(attentionc)
        out = self.rgb_scale * rgb * attentionc + self.t_scale * t * attentionc + self.high_scale * high * attentionc
        out = self.convout(out)
        return out

class UAFM_T_2(nn.Module):

    def __init__(self, in_ch, t_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__()
        self.resize_mode = resize_mode
        self.convt = nn.Sequential(
            nn.Conv2d(t_ch, in_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )
        self.convout = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.convs = nn.Sequential(
            nn.Conv2d(6, 3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1)
        )

        self.convc = nn.Sequential(
            nn.Conv2d(in_ch*6, in_ch*3,kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch*3, in_ch,kernel_size=1, bias=False)
        )

        self.rgb_scale = nn.Parameter(torch.ones(1), requires_grad=True)
        self.t_scale = nn.Parameter(torch.ones(1), requires_grad=True)
        self.high_scale = nn.Parameter(torch.ones(1), requires_grad=True)


    def mean_max_reduce_channel(self, x):
        # Reduce hw by avg and max
        # Return cat([avg_ch_0, max_ch_0, avg_ch_1, max_ch_1, ...])
        if not isinstance(x, (list, tuple)):
            return self.mean_max_reduce_channel_helper(x)
        elif len(x) == 1:
            return self.mean_max_reduce_channel_helper(x[0])
        else:
            res = []
            for xi in x:
                res.extend(self.mean_max_reduce_channel_helper(xi, use_concat=False))
            return torch.concat(res, dim=1)

    def avg_max_reduce_spatial(self, x):
        # Reduce hw by avg and max
        # Return cat([avg_ch_0, max_ch_0, avg_ch_1, max_ch_1, ...])
        if not isinstance(x, (list, tuple)):
            return self.avg_max_reduce_spatial_helper(x)
        elif len(x) == 1:
            return self.avg_max_reduce_spatial_helper(x[0])
        else:
            res = []
            for xi in x:
                res.extend(self.avg_max_reduce_spatial_helper(xi, use_concat=False))
            return torch.concat(res, dim=1)

    def avg_max_reduce_spatial_helper(self, x, use_concat=True):
        assert not isinstance(x, (list, tuple))
        avg = F.adaptive_avg_pool2d(x, 1)
        max = F.adaptive_max_pool2d(x, 1)
        if use_concat:
            res = torch.concat([avg, max], dim=1)
        else:
            res = [avg, max]
        return res

    def mean_max_reduce_channel_helper(self, x, use_concat=True):
        # Reduce hw by avg and max, only support single input
        assert not isinstance(x, (list, tuple))
        mean_value = torch.mean(x, dim=1, keepdim=True)
        max_value = torch.max(x, dim=1, keepdim=True)
        max_value = max_value[0]
        if use_concat:
            res = torch.concat([mean_value, max_value], dim=1)
        else:
            res = [mean_value, max_value]
        return res

    def forward(self, high, rgb, t):
        high = F.interpolate(high, size=rgb.size()[2:], mode=self.resize_mode, align_corners=False)
        t = self.convt(t)
        attentions = self.mean_max_reduce_channel([high, rgb, t])
        attentions = F.sigmoid(self.convs(attentions))
        rgb = rgb * attentions
        t = t * attentions
        high = high * attentions
        attentionc = self.avg_max_reduce_spatial([high, rgb, t])
        attentionc = self.convc(attentionc)
        attentionc = F.sigmoid(attentionc)
        out = self.rgb_scale * rgb * attentionc + self.t_scale * t * attentionc + self.high_scale * high * attentionc
        out = self.convout(out)
        return out

class base_afm(nn.Module):
    def __init__(self, in_ch, t_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__()
        self.resize_mode = resize_mode
        self.convt = nn.Sequential(
            nn.Conv2d(t_ch, in_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )
        self.convout = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, high, rgb, t):
        high = F.interpolate(high, size=rgb.size()[2:], mode=self.resize_mode, align_corners=False)
        t = self.convt(t)
        out = rgb + t + high
        out = self.convout(out)
        return out

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


if __name__ == '__main__':


    high = torch.randn(1, 256, 15, 20)
    rgb = torch.randn(1, 256, 30, 40)
    t = torch.randn(1, 160, 30, 40)
    model = UAFM_T(256, 160, 128)
    out = model(high, rgb, t)
    print(out.shape)

