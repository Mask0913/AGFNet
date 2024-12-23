from argparse import Namespace
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.multiformer.swift_transformer import get_model
from models.seg_former import ShortAttention, LongAttention, MidAttention, SegHead
from tools.benchmark import *


class MultiSeg(nn.Module):
    def __init__(
            self,
            num_classes=9,
            model_name='',
            multi_loss=False,
            train_config=None):
        super().__init__()
        XS_size = [48, 56, 112, 220]
        S_size = [48, 64, 168, 224]
        self.multi_loss = multi_loss

        if "XS" in model_name.split("+")[0]:
            self.RGB_size = XS_size
        elif "S" in model_name.split("+")[0]:
            self.RGB_size = S_size
        else:
            raise ValueError("model_name error!")
        if "XS" in model_name.split("+")[1]:
            self.T_size = XS_size
        elif "S" in model_name.split("+")[1]:
            self.T_size = S_size
        else:
            raise ValueError("model_name error!")
        self.RGB_raw_model = get_model(model_name.split("+")[0],
                                        "RGB")
        self.T_raw_model = get_model(model_name.split("+")[1],
                                        "T")
        self.RGB_patch_embed = self.RGB_raw_model.patch_embed
        self.RGB_network = self.RGB_raw_model.network
        self.RGB_norm = self.RGB_raw_model.get_normal()
        self.T_patch_embed = self.T_raw_model.patch_embed
        self.T_network = self.T_raw_model.network
        self.T_norm = self.T_raw_model.get_normal()

        # short fusion model
        short_attentions = []
        for i in range(4):
            short_attentions.append(ShortAttention(self.RGB_size[i], self.T_size[i]))
        self.short_attentions = nn.Sequential(*short_attentions)

        # mid fusion model
        self.mid_attention = MidAttention(self.RGB_size[3], self.T_size[3])

        # long fusion model
        self.long_attention1 = LongAttention(self.RGB_size[3], self.RGB_size[2], self.T_size[2])
        self.long_attention2 = LongAttention(self.RGB_size[2], self.RGB_size[1], self.T_size[1])

        # seg head
        if self.multi_loss:
            self.seg_head1 = SegHead(self.RGB_size[3],self.RGB_size[3] // 2, num_classes)
            self.seg_head2 = SegHead(self.RGB_size[2],self.RGB_size[2] // 2, num_classes)
            self.seg_head3 = SegHead(self.RGB_size[1],self.RGB_size[1] // 2, num_classes)
            self.seg_head_bundary = SegHead(self.RGB_size[1], self.RGB_size[1] // 2, 1)
        else:
            self.seg_head3 = SegHead(self.RGB_size[1],self.RGB_size[1] // 2, num_classes)

    def forward(self, x):# normal in 0 , 2 , 4, 6

        # prepare data
        N, C, H, W = x.shape
        RGB = x[:, :3, :, :]
        T = x[:, 3:, :, :]

        # stage 1
        RGB_x = self.RGB_patch_embed(RGB)
        RGB_x = self.RGB_network[0](RGB_x)
        RGB_x = self.RGB_norm[0](RGB_x)
        T_x = self.T_patch_embed(T)
        T_x = self.T_network[0](T_x)
        T_x = self.T_norm[0](T_x)
        RGB_x, T_x = self.short_attentions[0](RGB_x, T_x)

        # stage 2
        RGB_x = self.RGB_network[1](RGB_x)
        RGB_x = self.RGB_network[2](RGB_x)
        RGB_x = self.RGB_norm[1](RGB_x)
        T_x = self.T_network[1](T_x)
        T_x = self.T_network[2](T_x)
        T_x = self.T_norm[1](T_x)
        RGB_x, T_x = self.short_attentions[1](RGB_x, T_x)
        RGB_x_2, T_x_2 = RGB_x, T_x

        # stage 3
        RGB_x = self.RGB_network[3](RGB_x)
        RGB_x = self.RGB_network[4](RGB_x)
        RGB_x = self.RGB_norm[2](RGB_x)
        T_x = self.T_network[3](T_x)
        T_x = self.T_network[4](T_x)
        T_x = self.T_norm[2](T_x)
        RGB_x, T_x = self.short_attentions[2](RGB_x, T_x)
        RGB_x_3, T_x_3 = RGB_x, T_x

        # stage 4
        RGB_x = self.RGB_network[5](RGB_x)
        RGB_x = self.RGB_network[6](RGB_x)
        RGB_x = self.RGB_norm[3](RGB_x)
        T_x = self.T_network[5](T_x)
        T_x = self.T_network[6](T_x)
        T_x = self.T_norm[3](T_x)
        RGB_x, T_x = self.short_attentions[3](RGB_x, T_x)

        # fusion and upsample
        features1 = self.mid_attention(RGB_x, T_x)
        features2 = self.long_attention1(features1, RGB_x_3, T_x_3)
        features3 = self.long_attention2(features2, RGB_x_2, T_x_2)

        # seg head
        if self.training and self.multi_loss:
            features1 = F.interpolate(features1, size=(H, W), mode='bilinear', align_corners=True)
            seg_out1 = self.seg_head1(features1)
            features2 = F.interpolate(features2, size=(H, W), mode='bilinear', align_corners=True)
            seg_out2 = self.seg_head2(features2)
            features3 = F.interpolate(features3, size=(H, W), mode='bilinear', align_corners=True)
            seg_out3 = self.seg_head3(features3)
            bundary_out = self.seg_head_bundary(features3)
            return seg_out1, seg_out2, seg_out3, bundary_out
        else:
            seg = F.interpolate(features3, size=(H, W), mode='bilinear', align_corners=True)
            seg = self.seg_head3(seg)
            return seg


if __name__ == '__main__':
    model = MultiSeg(model_name="S+XS", multi_loss=True, num_classes=9)

    RGB_T = torch.randn(1, 4, 480, 640)
    features = model(RGB_T)
    for i in features:
        print(i.shape)


