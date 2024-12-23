from argparse import Namespace
import yaml
from models.fasternet import get_model
from models.segmodel import UAFM, SPPM, SegHead, SPPM_T, UAFM_T_2, base_afm, base_ppm, UAFM_T, UAFM_T_1
import torch
import torch.nn as nn
import torch.nn.functional as F
from tools.benchmark import *

class netSeg(nn.Module):

    def __init__(
            self,
            num_classes=9,
            model_name='fasternet_t2',
            train_config=None):
        super().__init__()
        self.model = get_model(model_name, train_config)
        if "t0" in model_name:
            self.finallch = 320
        elif "t1" in model_name:
            self.finallch = 512
        elif "t2" in model_name:
            self.finallch = 768
        elif "s" in model_name:
            self.finallch = 1024
        else:
            raise ValueError("model_name error!")
        self.SPPM = SPPM(self.finallch, self.finallch // 2, self.finallch // 2)
        self.UAFM1 = UAFM(
            self.finallch // 2,
            self.finallch // 2,
            self.finallch // 4)
        self.UAFM2 = UAFM(
            self.finallch // 4,
            self.finallch // 4,
            self.finallch // 8)
        #self.SegHead1 = SegHead(self.finallch // 4, num_classes, num_classes)
        self.SegHead2 = SegHead(self.finallch // 8, num_classes, num_classes)
        self.SegHead_1 = SegHead(self.finallch // 2, self.finallch // 16, num_classes)
        self.SegHead_2 = SegHead(self.finallch // 4, self.finallch // 16, num_classes)
        self.SegHead_3 = SegHead(self.finallch // 8, self.finallch // 16, num_classes)
        self.SegHead_bundary = SegHead(self.finallch // 8, self.finallch // 16, 1)

    def forward(self, x):
        N, C, H, W = x.shape
        x = x[:, :3, :, :]
        feature_map = self.model(x)
        x = self.SPPM(feature_map[-1])
        x1 = self.UAFM1(x, feature_map[-2])
        x2 = self.UAFM2(x1, feature_map[-3])
        if self.training:
            seg_out1 = self.SegHead_1(x)
            seg_out1 = F.interpolate(seg_out1, size=(H, W), mode='bilinear', align_corners=True)
            seg_out2 = self.SegHead_2(x1)
            seg_out2 = F.interpolate(seg_out2, size=(H, W), mode='bilinear', align_corners=True)
            seg_out3 = self.SegHead_3(x2)
            seg_out3 = F.interpolate(seg_out3, size=(H, W), mode='bilinear', align_corners=True)
            seg_out_bundary = self.SegHead_bundary(x2)
            seg_out_bundary = F.interpolate(seg_out_bundary, size=(H, W), mode='bilinear', align_corners=True)
            return seg_out1, seg_out2, seg_out3, seg_out_bundary
        else:
            out = self.SegHead_3(x2)
            out = F.interpolate(
                out,
                size=(
                    H,
                    W),
                mode='bilinear',
                align_corners=False)
            return out


class netSeg_T(nn.Module):

    def __init__(
            self,
            num_classes=9,
            model_name='fasternet_t2+t1',
            multi_loss=False,
            train_config=None):
        super().__init__()
        self.multi_loss = multi_loss
        t0_size = [40, 80, 160, 320]
        t1_size = [64, 128, 256, 512]
        t2_size = [96, 192, 384, 768]
        s_size = [128, 256, 512, 1024]
        m_size = [144, 288, 576, 1152]
        l_size = [192, 384, 768, 1536]

        if "t0" in model_name.split("+")[0]:
            RGB_size = t0_size
        elif "t1" in model_name.split("+")[0]:
            RGB_size = t1_size
        elif "t2" in model_name.split("+")[0]:
            RGB_size = t2_size
        elif "s" in model_name.split("+")[0]:
            RGB_size = s_size
        elif "m" in model_name.split("+")[0]:
            RGB_size = m_size
        elif "l" in model_name.split("+")[0]:
            RGB_size = l_size
        else:
            raise ValueError("model_name error!")
        if "t0" in model_name.split("+")[1]:
            T_size = t0_size
        elif "t1" in model_name.split("+")[1]:
            T_size = t1_size
        elif "t2" in model_name.split("+")[1]:
            T_size = t2_size
        elif "s" in model_name.split("+")[1]:
            T_size = s_size
        elif "m" in model_name.split("+")[1]:
            T_size = m_size
        elif "l" in model_name.split("+")[1]:
            T_size = l_size
        else:
            raise ValueError("model_name error!")

        self.model_RGB = get_model(
            model_name.split("+")[0],
            train_config,
            in_chans=3)
        self.model_T = get_model(
            model_name.split("+")[1],
            train_config,
            in_chans=1)
        self.SegHead_1 = SegHead(RGB_size[2], RGB_size[0] // 2, num_classes)
        self.SegHead_2 = SegHead(RGB_size[1], RGB_size[0] // 2, num_classes)
        self.SegHead_3 = SegHead(RGB_size[0], RGB_size[0] // 2, num_classes)
        self.SegHead_bundary = SegHead(RGB_size[0], RGB_size[0] // 2, 1)
        if train_config is None:
            self.SPPM_T = SPPM_T(RGB_size[3], T_size[3], RGB_size[2])
            self.UAFM_T1 = UAFM_T(RGB_size[2], T_size[2], RGB_size[1])
            self.UAFM_T2 = UAFM_T(RGB_size[1], T_size[1], RGB_size[0])
        elif train_config.exp == "baseline":
            self.SPPM_T = base_ppm(RGB_size[3], T_size[3], RGB_size[2])
            self.UAFM_T1 = base_afm(RGB_size[2], T_size[2], RGB_size[1])
            self.UAFM_T2 = base_afm(RGB_size[1], T_size[1], RGB_size[0])
        elif train_config.exp == "base+SPPM":
            self.SPPM_T = SPPM_T(RGB_size[3], T_size[3], RGB_size[2])
            self.UAFM_T1 = base_afm(RGB_size[2], T_size[2], RGB_size[1])
            self.UAFM_T2 = base_afm(RGB_size[1], T_size[1], RGB_size[0])
        elif train_config.exp == "base+UAFM":
            self.SPPM_T = base_ppm(RGB_size[3], T_size[3], RGB_size[2])
            self.UAFM_T1 = UAFM_T(RGB_size[2], T_size[2], RGB_size[1])
            self.UAFM_T2 = UAFM_T(RGB_size[1], T_size[1], RGB_size[0])
        elif train_config.exp == "base+UAFM_1":
            self.SPPM_T = base_ppm(RGB_size[3], T_size[3], RGB_size[2])
            self.UAFM_T1 = UAFM_T_1(RGB_size[2], T_size[2], RGB_size[1])
            self.UAFM_T2 = UAFM_T_1(RGB_size[1], T_size[1], RGB_size[0])
        elif train_config.exp == "base+UAFM_2":
            self.SPPM_T = base_ppm(RGB_size[3], T_size[3], RGB_size[2])
            self.UAFM_T1 = UAFM_T_2(RGB_size[2], T_size[2], RGB_size[1])
            self.UAFM_T2 = UAFM_T_2(RGB_size[1], T_size[1], RGB_size[0])
        elif train_config.exp == "base+SPPM+UAFM":
            self.SPPM_T = SPPM_T(RGB_size[3], T_size[3], RGB_size[2])
            self.UAFM_T1 = UAFM_T(RGB_size[2], T_size[2], RGB_size[1])
            self.UAFM_T2 = UAFM_T(RGB_size[1], T_size[1], RGB_size[0])
        elif train_config.exp == "base+SPPM+UAFM_1":
            self.SPPM_T = SPPM_T(RGB_size[3], T_size[3], RGB_size[2])
            self.UAFM_T1 = UAFM_T_1(RGB_size[2], T_size[2], RGB_size[1])
            self.UAFM_T2 = UAFM_T_1(RGB_size[1], T_size[1], RGB_size[0])
        elif train_config.exp == "base+SPPM+UAFM_2":
            self.SPPM_T = SPPM_T(RGB_size[3], T_size[3], RGB_size[2])
            self.UAFM_T1 = UAFM_T_2(RGB_size[2], T_size[2], RGB_size[1])
            self.UAFM_T2 = UAFM_T_2(RGB_size[1], T_size[1], RGB_size[0])
        else:
            raise ValueError("train_config error!")
        self.UAFM_T1.vision_path = "vision_attention3/"
        self.UAFM_T2.vision_path = "vision_attention4/"

    def forward(self, x):
        N, C, H, W = x.shape
        feature_map_RGB = self.model_RGB(x[:, :3, :, :])
        feature_map_T = self.model_T(x[:, 3:, :, :])
        sppm_out = self.SPPM_T(feature_map_RGB[-1], feature_map_T[-1])
        uafm_out1 = self.UAFM_T1(sppm_out, feature_map_RGB[-2], feature_map_T[-2])
        uafm_out2 = self.UAFM_T2(uafm_out1, feature_map_RGB[-3], feature_map_T[-3])
        if self.training and self.multi_loss:
            seg_out1 = self.SegHead_1(sppm_out)
            seg_out1 = F.interpolate(seg_out1, size=(H, W), mode='bilinear', align_corners=True)
            seg_out2 = self.SegHead_2(uafm_out1)
            seg_out2 = F.interpolate(seg_out2, size=(H, W), mode='bilinear', align_corners=True)
            seg_out3 = self.SegHead_3(uafm_out2)
            seg_out3 = F.interpolate(seg_out3, size=(H, W), mode='bilinear', align_corners=True)
            seg_out_bundary = self.SegHead_bundary(uafm_out2)
            seg_out_bundary = F.interpolate(seg_out_bundary, size=(H, W), mode='bilinear', align_corners=True)
            return seg_out1, seg_out2, seg_out3, seg_out_bundary
        else:
            seg_out = self.SegHead_3(uafm_out2)
            seg_out = F.interpolate(seg_out, size=(H, W), mode='bilinear', align_corners=True)
            return seg_out

