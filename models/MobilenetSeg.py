from argparse import Namespace

import yaml

from models.MobileNetV3 import get_model
from models.segmodel import UAFM, SPPM, SegHead, SPPM_T, UAFM_T_2, base_afm, base_ppm, UAFM_T, UAFM_T_1
import torch
import torch.nn as nn
import torch.nn.functional as F
from tools.benchmark import *

class MobileSeg(nn.Module):
    def __init__(
            self,
            num_classes=9,
            model_name='mobile_l+s',
            train_config=None):
        super().__init__()
        small = [16,24,40,96]
        large = [24,40,112,160]
        if "s" in model_name.split("+")[0]:
            RGB_size = small
        elif "l" in model_name.split("+")[0]:
            RGB_size = large
        else:
            raise NotImplementedError
        if "s" in model_name.split("+")[1]:
            T_size = small
        elif "l" in model_name.split("+")[1]:
            T_size = large
        else:
            raise NotImplementedError
        self.model_RGB = get_model(
            model_name.split("+")[0],
            None,
            in_chans=3)
        self.model_T = get_model(
            model_name.split("+")[1],
            None,
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
    def forward(self, x):
        N, C, H, W = x.shape
        feature_map_RGB = self.model_RGB(x[:, :3, :, :])
        feature_map_T = self.model_T(x[:, 3:, :, :])
        sppm_out = self.SPPM_T(feature_map_RGB[-1], feature_map_T[-1])
        uafm_out1 = self.UAFM_T1(sppm_out, feature_map_RGB[-2], feature_map_T[-2])
        uafm_out2 = self.UAFM_T2(uafm_out1, feature_map_RGB[-3], feature_map_T[-3])
        if self.training:
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

if __name__ == '__main__':
    with open("configs/train_MF_mobile.yaml", 'r') as f:
        args = yaml.safe_load(f)
    train_config = Namespace(**args)
    x = torch.randn(2, 4, 480, 640)
    model = MobileSeg(model_name="mobile_l+s", train_config=train_config)
    model.cuda()
    model.eval()
    x = x.cuda()
    eval_time = compute_eval_time(model, "cuda", 10, 50, (480, 640), False, channel=4)
    flops, params = compute_flops(model, "cuda", val_input_size=(480, 640), channels=4)
    print("netname:{} eval_time:{} flops:{} params:{}".format("mobileNet_l_s", eval_time, flops, params))
    #  flops:3.941968128 params:4.182270000000001