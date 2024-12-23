import torch
import torch.nn as nn
import torch.nn.functional as F
from models.multiformer.swift_transformer import get_model
from models.seg_former import ShortAttention, LongAttention, MidAttention, SegHead
from tools.benchmark import *
from models.segmodel import UAFM, SPPM


class FormerExp_RGB(nn.Module):

    def __init__(
            self,
            num_classes=9,
            model_name='S',
            multi_loss=False,
            exp="",
            train_config=None):
        super().__init__()
        XS_size = [48, 56, 112, 220]
        S_size = [48, 64, 168, 224]
        if "XS" in model_name:
            self.RGB_size = XS_size
        elif "S" in model_name:
            self.RGB_size = S_size
        else:
            raise ValueError("model_name error!")
        self.raw_model = get_model(model_name,
                                       "RGB")
        self.patch_embed = self.raw_model.get_patch_embed()
        self.network = self.raw_model.get_network()
        self.norm = self.raw_model.get_normal()
        del self.raw_model
        self.SPPM = SPPM(self.RGB_size[3], self.RGB_size[2], self.RGB_size[2])
        self.UAFM1 = UAFM(self.RGB_size[2], self.RGB_size[2], self.RGB_size[1])
        self.UAFM2 = UAFM(self.RGB_size[1], self.RGB_size[1], self.RGB_size[0])
        self.seg_head = SegHead(self.RGB_size[0], self.RGB_size[0] // 2, num_classes)

    def forward(self, x):
        x = x[:, :3, :, :]
        N,C,H,W = x.shape
        x = self.patch_embed(x)
        x0 = self.network[0](x)
        x0 = self.norm[0](x0)
        x1 = self.network[1](x0)
        x2 = self.network[2](x1)
        x2 = self.norm[1](x2)
        x3 = self.network[3](x2)
        x4 = self.network[4](x3)
        x4 = self.norm[2](x4)
        x5 = self.network[5](x4)
        x6 = self.network[6](x5)
        x6 = self.norm[3](x6)
        features = [x0, x2, x4, x6]
        x = self.SPPM(features[3])
        x = self.UAFM1(x, features[2])
        x = self.UAFM2(x, features[1])
        x = self.seg_head(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x


if __name__ == '__main__':
    x = torch.randn(1, 4, 480, 640)
    model = FormerExp_RGB(model_name="S", num_classes=9)
    y = model(x)
    print(y)

