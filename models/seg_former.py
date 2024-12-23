import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
import einops


class LongAttention(nn.Module):

    def __init__(self, High_channels, RGB_channels, T_channels,  drop_path=0., layer_scale_init_value=1e-5):
        super().__init__()
        self.low_channels = RGB_channels + T_channels
        self.high_CBR = CBR(High_channels, RGB_channels)
        self.low_CBR = CBR(self.low_channels, RGB_channels)

        self.High_conv0 = ConvEncoder(RGB_channels, RGB_channels * 4)
        self.High_conv1 = ConvEncoder(RGB_channels, RGB_channels)
        self.low_conv0 = ConvEncoder(RGB_channels, RGB_channels * 4)
        self.low_conv1 = ConvEncoder(RGB_channels, RGB_channels)

        self.Cross_attention = Cross_modal_attention(RGB_channels, RGB_channels)
        self.Mlp = Mlp(RGB_channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(
            RGB_channels).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(
            RGB_channels).unsqueeze(-1).unsqueeze(-1), requires_grad=True)

    def forward(self, high, RGB, T):
        low = torch.cat((RGB, T), dim=1)
        high = F.interpolate(high, size=low.size()[2:], mode="bilinear", align_corners=False)
        high = self.high_CBR(high)
        low = self.low_CBR(low)
        l_B, l_C, l_H, l_W = low.shape
        h_B, h_C, h_H, h_W = high.shape

        high = self.High_conv0(high)
        high = self.High_conv1(high)
        low = self.low_conv0(low)
        low = self.low_conv1(low)
        high = high + self.drop_path(self.layer_scale_1 * self.Cross_attention(
            high.permute(0, 2, 3, 1).reshape(h_B, h_H * h_W, h_C),
            low.permute(0, 2, 3, 1).reshape(l_B, l_H * l_W, l_C)
        ).reshape(h_B, h_H, h_W, h_C).permute(0, 3, 1, 2))
        high = high + self.drop_path(self.layer_scale_2 * self.Mlp(high))
        return high

class MidAttention(nn.Module):

    def __init__(self, RGB_channels, T_channels, drop_path=0.,
            layer_scale_init_value=1e-5):
        super().__init__()
        self.CBR = CBR(RGB_channels + T_channels, RGB_channels)
        self.conv0 = ConvEncoder(RGB_channels, RGB_channels * 4)
        self.conv1 = ConvEncoder(RGB_channels, RGB_channels)
        self.Cross_attention = Cross_modal_attention(RGB_channels, RGB_channels)
        self.Mlp = Mlp(RGB_channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(
            RGB_channels).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(
            RGB_channels).unsqueeze(-1).unsqueeze(-1), requires_grad=True)

    def forward(self, RGB, T):
        x = torch.cat((RGB, T), dim=1)
        x = self.CBR(x)
        B, C, H, W = x.shape
        x = self.conv0(x)
        x = self.conv1(x)
        x = x + self.drop_path(self.layer_scale_1 * self.Cross_attention(
            x.permute(0, 2, 3, 1).reshape(B, H * W, C),
            x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        ).reshape(B, H, W, C).permute(0, 3, 1, 2))
        x = x + self.drop_path(self.layer_scale_2 * self.Mlp(x))
        return x

class ShortAttention(nn.Module):

    def __init__(
            self,
            RGB_in_channels,
            T_in_channels,
            drop_path=0.,
            layer_scale_init_value=1e-5):
        super().__init__()
        self.RGB_conv0 = ConvEncoder(RGB_in_channels, RGB_in_channels * 4)
        self.T_conv0 = ConvEncoder(T_in_channels, T_in_channels * 4)
        self.RGB_conv1 = ConvEncoder(RGB_in_channels, RGB_in_channels)
        self.T_conv1 = ConvEncoder(T_in_channels, T_in_channels)
        self.RGB_cross_attention = Cross_modal_attention(
            RGB_in_channels, T_in_channels)
        self.T_cross_attention = Cross_modal_attention(
            T_in_channels, RGB_in_channels)
        self.RGB_Mlp = Mlp(RGB_in_channels)
        self.T_Mlp = Mlp(T_in_channels)

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.RGB_layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(
            RGB_in_channels).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
        self.RGB_layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(
            RGB_in_channels).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
        self.T_layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(
            T_in_channels).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
        self.T_layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(
            T_in_channels).unsqueeze(-1).unsqueeze(-1), requires_grad=True)

    def forward(self, RGB, T):
        R_B, R_C, R_H, R_W = RGB.shape
        T_B, T_C, T_H, T_W = T.shape
        RGB_x = self.RGB_conv0(RGB)
        RGB_x = self.RGB_conv1(RGB_x)
        T_x = self.T_conv0(T)
        T_x = self.T_conv1(T_x)
        RGB_x_copy = RGB_x
        RGB_x = RGB_x + self.drop_path(self.RGB_layer_scale_1 * self.RGB_cross_attention(
            RGB_x.permute(0, 2, 3, 1).reshape(R_B, R_H * R_W, R_C),
            T_x.permute(0, 2, 3, 1).reshape(T_B, T_H * T_W, T_C)
        ).reshape(R_B, R_H, R_W, R_C).permute(0, 3, 1, 2))
        T_x = T_x + self.drop_path(self.T_layer_scale_1 * self.T_cross_attention(
            T_x.permute(0, 2, 3, 1).reshape(T_B, T_H * T_W, T_C),
            RGB_x_copy.permute(0, 2, 3, 1).reshape(R_B, R_H * R_W, R_C)
        ).reshape(T_B, T_H, T_W, T_C).permute(0, 3, 1, 2))
        RGB_x = RGB_x + self.drop_path(self.RGB_layer_scale_2 * self.RGB_Mlp(RGB_x))
        T_x = T_x + self.drop_path(self.T_layer_scale_2 * self.T_Mlp(T_x))
        return RGB_x, T_x


class CBR(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(CBR, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class ConvEncoder(nn.Module):
    def __init__(
            self,
            dim,
            hidden_dim=64,
            kernel_size=3,
            drop_path=0.,
            use_layer_scale=True):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim)
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(torch.ones(
                dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.use_layer_scale:
            x = input + self.drop_path(self.layer_scale * x)
        else:
            x = input + self.drop_path(x)
        return x


class Cross_modal_attention(nn.Module):

    def __init__(self, in_a_dims=512, in_b_dims=512):
        super().__init__()
        self.to_query = nn.Linear(in_a_dims, in_a_dims)
        self.to_key = nn.Linear(in_b_dims, in_a_dims)
        self.w_g = nn.Parameter(torch.randn(in_a_dims, 1))
        self.scale_factor = in_a_dims ** -0.5
        self.Proj = nn.Linear(in_a_dims, in_a_dims)
        self.final = nn.Linear(in_a_dims, in_a_dims)

    def forward(self, a, b):
        query = self.to_query(a)
        key = self.to_key(b)
        query = torch.nn.functional.normalize(query, dim=-1)  # BxNxD
        key = torch.nn.functional.normalize(key, dim=-1)  # BxNxD
        query_weight = query @ self.w_g  # BxNx1 (BxNxD @ Dx1) w_g: 可学习令牌参数
        A = query_weight * self.scale_factor  # BxNx1
        A = torch.nn.functional.normalize(A, dim=1)  # BxNx1
        G = torch.sum(A * query, dim=1)  # BxD
        G = einops.repeat(
            G, "b d -> b repeat d", repeat=key.shape[1]
        )  # BxNxD
        out = self.Proj(G * key) + query  # BxNxD
        out = self.final(out)  # BxNxD
        return out


class Mlp(nn.Module):
    def __init__(self, in_dim, drop=0.):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_dim)
        self.fc1 = nn.Conv2d(in_dim, in_dim * 4, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(in_dim * 4, in_dim, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

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
    RGB = torch.randn(1, 400, 20, 20)
    T = torch.randn(1, 380, 20, 20)
    midatten = MidAttention(400, 380)
    x = midatten(RGB, T)
    print(x.shape)

