import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskGuidedDualAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels//8, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels//8, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x, mask):
        spatial_att = self.spatial_att(x)
        guided_att = spatial_att * mask + spatial_att

        channel_att = self.channel_att(x)

        return x * guided_att * channel_att

class LightResDenseBlock(nn.Module):
    """ (50%)"""
    def __init__(self, in_channels, growth_rate=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, growth_rate, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels + growth_rate, growth_rate, 3, padding=1)
        self.conv3 = nn.Conv2d(in_channels + 2*growth_rate, in_channels, 1)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.conv3(torch.cat([x, x1, x2], 1))
        return x + x3

class AdaIN(nn.Module):
    """"""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels)

    def forward(self, x, style):
        style_mean = style.mean(dim=[2,3], keepdim=True)
        style_std = style.std(dim=[2,3], keepdim=True) + 1e-8

        normalized = self.norm(x)
        return normalized * style_std + style_mean

class MDAGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64):
        super().__init__()
        self.init_conv = nn.Conv2d(in_channels, base_channels, 7, padding=3)

        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*4, 3, stride=2, padding=1),
            nn.ReLU()
        )

        self.resblock1 = LightResDenseBlock(base_channels*4)
        self.resblock2 = LightResDenseBlock(base_channels*4)
        self.resblock3 = LightResDenseBlock(base_channels*4)

        self.att = MaskGuidedDualAttention(base_channels*4)

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*4, base_channels*2, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*2, base_channels, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )

        self.adain = AdaIN(base_channels)

        self.out_conv = nn.Conv2d(base_channels, out_channels, 7, padding=3)

    def forward(self, x, mask):
        x0 = F.relu(self.init_conv(x))

        x1 = self.down1(x0)
        x2 = self.down2(x1)

        res = self.resblock1(x2)
        res = self.resblock2(res)
        res = self.resblock3(res)

        mask = F.interpolate(mask, size=res.shape[2:], mode='bilinear', align_corners=False)
        att_res = self.att(res, mask)

        up1 = self.up1(att_res)
        up2 = self.up2(up1 + x1)

        adain_out = self.adain(up2 + x0, res)

        return torch.tanh(self.out_conv(adain_out))

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.scale1 = self._make_scale(in_channels)
        self.scale2 = self._make_scale(in_channels)
        self.scale3 = self._make_scale(in_channels)
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1)

    def _make_scale(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=2),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=2),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, stride=1, padding=2),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, stride=1, padding=2)
        )

    def forward(self, x):
        out1 = self.scale1(x)
        x = self.downsample(x)
        out2 = self.scale2(x)
        x = self.downsample(x)
        out3 = self.scale3(x)
        return [out1, out2, out3]