import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * scale_factor ** 2, kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.block(x)

class Generator(nn.Module):
    def __init__(self, which_scale, in_channels=3, num_filters=64, num_res_blocks=23):
        super().__init__()

        self.scale = which_scale

        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(*[ResidualBlock(num_filters) for _ in range(num_res_blocks)])
        self.mid_conv = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)

        self.upsampling = UpsampleBlock(num_filters, scale_factor=2) #255->510
        self.upsampling2 = UpsampleBlock(num_filters, scale_factor=2)
        self.upsampling3 = UpsampleBlock(num_filters, scale_factor=2) #1020->2040

        self.output1 = self.output1 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(num_filters, in_channels, kernel_size=3, padding=1),
        )

        self.output2 = self.output1 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(num_filters, in_channels, kernel_size=5, padding=2),
        )

        self.output3 = self.output1 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=9, padding=4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(num_filters, in_channels, kernel_size=9, padding=4),
        )

    def forward(self, x):

        x1 = self.conv1(x)
        res = self.res_blocks(x1)
        mid = self.mid_conv(res)
        out = x1 + mid  

        if self.scale == 8:

            out = self.upsampling(out)
            out = self.upsampling2(out)
            out = self.upsampling3(out)

            out1 = self.output1(out)
            out2 = self.output2(out)
            out3 = self.output3(out)

            return out1, out2, out3

        elif self.scale == 4:

            out = self.upsampling(out)
            out = self.upsampling2(out)

            out1 = self.output1(out)
            out2 = self.output2(out)
            out3 = self.output2(out)

            return out1, out2, out3

        else:
            out = self.upsampling(out)

            out1 = self.output1(out)
            out2 = self.output2(out)
            out3 = self.output2(out)

            return out1, out2, out3
