import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    '''
    Camada de bloco residual (RRDB)
    '''
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
    '''
    Camada de UpSampling modificada 
    com uma convolução antes do PixelShuffle
    '''
    def __init__(self, in_channels, scale_factor=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels*scale_factor**2, kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.block(x)

#Camada geradora modificada
#Os valores de canais, filtros e número de camadas de bloco são baseadas no artigo
class Generator(nn.Module):
    '''
    Camada geradora modificada para gerar 3 saídas
    Cada saída possui uma camada de output diferente
    Os valores de canais, filtros e número de camadas de bloco são baseadas no artigo
    '''
    def __init__(self, which_scale, in_channels=3, num_filters=64, num_res_blocks=23):
        super().__init__()

        self.scale = which_scale 

        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(*[ResidualBlock(num_filters) for _ in range(num_res_blocks)])
        self.mid_conv = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)

        #Importante notar que essas camadas 
        #só são ativadas dependendo da escala escolhida
        self.upsampling = UpsampleBlock(num_filters, scale_factor=2) #128-256
        self.upsampling2 = UpsampleBlock(num_filters, scale_factor=2) #256-512
        self.upsampling3 = UpsampleBlock(num_filters, scale_factor=2) #512-1024

        #Essas camadas foram criadas com base na condicional 
        self.output1 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters*2, kernel_size=9, padding=4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(num_filters*2, num_filters, kernel_size=9, padding=4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(num_filters, in_channels, kernel_size=9, padding=4),
        )

        self.output2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=9, padding=4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(num_filters, in_channels, kernel_size=9, padding=4),
        )

        self.output3 = nn.Sequential(
            nn.Conv2d(num_filters, in_channels, kernel_size=9, padding=4),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        res = self.res_blocks(x1)
        mid = self.mid_conv(res)
        out = x1 + mid  # skip connection

        #condicionais dependendo da escala escolhida
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
    
class Discriminator(nn.Module):
    '''
    Discriminadora seguindo o artigo Wang et al. 2018
    '''
    def __init__(self, in_channels=3):
        super().__init__()

        def block(in_f, out_f, kernel_size=3, stride=1):
            layers = [nn.Conv2d(in_f, out_f, kernel_size, stride, padding=1)]
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, 64, stride=1),  
            *block(64, 64, stride=2),  
            *block(64, 128, stride=1),
            *block(128, 128, stride=2),
            *block(128, 256, stride=1),
            *block(256, 256, stride=2),
            *block(256, 512, stride=1),
            *block(512, 512, stride=2),
            nn.AdaptiveAvgPool2d(1),  #reduz para (batch_size, 512, 1, 1)
            nn.Conv2d(512, 1, kernel_size=1)  
        )

    def forward(self, x):
        out = self.model(x)
        return out.view(out.size(0), -1)  # shape (batch, 1)
