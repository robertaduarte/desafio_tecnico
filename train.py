import os
from PIL import Image

from import_data import proximo_batch
from evaluation import evaluation 
from architecture import *
from perceptual_loss import *

from torch.utils.data import DataLoader
import torchvision.transforms.functional as F

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from tqdm import tqdm

import matplotlib.pyplot as plt

lr_dir = "/diretorio/das/imagens/de/baixa/resolucao/" 
hr_dir = "/diretorio/das/imagens/de/alta/resolucao/" 

lr_dir_val = "/diretorio/das/imagens/de/baixa/resolucao/para/validacao/" 
hr_dir_val = "/diretorio/das/imagens/de/alta/resolucao/para/validacao/" 

output_dir = "/diretorio/para/salvar/modelo/" 

'''
Escolha a escala que deseja aumentar: 2x, 4x, 8x
Os condicionais devem ser alteradores dependendo do tamanho das imagens
mas respeitando os aumento de 2x, 4x, 8x
'''

scale = 4

if scale == 2:

  img_size_input = (256, 128)
  img_size_target = (512, 256)

elif scale == 4:

  img_size_input = (128, 64)
  img_size_target = (512, 256)

else: 

  img_size_input = (128, 64)
  img_size_target = (1024, 512)


'''
Definições iniciais

lr: escolhido como default
beta: escolhido como default
lambda_pixel, lambda_perc, lambda_adv: escolhidos como default
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

generator = Generator(scale).to(device)
discriminator = Discriminator().to(device)

perceptual_loss = PerceptualLoss().to(device)
pixel_loss = nn.L1Loss()
bce_loss = nn.BCEWithLogitsLoss()

g_optimizer = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.9, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.9, 0.999))

lambda_pixel = 1.0
lambda_perc = 1e-3
lambda_adv = 1e-3




'''
O treinamento é feito através de batches onde 
a função presente no data.py chama e trata o 
número batch_size de cada vez.

Em testes com data_loader, a variação do tempo
não é tão diferente e esse método funciona melhor
para imagens maiores e que use muita RAM
'''
def train(img_size_input, img_size_target, lr_dir, hr_dir, lr_dir_val, hr_dir_val, out_dir, epochs=100):

  '''
  Args:

  img_size_input: tamanho da imagem de input
  img_size_output: tamanho da imagem de output
  lr_dir: diretório das imagens de baixa resolução
  hr_dir: diretório das imagens de alta resolução
  lr_dir_val: diretório das imagens de validação de baixa resolução
  hr_dir_val: diretório das imagens de validação de alta resolução
  epochs: número de épocas
  '''

    batch_size = 8

    for epoch in range(epochs):

        generator.train()
        discriminator.train()

        index = 0

        for _ in tqdm(range(800//batch_size)):

            lr, _ = proximo_batch(lr_dir, index, img_size_input, batch_size)
            hr, index = proximo_batch(hr_dir, index, img_size_target, batch_size)

            '''
            Treinando a discriminadora
            para reconhecer verdadeiro (1) e falso (0)

            Note que a discriminadora acaba 
            devolvendo um valor entre 0 e 1 na prática.
            Esse valor x 100 pode ser interpretado
            como a porcentagem de quanto as imagens se assemelham ao target 
            '''
            with torch.no_grad():
                fake_hr1, fake_hr2, fake_hr3 = generator(lr)

            real_out = discriminator(hr)
            fake_out1 = discriminator(fake_hr1.detach())
            fake_out2 = discriminator(fake_hr2.detach())
            fake_out3 = discriminator(fake_hr3.detach())

            real_labels = torch.ones_like(real_out)
            fake_labels = torch.zeros_like(fake_out1)

            d_loss_real = bce_loss(real_out, real_labels)
            d_loss_fake1 = bce_loss(fake_out1, fake_labels)
            d_loss_fake2 = bce_loss(fake_out2, fake_labels)
            d_loss_fake3 = bce_loss(fake_out3, fake_labels)
            d_loss = (d_loss_real + d_loss_fake1 + d_loss_fake2 + d_loss_fake3) / 4

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            '''
            Treinando a geradora que retorna 3 imagens
            '''

            fake_hr1, fake_hr2, fake_hr3 = generator(lr)

            g_pixel1 = pixel_loss(fake_hr1, hr)
            g_perc1 = perceptual_loss(fake_hr1, hr)
            g_adv1 = bce_loss(discriminator(fake_hr1), torch.ones_like(real_out))

            g_pixel2 = pixel_loss(fake_hr2, hr)
            g_perc2 = perceptual_loss(fake_hr2, hr)
            g_adv2 = bce_loss(discriminator(fake_hr2), torch.ones_like(real_out))

            g_pixel3 = pixel_loss(fake_hr3, hr)
            g_perc3 = perceptual_loss(fake_hr3, hr)
            g_adv3 = bce_loss(discriminator(fake_hr3), torch.ones_like(real_out))

            g_pixel = (g_pixel1 + g_pixel2 + g_pixel3) / 3
            g_perc = (g_perc1 + g_perc2 + g_perc3) / 3
            g_adv = (g_adv1 + g_adv2 + g_adv3) / 3

            g_loss = lambda_pixel * g_pixel + lambda_perc * g_perc + lambda_adv * g_adv

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
            
        print({'D_loss': f"{d_loss.item():.4f}", 'G_loss': f"{g_loss.item():.4f}"})
        generator.eval()
        
        '''
        Avaliação do set de validação
        '''     
        evaluation(img_size_input, img_size_target, lr_dir_val, hr_dir_val, generator, epoch, out_dir)

        '''
        Salvando o modelo
        '''
        torch.save(generator.state_dict(), out_dir+str(epoch)+".pth")
        #torch.save(discriminator.state_dict(), out_dir+str(epoch)+".pth")


train(img_size_input, img_size_target, lr_dir, hr_dir, lr_dir_val, hr_dir_val, output_dir, epochs=100)
