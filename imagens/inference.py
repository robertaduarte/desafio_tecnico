import os
from PIL import Image

import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from architecture import *
from data import padronizar

import matplotlib.pyplot as plt 

from torchvision import transforms
import torchvision.transforms as T

lr_dir_val = "C:\\Users\\impor\\Downloads\\prova_tecnica\\lr_low_val\\"
hr_dir_val =  "C:\\Users\\impor\\Downloads\\prova_tecnica\\hr_val\\"

output_dir = "C:\\Users\\impor\\Downloads\\prova_tecnica\\outputs1\\2x\\"

scale = 2

if scale == 2:

  img_size_input = (256, 128)
  img_size_target = (512, 256)

elif scale == 4:

  img_size_input = (128, 64)
  img_size_target = (512, 256)

else: 

  img_size_input = (128, 64)
  img_size_target = (1024, 512)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

generator = Generator(scale).to(device)
generator.load_state_dict(torch.load("C:\\Users\\impor\\Downloads\\prova_tecnica\\outputs1\\68.pth"))
generator.eval()


def carregar(img_size, paths):

    to_tensor = T.ToTensor()
    img = Image.open(paths).convert('RGB')
    img = padronizar(img, img_size)
    img = to_tensor(img).permute(0, 2, 1)

    return img 


def plotar(img_size_input, img_size_target, lr_dir_val, hr_dir_val, generator, output_dir):
    '''
    Args:

    img_size_input: tamanho da imagem de input
    img_size_output: tamanho da imagem de output
    lr_dir_val: diretório da imagem de validação de baixa resolução
    hr_dir_val: diretório da imagem de validação de alta resolução
    generator: geradora treinada 
    epoch: época
    output_dir: diretório para salvar as imagens
    '''
    paths_lr = sorted([os.path.join(lr_dir_val, fname) for fname in os.listdir(lr_dir_val) if fname.lower()])
    paths_hr = sorted([os.path.join(hr_dir_val, fname) for fname in os.listdir(hr_dir_val) if fname.lower()])


    for i in range(len(paths_lr)):

        lr = carregar(img_size_input, paths_lr[i]).to(device)
        hr = carregar(img_size_target, paths_hr[i]).to(device)

        fig, axs = plt.subplots(1, 4, figsize=(12, 6))

        with torch.no_grad():
            fake_hr1, fake_hr2, fake_hr3 = generator(lr)

        fake_hr1 = fake_hr1.clamp(0, 1)
        fake_hr2 = fake_hr2.clamp(0, 1)
        fake_hr3 = fake_hr3.clamp(0, 1)
        hr = hr.clamp(0, 1)

        hr_img = hr.cpu().permute(2, 1, 0).numpy()

        for j, (fake, label) in enumerate(zip([fake_hr1, fake_hr2, fake_hr3], ["1", "2", "3"])):
            fake_img = fake.cpu().permute(2, 1, 0).numpy()

            mse_val = F.mse_loss(fake, hr).item()
            psnr_val = psnr(hr_img, fake_img, data_range=1.0)
            ssim_val = ssim(hr_img, fake_img, data_range=1.0, channel_axis=-1)

            axs[j].imshow(fake_img)
            axs[j].set_title(f"Imagem {label}\nMSE: {mse_val:.4f}\nPSNR: {psnr_val:.2f}dB\nSSIM: {ssim_val:.4f}")
            axs[j].axis("off")

        axs[3].imshow(hr_img)
        axs[3].set_title("Target")
        axs[3].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"img_{i}.png"))
        plt.close()


plotar(img_size_input, img_size_target, lr_dir_val, hr_dir_val, generator, output_dir)