import os
from PIL import Image

import torch
from torchvision import transforms
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt

from data import proximo_batch

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch.nn.functional as F

def evaluation(img_size_input, img_size_target, lr_dir_val, hr_dir_val, generator, epoch, output_dir):
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

    batch_size = 2
    index = np.random.randint(0, 100 - batch_size)

    lr, _ = proximo_batch(lr_dir_val, index, img_size_input, batch_size)
    hr, index = proximo_batch(hr_dir_val, index, img_size_target, batch_size)

    fig, axs = plt.subplots(batch_size, 4, figsize=(12, 6))

    with torch.no_grad():
        fake_hr1, fake_hr2, fake_hr3 = generator(lr)

    fake_hr1 = fake_hr1.clamp(0, 1)
    fake_hr2 = fake_hr2.clamp(0, 1)
    fake_hr3 = fake_hr3.clamp(0, 1)
    hr = hr.clamp(0, 1)

    for i in range(batch_size):
        hr_img = hr[i].cpu().permute(2, 1, 0).numpy()

        for j, (fake, label) in enumerate(zip([fake_hr1, fake_hr2, fake_hr3], ["1", "2", "3"])):
            fake_img = fake[i].cpu().permute(2, 1, 0).numpy()

            '''
            Metricas usadas

            MSE: mean squared error
            PSNR: peak signal-to-noise ratio
            SSIM: structural similarity index measure) 
            '''
            mse_val = F.mse_loss(fake[i], hr[i]).item()
            psnr_val = psnr(hr_img, fake_img, data_range=1.0)
            ssim_val = ssim(hr_img, fake_img, data_range=1.0, channel_axis=-1)

            axs[i, j].imshow(fake_img)
            axs[i, j].set_title(f"Imagem {label}\nMSE: {mse_val:.4f}\nPSNR: {psnr_val:.2f}dB\nSSIM: {ssim_val:.4f}")
            axs[i, j].axis("off")

        axs[i, 3].imshow(hr_img)
        axs[i, 3].set_title("Target")
        axs[i, 3].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"comparison_epoch_{epoch+1}.png"))
    plt.close()