import os
from PIL import Image

import torch
from torchvision import transforms
import torchvision.transforms as T


def padronizar(img, size):
        '''
        A função padronizar tem como objetivos:
        1. Deixar as imagens na horizontal (largura > altura)
        2. Padronizar os tamanhos apropriados
        '''
        w, h = size
        if h > w:
            img = img.rotate(90, expand=True)
        img = img.resize(size, Image.Resampling.BICUBIC)
        return img


def proximo_batch(im_dir, start_index, img_size, batch_size, device='cuda'):
    '''
    Prepara as imagens e padroniza elas
    Cria batches de dados para treinar
    '''
    paths = sorted([os.path.join(im_dir, fname) for fname in os.listdir(im_dir) if fname.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    #transform = transforms.ToTensor()
    to_tensor = T.ToTensor()

    batch_imgs = []
    end_index = min(start_index + batch_size, len(paths))

    for i in range(start_index, end_index):
        img = Image.open(paths[i]).convert('RGB')
        img = padronizar(img, img_size)
        img = to_tensor(img).permute(0, 2, 1)
        batch_imgs.append(img.unsqueeze(0))  # [1, 3, H, W]

    if len(batch_imgs) < batch_size:
        return None, None  # acabou

    batch = torch.cat(batch_imgs, dim=0).to(device)
    next_index = end_index

    return batch, next_index