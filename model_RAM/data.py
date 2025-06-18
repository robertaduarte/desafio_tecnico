import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class createDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, input_size, output_size):

        self.input_size = input_size
        self.output_size = output_size

        self.path_for_lr = sorted([os.path.join(lr_dir, f) for f in os.listdir(lr_dir)])
        self.path_for_hr = sorted([os.path.join(hr_dir, f) for f in os.listdir(hr_dir)])

        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.path_for_lr)
    
    def padronizar(self, img, size):
        w, h = size
        if h > w:
            img = img.rotate(90, expand=True)
        img = img.resize(size, Image.Resampling.BICUBIC)

        return img

    def __getitem__(self, idx):

        lr = Image.open(self.path_for_lr[idx]).convert('RGB')
        hr = Image.open(self.path_for_hr[idx]).convert('RGB')

        lr = self.padronizar(lr, size = self.input_size)
        hr = self.padronizar(hr, size = self.output_size)

        lr = self.to_tensor(lr)
        hr = self.to_tensor(hr)

        return lr, hr

    

