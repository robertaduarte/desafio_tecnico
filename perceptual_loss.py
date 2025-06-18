import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights
from torchvision import transforms


class PerceptualLoss(nn.Module):
    def __init__(self, layer='features.35'):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:35].eval().to(device)

        for param in self.vgg.parameters():
            param.requires_grad = False

        self.criterion = nn.L1Loss()

    def forward(self, fake_hr, hr):
        '''
        para usar a loss function baseada no VGG, é necessário normalizar para os dados do VGG
        a normalização só acontece nessa fase para comparar os features através de uma L1Loss
        '''

        normalize_vgg = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        fake_hr = normalize_vgg(fake_hr)
        hr = normalize_vgg(hr)

        fake_hr_features = self.vgg(fake_hr)
        hr_features = self.vgg(hr) 

        return self.criterion(fake_hr_features, hr_features)
