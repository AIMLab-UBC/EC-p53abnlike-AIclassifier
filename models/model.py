import timm
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from einops import rearrange, repeat

from utils.utils import print_mix


####################################################
out_channel = {'alexnet': 256, 'vgg16': 512, 'vgg19': 512, 'vgg16_bn': 512, 'vgg19_bn': 512,
               'resnet18': 512, 'resnet34': 512, 'resnet50': 2048, 'resnext50_32x4d': 2048,
               'resnext101_32x8d': 2048, 'mobilenet_v2': 1280, 'mobilenet_v3_small': 576,
               'mobilenet_v3_large': 960 ,'mnasnet1_3': 1280, 'shufflenet_v2_x1_5': 1024,
               'squeezenet1_1': 512, 'efficientnet-b0': 1280, 'efficientnet-l2': 5504,
               'efficientnet-b1': 1280, 'efficientnet-b2': 1408, 'efficientnet-b3': 1536,
               'efficientnet-b4': 1792, 'efficientnet-b5': 2048, 'efficientnet-b6': 2304,
               'efficientnet-b7': 2560, 'efficientnet-b8': 2816, 'vit_deit_small_patch16_224': 384}

feature_map = {'alexnet': -2, 'vgg16': -2,  'vgg19': -2, 'vgg16_bn': -2,  'vgg19_bn': -2,
               'resnet18': -2, 'resnet34': -2, 'resnet50': -2, 'resnext50_32x4d': -2,
               'resnext101_32x8d': -2, 'mobilenet_v2': 0, 'mobilenet_v3_large': -2,
               'mobilenet_v3_small': -2, 'mnasnet1_3': 0, 'shufflenet_v2_x1_5': -1,
               'squeezenet1_1': 0, 'vit_deit_small_patch16_224': 'inf'}
####################################################

class VanillaModel(nn.Module):
    def __init__(self,
                 backbone: str) -> None:
        super(VanillaModel, self).__init__()

        self.backbone  = backbone
        # Vision Transformer
        if 'vit' in self.backbone:
            model = timm.create_model(self.backbone, pretrained=False, num_classes=0)
            self.feature_extract = model
        else:
            model = getattr(models, self.backbone)
            model = model(weights=None)
            # Seperate feature and classifier layers
            self.feature_extract = nn.Sequential(*list(model.children())[0]) if feature_map[self.backbone]==0 \
                                   else nn.Sequential(*list(model.children())[:feature_map[self.backbone]])

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        feature = self.feature_extract(x)
        feature = F.adaptive_avg_pool2d(feature, 1)
        out     = torch.flatten(feature, 1)
        return out

class VarMIL(nn.Module):
    """
    Our modified implementation of https://arxiv.org/abs/2107.09405
    """
    def __init__(self,
                 cfg: dict) -> None:
        super().__init__()
        dim = 128
        self.device     = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.attention  = nn.Sequential(nn.Linear(out_channel[cfg['backbone']], dim),
                                       nn.Tanh(),
                                       nn.Linear(dim, 1))
        self.classifier = nn.Sequential(nn.Linear(2*out_channel[cfg['backbone']], dim),
                                       nn.ReLU(),
                                       nn.Linear(dim, cfg['num_classes']))

    def forward(self,
                x: torch.Tensor) -> torch.Tensor | torch.Tensor:
        """
        x   (input)            : B (batch size) x K (nb_patch) x out_channel
        A   (attention weights): B (batch size) x K (nb_patch) x 1
        M   (weighted mean)    : B (batch size) x out_channel
        S   (std)              : B (batch size) x K (nb_patch) x out_channel
        V   (weighted variance): B (batch size) x out_channel
        nb_patch (nb of patch) : B (batch size)
        M_V (concate M and V)  : B (batch size) x 2*out_channel
        out (final output)     : B (batch size) x num_classes
        """
        b, k, c = x.shape
        A = self.attention(x)
        A = A.masked_fill((x == 0).all(dim=2).reshape(A.shape), -9e15) # filter padded rows
        A = F.softmax(A, dim=1)                                        # softmax over K
        M = torch.einsum('b k d, b k o -> b o', A, x)                  # d is 1 here
        S = torch.pow(x-M.reshape(b,1,c), 2)
        V = torch.einsum('b k d, b k o -> b o', A, S)
        nb_patch = (torch.tensor(k).expand(b)).to(self.device)
        nb_patch = nb_patch - torch.sum((x == 0).all(dim=2), dim=1)    # filter padded rows
        nb_patch = nb_patch / (nb_patch - 1)                           # I / I-1
        nb_patch = torch.nan_to_num(nb_patch, posinf=1)                # for cases, when we have only 1 patch (inf)
        V = V * nb_patch[:, None]                                      # broadcasting
        M_V = torch.cat((M, V), dim=1)
        out = self.classifier(M_V)
        return A, out


class Attention(nn.Module):
    def __init__(self,
                 cfg: dict) -> None:
        super().__init__()
        if cfg['model'] == 'VarMIL':
            self.model = VarMIL(cfg)
        else:
            raise NotImplementedError()

    def trainable_parameters(self) -> None:
        """
        Copy from: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
        """
        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # print_mix(f'Total trainable parameters are __{params}__.', color='RED')
        print(f'Total trainable parameters are {params}.')

    def forward(self,
                x: torch.Tensor) -> torch.Tensor | torch.Tensor:
        """
        x (input) : B (batch size) x K (nb_patch) x out_channel
        """
        attention, out = self.model(x)
        return attention, out


if __name__ == "__main__":
    model = Attention({'backbone': "vit_deit_small_patch16_224",
                       'model': 'VarMIL',
                       'num_classes': 2})
    print(model)
    print(model(torch.rand(4,3,224,224)).shape)
