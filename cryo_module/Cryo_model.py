import torch.nn as nn
from cryo_module.unet3d.emmodel import ResUNet3D4EM
import torch

class Cryo_Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.models = nn.ModuleDict()
        self.models['AA_model'] = ResUNet3D4EM()
        self.models['BB_model'] = ResUNet3D4EM()
        self.models['CA_model'] = ResUNet3D4EM()
        
