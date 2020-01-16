import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class ModelD(nn.Module):
    def __init__(self):
        super(ModelD, self).__init__()
        
        # 30x30x1
        self.conv1 = spectral_norm(nn.Conv2d( 1,   32, kernel_size=(3,3), stride=(1,1), padding=(1,1))) # 30x30x32
        #self.ln1   = nn.LayerNorm([32, 30, 30])
        self.conv2 = spectral_norm(nn.Conv2d( 32,  64, kernel_size=(3,3), stride=(1,1), padding=(1,1))) # 30x30x64
        #self.ln2   = nn.LayerNorm([64, 30, 30])        
        self.pool1 = nn.MaxPool2d(2, 2)                                                  # 15x15x64

        self.conv3 = spectral_norm(nn.Conv2d( 64, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1))) # 15x15x128
        #self.ln3   = nn.LayerNorm([128, 15, 15])
        self.conv4 = spectral_norm(nn.Conv2d(128, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1))) # 15x15x256
        #self.ln4   = nn.LayerNorm([256, 15, 15])
        self.pool2 = nn.MaxPool2d(3, 3)                                                  # 15x15x256
        
        # 5x5x256 = 6400
        self.fc1 = spectral_norm(nn.Linear(6400 + 5, 1))
        
    def forward(self, EnergyDeposit, ParticleMomentum_ParticlePoint):
        X = EnergyDeposit
        X = F.leaky_relu(self.conv1(X))
        X = F.leaky_relu(self.conv2(X))
        X = self.pool1(X)

        X = F.leaky_relu(self.conv3(X))
        X = F.leaky_relu(self.conv4(X))
        X = self.pool2(X)
        
        X = X.view(len(X), -1)
        mom_point = (ParticleMomentum_ParticlePoint - MEAN_TRAIN_MOM_POINT) / STD_TRAIN_MOM_POINT
        X = torch.cat([X, mom_point], dim=1)
        
        #X = F.leaky_relu(self.fcbn1(self.fc1(X)))
        #X = F.leaky_relu(self.fcbn2(self.fc2(X)))
        #X = F.leaky_relu(self.fcbn3(self.fc3(X)))
        if TASK in ['WASSERSTEIN', 'HINGE']:
            return self.fc1(X)
        else:
            return torch.sigmoid(self.fc1(X))