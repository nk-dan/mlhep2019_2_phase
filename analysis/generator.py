import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
MEAN_TRAIN_MOM_POINT = torch.Tensor([-0.08164814, -0.02489864, 20.8446184, -0.01204223,  0.02772552]).to(device)
STD_TRAIN_MOM_POINT  = torch.Tensor([ 5.4557047,   5.38253167, 24.26102735, 2.69435522,  2.65776869]).to(device)
NOISE_DIM = 10
TASK = 'HINGE'

class ModelGConvTranspose(nn.Module):
    def __init__(self, z_dim):
        self.z_dim = z_dim
        super(ModelGConvTranspose, self).__init__()
        self.fc1 = nn.Linear(self.z_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 512)
        self.fc4 = nn.Linear(512, 20736)

        self.conv1 = spectral_norm(nn.ConvTranspose2d(256, 256, 3, stride=2, output_padding=1))
        #self.bn1 = ConditionalBatchNorm2d(5, 256)
        self.bn1 = nn.BatchNorm2d(256)
        #self.ln1 = nn.LayerNorm([256, 20, 20])
        self.conv2 = spectral_norm(nn.ConvTranspose2d(256, 128, 3))
        #self.bn2 = ConditionalBatchNorm2d(5, 128)
        self.bn2 = nn.BatchNorm2d(128)
        #self.ln2 = nn.LayerNorm([128, 22, 22])
        self.conv3 = spectral_norm(nn.ConvTranspose2d(128, 64, 3))
        #self.bn3 = ConditionalBatchNorm2d(5, 64)
        self.bn3 = nn.BatchNorm2d(64)
        #self.ln3 = nn.LayerNorm([64, 24, 24])
        self.conv4 = spectral_norm(nn.ConvTranspose2d(64, 32, 3))
        #self.bn4 = ConditionalBatchNorm2d(5, 32)
        self.bn4 = nn.BatchNorm2d(32)
        #self.ln4 = nn.LayerNorm([32, 26, 26])
        self.conv5 = spectral_norm(nn.ConvTranspose2d(32, 16, 3))
        self.bn5 = nn.BatchNorm2d(16)
        #self.ln5 = nn.LayerNorm([16, 28, 28])
        self.conv6 = spectral_norm(nn.ConvTranspose2d(16, 1, 3))
        
    def forward(self, z, ParticleMomentum_ParticlePoint):
        mom_point = (ParticleMomentum_ParticlePoint - MEAN_TRAIN_MOM_POINT) / STD_TRAIN_MOM_POINT
        #mom_point = ParticleMomentum_ParticlePoint
        x = torch.cat([z, mom_point], dim=1)
        x = F.leaky_relu(self.fc1(z))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        EnergyDeposit = x.view(-1, 256, 9, 9)
        
        #print(EnergyDeposit.shape)
        #EnergyDeposit = F.leaky_relu(self.bn2(self.conv2(EnergyDeposit), mom_point))
        #EnergyDeposit = F.leaky_relu(self.bn3(self.conv3(EnergyDeposit), mom_point))
        #EnergyDeposit = F.leaky_relu(self.bn4(self.conv4(EnergyDeposit), mom_point))
        EnergyDeposit = F.leaky_relu(self.bn1(self.conv1(EnergyDeposit)))
        EnergyDeposit = F.leaky_relu(self.bn2(self.conv2(EnergyDeposit)))
        EnergyDeposit = F.leaky_relu(self.bn3(self.conv3(EnergyDeposit)))
        EnergyDeposit = F.leaky_relu(self.bn4(self.conv4(EnergyDeposit)))
        EnergyDeposit = F.leaky_relu(self.bn5(self.conv5(EnergyDeposit)))
        EnergyDeposit = self.conv6(EnergyDeposit)

        return EnergyDeposit