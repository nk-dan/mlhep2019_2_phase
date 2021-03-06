import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
MEAN_TRAIN_MOM_POINT = torch.Tensor([-0.08164814, -0.02489864, 20.8446184, -0.01204223,  0.02772552]).to(device)
STD_TRAIN_MOM_POINT  = torch.Tensor([ 5.4557047,   5.38253167, 24.26102735, 2.69435522,  2.65776869]).to(device)
TASK = 'HINGE'



class ModelD(nn.Module):
    def __init__(self):
        super(ModelD, self).__init__()
        
        # 30x30x1
        self.conv1 = spectral_norm(nn.Conv2d( 1,   32, kernel_size=(3,3), stride=(1,1), padding=(1,1))) # 30x30x32
        self.conv2 = spectral_norm(nn.Conv2d( 32,  64, kernel_size=(3,3), stride=(2,2), padding=(1,1))) # 15x15x64       
        #self.pool1 = nn.MaxPool2d(2, 2)
        #self.conv3 = spectral_norm(nn.Conv2d(64, 64, kernel_size=(3,3), stride=(2,2), padding=(1,1)))                                               # 15x15x64
        #self.attn1 = SelfAttention(64, 'relu')

        self.conv3 = spectral_norm(nn.Conv2d( 64, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1))) # 15x15x128
        self.conv4 = spectral_norm(nn.Conv2d(128, 256, kernel_size=(3,3), stride=(3,3), padding=(1,1))) # 5x5x256
        
        #self.pool2 = nn.MaxPool2d(3, 3)                                                  # 15x15x256
        #self.conv6 = spectral_norm(nn.Conv2d(256, 256, kernel_size=(3,3), stride=(3,3), padding=(1,1)))
        #self.attn2 = SelfAttention(256, 'relu')
        
        # 5x5x256 = 6400
        self.fc1 = spectral_norm(nn.Linear(6400, 1024))
        #self.fc2 = spectral_norm(nn.Linear(1024,32))
        self.cl = spectral_norm(nn.Linear(5, 1024)) #cgan projection, in original paper: res_block + global_max_pool + linear
        self.fc3 = spectral_norm(nn.Linear(1024,1))
        
    def forward(self, EnergyDeposit, ParticleMomentum_ParticlePoint):
        X = EnergyDeposit
        X = F.leaky_relu(self.conv1(X))
        X = F.leaky_relu(self.conv2(X))
        #X = self.pool1(X)

        X = F.leaky_relu(self.conv3(X))
        X = F.leaky_relu(self.conv4(X))
        #X = F.leaky_relu(self.conv5(X))
        #X = F.leaky_relu(self.conv6(X))
        #X = self.pool2(X)
        X = X.view(len(X), -1)
        mom_point = (ParticleMomentum_ParticlePoint - MEAN_TRAIN_MOM_POINT) / STD_TRAIN_MOM_POINT
        #X = torch.cat([X, mom_point], dim=1)
        X = F.leaky_relu(self.fc1(X))
        #X = F.leaky_relu(self.fc2(X))
        #X = F.leaky_relu(self.fcbn1(self.fc1(X)))
        #X = F.leaky_relu(self.fcbn2(self.fc2(X)))
        #X = F.leaky_relu(self.fcbn3(self.fc3(X)))
        if TASK in ['WASSERSTEIN', 'HINGE', 'RAHINGE']:
            return self.fc3(X) + torch.sum(self.cl(mom_point) * X, dim=1, keepdim=True)
        else:
            return torch.sigmoid(self.fc1(X))