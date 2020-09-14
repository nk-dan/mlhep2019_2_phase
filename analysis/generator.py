import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

NOISE_DIM = 256
TASK = 'WASSERSTEIN'
MEAN_TRAIN_MOM_POINT = torch.Tensor([-0.08164814, -0.02489864, 20.8446184, -0.01204223,  0.02772552])
STD_TRAIN_MOM_POINT  = torch.Tensor([ 5.4557047,   5.38253167, 24.26102735, 2.69435522,  2.65776869])

class ConditionalBatchNorm2d(nn.Module):
  def __init__(self, in_features, out_features):
    super().__init__()
    self.num_features = out_features
    self.bn = nn.BatchNorm2d(out_features, affine=False)
    self.gammas = nn.Linear(in_features, out_features)
    self.bethas = nn.Linear(in_features, out_features)
    self._initialize()

  def _initialize(self):
    nn.init.ones_(self.gammas.weight.data)
    nn.init.zeros_(self.bethas.weight.data)

  def forward(self, x, y):
    batchnorm = self.bn(x)
    gamma = self.gammas(y)
    beta = self.bethas(y)
    out = (gamma.view(-1, self.num_features, 1, 1)) * batchnorm + beta.view(-1, self.num_features, 1, 1)
    return out
  
class SelfAttention(nn.Module):

    def __init__(self, in_dim, activation):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1)

    def forward(self,x):

        m_batchsize, C, width, height = x.size()
        proj_query = self.query(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key = self.key(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy = torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out, attention
      
'''
class ModelGConvTranspose(nn.Module):
    def __init__(self, z_dim):
        self.z_dim = z_dim
        super(ModelGConvTranspose, self).__init__()
        self.fc1 = nn.Linear(self.z_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 512)
        self.fc4 = nn.Linear(512, 20736)

        self.conv1 = spectral_norm(nn.ConvTranspose2d(256, 256, 3, stride=2, output_padding=1))
        self.bn1 = ConditionalBatchNorm2d(5, 256)
        #self.bn1 = nn.BatchNorm2d(256)
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
        self.bn4 = ConditionalBatchNorm2d(5, 32)
        #self.bn4 = nn.BatchNorm2d(32)
        #self.ln4 = nn.LayerNorm([32, 26, 26])
        self.conv5 = spectral_norm(nn.ConvTranspose2d(32, 16, 3))
        self.bn5 = nn.BatchNorm2d(16)
        #self.ln5 = nn.LayerNorm([16, 28, 28])
        self.conv6 = spectral_norm(nn.ConvTranspose2d(16, 1, 3))
        
    def forward(self, z, ParticleMomentum_ParticlePoint):
        mom_point = (ParticleMomentum_ParticlePoint - MEAN_TRAIN_MOM_POINT) / STD_TRAIN_MOM_POINT
        #mom_point = ParticleMomentum_ParticlePoint
        #x = torch.cat([z, mom_point], dim=1)
        #x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc1(z))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        EnergyDeposit = x.view(-1, 256, 9, 9)
        
        #print(EnergyDeposit.shape)
        EnergyDeposit = F.leaky_relu(self.bn1(self.conv1(EnergyDeposit), mom_point))
        #EnergyDeposit = F.leaky_relu(self.bn1(self.conv1(EnergyDeposit)))
        #EnergyDeposit = F.leaky_relu(self.bn2(self.conv2(EnergyDeposit), mom_point))
        EnergyDeposit = F.leaky_relu(self.bn2(self.conv2(EnergyDeposit)))
        #EnergyDeposit = F.leaky_relu(self.bn3(self.conv3(EnergyDeposit), mom_point))
        EnergyDeposit = F.leaky_relu(self.bn3(self.conv3(EnergyDeposit)))
        EnergyDeposit = F.leaky_relu(self.bn4(self.conv4(EnergyDeposit), mom_point))
        #EnergyDeposit = F.leaky_relu(self.bn4(self.conv4(EnergyDeposit)))
        EnergyDeposit = F.leaky_relu(self.bn5(self.conv5(EnergyDeposit)))
        EnergyDeposit = F.relu(self.conv6(EnergyDeposit))

        return EnergyDeposit
'''

class Generator_alt(nn.Module):
    def __init__(self, z_dim):
        super(Generator_alt, self).__init__()
        self.z_dim = z_dim
        self.fc1 = nn.Linear(self.z_dim, self.z_dim * 2 * 2)

        self.conv1 = nn.ConvTranspose2d(self.z_dim, 128, 3, padding=1, stride=2, output_padding=1)
        self.bn1 = ConditionalBatchNorm2d(5, 128)
        #self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.bn2 = ConditionalBatchNorm2d(5, 64)
        #self.bn2 = nn.BatchNorm2d(64)
        self.attn1 = SelfAttention(64, 'relu')
        self.conv3 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.bn3 = ConditionalBatchNorm2d(5, 32)
        #self.bn3 = nn.BatchNorm2d(32)
        #self.attn1 = SelfAttention(64, 'relu')
        self.conv4 = nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1)
        ##self.bn4 = ConditionalBatchNorm2d(5, 1)
        #self.attn2 = SelfAttention(32, 'relu')

        
    def forward(self, z, ParticleMomentum_ParticlePoint):
        mom_point = (ParticleMomentum_ParticlePoint - MEAN_TRAIN_MOM_POINT) / STD_TRAIN_MOM_POINT
        #mom_point = ParticleMomentum_ParticlePoint
        #x = torch.cat([z, mom_point], dim=1)
        #x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc1(z.float()))
        EnergyDeposit = x.view(-1, self.z_dim, 2, 2)
        
        #print(EnergyDeposit.shape)
        EnergyDeposit = F.relu(self.bn1(self.conv1(EnergyDeposit), mom_point))
        #EnergyDeposit = F.relu(self.bn1(self.conv1(EnergyDeposit)))
        EnergyDeposit = F.relu(self.bn2(self.conv2(EnergyDeposit), mom_point))
        EnergyDeposit, _ = self.attn1(EnergyDeposit)
        EnergyDeposit = F.relu(self.bn3(self.conv3(EnergyDeposit), mom_point))
        #EnergyDeposit, _ = self.attn1(EnergyDeposit)
        ##EnergyDeposit = F.relu(self.bn4(self.conv4(EnergyDeposit), mom_point))
        EnergyDeposit = F.relu(self.conv4(EnergyDeposit))
        #EnergyDeposit, _ = self.attn2(EnergyDeposit)
        EnergyDeposit = EnergyDeposit[:,:,1:31,1:31]

        return EnergyDeposit
