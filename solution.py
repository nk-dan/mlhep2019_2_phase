#!/usr/bin/python
import sys
import numpy as np
from analysis.generator import ConditionalBatchNorm2d, Generator_alt, NOISE_DIM
from analysis.calc_loss import GANLosses
from analysis.critic import ModelD
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils
from torch.nn.utils import spectral_norm
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
MEAN_TRAIN_MOM_POINT = torch.Tensor([-0.08164814, -0.02489864, 20.8446184, -0.01204223,  0.02772552]).to(device)
STD_TRAIN_MOM_POINT  = torch.Tensor([ 5.4557047,   5.38253167, 24.26102735, 2.69435522,  2.65776869]).to(device)
GRAD_PENALTY = True
ZERO_CENTERED_GRAD_PENALTY = False

INSTANCE_NOISE = True

MEAN_TRAIN_MOM   = np.array([-0.08164814, -0.02489864, 20.8446184])
MEAN_TRAIN_POINT = np.array([-0.01204223,  0.02772552])
STD_TRAIN_MOM    = np.array([ 5.4557047,   5.38253167, 24.26102735])
STD_TRAIN_POINT  = np.array([ 2.69435522,  2.65776869])

TASKS = ['KL', 'REVERSED_KL', 'WASSERSTEIN', 'HINGE']

TASK = 'HINGE'

LIPSITZ_WEIGHTS = False

class ModelG(nn.Module):
    def __init__(self, z_dim):
        self.z_dim = z_dim
        super(ModelG, self).__init__()
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

def dataset_with_one_particle_type(type_id, EnergyDeposit, ParticleMomentum, ParticlePoint, PDG):
    ind = [i for i, x in enumerate(PDG) if x == type_id]
    
    return utils.TensorDataset(EnergyDeposit[ind], ParticleMomentum[ind], ParticlePoint[ind], PDG[ind])

def add_instance_noise(data, std=0.01):
    return data + torch.distributions.Normal(0, std).sample(data.shape).to(device)


def trainer(data_train):

    MAX_TRAIN_SIZE = data_train['EnergyDeposit'].shape[0]
    TRAIN_SIZE = 5000 * 5
    TRAIN_IND_ARR = np.random.choice(MAX_TRAIN_SIZE, TRAIN_SIZE)

    EnergyDeposit    = data_train['EnergyDeposit'][TRAIN_IND_ARR].reshape(-1, 1, 30, 30)
    ParticleMomentum = data_train['ParticleMomentum'][TRAIN_IND_ARR]
    ParticlePoint    = data_train['ParticlePoint'][:, :2][TRAIN_IND_ARR]
    ParticlePDG      = data_train['ParticlePDG'][TRAIN_IND_ARR]

    EnergyDeposit    = torch.tensor(EnergyDeposit).float()
    ParticleMomentum = torch.tensor(ParticleMomentum).float()
    ParticlePoint    = torch.tensor(ParticlePoint).float()
    ParticlePDG      = torch.tensor(ParticlePDG).float()

    BATCH_SIZE = 512

    calo_dataset = dataset_with_one_particle_type(11., EnergyDeposit, ParticleMomentum, ParticlePoint, ParticlePDG)
    TRAIN_SIZE = len(calo_dataset)

    calo_dataloader = utils.DataLoader(calo_dataset, batch_size=BATCH_SIZE, pin_memory=True, shuffle=True)

    # Normalising stuff

    type_id = 11.

    ind = [i for i, x in enumerate(data_train['ParticlePDG']) if x == type_id]

    mean_train_mom   = np.mean(data_train['ParticleMomentum'][ind], axis=0)
    mean_train_point = np.mean(data_train['ParticlePoint'][:,:2][ind], axis=0)

    std_train_mom   = np.std(data_train['ParticleMomentum'][ind], axis=0)
    std_train_point = np.std(data_train['ParticlePoint'][:,:2][ind], axis=0)


    clamp_lower, clamp_upper = -0.01, 0.01

    gan_losses = GANLosses(TASK, device)
    discriminator = ModelD().to(device)
    generator = ModelG(z_dim=NOISE_DIM).to(device)

    epoch_num = 5
    lr_dis, lr_gen = 4e-4, 1e-4

    g_optimizer = optim.Adam(generator.parameters(), betas=(0.0, 0.999), lr=lr_gen)
    d_optimizer = optim.Adam(discriminator.parameters(), betas=(0.0, 0.999), lr=lr_dis)

    k_d, k_g = 4, 1

    dis_epoch_loss,  gen_epoch_loss  = [], []
    predictions_dis, predictions_gen = [], [] 

    for epoch in tqdm(range(epoch_num)):
        first = True
        
        for energy_b, mom_b, point_b, pdg_b in calo_dataloader:
            energy_b, mom_b = energy_b.to(device), mom_b.to(device)
            point_b,  pdg_b = point_b.to(device), pdg_b.to(device)

            mom_point_b = torch.cat([mom_b.to(device), point_b.to(device)], dim=1)
                
                
            # Optimize D
            for _ in range(k_d):
                noise = torch.randn(len(energy_b), NOISE_DIM).to(device)
                energy_gen = generator(noise, mom_point_b)

                if INSTANCE_NOISE:
                    energy_b   = add_instance_noise(energy_b)
                    energy_gen = add_instance_noise(energy_gen)
            
                loss = gan_losses.d_loss(discriminator(energy_gen, mom_point_b),
                                 discriminator(energy_b,   mom_point_b))
       
                coef = 0
                if GRAD_PENALTY:
                    coef = +1.
                elif ZERO_CENTERED_GRAD_PENALTY:
                    coef = -1.

                loss += coef * gan_losses.calc_gradient_penalty(discriminator,
                                                        energy_gen.data,
                                                        mom_point_b,
                                                        energy_b.data)
                d_optimizer.zero_grad()
                loss.backward()
                d_optimizer.step()

                if LIPSITZ_WEIGHTS:                    
                    [p.data.clamp_(clamp_lower, clamp_upper) for p in discriminator.parameters()]

            dis_loss_item = loss.item()

            # Optimize G
            for _ in range(k_g):
                noise = torch.randn(len(energy_b), NOISE_DIM).to(device)
                energy_gen = generator(noise, mom_point_b)
        
                if INSTANCE_NOISE:
                    energy_b = add_instance_noise(energy_b)
                    energy_gen = add_instance_noise(energy_gen)
        
                loss = gan_losses.g_loss(discriminator(energy_gen, mom_point_b))
                g_optimizer.zero_grad()
                loss.backward()
                g_optimizer.step()

            gen_loss_item = loss.item()
    
    return generator.state_dict()




def main():
    input_dir, output_dir = sys.argv[1:]
    
    data_train = np.load(input_dir + '/data_train.npz', allow_pickle=True)

    
    data_val = np.load(input_dir + '/data_val.npz', allow_pickle=True)
    val_data_path_out = output_dir + '/data_val_prediction.npz'

    data_test = np.load(input_dir + '/data_test.npz', allow_pickle=True)
    test_data_path_out = output_dir + '/data_test_prediction.npz'
    
    generator_cpu = ModelGConvTranspose(z_dim=NOISE_DIM)
    #generator_cpu.load_state_dict(trainer(data_train))
    generator_cpu.load_state_dict(torch.load(os.path.dirname(os.path.abspath(__file__)) + '/model_fed_sa.pt'))
    generator_cpu.eval()
    
    # val
    ParticleMomentum_val = torch.tensor(data_val['ParticleMomentum']).float()
    ParticlePoint_val = torch.tensor(data_val['ParticlePoint'][:, :2]).float()
    ParticleMomentum_ParticlePoint_val = torch.cat([ParticleMomentum_val, ParticlePoint_val], dim=1)
    calo_dataset_val = utils.TensorDataset(ParticleMomentum_ParticlePoint_val)
    calo_dataloader_val = torch.utils.data.DataLoader(calo_dataset_val, batch_size=1024, shuffle=False)

    with torch.no_grad():
        EnergyDeposit_val = []
        for ParticleMomentum_ParticlePoint_val_batch in tqdm(calo_dataloader_val):
            noise = torch.randn(len(ParticleMomentum_ParticlePoint_val_batch[0]), NOISE_DIM)
            EnergyDeposit_val_batch = generator_cpu(noise, ParticleMomentum_ParticlePoint_val_batch[0]).detach().numpy()
            EnergyDeposit_val.append(EnergyDeposit_val_batch)
        np.savez_compressed(val_data_path_out, 
                            EnergyDeposit=np.concatenate(EnergyDeposit_val, axis=0).reshape(-1, 30, 30))

        del EnergyDeposit_val
    del data_val; del ParticleMomentum_val; del ParticlePoint_val; del ParticleMomentum_ParticlePoint_val;
    del calo_dataset_val; calo_dataloader_val
    
    
    ParticleMomentum_test = torch.tensor(data_test['ParticleMomentum']).float()
    ParticlePoint_test = torch.tensor(data_test['ParticlePoint'][:, :2]).float()
    ParticleMomentum_ParticlePoint_test = torch.cat([ParticleMomentum_test, ParticlePoint_test], dim=1)
    calo_dataset_test = utils.TensorDataset(ParticleMomentum_ParticlePoint_test)
    calo_dataloader_test = torch.utils.data.DataLoader(calo_dataset_test, batch_size=1024, shuffle=False)

    with torch.no_grad():
        EnergyDeposit_test = []
        for ParticleMomentum_ParticlePoint_test_batch in tqdm(calo_dataloader_test):
            noise = torch.randn(len(ParticleMomentum_ParticlePoint_test_batch[0]), NOISE_DIM)
            EnergyDeposit_test_batch = generator_cpu(noise, ParticleMomentum_ParticlePoint_test_batch[0]).detach().numpy()
            EnergyDeposit_test.append(EnergyDeposit_test_batch)
        np.savez_compressed(test_data_path_out, 
                            EnergyDeposit=np.concatenate(EnergyDeposit_test, axis=0).reshape(-1, 30, 30))

        del EnergyDeposit_test
    del data_test; del ParticleMomentum_test; del ParticlePoint_test; del ParticleMomentum_ParticlePoint_test;
    del calo_dataset_test; calo_dataloader_test


    return 0

if __name__ == "__main__":
    main()
