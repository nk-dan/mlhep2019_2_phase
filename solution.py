#!/usr/bin/python
import sys
import numpy as np
from analysis.generator import ModelGConvTranspose, NOISE_DIM
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils
import os


def model_prediction(input_file, output_file, generator_cpu, batch_size=1024, shuffle=False):
    """
    Model test/validation

        Parameters
        ----------
            input_file : str
                Input file name with path (e.g. data/data_val.npz).
            
            output_file : str
                Output file name with path (e.g. output/data_val_prediction.npz).
            
            generator_cpu :

            
            batch_size : int, optional
                Number of elements in a dataloader batch.
            
            shuffle : bool, optional
                If to enable shuffling in batches.

        Returns:
        ----------
            None
    """
    
    dataset = np.load(input_file, allow_pickle=True)
    val_data_path_out = output_file

    part_mom_val   = torch.tensor(dataset['ParticleMomentum']).float()
    part_point_val = torch.tensor(dataset['ParticlePoint'][:, :2]).float()
    part_mom_point = torch.cat([part_mom_val, part_point_val], dim=1)
    
    calo_dataset    = utils.TensorDataset(part_mom_point)
    calo_dataloader = torch.utils.data.DataLoader(calo_dataset, batch_size=batch_size, shuffle=shuffle)

    with torch.no_grad():
        EnergyDeposit_val = []

        for part_mom_point_batch in tqdm(calo_dataloader):
            noise = torch.randn(len(part_mom_point_batch[0]), NOISE_DIM)
            EnergyDeposit_val_batch = generator_cpu(noise, part_mom_point_batch[0]).detach().numpy()
            EnergyDeposit_val.append(EnergyDeposit_val_batch)
        
        np.savez_compressed(output_file, 
                            EnergyDeposit=np.concatenate(EnergyDeposit_val, axis=0).reshape(-1, 30, 30))

        del EnergyDeposit_val
    del dataset; del part_mom_val; del part_point_val; del part_mom_point;
    del calo_dataset; calo_dataloader
    return


def main():
    input_dir, output_dir = sys.argv[1:]
        
    generator_cpu = ModelGConvTranspose(NOISE_DIM)
    generator_cpu.load_state_dict(torch.load(os.path.dirname(os.path.abspath(__file__)) + '/gan.pt'))
    generator_cpu.eval()
    
    # Validation
    model_evaluation(input_dir  + '/data_val.npz' , 
                     output_dir + '/data_val_prediction.npz', generator_cpu)
    # Test
    model_evaluation(input_dir  + '/data_test.npz', 
                     output_dir + '/data_test_prediction.npz', generator_cpu)

    return 0

if __name__ == "__main__":
    main()
