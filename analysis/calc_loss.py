import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils
import numpy as np

def l2_norm(tensor):
    """Return the l2 norm of a tensor."""
    return torch.sqrt(1e-8 + torch.sum(tensor**2))

class GANLosses(object):
    def __init__(self, task, device, gp = 'local-two-sided'):
        self.TASK = task
        self.device = device
        self.GP = gp
    
    def g_loss(self, discrim_output):
        eps = 1e-10
        if self.TASK == 'KL': 
            loss = torch.log(1 - discrim_output + eps).mean()    
        elif self.TASK == 'REVERSED_KL':
            loss = - torch.log(discrim_output + eps).mean()
        elif self.TASK in ['WASSERSTEIN', 'HINGE']:
            loss = - discrim_output.mean()
        return loss

    def d_loss(self, discrim_output_gen, discrim_output_real):
        eps = 1e-10
        if self.TASK in ['KL', 'REVERSED_KL']: 
            loss = - torch.log(discrim_output_real + eps).mean() - torch.log(1 - discrim_output_gen + eps).mean()
        elif self.TASK == 'WASSERSTEIN':
            loss = - (discrim_output_real.mean() - discrim_output_gen.mean())
        elif self.TASK == 'HINGE':
            loss = torch.nn.ReLU()(1.0 - discrim_output_real).mean() + torch.nn.ReLU()(1.0 + discrim_output_gen).mean()
        return loss

    def calc_gradient_penalty(self, discriminator, data_gen, inputs_batch, inp_data, lambda_reg = .1):
        gradient_penalty = 0
        k = 1

        if self.GP == 'local-two-sided':
            N = 10

            noise_y = torch.distributions.Normal(0, N).sample(inp_data.shape).to(self.device)

            perturbed_y = (inp_data + noise_y).to(self.device)

            perturbed_y.requires_grad = True

            disc_interpolates = discriminator(perturbed_y, inputs_batch)

            gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=perturbed_y,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]

            gradient_penalty = ((l2_norm(gradients) - k) ** 2).mean() * lambda_reg

        elif self.GP == 'original':
            alpha = torch.rand(inp_data.shape[0], 1).to(self.device)
            dims_to_add = len(inp_data.size()) - 2
            for i in range(dims_to_add):
                alpha = alpha.unsqueeze(-1)
                # alpha = alpha.expand(inp_data.size())

            interpolates = (alpha * inp_data + ((1 - alpha) * data_gen)).to(self.device)

            interpolates.requires_grad = True

            disc_interpolates = discriminator(interpolates, inputs_batch)

            gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]

            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_reg

        return gradient_penalty


    def calc_zero_centered_GP(self, discriminator, data_gen, inputs_batch, inp_data, gamma_reg = .1):
        
        local_input = inp_data.clone().detach().requires_grad_(True)
        disc_interpolates = discriminator(local_input, inputs_batch)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=local_input,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        return gamma_reg / 2 * (gradients.norm(2, dim=1) ** 2).mean()
