import typing as tp

import torch
from torch import nn    
import numpy as np


# helper function; removes the predicted noise (but adds some noise back in to avoid collapse)
def denoise_add_noise(
        a_t: torch.Tensor, 
        ab_t: torch.Tensor,
        b_t: torch.Tensor, 
        x: torch.Tensor, 
        t: int, 
        pred_noise: torch.Tensor, 
        z: torch.Tensor = None
) -> torch.Tensor:
 
    if z is None:
        z = torch.randn_like(x)
    noise = b_t.sqrt()[t] * z
    mean = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
    return mean + noise


def denoise_ddim(
        ab_t: torch.Tensor, 
        x: torch.Tensor, 
        t: torch.Tensor, 
        t_prev: int, 
        pred_noise: torch.Tensor
) -> torch.Tensor:

    ab = ab_t[t]
    ab_prev = ab_t[t_prev]
    
    x0_pred = ab_prev.sqrt() / ab.sqrt() * (x - (1 - ab).sqrt() * pred_noise)
    dir_xt = (1 - ab_prev).sqrt() * pred_noise

    return x0_pred + dir_xt


# sample quickly using DDIM
@torch.no_grad()
def sample_ddim(model: nn.Module, n_sample: int, image_size: int, device: torch.device, timesteps: int, n=20):
    # x_T ~ N(0, 1), sample initial noise
    samples = torch.randn(n_sample, 3, image_size, image_size).to(device)  

    # array to keep track of generated steps for plotting
    intermediate = [] 
    step_size = timesteps // n
    for i in range(timesteps, 0, -step_size):
        print(f'sampling timestep {i:3d}', end='\r')

        # reshape time tensor
        t = torch.tensor([i / timesteps])[:, None, None, None].to(device)

        eps = model(samples, t)    # predict noise e_(x_t,t)
        samples = denoise_ddim(samples, i, i - step_size, eps)
        intermediate.append(samples.detach().cpu().numpy())

    intermediate = np.stack(intermediate)
    return samples, intermediate


# sample using standard algorithm
@torch.no_grad()
def sample_ddpm(
    model: nn.Module, 
    n_sample: int, 
    image_size: int, 
    device: torch.device, 
    timesteps: int, 
    n: int = 20, 
    save_rate: int = 1
) -> tp.Tuple[tp.List[np.ndarray], tp.List[np.ndarray]]:
    
    # x_T ~ N(0, 1), sample initial noise
    samples = torch.randn(n_sample, 3, image_size, image_size).to(device)  

    # array to keep track of generated steps for plotting
    intermediate = [] 
    for i in range(timesteps, 0, -1):
        print(f'sampling timestep {i:3d}', end='\r')

        # reshape time tensor
        t = torch.tensor([i / timesteps])[:, None, None, None].to(device)

        # sample some random noise to inject back in. For i = 1, don't add back in noise
        z = torch.randn_like(samples) if i > 1 else 0

        eps = model(samples, t)    # predict noise e_(x_t,t)
        samples = denoise_add_noise(samples, i, eps, z)
        if i % save_rate == 0 or i==timesteps or i<8:
            intermediate.append(samples.detach().cpu().numpy())

    intermediate = np.stack(intermediate)
    return samples, intermediate
