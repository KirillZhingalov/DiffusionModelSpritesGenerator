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
def sample_ddim(
    model: nn.Module, 
    ctx: torch.Tensor, 
    n_samples: int, 
    image_size: int, 
    device: torch.device, 
    timestaps: int, 
    n: int = 20
) -> torch.Tensor:
    
    # x_T ~ N(0, 1), sample initial noise
    samples = torch.randn(n_samples, 3, image_size, image_size).to(device)  

    # array to keep track of generated steps for plotting 
    step_size = timestaps // n
    for i in range(timestaps, 0, -step_size):
        print(f'sampling timestep {i:3d}', end='\r')

        # reshape time tensor
        t = torch.tensor([i / timestaps])[:, None, None, None].to(device)

        eps = model(samples, t, c=ctx)    # predict noise e_(x_t,t)
        samples = denoise_ddim(samples, i, i - step_size, eps)

    return samples


# sample using standard algorithm
@torch.no_grad()
def sample_ddpm(
    model: nn.Module, 
    ctx: torch.Tensor,
    n_samples: int, 
    image_size: int, 
    device: torch.device, 
    timestaps: int, 
) -> torch.Tensor:
    
    # x_T ~ N(0, 1), sample initial noise
    samples = torch.randn(n_samples, 3, image_size, image_size).to(device)  

    # array to keep track of generated steps for plotting
    intermediate = [] 
    for i in range(timestaps, 0, -1):
        print(f'sampling timestep {i:3d}', end='\r')

        # reshape time tensor
        t = torch.tensor([i / timestaps])[:, None, None, None].to(device)

        # sample some random noise to inject back in. For i = 1, don't add back in noise
        z = torch.randn_like(samples) if i > 1 else 0

        eps = model(samples, t, c=ctx)    # predict noise e_(x_t,t)
        samples = denoise_add_noise(samples, i, eps, z)

    return samples
