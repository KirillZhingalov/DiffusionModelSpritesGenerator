import argparse as ap
import numpy as np

import torch
from torch import nn

from unet import ContextUnet
from samplers import (
    sample_ddim, 
    sample_ddpm
)


def parse_args():
    parser = ap.ArgumentParser(description='Train a Context Unet model')
    
    # diffusion hyperparameters
    parser.add_argument('--diffusion-timestamps', type=int, default=500, help='Number of diffusion timesteps')
    parser.add_argument('--diffusion-beta1', type=float, default=1e-4, help='Diffusion beta1 hyperparameter')
    parser.add_argument('--diffusion-beta2', type=float, default=0.02, help='Diffusion beta2 hyperparameter')
    
    # network hyperparameters
    parser.add_argument('--n-feat', type=int, default=256, help='Number of intermediate feature maps')
    parser.add_argument('--n-cfeat', type=int, default=10, help='Number of context features')
    parser.add_argument('--image-size', type=int, default=28, help='Size of input image')
    parser.add_argument('--save-dir', type=str, default='./weights/', help='Directory to save model weights')

    # inferece params
    parser.add_argument('--n-samples', type=int, default=1, help='Number of samples to generate')
    parser.add_argument('--sampler', type=str, default='ddim', help='Sampler to use for inference')

    return parser.parse_args()


# define sampling function for DDIM   
# removes the noise using ddim


def inference(args: ap.Namespace) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # construct model
    nn_model = ContextUnet(in_channels=3, n_feat=args.n_feat, n_cfeat=args.n_cfeat, height=args.image_size).to(device)

    if args.sampler == 'ddim':
        samples, _ = sample_ddim(nn_model, args.n_samples, args.image_size, device, args.diffusion_timestamps)

    elif args.sampler == 'ddpm':
        samples, _ = sample_ddpm(nn_model, args.n_samples, args.image_size, device, args.diffusion_timestamps)

    else:
        raise ValueError(f'Unknown sampler {args.sampler}')

    # TODO save samples or show in window


if __name__ == '__main__':
    inference(parse_args())