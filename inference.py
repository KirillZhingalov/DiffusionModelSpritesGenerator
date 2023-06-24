import matplotlib.pyplot as plt
import argparse as ap
import numpy as np
import os

import torch
from torch import nn
import torch.nn.functional as F

from unet import ContextUnet
from samplers import (
    sample_ddim, 
    sample_ddpm
)


def parse_args():
    parser = ap.ArgumentParser(description='Train a Context Unet model')
    
    # diffusion hyperparameters
    parser.add_argument('--diffusion-timestaps', type=int, default=500, help='Number of diffusion timesteps')
    parser.add_argument('--diffusion-beta1', type=float, default=1e-4, help='Diffusion beta1 hyperparameter')
    parser.add_argument('--diffusion-beta2', type=float, default=0.02, help='Diffusion beta2 hyperparameter')
    
    # network hyperparameters
    parser.add_argument('--n-feat', type=int, default=256, help='Number of intermediate feature maps')
    parser.add_argument('--n-cfeat', type=int, default=10, help='Number of context features')
    parser.add_argument('--image-size', type=int, default=28, help='Size of input image')
    
    # inferece params
    # hero, non-hero, food, spell, side-facing
    parser.add_argument('--weights-path', type=str, default='./weights/context_model_final.pth', help='Path to model weights')
    parser.add_argument('--sprite-class', type=str, default='hero', help='Sprite class to generate')
    parser.add_argument('--n-samples', type=int, default=1, help='Number of samples to generate')
    parser.add_argument('--sampler', type=str, default='ddim', help='Sampler to use for inference')
    parser.add_argument('--save-dir', type=str, default='./generated_samples/', help='Directory to save generated sprites')
    parser.add_argument('--visualize', action='store_true', help='Visualize generated sprites')

    return parser.parse_args()


CLASSES_MAPPING = {
    'hero': 0,
    'non-hero': 1,
    'food': 2,
    'spell': 3,
    'side-facing': 4
}


def inference(args: ap.Namespace) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # construct model
    nn_model = ContextUnet(in_channels=3, n_feat=args.n_feat, n_cfeat=args.n_cfeat, height=args.image_size).to(device)
    
    # Load model weights
    if os.path.exists(args.weights_path):
        nn_model.load_state_dict(torch.load(args.weights_path, map_location=device))
        nn_model.eval()

    if sprite_class_idx := CLASSES_MAPPING.get(args.sprite_class, None) is None:
        raise ValueError(f'Unknown sprite class {args.sprite_class}')

    t = torch.Tensor([sprite_class_idx for _ in range(args.n_samples)]).long()
    ctx = F.one_hot(t.long(), num_classes=5).float().to(device)

    if args.sampler == 'ddim':
        samples, _ = sample_ddim(
            model = nn_model, 
            ctx = ctx,
            n_samples = args.n_samples, 
            image_size = args.image_size, 
            device = device, 
            timestaps = args.diffusion_timestaps, 
            n = 20
        )

    elif args.sampler == 'ddpm':
        samples, _ = sample_ddpm(
            model = nn_model, 
            ctx=ctx,
            n_samples = args.n_samples, 
            image_size = args.image_size, 
            device = device, 
            timestaps = args.diffusion_timestaps
        )

    else:
        raise ValueError(f'Unknown sampler {args.sampler}')

    if args.visualize:
        plt.imshow(samples[0].permute(1, 2, 0).cpu().numpy())
        plt.show()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    for i in range(args.n_samples):
        np.save(os.path.join(args.save_dir, f'sample_{i}.npy'), samples[i].cpu().numpy())

if __name__ == '__main__':
    inference(parse_args())