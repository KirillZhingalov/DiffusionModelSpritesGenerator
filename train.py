import os
import argparse as ap
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


from dataset import (
    CustomDataset, 
    transform
)
from unet import ContextUnet



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

    # training hyperparameters
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size')
    parser.add_argument('--n-epoch', type=int, default=32, help='Number of epochs')
    parser.add_argument('--lrate', type=float, default=1e-3, help='Learning rate')
    
    return parser.parse_args()


# helper function: perturbs an image to a specified noise level
def perturb_input(ab_t: torch.Tensor, x: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
    return ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]) * noise


# Main function for model training
def train(args: ap.Namespace) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # construct model
    nn_model = ContextUnet(in_channels=3, n_feat=args.n_feat, n_cfeat=args.c_feat, height=args.image_size).to(device)

    # construct DDPM noise schedule
    b_t = (args.diffusion_beta1 - args.diffusion_beta2) * torch.linspace(0, 1, args.diffusion_timestamps + 1, device=device) + args.diffusion_beta1
    a_t = 1 - b_t
    ab_t = torch.cumsum(a_t.log(), dim=0).exp()    
    ab_t[0] = 1

    # setup optimizer
    optim = torch.optim.Adam(nn_model.parameters(), lr=args.lrate)

    # load dataset and construct optimizer
    dataset = CustomDataset("./data/sprites_1788_16x16.npy", "./data/sprite_labels_nc_1788_16x16.npy", transform, null_context=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)

    # training with context code
    # set into train mode
    nn_model.train()

    for ep in range(args.n_epoch):
        print(f'epoch {ep}')
        
        # linearly decay learning rate
        optim.param_groups[0]['lr'] = args.lrate * (1 - ep/args.n_epoch)
        
        pbar = tqdm(dataloader, mininterval=2 )
        for x, c in pbar:   # x: images  c: context
            optim.zero_grad()
            x = x.to(device)
            c = c.to(x)
            
            # randomly mask out c
            context_mask = torch.bernoulli(torch.zeros(c.shape[0]) + 0.9).to(device)
            c = c * context_mask.unsqueeze(-1)
            
            # perturb data
            noise = torch.randn_like(x)
            t = torch.randint(1, args.timestamps + 1, (x.shape[0],)).to(device) 
            x_pert = perturb_input(x, t, noise)
            
            # use network to recover noise
            pred_noise = nn_model(x_pert, t / args.timestamps, c=c)
            
            # loss is mean squared error between the predicted and true noise
            loss = F.mse_loss(pred_noise, noise)
            loss.backward()
            
            optim.step()

        # save model periodically
        if ep and not ep % 4:
            if not os.path.exists(args.save_dir):
                os.mkdir(args.save_dir)
            torch.save(nn_model.state_dict(), args.save_dir + f"context_model_{ep}.pth")
            print('saved model at ' + args.save_dir + f"context_model_{ep}.pth")

    torch.save(nn_model.state_dict(), args.save_dir + f"context_model_final.pth")


if __name__ == '__main__':
    train(parse_args())
    