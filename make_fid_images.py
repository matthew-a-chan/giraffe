import torch
import os
import argparse
from tqdm import tqdm
import time
from im2scene import config
from im2scene.checkpoints import CheckpointIO
import numpy as np
from im2scene.eval import (
    calculate_activation_statistics, calculate_frechet_distance)
from math import ceil
from torchvision.utils import save_image, make_grid


parser = argparse.ArgumentParser(
    description='Evaluate a GIRAFFE model.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

out_dir = cfg['training']['out_dir']
out_vis_file = os.path.join(out_dir, 'fid_images.jpg')

# Model
model = config.get_model(cfg, device=device)

checkpoint_io = CheckpointIO(out_dir, model=model)
try:
    checkpoint_io.load(cfg['test']['model_file'])
except:
    print(cfg['test']['model_file'], 'not found. Using:', cfg['test']['model_file'].replace('_best', ''))
    checkpoint_io.load(cfg['test']['model_file'].replace('_best', ''))

# Generate
model.eval()

n_images = 256
batch_size = cfg['training']['batch_size']
n_iter = ceil(n_images / batch_size)

img_fake = []
for i in tqdm(range(n_iter)):
    with torch.no_grad():
        img_fake.append(model(batch_size).cpu())
img_fake = torch.cat(img_fake, dim=0)[:n_images]
img_fake.clamp_(0., 1.)

img_uint8 = (img_fake * 255).cpu().numpy().astype(np.uint8)

# use uint for eval to fairly compare
img_fake = torch.from_numpy(img_uint8).float() / 255.

# Save a grid of 16x16 images for visualization
save_image(img_fake[:256],  out_vis_file, nrow=16, normalize=False, scale_each=True)
