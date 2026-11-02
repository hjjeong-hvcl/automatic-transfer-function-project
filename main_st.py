import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch

from nerf.color_network import ColorNetwork
from nerf.provider import NeRFDataset
from nerf.utils import Trainer, seed_everything

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

AXES = ['+x', '-x', '+y', '-y', '+z', '-z']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-O', dest='fp16', action='store_true', help="use mixed precision")
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--data_str', type=str, required=True, help="<name>_<W>_<H>_<D>, loaded from /dataset/<data_str>.raw")
    parser.add_argument('--spacing', type=float, nargs=3, required=True, help="voxel spacing (sz sy sx)")
    parser.add_argument('--dtype', type=str, default=None, choices=['uint8', 'uint16', 'float32'])
    parser.add_argument('--object_hint', type=str, required=True)

    parser.add_argument('--mip_axis', type=str, default='z', choices=['x', 'y', 'z'])
    parser.add_argument('--mip_view', type=str, default='side', choices=['top', 'side'])
    parser.add_argument('--mip_rot', type=int, default=0)
    parser.add_argument('--mip_best_phi', type=int, default=90)

    parser.add_argument('--direction', type=str, default='+x', choices=AXES)
    parser.add_argument('--up', type=str, default=None, choices=AXES)
    parser.add_argument('--yaw_deg', type=float, default=0.0)
    parser.add_argument('--radius', type=float, default=2.5)
    parser.add_argument('--image_size', type=int, default=512)

    parser.add_argument('--bound', type=float, default=1.0)
    parser.add_argument('--min_near', type=float, default=0.0)
    parser.add_argument('--density_thresh', type=float, default=10)
    parser.add_argument('--dt_gamma', type=float, default=0.0)
    parser.add_argument('--max_steps', type=int, default=1024)
    parser.add_argument('--alpha0', type=float, default=0.5)
    parser.add_argument('--d_scale', type=float, default=5000.0)

    parser.add_argument('--num_levels', type=int, default=16)
    parser.add_argument('--level_dim', type=int, default=8)
    parser.add_argument('--base_resolution', type=int, default=16)
    parser.add_argument('--per_level_scale', type=float, default=1.5)
    parser.add_argument('--log2_hashmap_size', type=int, default=19)

    parser.add_argument('--remove_gm', action='store_true')
    parser.add_argument('--fusion_mode', type=str, default='learnable', choices=['learnable', 'fixed'])
    parser.add_argument('--fixed_beta', type=float, default=0.5)
    parser.add_argument('--lambda_gt_prop', type=float, default=70.0)
    parser.add_argument('--use_conf_teacher_for_wprop', type=int, default=1)
    parser.add_argument('--affinity_sigma', type=float, default=0.1)
    return parser.parse_args()


def main():
    opt = parse_args()
    opt.spacing = tuple(opt.spacing)
    if opt.dtype is not None:
        opt.dtype = getattr(np, opt.dtype)

    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_split = opt.data_str.split("_")
    opt.data_name = data_split[0]
    opt.data_w_int = int(data_split[1])
    opt.data_h_int = int(data_split[2])
    opt.data_d_int = int(data_split[3])

    opt.time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    opt.workspace = os.path.join("results", opt.data_name, "workspaces", opt.time_stamp)
    os.makedirs(opt.workspace, exist_ok=False)

    with open(os.path.join(opt.workspace, "command.txt"), "w") as f:
        f.write(' '.join(sys.argv) + '\n')

    seed_everything(opt.seed)

    model = ColorNetwork(opt)
    trainer = Trainer(opt, model, opt.device)

    valid_loader = NeRFDataset(opt, opt.device, opt.image_size).dataloader()
    trainer.run_gemini(valid_loader)


if __name__ == '__main__':
    main()
