import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from .utils import get_rays


AXIS_VECTORS = {
    "+x": [1.0, 0.0, 0.0],
    "-x": [-1.0, 0.0, 0.0],
    "+y": [0.0, 1.0, 0.0],
    "-y": [0.0, -1.0, 0.0],
    "+z": [0.0, 0.0, 1.0],
    "-z": [0.0, 0.0, -1.0],
}


def axis_pose(device, direction="+x", radius=1.0, yaw_deg=0.0, up=None, dtype=torch.float32):
    def normalize(v):
        return v / (torch.norm(v, dim=-1, keepdim=True) + 1e-10)

    base = torch.tensor(AXIS_VECTORS[direction], device=device, dtype=dtype)

    yaw = math.radians(yaw_deg)
    Rz = torch.tensor([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw),  math.cos(yaw), 0],
        [0,              0,             1],
    ], device=device, dtype=dtype)

    center = (Rz @ base) * radius
    forward = -normalize(center.unsqueeze(0)).squeeze(0)

    if up is None:
        up_vec = [0.0, -1.0, 0.0] if direction in ["+z", "-z"] else [0.0, 0.0, 1.0]
    else:
        up_vec = AXIS_VECTORS[up]
    up = torch.tensor(up_vec, device=device, dtype=dtype)

    right = normalize(torch.cross(forward, up, dim=-1).unsqueeze(0)).squeeze(0)
    up = normalize(torch.cross(right, forward, dim=-1).unsqueeze(0)).squeeze(0)

    pose = torch.eye(4, dtype=dtype, device=device).unsqueeze(0)
    pose[0, :3, :3] = torch.stack([right, up, -forward], dim=-1)
    pose[0, :3, 3] = -center
    return pose


class NeRFDataset:
    def __init__(self, opt, device, image_size, fov_deg=75.0):
        self.device = device
        self.H = self.W = image_size

        self.poses = axis_pose(device, direction=opt.direction, radius=opt.radius, yaw_deg=opt.yaw_deg, up=opt.up)

        fov = math.radians(fov_deg)
        fl = (self.W / 2) / np.tan(fov / 2)
        self.intrinsics = np.array([fl, fl, self.W / 2, self.H / 2])

    def collate(self, index):
        poses = self.poses[index].to(self.device)
        rays = get_rays(poses, self.intrinsics, self.H, self.W, -1)
        return {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
        }

    def dataloader(self):
        loader = DataLoader(list(range(len(self.poses))), batch_size=1, collate_fn=self.collate, shuffle=False, num_workers=0)
        loader._data = self
        return loader
