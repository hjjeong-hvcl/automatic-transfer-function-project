import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from .renderer import NeRFRenderer

COLOR_TABLE = {
    "red":      [0.85, 0.20, 0.20],
    "green":    [0.25, 0.85, 0.25],
    "blue":     [0.25, 0.45, 0.95],
    "yellow":   [0.95, 0.90, 0.25],
    "magenta":  [0.85, 0.25, 0.85],
    "cyan":     [0.25, 0.85, 0.85],
    "white":    [1.0, 1.0, 1.0],
}

class DualBranchHead(nn.Module):
    def __init__(self, pos_dim, attr_dim, hidden_dim, num_parts, fusion_mode="learnable", fixed_alpha=0.5):
        super().__init__()
        self.fusion_mode = fusion_mode

        self.pos_branch = nn.Sequential(
            nn.Linear(pos_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_parts),
        )
        self.attr_branch = nn.Sequential(
            nn.Linear(attr_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_parts),
        )

        if fusion_mode == "learnable":
            self.alpha = nn.Parameter(torch.zeros(num_parts))
        elif fusion_mode == "fixed":
            self.register_buffer("alpha", torch.ones(num_parts) * fixed_alpha)
        else:
            raise ValueError("fusion_mode must be 'learnable' or 'fixed'")

    def forward(self, pos_feat, attr_feat):
        logit_pos = self.pos_branch(pos_feat)
        logit_attr = self.attr_branch(attr_feat)

        alpha = torch.sigmoid(self.alpha) if self.fusion_mode == "learnable" else self.alpha
        alpha = alpha[None, :]

        return alpha * logit_pos + (1.0 - alpha) * logit_attr

class ColorNetwork(NeRFRenderer):
    def __init__(self, opt):
        super().__init__(opt)

        self.data_w_int = opt.data_w_int
        self.data_h_int = opt.data_h_int
        self.data_d_int = opt.data_d_int

        self.num_classes = None
        self.alpha0 = opt.alpha0
        self.d_scale = torch.tensor(opt.d_scale)
        self.remove_gm = opt.remove_gm

        self.hash_enc, self.hash_dim = get_encoder(
            "hashgrid",
            num_levels=opt.num_levels,
            level_dim=opt.level_dim,
            base_resolution=opt.base_resolution,
            per_level_scale=opt.per_level_scale,
            log2_hashmap_size=opt.log2_hashmap_size,
        )

        self.attr_dim = 64
        self.hidden_dim = 64

        self.attr_mlp = nn.Sequential(
            nn.Linear(1 if self.remove_gm else 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.attr_dim),
            nn.ReLU(inplace=True),
        )

    def gen_head(self, fusion_mode, fixed_beta):
        self.head = DualBranchHead(
            pos_dim=self.hash_dim,
            attr_dim=self.attr_dim,
            hidden_dim=self.hidden_dim,
            num_parts=self.num_classes,
            fusion_mode=fusion_mode,
            fixed_alpha=fixed_beta,
        )

    def set_class_colors(self, part_color_json):
        colors = []
        for part in part_color_json:
            color_name = part["color"].lower()
            if color_name not in COLOR_TABLE:
                raise ValueError(f"Unknown color in JSON: {color_name}")
            colors.append(COLOR_TABLE[color_name])
        colors.append([1.0, 1.0, 1.0])

        colors = torch.tensor(colors, dtype=torch.float32, device=self.device)
        self.register_buffer("class_colors", colors)

    def set_gradient(self):
        vol = self.volume

        grad_z = F.pad(vol[2:, :, :] - vol[:-2, :, :], (0, 0, 0, 0, 1, 1))
        grad_y = F.pad(vol[:, 2:, :] - vol[:, :-2, :], (0, 0, 1, 1, 0, 0))
        grad_x = F.pad(vol[:, :, 2:] - vol[:, :, :-2], (1, 1, 0, 0, 0, 0))

        self.grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2 + 1e-12).to(torch.float16)

    def prepare_intensity_stats(self):
        vol = self.volume.float()

        intensity_mean = vol.mean().item()
        intensity_std = max(vol.std().item(), 1e-6)

        self.volume_intensity = (vol - intensity_mean) / intensity_std

    def density(self, xyzs_len, dt, device, eps=1e-6):
        sigma_fixed = math.log(1.0 / max(1e-6, 1.0 - self.alpha0)) / (dt + eps)
        return torch.full((xyzs_len,), sigma_fixed, device=device).detach()

    def forward_with_emb(self, points, intens, grad):
        pos_feat = self.hash_enc(points)

        if self.remove_gm:
            attr_feat = self.attr_mlp(intens.unsqueeze(-1))
        else:
            attr_feat = self.attr_mlp(torch.stack([intens, grad], dim=-1))

        return self.head(pos_feat, attr_feat)
