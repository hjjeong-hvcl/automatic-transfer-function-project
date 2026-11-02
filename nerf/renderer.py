import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version as pver

import raymarching


@torch.no_grad()
def sample_normals_like_precomputed_grad(vol, grid, eps=1e-8):
    D, H, W = vol.shape[-3:]

    eps_x = 2.0 / max(W - 1, 1)
    eps_y = 2.0 / max(H - 1, 1)
    eps_z = 2.0 / max(D - 1, 1)

    def sample(g):
        return F.grid_sample(vol, g.type_as(vol), mode="bilinear", align_corners=True).view(-1)

    gx_p = grid.clone(); gx_p[..., 0] += eps_x
    gx_m = grid.clone(); gx_m[..., 0] -= eps_x
    gy_p = grid.clone(); gy_p[..., 1] += eps_y
    gy_m = grid.clone(); gy_m[..., 1] -= eps_y
    gz_p = grid.clone(); gz_p[..., 2] += eps_z
    gz_m = grid.clone(); gz_m[..., 2] -= eps_z

    grad_x = 0.5 * (sample(gx_p) - sample(gx_m))
    grad_y = 0.5 * (sample(gy_p) - sample(gy_m))
    grad_z = 0.5 * (sample(gz_p) - sample(gz_m))

    normals = torch.stack([grad_x, grad_y, grad_z], dim=-1)
    return F.normalize(normals, dim=-1, eps=eps)


def custom_meshgrid(*args):
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    return torch.meshgrid(*args, indexing='ij')


class NeRFRenderer(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.bound = opt.bound
        self.cascade = 1 + math.ceil(math.log2(opt.bound))
        self.grid_size = 128
        self.min_near = opt.min_near
        self.density_thresh = opt.density_thresh
        self.device = opt.device

        aabb_infer = torch.FloatTensor([-self.bound, -self.bound, -self.bound, self.bound, self.bound, self.bound])
        self.register_buffer('aabb_infer', aabb_infer)

        density_grid = torch.zeros([self.cascade, self.grid_size ** 3])
        density_bitfield = torch.zeros(self.cascade * self.grid_size ** 3 // 8, dtype=torch.uint8)
        self.register_buffer('density_grid', density_grid)
        self.register_buffer('density_bitfield', density_bitfield)
        self.mean_density = 0

    def density(self, xyzs_len, dt, device):
        raise NotImplementedError()

    def run_cuda(self, rays_o, rays_d, dt_gamma=0, max_steps=1024, T_thresh=1e-4):
        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)
        N = rays_o.shape[0]
        device = rays_o.device

        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, self.aabb_infer, self.min_near)
        bg_color = 1

        weights_sum = torch.zeros(N, dtype=torch.float32, device=device)
        depth = torch.zeros(N, dtype=torch.float32, device=device)
        image = torch.zeros(N, 3, dtype=torch.float32, device=device)

        rays_alive = torch.arange(N, dtype=torch.int32, device=device)
        rays_t = nears.clone()
        step = 0

        D, H, W = self.data_d_int, self.data_h_int, self.data_w_int

        while step < max_steps:
            n_alive = rays_alive.shape[0]
            if n_alive == 0:
                break

            n_step = max(min(N // n_alive, 8), 1)

            xyzs, dirs, deltas = raymarching.march_rays(
                n_alive, n_step, rays_alive, rays_t,
                rays_o, rays_d,
                self.bound, self.density_bitfield,
                self.cascade, self.grid_size,
                nears, fars,
                128, False,
                dt_gamma, max_steps
            )

            xyzs01_linear = (xyzs / self.bound + 1) * 0.5
            dims = xyzs01_linear.new_tensor([W, H, D]).float()
            scale = dims.max() / dims
            xyzs01_volume = (xyzs01_linear - 0.5) * scale + 0.5

            inside_mask = (
                (xyzs01_volume[:, 0] >= 0) & (xyzs01_volume[:, 0] <= 1) &
                (xyzs01_volume[:, 1] >= 0) & (xyzs01_volume[:, 1] <= 1) &
                (xyzs01_volume[:, 2] >= 0) & (xyzs01_volume[:, 2] <= 1)
            )

            coord_zyx = xyzs01_volume[:, [2, 1, 0]].clamp(0, 1)
            ijk = torch.floor(coord_zyx * xyzs01_linear.new_tensor([D, H, W])).long()
            ijk[:, 0] = ijk[:, 0].clamp(0, D - 1)
            ijk[:, 1] = ijk[:, 1].clamp(0, H - 1)
            ijk[:, 2] = ijk[:, 2].clamp(0, W - 1)

            grid = xyzs01_volume.view(1, -1, 1, 1, 3) * 2 - 1

            sigmas = F.grid_sample(
                self.volume.unsqueeze(0).unsqueeze(0), grid, mode="bilinear", align_corners=True
            ).view(-1) * self.d_scale
            sigmas = sigmas * inside_mask.float()

            vol = self.volume_intensity.unsqueeze(0).unsqueeze(0)

            intens_sampled = F.grid_sample(vol, grid, mode="bilinear", align_corners=True).view(-1)
            gm_vol = self.grad_mag.unsqueeze(0).unsqueeze(0)
            grad_sampled = F.grid_sample(gm_vol, grid, mode="bilinear", align_corners=True).view(-1)
            logits = self.forward_with_emb(xyzs01_volume, intens_sampled, grad_sampled)
            probs = torch.softmax(logits, dim=-1)

            keep_mask_per_class = torch.ones_like(probs)
            keep_mask_per_class[:, -1] = 0.0
            probs_keep = probs * keep_mask_per_class

            rgbs_raw = self.class_colors[probs_keep.argmax(dim=-1)]

            assigned_class = probs.argmax(dim=-1)

            p_bg = probs[:, -1]
            sigmas = sigmas * (1.0 - p_bg)

            intensity_vals = self.volume[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
            sigmas = sigmas * (intensity_vals > self.max_back_int).float()

            keep_set = torch.tensor(self.keep_classes, device=assigned_class.device)
            keep_mask = (assigned_class.unsqueeze(-1) == keep_set).any(dim=-1)
            sigmas = sigmas.clone()
            sigmas[~keep_mask] = 0.0

            keep_prob_sum = probs_keep.sum(dim=-1).clamp(0.0, 1.0)
            sigmas = sigmas * keep_prob_sum ** 3.0

            normals = sample_normals_like_precomputed_grad(vol, grid)

            world_up = torch.tensor([0.0, 1.0, 0.0], device=self.device, dtype=normals.dtype)
            view_dir = F.normalize(dirs, dim=-1)
            up = world_up - (world_up * view_dir).sum(-1, keepdim=True) * view_dir
            up = F.normalize(up, dim=-1)

            ndotu = (normals * up).sum(-1, keepdim=True)
            hemi = 0.5 * (ndotu + 1.0)

            ambient = 0.35
            diffuse_w = 0.75
            shading = ambient + diffuse_w * torch.pow(hemi, 1.2)
            brightness_gain = 1.2

            rgbs = rgbs_raw * shading * brightness_gain

            max_val = rgbs.max(dim=-1, keepdim=True)[0].clamp(min=1e-6)
            rgbs = torch.where(rgbs > 1.0, rgbs / max_val, rgbs)
            rgbs = rgbs.clamp(0, 1)
            rgbs = rgbs * inside_mask[:, None].float()
            rgbs = rgbs * (sigmas > 0).float().unsqueeze(-1)

            raymarching.composite_rays(
                n_alive, n_step, rays_alive, rays_t,
                sigmas, rgbs, deltas,
                weights_sum, depth, image,
                T_thresh
            )

            rays_alive = rays_alive[rays_alive >= 0]
            step += n_step

        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
        depth = torch.clamp(depth - nears, min=0) / (fars - nears)

        return {
            'image': image.view(*prefix, 3),
            'depth': depth.view(*prefix),
            'weights_sum': weights_sum,
        }

    @torch.no_grad()
    def update_extra_state(self, decay=0.95, S=128):
        tmp_grid = -torch.ones_like(self.density_grid)

        X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)

        for xs in X:
            for ys in Y:
                for zs in Z:
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    indices = raymarching.morton3D(coords).long()
                    xyzs = 2 * coords.float() / (self.grid_size - 1) - 1
                    for cas in range(self.cascade):
                        bound = min(2 ** cas, self.bound)
                        half_grid_size = bound / self.grid_size
                        cas_xyzs = xyzs * (bound - half_grid_size)
                        cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                        sigmas = self.density(xyzs_len=cas_xyzs.shape[0], dt=2 * half_grid_size, device=self.device)
                        tmp_grid[cas, indices] = sigmas

        valid_mask = (self.density_grid >= 0) & (tmp_grid >= 0)
        self.density_grid[valid_mask] = torch.maximum(self.density_grid[valid_mask] * decay, tmp_grid[valid_mask])
        self.mean_density = torch.mean(self.density_grid.clamp(min=0)).item()

        density_thresh = min(self.mean_density, self.density_thresh)
        self.density_bitfield = raymarching.packbits(self.density_grid, density_thresh, self.density_bitfield)

    def render(self, rays_o, rays_d, dt_gamma=0, max_steps=1024):
        return self.run_cuda(rays_o, rays_d, dt_gamma=dt_gamma, max_steps=max_steps)
