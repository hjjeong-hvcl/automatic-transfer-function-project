import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def canonicalize_to_z_forward(vol, axis):
    if axis == "z":
        return vol
    elif axis == "y":
        return vol.permute(1, 0, 2)
    elif axis == "x":
        return vol.permute(2, 1, 0)
    else:
        raise ValueError(axis)


def crop_black_padding(mip, thresh=1e-4, pad=4):
    mask = mip > thresh

    if not mask.any():
        H, W = mip.shape
        return mip, (0, H, 0, W)

    ys, xs = torch.where(mask)

    y0 = max(ys.min().item() - pad, 0)
    y1 = min(ys.max().item() + pad + 1, mip.shape[0])
    x0 = max(xs.min().item() - pad, 0)
    x1 = min(xs.max().item() + pad + 1, mip.shape[1])

    return mip[y0:y1, x0:x1], (y0, y1, x0, x1)


def save_mip_image(mip, save_path, crop_thresh=1e-4, crop_pad=6):
    mip, bbox = crop_black_padding(mip, thresh=crop_thresh, pad=crop_pad)

    mip_np = mip.detach().cpu().numpy()
    vmin, vmax = mip_np.min(), mip_np.max()
    if vmax > vmin:
        mip_np = (mip_np - vmin) / (vmax - vmin)
    else:
        mip_np = np.zeros_like(mip_np)

    img = Image.fromarray((mip_np * 255).astype(np.uint8), mode="L")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path)

    return bbox


def save_axis_conditioned_mips(mips, out_dir, prefix="mip", pad0=None, rot_angle=0):
    os.makedirs(out_dir, exist_ok=True)

    for item in mips:
        k = (rot_angle // 90) % 4
        mip = torch.rot90(item["mip"], k=k, dims=(0, 1))

        fname = f"{prefix}_axis-{item['axis']}_theta-{item['theta']:+03d}_phi-{item['phi']:03d}.png"
        item["bbox"] = save_mip_image(mip, os.path.join(out_dir, fname))

    meta = [{k: v for k, v in item.items() if k != "mip"} for item in mips]
    np.save(os.path.join(out_dir, "views.npy"), {"pad0": pad0, "views": meta}, allow_pickle=True)


def pad_volume_for_rotation(volume_dhw, theta_max_deg=45, margin=0.15):
    D, H, W = volume_dhw.shape

    th = math.radians(theta_max_deg)
    factor = (abs(math.cos(th)) + abs(math.sin(th))) * (1.0 + margin)

    padD = max(0, int(math.ceil(D * factor)) - D)
    padH = max(0, int(math.ceil(H * factor)) - H)
    padW = max(0, int(math.ceil(W * factor)) - W)

    pad_d0, pad_h0, pad_w0 = padD // 2, padH // 2, padW // 2
    pad_d1, pad_h1, pad_w1 = padD - pad_d0, padH - pad_h0, padW - pad_w0

    vol5 = volume_dhw.unsqueeze(0).unsqueeze(0)
    vol5 = F.pad(vol5, (pad_w0, pad_w1, pad_h0, pad_h1, pad_d0, pad_d1), mode="constant", value=0)

    return vol5[0, 0], (pad_d0, pad_h0, pad_w0)


def camera_rotation(theta_deg, phi_deg, device="cuda"):
    th = math.radians(theta_deg)
    ph = math.radians(phi_deg)

    R_az = torch.tensor([
        [math.cos(ph), -math.sin(ph), 0],
        [math.sin(ph),  math.cos(ph), 0],
        [0,             0,            1],
    ], device=device)

    R_tlt = torch.tensor([
        [1, 0,             0],
        [0, math.cos(th), -math.sin(th)],
        [0, math.sin(th),  math.cos(th)],
    ], device=device)

    return R_az @ R_tlt


def similarity_transform_for_affine_grid(R_xyz, D, H, W, device="cuda"):
    S = torch.diag(torch.tensor([W - 1.0, H - 1.0, D - 1.0], device=device))
    S_inv = torch.diag(torch.tensor([1.0 / (W - 1.0), 1.0 / (H - 1.0), 1.0 / (D - 1.0)], device=device))
    return (S_inv @ R_xyz @ S).float()


def compose_true_rotation(theta_deg, phi_deg, D, H, W, device="cuda"):
    R = camera_rotation(theta_deg, phi_deg, device=device).float()
    return similarity_transform_for_affine_grid(R, D, H, W, device=device)


def rotate_volume(volume_5d, R):
    device = volume_5d.device
    D, H, W = volume_5d.shape[-3:]

    A = torch.zeros((1, 3, 4), device=device)
    A[0, :, :3] = R

    grid = F.affine_grid(A, size=(1, 1, D, H, W), align_corners=True)
    return F.grid_sample(volume_5d, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def camera_mip(volume_dhw, axis="z", theta_deg=0.0, phi_deg=0.0):
    vol = canonicalize_to_z_forward(volume_dhw, axis)
    D, H, W = vol.shape

    R = compose_true_rotation(theta_deg, phi_deg, D, H, W, device=volume_dhw.device)
    vol_rot = rotate_volume(vol.unsqueeze(0).unsqueeze(0), R)

    return vol_rot[0, 0].max(dim=0).values


def generate_axis_conditioned_mips(volume, axis="z", thetas=(0,), phis=(0,)):
    mips = []
    for theta in thetas:
        phi_list = (0,) if theta == 0 else phis
        for phi in phi_list:
            mips.append({
                "axis": axis,
                "theta": theta,
                "phi": phi,
                "mip": camera_mip(volume, axis=axis, theta_deg=theta, phi_deg=phi),
            })
    return mips
