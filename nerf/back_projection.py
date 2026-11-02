import cv2
import numpy as np
import torch
from PIL import Image

from nerf.MIP_axis import canonicalize_to_z_forward, compose_true_rotation, rotate_volume


COLOR_RANGES = {
    "red":     ([150,   0,   0], [255, 100, 100]),
    "blue":    ([  0,   0, 150], [100, 100, 255]),
    "yellow":  ([150, 150,   0], [255, 255, 100]),
    "green":   ([  0, 120,   0], [100, 255, 100]),
    "cyan":    ([  0, 150, 150], [100, 255, 255]),
    "magenta": ([150,   0, 150], [255, 100, 255]),
    "purple":  ([100,   0, 100], [200, 100, 255]),
    "black":   ([  0,   0,   0], [ 40,  40,  40]),
}


def build_color_to_part(color_json):
    part_ranges = []

    for idx, part in enumerate(color_json):
        color_name = part["color"].lower()

        matched = False
        for key, (lo, hi) in COLOR_RANGES.items():
            if key in color_name:
                part_ranges.append({
                    "part_id": idx,
                    "name": part["name"],
                    "color": key,
                    "lower": np.array(lo, dtype=np.uint8),
                    "upper": np.array(hi, dtype=np.uint8),
                })
                matched = True
                break

        if not matched:
            raise ValueError(f"Unsupported color '{part['color']}' for part '{part['name']}'")

    return part_ranges, len(part_ranges)


def pixel_to_part_id(rgb, part_ranges, background_part_id):
    r, g, b = rgb

    for pr in part_ranges:
        lo, hi = pr["lower"], pr["upper"]
        if lo[0] <= r <= hi[0] and lo[1] <= g <= hi[1] and lo[2] <= b <= hi[2]:
            return pr["part_id"]

    lo, hi = COLOR_RANGES["black"]
    if lo[0] <= r <= hi[0] and lo[1] <= g <= hi[1] and lo[2] <= b <= hi[2]:
        return background_part_id

    return None


def unrot90_coords(xr, yr, H, W, k):
    k %= 4
    if k == 0:
        return xr, yr
    elif k == 1:
        return (W - 1 - yr), xr
    elif k == 2:
        return (W - 1 - xr), (H - 1 - yr)
    elif k == 3:
        return yr, (H - 1 - xr)


def idx_to_norm(i, size):
    return (2.0 * i / (size - 1.0)) - 1.0


def norm_to_idx(xn, size):
    return (xn + 1.0) * 0.5 * (size - 1.0)


def canon_zyx_to_orig_zyx(z, y, x, axis):
    if axis == "z":
        return z, y, x
    elif axis == "y":
        return y, z, x
    elif axis == "x":
        return x, y, z
    else:
        raise ValueError(axis)


@torch.no_grad()
def select_peak_runs_gpu(rays, min_ratio=0.8, prom_ratio=0.1):
    device = rays.device
    R, D = rays.shape

    ray_max = rays.max(dim=1).values
    mask_high = rays >= (min_ratio * ray_max).unsqueeze(1)

    left = torch.cat([rays[:, :1], rays[:, :-1]], dim=1)
    right = torch.cat([rays[:, 1:], rays[:, -1:]], dim=1)
    local_max = (rays >= left) & (rays >= right) & mask_high
    global_max = (rays == ray_max.unsqueeze(1)) & mask_high

    ray_min = rays.min(dim=1).values
    prom_ok = (rays - ray_min.unsqueeze(1)) >= (prom_ratio * ray_max).unsqueeze(1)

    peak_mask = (local_max | global_max) & prom_ok

    prev = torch.cat([torch.zeros((R, 1), dtype=torch.bool, device=device), mask_high[:, :-1]], dim=1)
    run_start = mask_high & (~prev)
    run_id = torch.cumsum(run_start.to(torch.int32), dim=1) - 1

    max_runs = int(run_start.sum(dim=1).max().item()) if R > 0 else 0
    if max_runs == 0:
        return torch.zeros_like(mask_high), ray_max

    peak_pos = peak_mask.nonzero(as_tuple=False)
    if peak_pos.numel() == 0:
        return torch.zeros_like(mask_high), ray_max

    pr = peak_pos[:, 0]
    pz = peak_pos[:, 1]
    prid = run_id[pr, pz].clamp(min=0)

    lin = pr * max_runs + prid
    run_has_peak = torch.zeros((R * max_runs,), device=device, dtype=torch.int32)
    run_has_peak.index_add_(0, lin, torch.ones_like(lin, dtype=torch.int32))

    rid = run_id.clamp(min=0)
    lin_all = torch.arange(R, device=device).unsqueeze(1) * max_runs + rid
    has_peak = run_has_peak[lin_all] > 0

    return mask_high & has_peak, ray_max


def rot_zyx_to_unrot_padded_zyx_splat(z, y, x, R, Dp, Hp, Wp):
    xn = idx_to_norm(x.float(), Wp)
    yn = idx_to_norm(y.float(), Hp)
    zn = idx_to_norm(z.float(), Dp)

    p_unrot = torch.stack([xn, yn, zn], dim=-1) @ R.T

    xf = norm_to_idx(p_unrot[:, 0], Wp)
    yf = norm_to_idx(p_unrot[:, 1], Hp)
    zf = norm_to_idx(p_unrot[:, 2], Dp)

    x0 = torch.floor(xf).long()
    y0 = torch.floor(yf).long()
    z0 = torch.floor(zf).long()
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1

    x0.clamp_(0, Wp - 1); x1.clamp_(0, Wp - 1)
    y0.clamp_(0, Hp - 1); y1.clamp_(0, Hp - 1)
    z0.clamp_(0, Dp - 1); z1.clamp_(0, Dp - 1)

    wx = (xf - x0.float()).clamp(0, 1)
    wy = (yf - y0.float()).clamp(0, 1)
    wz = (zf - z0.float()).clamp(0, 1)

    zyx8 = torch.stack([
        torch.stack([z0, y0, x0], dim=-1),
        torch.stack([z0, y0, x1], dim=-1),
        torch.stack([z0, y1, x0], dim=-1),
        torch.stack([z0, y1, x1], dim=-1),
        torch.stack([z1, y0, x0], dim=-1),
        torch.stack([z1, y0, x1], dim=-1),
        torch.stack([z1, y1, x0], dim=-1),
        torch.stack([z1, y1, x1], dim=-1),
    ], dim=1)

    w8 = torch.stack([
        (1 - wz) * (1 - wy) * (1 - wx),
        (1 - wz) * (1 - wy) * wx,
        (1 - wz) * wy * (1 - wx),
        (1 - wz) * wy * wx,
        wz * (1 - wy) * (1 - wx),
        wz * (1 - wy) * wx,
        wz * wy * (1 - wx),
        wz * wy * wx,
    ], dim=1)

    return zyx8, w8


@torch.no_grad()
def build_peak_voxels_multiview_prob_gpu(
    D, H, W,
    padded_volume,
    pad0,
    views,
    part_json,
    rot_angle,
    device="cuda",
    min_ratio=0.95,
    prom_ratio=0.1,
    ray_chunk=4096,
    tau_px=0.5,
):
    padded_volume = padded_volume.to(device)
    pad_d0, pad_h0, pad_w0 = pad0

    K = len(part_json) + 1
    V = len(views)

    zyx_all_list = []
    label_all_list = []
    conf_all_list = []
    back_int_list = []

    part_ranges, bg_id = build_color_to_part(part_json)

    for v in views:
        axis = v["axis"]

        vol_can = canonicalize_to_z_forward(padded_volume, axis)
        Dpc, Hc, Wc = vol_can.shape
        z_idx_full = torch.arange(Dpc, device=device)

        mask_np = np.array(Image.open(v["mask_path"]).convert("RGB"))
        Hm, Wm = mask_np.shape[:2]

        mask_part = np.full((Hm, Wm), -1, dtype=np.int32)
        for yy in range(Hm):
            for xx in range(Wm):
                p = pixel_to_part_id(tuple(mask_np[yy, xx]), part_ranges, bg_id)
                if p is not None:
                    mask_part[yy, xx] = int(p)
        fg = (mask_part != -1)

        up = np.full_like(mask_part, -1); up[1:] = mask_part[:-1]
        down = np.full_like(mask_part, -1); down[:-1] = mask_part[1:]
        left = np.full_like(mask_part, -1); left[:, 1:] = mask_part[:, :-1]
        right = np.full_like(mask_part, -1); right[:, :-1] = mask_part[:, 1:]

        boundary_all = fg & ((up != mask_part) | (down != mask_part) | (left != mask_part) | (right != mask_part))

        fg_u8 = fg.astype(np.uint8)
        inv = fg_u8.copy()
        inv[boundary_all] = 0
        inv[(fg_u8 == 1) & (~boundary_all)] = 1
        dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)

        w_soft = 1.0 - np.exp(-dist / tau_px)
        w_soft[~fg] = 0.0

        R_eff = compose_true_rotation(v["theta"], v["phi"], Dpc, Hc, Wc, device=device)
        vol_rot = rotate_volume(vol_can[None, None], R_eff)[0, 0]

        k = (rot_angle // 90) % 4
        y0b, y1b, x0b, x1b = v["bbox"]

        x0_list, y0_list, parts, w_full = [], [], [], []
        for y in range(Hm):
            for x in range(Wm):
                part = mask_part[y, x]
                if part == -1:
                    continue
                x0, y0 = unrot90_coords(x + x0b, y + y0b, Hc, Wc, k)
                x0_list.append(x0)
                y0_list.append(y0)
                parts.append(int(part))
                w_full.append(float(w_soft[y, x]))

        if len(parts) == 0:
            continue

        x0_all = torch.tensor(x0_list, device=device, dtype=torch.long)
        y0_all = torch.tensor(y0_list, device=device, dtype=torch.long)
        part_all = torch.tensor(parts, device=device, dtype=torch.long)
        w_all = torch.tensor(w_full, device=device, dtype=torch.float32)

        Rrays = x0_all.shape[0]

        for s in range(0, Rrays, ray_chunk):
            e = min(s + ray_chunk, Rrays)

            x0 = x0_all[s:e]
            y0 = y0_all[s:e]
            part_ids = part_all[s:e]
            w_pix = w_all[s:e]

            rays = vol_rot[:, y0, x0].transpose(0, 1).contiguous()

            keep_mask, ray_max = select_peak_runs_gpu(rays, min_ratio=min_ratio, prom_ratio=prom_ratio)

            bg_mask = (part_ids == (K - 1))
            if bg_mask.any():
                back_int_list.append(ray_max[bg_mask])

            nz = keep_mask.nonzero(as_tuple=False)
            if nz.numel() == 0:
                continue

            r_idx = nz[:, 0]
            z_sel = nz[:, 1]

            y_sel = y0[r_idx]
            x_sel = x0[r_idx]
            part_sel = part_ids[r_idx]

            I_peak = rays[r_idx, z_sel]
            conf = (I_peak / (ray_max[r_idx] + 1e-6)).clamp(0, 1)
            conf = conf * w_pix[r_idx]

            zyx8_can, w8 = rot_zyx_to_unrot_padded_zyx_splat(
                z_idx_full[z_sel], y_sel, x_sel, R_eff, Dpc, Hc, Wc
            )
            zyx8_can = zyx8_can.long()

            z_pad, y_pad, x_pad = canon_zyx_to_orig_zyx(zyx8_can[..., 0], zyx8_can[..., 1], zyx8_can[..., 2], axis)
            zyx8_pad = torch.stack([z_pad, y_pad, x_pad], dim=-1)

            zyx8_pad[..., 0] -= pad_d0
            zyx8_pad[..., 1] -= pad_h0
            zyx8_pad[..., 2] -= pad_w0

            zyx_flat = zyx8_pad.view(-1, 3)
            w_flat = w8.view(-1)

            part_rep = part_sel.repeat_interleave(8)
            conf_rep = conf.repeat_interleave(8) * w_flat

            inside = (
                (zyx_flat[:, 0] >= 0) & (zyx_flat[:, 0] < D) &
                (zyx_flat[:, 1] >= 0) & (zyx_flat[:, 1] < H) &
                (zyx_flat[:, 2] >= 0) & (zyx_flat[:, 2] < W)
            )

            if inside.any():
                zyx_all_list.append(zyx_flat[inside].long())
                label_all_list.append(part_rep[inside])
                conf_all_list.append(conf_rep[inside])

    if len(zyx_all_list) == 0:
        raise RuntimeError("No peak voxels extracted.")

    zyx_all = torch.cat(zyx_all_list, dim=0)
    label_all = torch.cat(label_all_list, dim=0)
    conf_all = torch.cat(conf_all_list, dim=0)

    idx = zyx_all[:, 0] * (H * W) + zyx_all[:, 1] * W + zyx_all[:, 2]
    uniq_idx, inv = torch.unique(idx, return_inverse=True)
    N = uniq_idx.shape[0]

    gt_prob = torch.zeros((N, K), device=device)
    gt_prob.index_put_((inv, label_all), torch.ones_like(conf_all), accumulate=True)

    gt_conf = torch.zeros((N,), device=device)
    gt_conf.index_add_(0, inv, conf_all)

    gt_prob /= V
    gt_conf /= V

    z = uniq_idx // (H * W)
    y = (uniq_idx % (H * W)) // W
    x = uniq_idx % W
    gt_zyx = torch.stack([z, y, x], dim=1)

    back_int = torch.cat(back_int_list, dim=0) if len(back_int_list) > 0 else torch.empty(0, device=device)

    return gt_zyx, gt_prob, gt_conf, back_int
