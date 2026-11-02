import torch
import torch.nn.functional as F


def gather_6n_zyx(zyx, D, H, W):
    z, y, x = zyx[:, 0], zyx[:, 1], zyx[:, 2]

    return torch.stack([
        torch.stack([torch.clamp(z - 1, 0, D - 1), y, x], dim=1),
        torch.stack([torch.clamp(z + 1, 0, D - 1), y, x], dim=1),
        torch.stack([z, torch.clamp(y - 1, 0, H - 1), x], dim=1),
        torch.stack([z, torch.clamp(y + 1, 0, H - 1), x], dim=1),
        torch.stack([z, y, torch.clamp(x - 1, 0, W - 1)], dim=1),
        torch.stack([z, y, torch.clamp(x + 1, 0, W - 1)], dim=1),
    ], dim=1)


def zyx_to_points01_xyz(zyx, D, H, W):
    z = zyx[:, 0].float() / (D - 1)
    y = zyx[:, 1].float() / (H - 1)
    x = zyx[:, 2].float() / (W - 1)
    return torch.stack([x, y, z], dim=1)


def make_query_logits_fn(model):
    device = model.volume_intensity.device

    vol = model.volume_intensity.unsqueeze(0).unsqueeze(0)
    gm_vol = model.grad_mag.unsqueeze(0).unsqueeze(0)

    def query_logits(points01_xyz):
        N = points01_xyz.shape[0]
        grid = (points01_xyz.clone() * 2.0 - 1.0).view(1, N, 1, 1, 3).to(device)

        intens_sampled = F.grid_sample(vol, grid, mode="bilinear", align_corners=True).view(-1)
        grad_sampled = F.grid_sample(gm_vol, grid.type_as(gm_vol), mode="bilinear", align_corners=True).view(-1)

        return model.forward_with_emb(points01_xyz, intens_sampled, grad_sampled)

    return query_logits


def sparse_to_dense_prob_conf(gt_zyx, gt_prob, gt_conf, D, H, W, K, device):
    prob_dense = torch.zeros((D * H * W, K), device=device)
    conf_dense = torch.zeros((D * H * W,), device=device)

    linear = gt_zyx[:, 0] * (H * W) + gt_zyx[:, 1] * W + gt_zyx[:, 2]
    prob_dense[linear] = gt_prob
    conf_dense[linear] = gt_conf

    return prob_dense, conf_dense


@torch.no_grad()
def _sample_mix_gt_and_random(linear_all, is_gt_lookup, M, device):
    DHW = linear_all.numel()

    gt_idx = torch.nonzero(is_gt_lookup, as_tuple=False).flatten()
    n_gt = gt_idx.numel()

    if n_gt == 0:
        return linear_all[torch.randint(0, DHW, (M,), device=device)]

    Mg = min(M // 2, n_gt)
    Mu = M - Mg

    perm = torch.randperm(n_gt, device=device)[:Mg]
    lin_g = linear_all[gt_idx[perm]]
    lin_u = linear_all[torch.randint(0, DHW, (Mu,), device=device)]

    return torch.cat([lin_g, lin_u], dim=0)


def propagation_loss(
    query_logits_fn,
    volume_intensity,
    grad_mag,
    prob,
    conf,
    sigma=0.1,
    lambda_gt_prop=70.0,
    use_conf_teacher_for_wprop=True,
    M=65536,
    tau=1.0,
    lambda_gt_ce=1.0,
    gamma=1.0,
    conf_teacher_beta=0.1,
    eps=1e-6,
):
    device = volume_intensity.device
    D, H, W = volume_intensity.shape
    DHW = D * H * W

    is_gt_lookup = (conf > 0) & (prob.sum(dim=1) > 0)

    linear_all = torch.arange(DHW, device=device, dtype=torch.long)
    linear = _sample_mix_gt_and_random(linear_all, is_gt_lookup, M, device)

    z = linear // (H * W)
    y = (linear % (H * W)) // W
    x = linear % W
    zyx = torch.stack([z, y, x], dim=1)

    p_gt = prob[linear]
    gt_conf = conf[linear]
    prob_mass = p_gt.sum(dim=1)
    is_gt = (gt_conf > 0) & (prob_mass > 0)
    p_gt_norm = p_gt / (prob_mass[:, None] + eps)

    zyx_n = gather_6n_zyx(zyx, D, H, W)
    zn, yn, xn = zyx_n[..., 0], zyx_n[..., 1], zyx_n[..., 2]

    f_c = torch.stack([volume_intensity[z, y, x], grad_mag[z, y, x]], dim=-1)
    f_n = torch.stack([volume_intensity[zn, yn, xn], grad_mag[zn, yn, xn]], dim=-1)
    dist = torch.sum((f_n - f_c[:, None, :]) ** 2, dim=-1)

    conf_n = conf[(zn * (H * W) + yn * W + xn).view(-1)].view(-1, 6)

    weights = torch.softmax(-dist / (2.0 * sigma * sigma + eps), dim=1)
    conf_teacher = torch.sum(weights * conf_n, dim=1)
    conf_teacher = (conf_teacher * conf_teacher_beta).clamp(0.0, 1.0).detach()

    if use_conf_teacher_for_wprop:
        effective_conf = torch.where(is_gt, gt_conf, conf_teacher)
    else:
        effective_conf = gt_conf

    pts_c = zyx_to_points01_xyz(zyx, D, H, W)
    pts_n = zyx_to_points01_xyz(zyx_n.reshape(-1, 3), D, H, W)

    logits_c = query_logits_fn(pts_c)
    logits_n = query_logits_fn(pts_n).view(-1, 6, logits_c.shape[-1])

    logp_c = F.log_softmax(logits_c / tau, dim=-1)
    p_n = F.softmax(logits_n / tau, dim=-1)

    q = torch.sum(weights[..., None] * p_n, dim=1).detach()
    q = q / (q.sum(dim=-1, keepdim=True) + eps)

    kl = F.kl_div(logp_c, q, reduction="none").sum(-1)

    w_prop = torch.clamp((1.0 - effective_conf) ** gamma, min=0.0)
    loss_prop = (w_prop * kl).mean()

    if is_gt.any():
        ce_per = -(p_gt_norm * logp_c).sum(dim=1)
        ce = (ce_per[is_gt] * gt_conf[is_gt]).mean()
    else:
        ce = logits_c.sum() * 0.0

    total = lambda_gt_prop * loss_prop + lambda_gt_ce * ce
    return total, ce, loss_prop
