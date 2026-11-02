import numpy as np
import torch
import torch.nn.functional as F


def load_and_resample_volume(volume_path, shape, spacing, device="cuda", mode="trilinear", dtype=None):
    if dtype is None:
        file_bytes = np.fromfile(volume_path, dtype=np.uint8).size
        num_voxels = shape[0] * shape[1] * shape[2]
        if file_bytes == num_voxels:
            dtype = np.uint8
        elif file_bytes == num_voxels * 2:
            dtype = np.uint16
        elif file_bytes == num_voxels * 4:
            dtype = np.float32
        else:
            raise ValueError(
                f"Cannot infer dtype. file_bytes={file_bytes}, "
                f"expected {num_voxels} (uint8) or {num_voxels*2} (uint16) "
                f"or {num_voxels*4} (float32)"
            )

    vol = np.fromfile(volume_path, dtype=dtype).reshape(shape)

    vol = vol.astype(np.float32)
    if dtype == np.uint8:
        vol /= 255.0
    elif dtype == np.uint16:
        vol /= 65535.0
    else:
        vmin, vmax = vol.min(), vol.max()
        if vmax > vmin:
            vol = (vol - vmin) / (vmax - vmin)

    volume = torch.from_numpy(vol).to(device)
    D, H, W = shape
    sz, sy, sx = spacing

    if abs(sz - sy) < 1e-6 and abs(sy - sx) < 1e-6:
        return volume, D, H, W

    s_target = min(sz, sy, sx)
    D_new = int(round(D * sz / s_target))
    H_new = int(round(H * sy / s_target))
    W_new = int(round(W * sx / s_target))

    vol_iso = F.interpolate(
        volume[None, None],
        size=(D_new, H_new, W_new),
        mode=mode,
        align_corners=False,
    )
    return vol_iso[0, 0], D_new, H_new, W_new
