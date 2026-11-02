import io
import json
import os
import random
import re

import cv2
import numpy as np
import torch
from google import genai
from packaging import version as pver
from PIL import Image

from nerf.MIP_axis import generate_axis_conditioned_mips, pad_volume_for_rotation, save_axis_conditioned_mips
from nerf.back_projection import build_peak_voxels_multiview_prob_gpu
from nerf.partnet_utils import (
    make_query_logits_fn,
    propagation_loss,
    sparse_to_dense_prob_conf,
)
from nerf.volume import load_and_resample_volume

RECOLOR_PROMPT = '''
Preserve the exact silhouette and structure without any additions.

STRICT GEOMETRIC PRESERVATION RULES: 1. Negative Space Integrity: Maintain all black pixels between fine structures. Do not bridge or merge separate elements of the object, no matter how close they are. Every gap must be preserved. 2. High-Frequency Detail Retention: Capture and colorize every single foreground pixel. Do not simplify, smooth, or round off the edges of thin, jagged, or pointed structures. 3. Anti-Dilation Policy: Do not thicken or expand the original lines. The tinting must follow the exact 1-pixel-wide fidelity of the original luminance data. 4. Sharp Boundary Definition: Ensure a distinct color transition between different identified parts. Colors must not bleed across the structural boundaries of the object.

IMPORTANT: Identify and distinguish distinct 3 to 5 parts of the object.
After generating the image, you MUST also output the color information for each part in the following JSON format:
[
    {"name": "part_name", "color": "color_description"}
]
STRICT COLOR PALETTE CONTROL: Allowed Colors Only: You must use ONLY the following 7 colors for both image generation and JSON labeling: [red, blue, yellow, green, cyan, magenta, purple].
'''

ANALYSIS_PROMPT = (
    "Identify the colors applied to each part and output them in the following JSON format only:\n"
    + "[\n"
    + "  {\"name\": \"part_name\", \"color\": \"color_description\"}\n"
    + "]\n"
    + "Output ONLY the raw JSON code. No conversational text.\n"
    + "JSON OUTPUT RULES: 1. Atomic Naming Only: Each \"name\" must be a single, distinct anatomical or structural term. Do not use slashes (/), conjunctions (and), or multiple labels for a single color. 2. Primary Labeling: If a part could be described by multiple names, choose only the single most dominant name. 3. No Ambiguity: Each part name must be unique and represent exactly one identified region."
    + "STRICT COLOR PALETTE CONTROL: 1. Allowed Colors Only: You must use ONLY the following 7 colors for both image generation and JSON labeling: [red, blue, yellow, green, cyan, magenta, purple]. 2. Exact Matching: Do not use variations or similar colors (e.g., do not use \"pink\" instead of \"magenta\", do not use \"lime\" instead of \"green\"). 3. Palette Consistency: The color assigned to a part in the generated image must exactly match the color name provided in the JSON output. 4. Mandatory Selection: Map each identified part to the most appropriate color from the allowed list above."
)

def custom_meshgrid(*args):
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    return torch.meshgrid(*args, indexing='ij')

@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1):
    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = custom_meshgrid(torch.linspace(0, W - 1, W, device=device), torch.linspace(0, H - 1, H, device=device))
    i = i.t().reshape([1, H * W]).expand([B, H * W]) + 0.5
    j = j.t().reshape([1, H * W]).expand([B, H * W]) + 0.5

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2)

    rays_o = poses[..., :3, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d)

    return {'rays_o': rays_o, 'rays_d': rays_d}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def save_render(preds, path):
    pred = (preds[0].detach().cpu().numpy() * 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))

class Trainer(object):
    def __init__(self, opt, model, device):
        self.opt = opt
        self.device = device
        self.fp16 = opt.fp16
        self.workspace = opt.workspace

        self.data_str = opt.data_str
        self.data_name = opt.data_name
        self.data_w_int = opt.data_w_int
        self.data_h_int = opt.data_h_int
        self.data_d_int = opt.data_d_int
        self.spacing = opt.spacing
        self.dtype = opt.dtype

        self.mip_axis = opt.mip_axis
        self.mip_view = opt.mip_view
        self.mip_rot = opt.mip_rot
        self.mip_best_phi = opt.mip_best_phi
        self.object_hint = opt.object_hint

        self.fusion_mode = opt.fusion_mode
        self.fixed_beta = opt.fixed_beta
        self.lambda_gt_prop = opt.lambda_gt_prop
        self.use_conf_teacher_for_wprop = opt.use_conf_teacher_for_wprop
        self.affinity_sigma = opt.affinity_sigma

        model.to(self.device)
        self.model = model

    def render_step(self, data):
        H, W = data['H'], data['W']
        outputs = self.model.render(
            data['rays_o'], data['rays_d'],
            dt_gamma=self.opt.dt_gamma, max_steps=self.opt.max_steps,
        )
        return outputs['image'].reshape(-1, H, W, 3), outputs['depth'].reshape(-1, H, W)

    def generate_mask_with_gemini(self, best_mip, best_mask_dir, json_dir):
        client = genai.Client()

        recolor_prompt = (
            f"Create a solid, non-transparent segmentation mask strictly within the existing boundaries "
            f"of the provided image of {self.object_hint}." + RECOLOR_PROMPT
        )

        width, height = best_mip.size

        response_step1 = client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=[recolor_prompt, best_mip],
        )

        generated_image_bytes = None
        for part in response_step1.candidates[0].content.parts:
            if hasattr(part, 'inline_data') and part.inline_data:
                generated_image_bytes = part.inline_data.data

        if not generated_image_bytes:
            raise RuntimeError("Fail to generate an image.")

        generated_img = Image.open(io.BytesIO(generated_image_bytes))
        if generated_img.size != (width, height):
            generated_img = generated_img.resize((width, height), Image.Resampling.NEAREST)

        generated_img.save(best_mask_dir, "PNG")

        analysis_prompt = f"Analyze the provided recolored {self.object_hint} image.\n" + ANALYSIS_PROMPT

        response_step2 = client.models.generate_content(
            model="gemini-3.1-pro-preview",
            contents=[analysis_prompt, Image.open(io.BytesIO(generated_image_bytes))],
        )

        json_text = response_step2.text
        if json_text:
            clean_json = re.sub(r'```json|```', '', json_text).strip()
            try:
                color_data = json.loads(clean_json)
            except json.JSONDecodeError:
                raise RuntimeError(f"JSON parsing failed. Original text:\n{json_text}")
            with open(json_dir, "w") as f:
                json.dump(color_data, f, indent=2, ensure_ascii=False)
        else:
            raise RuntimeError("Gemini returned no JSON text.")

    def run_gemini(self, loader):
        cur_eval_dir = self.workspace

        volume, D_new, H_new, W_new = load_and_resample_volume(
            f'dataset/{self.data_str}.raw',
            (self.data_d_int, self.data_h_int, self.data_w_int),
            self.spacing,
            device=self.device,
            dtype=self.dtype,
        )

        self.data_d_int, self.data_h_int, self.data_w_int = D_new, H_new, W_new
        self.model.data_d_int, self.model.data_h_int, self.model.data_w_int = D_new, H_new, W_new
        D, H, W = D_new, H_new, W_new

        self.model.volume = volume
        self.model.prepare_intensity_stats()
        self.model.set_gradient()

        dataset = loader._data
        pose = dataset.poses[[0]]

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.fp16):
                self.model.update_extra_state()

        if self.mip_view == 'top':
            best_theta, best_phi = 0, 0
        elif self.mip_view == 'side':
            best_theta, best_phi = 90, self.mip_best_phi
        thetas, phis = (best_theta,), (best_phi,)

        mips_dir = f"results/{self.data_name}/mips"
        best_mip_fname = f"{self.data_name}_axis-{self.mip_axis}_theta-{best_theta:+03d}_phi-{best_phi:03d}.png"
        best_mip_dir = os.path.join(mips_dir, best_mip_fname)

        padded_volume, pad0 = pad_volume_for_rotation(volume, theta_max_deg=max(thetas), margin=0.9)

        ray = get_rays(pose, dataset.intrinsics, dataset.H, dataset.W, -1)
        ray_dic = {
            'H': dataset.H,
            'W': dataset.W,
            'rays_o': ray['rays_o'],
            'rays_d': ray['rays_d'],
        }

        mips = generate_axis_conditioned_mips(padded_volume, axis=self.mip_axis, thetas=thetas, phis=phis)

        save_axis_conditioned_mips(mips, out_dir=mips_dir, prefix=self.data_name, pad0=pad0, rot_angle=self.mip_rot)
        best_mip = Image.open(best_mip_dir).convert("RGB")

        best_mask_dir = os.path.join(mips_dir, f"mask_{best_mip_fname}")
        json_dir = os.path.join(mips_dir, f"{self.data_name}_part_color.json")

        if not os.path.exists(best_mask_dir):
            self.generate_mask_with_gemini(best_mip, best_mask_dir, json_dir)

        views_meta = np.load(os.path.join(mips_dir, "views.npy"), allow_pickle=True).item()
        pad0 = views_meta["pad0"]
        views = views_meta["views"]

        with open(json_dir, "r", encoding="utf-8") as f:
            part_color_json = json.load(f)

        view_entries = []
        for v in views:
            mip_name = f"{self.data_name}_axis-{v['axis']}_theta-{v['theta']:+03d}_phi-{v['phi']:03d}.png"
            mask_path = os.path.join(mips_dir, f"mask_{mip_name}")

            if not os.path.exists(mask_path):
                continue

            view_entries.append({
                "axis": v["axis"],
                "theta": v["theta"],
                "phi": v["phi"],
                "bbox": v["bbox"],
                "mask_path": mask_path,
            })

        self.model.num_classes = len(part_color_json) + 1
        self.model.gen_head(self.fusion_mode, self.fixed_beta)
        self.model.to(self.device)

        self.model.set_class_colors(part_color_json)

        colors_np = (self.model.class_colors.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        block_size = 50
        palette = np.zeros((block_size, colors_np.shape[0] * block_size, 3), dtype=np.uint8)
        for i, rgb in enumerate(colors_np):
            palette[:, i * block_size:(i + 1) * block_size, :] = rgb
        Image.fromarray(palette).save(os.path.join(cur_eval_dir, 'class_colors.png'))

        gt_zyx, gt_prob, gt_conf, back_int = build_peak_voxels_multiview_prob_gpu(
            D, H, W,
            padded_volume, pad0,
            view_entries, part_color_json,
            self.mip_rot,
            device=self.device,
        )

        self.model.max_back_int = torch.quantile(back_int, 0.9)

        num_parts = self.model.num_classes - 1

        model = self.model
        steps = 500
        lr = 1e-3

        query_logits_fn = make_query_logits_fn(model)

        params = [
            {"params": model.head.pos_branch.parameters(), "lr": lr},
            {"params": model.head.attr_branch.parameters(), "lr": lr},
            {"params": model.hash_enc.parameters(), "lr": lr * 0.3},
            {"params": model.attr_mlp.parameters(), "lr": lr * 0.5},
        ]
        if model.head.fusion_mode == "learnable":
            params.append({"params": [model.head.alpha], "lr": lr * 2.0})

        optimizer = torch.optim.Adam(params)

        prob_dense, conf_dense = sparse_to_dense_prob_conf(
            gt_zyx, gt_prob, gt_conf, D, H, W, self.model.num_classes, device=self.device,
        )

        for step in range(steps):
            self.model.train()

            loss, loss_sup, loss_prop = propagation_loss(
                query_logits_fn=query_logits_fn,
                volume_intensity=model.volume_intensity,
                grad_mag=model.grad_mag,
                prob=prob_dense,
                conf=conf_dense,
                sigma=self.affinity_sigma,
                lambda_gt_prop=self.lambda_gt_prop,
                use_conf_teacher_for_wprop=self.use_conf_teacher_for_wprop,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.fp16):
                self.model.keep_classes = list(range(num_parts))
                preds, _ = self.render_step(ray_dic)
                save_render(preds, os.path.join(cur_eval_dir, f'step{steps - 1}_all_parts.png'))

                print(f"[{steps - 1:05d}] loss={loss.item():.4f} sup={loss_sup.item():.4f} prop={loss_prop.item():.4f}")

                for i in range(num_parts):
                    self.model.keep_classes = i
                    preds, _ = self.render_step(ray_dic)
                    save_render(preds, os.path.join(cur_eval_dir, f'step{steps - 1}_part{i}.png'))
