import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter


def _make_displacement(h, w, amplitude_px, smoothness_px, seed):
    rng = np.random.RandomState(seed)
    dx = rng.randn(h, w).astype(np.float32)
    dy = rng.randn(h, w).astype(np.float32)
    dx = gaussian_filter(dx, sigma=smoothness_px)
    dy = gaussian_filter(dy, sigma=smoothness_px)
    dx = dx / (dx.std() + 1e-8) * amplitude_px
    dy = dy / (dy.std() + 1e-8) * amplitude_px
    return dx, dy


def _make_pencil_texture(h, w, strength, scale, seed):
    rng = np.random.RandomState(seed)
    noise = rng.randn(h, w).astype(np.float32)
    noise = gaussian_filter(noise, sigma=(scale * 0.5, scale * 1.5))
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    return 1.0 - noise * strength


def _warp(tensor_chw, dx, dy):
    c, h, w = tensor_chw.shape
    t = tensor_chw.unsqueeze(0)
    gy = torch.linspace(-1, 1, h, dtype=torch.float32)
    gx = torch.linspace(-1, 1, w, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
    grid_x = grid_x + torch.from_numpy(dx) * 2.0 / w
    grid_y = grid_y + torch.from_numpy(dy) * 2.0 / h
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    return F.grid_sample(t, grid, mode="bilinear", padding_mode="border", align_corners=True).squeeze(0)


def _hex_to_rgb(hex_str):
    h = hex_str.strip().lstrip("#").ljust(6, "0")
    return (int(h[0:2], 16) / 255.0,
            int(h[2:4], 16) / 255.0,
            int(h[4:6], 16) / 255.0)


class BoilEffectNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images":          ("IMAGE",),
                "amplitude":       ("FLOAT", {"default": 2.0,  "min": 0.1,  "max": 20.0,  "step": 0.1}),
                "smoothness":      ("FLOAT", {"default": 30.0, "min": 1.0,  "max": 200.0, "step": 1.0}),
                "step":            ("INT",   {"default": 3,    "min": 1,    "max": 60}),
                "seed":            ("INT",   {"default": 42,   "min": 0,    "max": 9999}),
                "pencil_strength": ("FLOAT", {"default": 0.25, "min": 0.0,  "max": 1.0,   "step": 0.01}),
                "pencil_scale":    ("FLOAT", {"default": 1.5,  "min": 0.5,  "max": 10.0,  "step": 0.1}),
                "fill_color":      ("STRING",  {"default": "#ED8AB6", "tooltip": "Fallback flat fill color (used when no fill_image connected)"}),
                "boil_fill":       ("BOOLEAN", {"default": True, "tooltip": "Apply same boil displacement to fill layer"}),
            },
            "optional": {
                "mask":       ("MASK",  {"tooltip": "Stroke mask from Blender GP Trace"}),
                "fill_mask":  ("MASK",  {"tooltip": "Pipo silhouette mask from SAM3"}),
                "fill_image": ("IMAGE", {"tooltip": "SAM3 video frames — used as fill instead of flat color"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("images", "stroke_mask", "combined_mask")
    FUNCTION = "apply"
    CATEGORY = "fae/image"

    def apply(self, images, amplitude, smoothness, step, seed,
              pencil_strength, pencil_scale, fill_color, boil_fill,
              mask=None, fill_mask=None, fill_image=None):

        n, h, w, c = images.shape
        fill_rgb = _hex_to_rgb(fill_color)
        fill_tensor = torch.tensor(fill_rgb, dtype=torch.float32)  # [3]

        out_images, out_masks, out_combined = [], [], []
        pencil_cache = {}

        for i in range(n):
            hold      = i // step
            hold_seed = seed * 10000 + hold

            dx, dy = _make_displacement(h, w, amplitude, smoothness, hold_seed)

            # ── Warp outline image ────────────────────────────────────────────
            frame  = images[i].permute(2, 0, 1)
            warped = _warp(frame, dx, dy).permute(1, 2, 0)  # [H, W, C]

            # ── Warp stroke mask ──────────────────────────────────────────────
            if mask is not None and i < mask.shape[0]:
                stroke_mask = _warp(mask[i].unsqueeze(0), dx, dy).squeeze(0)
            else:
                stroke_mask = warped.mean(dim=-1).clamp(0, 1)

            # ── Pencil texture on strokes ─────────────────────────────────────
            if pencil_strength > 0:
                if hold not in pencil_cache:
                    pencil_cache[hold] = _make_pencil_texture(
                        h, w, pencil_strength, pencil_scale, hold_seed + 99999
                    )
                texture = torch.from_numpy(pencil_cache[hold])
                sm = stroke_mask.clamp(0, 1).unsqueeze(-1)
                warped = warped * (1.0 - sm + sm * texture.unsqueeze(-1))

            # ── Fill layer ────────────────────────────────────────────────────
            if fill_mask is not None and i < fill_mask.shape[0]:
                fm = fill_mask[i]  # [H, W]

                # Build the fill pixels — SAM3 image if connected, else flat color
                if fill_image is not None and i < fill_image.shape[0]:
                    fi = fill_image[i]  # [H, W, C] already in same space
                    if boil_fill:
                        fi = _warp(fi.permute(2, 0, 1), dx, dy).permute(1, 2, 0)
                    fill_layer = fi
                else:
                    fill_layer = fill_tensor.view(1, 1, 3).expand(h, w, 3)

                if boil_fill:
                    fm = _warp(fm.unsqueeze(0), dx, dy).squeeze(0)
                fm = fm.clamp(0, 1).unsqueeze(-1)  # [H, W, 1]

                # Composite: fill behind outlines
                result = fill_layer * fm + warped * (1.0 - fm)
                # Overlay outlines on top
                result = result * (1.0 - stroke_mask.unsqueeze(-1)) + warped * stroke_mask.unsqueeze(-1)
            else:
                result = warped

            # Combined mask: union of fill + stroke
            if fill_mask is not None and i < fill_mask.shape[0]:
                warped_fm = _warp(fill_mask[i].unsqueeze(0), dx, dy).squeeze(0) if boil_fill else fill_mask[i]
                combined = (warped_fm.clamp(0, 1) + stroke_mask.clamp(0, 1)).clamp(0, 1)
            else:
                combined = stroke_mask.clamp(0, 1)

            out_images.append(result.clamp(0, 1))
            out_masks.append(stroke_mask.clamp(0, 1))
            out_combined.append(combined)

        return (torch.stack(out_images), torch.stack(out_masks), torch.stack(out_combined))
