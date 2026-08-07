import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, binary_erosion


# ── Displacement helpers ──────────────────────────────────────────────────────

def _squiggle_displacement(img_h, img_w, noise_tex, amplitude_px, uv_scale, time_offset):
    """
    SquiggleVision-style displacement using a tiled noise texture.
    noise_tex: [C, H_n, W_n] float tensor
    Returns dx, dy numpy arrays [H, W] in pixels.
    """
    # Build UV grid for the image, scaled + offset
    gy = torch.linspace(0, uv_scale, img_h)
    gx = torch.linspace(0, uv_scale, img_w)
    grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")

    # Scroll the noise by time_offset (wraps via periodic/border sampling)
    sample_x = ((grid_x + time_offset) % 1.0) * 2 - 1
    sample_y = ((grid_y + time_offset * 0.7) % 1.0) * 2 - 1  # offset Y phase slightly

    sample_grid = torch.stack([sample_x, sample_y], dim=-1).unsqueeze(0)  # [1, H, W, 2]

    sampled = F.grid_sample(
        noise_tex.unsqueeze(0), sample_grid,
        mode="bilinear", padding_mode="border", align_corners=True
    ).squeeze(0)  # [C, H, W]

    # Use channel 0 for X, channel 1 (or 0 again with phase shift) for Y
    nc = sampled.shape[0]
    dx_raw = sampled[0]                    # [H, W]
    dy_raw = sampled[1 % nc]              # [H, W]

    # Remap from [0,1] to [-1,1] then scale to pixels
    dx = (dx_raw - 0.5) * 2 * amplitude_px
    dy = (dy_raw - 0.5) * 2 * amplitude_px

    return dx.numpy(), dy.numpy()


def _fallback_displacement(h, w, amplitude_px, smoothness_px, seed):
    """Gaussian noise fallback when no noise texture is provided."""
    rng = np.random.RandomState(seed)
    dx = gaussian_filter(rng.randn(h, w).astype(np.float32), sigma=smoothness_px)
    dy = gaussian_filter(rng.randn(h, w).astype(np.float32), sigma=smoothness_px)
    dx = dx / (dx.std() + 1e-8) * amplitude_px
    dy = dy / (dy.std() + 1e-8) * amplitude_px
    return dx, dy


def _make_pencil_texture(h, w, strength, scale, seed):
    rng = np.random.RandomState(seed)
    noise = rng.randn(h, w).astype(np.float32)
    noise = gaussian_filter(noise, sigma=(scale * 0.5, scale * 1.5))
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    return 1.0 - noise * strength


def _shape_fill_mask(mask_hw, erode_px, sigma, gamma):
    """Shrink and optionally soften the fill mask edge.

    erode_px: hard inward shrink in pixels — fully opaque, no transparency.
    sigma:    Gaussian feather; the mask is auto-eroded by an extra ~2*sigma
              first so the soft edge falls inside the original boundary
              instead of bleeding alpha outward.
    gamma:    transparency curve of the feathered edge — <1 more opaque,
              >1 more transparent. No effect when sigma is 0.
    """
    m = mask_hw.numpy() > 0.5
    r = int(round(erode_px))
    if sigma > 0:
        r += max(1, int(round(sigma * 2)))
    if r > 0:
        m = binary_erosion(m, iterations=r)
    m = m.astype(np.float32)
    if sigma > 0:
        m = gaussian_filter(m, sigma=sigma)
    out = torch.from_numpy(m).clamp(0, 1)
    if gamma != 1.0:
        out = out.pow(gamma)
    return out


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


def _rgb_to_hsv(rgb):
    """rgb: [..., 3] in [0,1] -> hsv: [..., 3], h in [0,1]"""
    r, g, b = rgb.unbind(-1)
    maxc, _ = rgb.max(-1)
    minc, _ = rgb.min(-1)
    v = maxc
    deltac = maxc - minc
    s = torch.where(maxc > 0, deltac / maxc.clamp(min=1e-8), torch.zeros_like(maxc))

    deltac_safe = deltac.clamp(min=1e-8)
    rc = (maxc - r) / deltac_safe
    gc = (maxc - g) / deltac_safe
    bc = (maxc - b) / deltac_safe

    h = torch.zeros_like(maxc)
    h = torch.where(maxc == r, bc - gc, h)
    h = torch.where(maxc == g, 2.0 + rc - bc, h)
    h = torch.where(maxc == b, 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0
    h = torch.where(deltac == 0, torch.zeros_like(h), h)

    return torch.stack([h, s, v], dim=-1)


def _hsv_to_rgb(hsv):
    """hsv: [..., 3], h in [0,1] -> rgb: [..., 3] in [0,1]"""
    h, s, v = hsv.unbind(-1)
    i = (h * 6.0).floor()
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    i_mod = (i % 6).long()

    r = torch.zeros_like(h)
    g = torch.zeros_like(h)
    b = torch.zeros_like(h)

    masks = [i_mod == k for k in range(6)]
    r = torch.where(masks[0], v, r); g = torch.where(masks[0], t, g); b = torch.where(masks[0], p, b)
    r = torch.where(masks[1], q, r); g = torch.where(masks[1], v, g); b = torch.where(masks[1], p, b)
    r = torch.where(masks[2], p, r); g = torch.where(masks[2], v, g); b = torch.where(masks[2], t, b)
    r = torch.where(masks[3], p, r); g = torch.where(masks[3], q, g); b = torch.where(masks[3], v, b)
    r = torch.where(masks[4], t, r); g = torch.where(masks[4], p, g); b = torch.where(masks[4], v, b)
    r = torch.where(masks[5], v, r); g = torch.where(masks[5], p, g); b = torch.where(masks[5], q, b)

    return torch.stack([r, g, b], dim=-1)


def _match_fill_color(fill_img, src_hex, dst_hex):
    """Shift fill_img's hue/saturation/brightness so src_hex maps onto dst_hex."""
    src_rgb = torch.tensor(_hex_to_rgb(src_hex), dtype=torch.float32)
    dst_rgb = torch.tensor(_hex_to_rgb(dst_hex), dtype=torch.float32)
    if torch.allclose(src_rgb, dst_rgb, atol=1e-3):
        return fill_img

    src_hsv = _rgb_to_hsv(src_rgb)
    dst_hsv = _rgb_to_hsv(dst_rgb)

    h, s, v = _rgb_to_hsv(fill_img).unbind(-1)

    h = (h + (dst_hsv[0] - src_hsv[0])) % 1.0
    s = (s * (dst_hsv[1] / src_hsv[1].clamp(min=1e-3))).clamp(0, 1)
    v = (v * (dst_hsv[2] / src_hsv[2].clamp(min=1e-3))).clamp(0, 1)

    return _hsv_to_rgb(torch.stack([h, s, v], dim=-1))


# ── Node ──────────────────────────────────────────────────────────────────────

class BoilEffectNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images":          ("IMAGE",),
                "amplitude":       ("FLOAT",  {"default": 2.0,  "min": 0.1,  "max": 20.0,  "step": 0.1,
                                               "tooltip": "Max displacement in pixels"}),
                "uv_scale":        ("FLOAT",  {"default": 2.0,  "min": 0.1,  "max": 10.0,  "step": 0.1,
                                               "tooltip": "How many times the noise texture tiles across the image — higher = tighter squiggle"}),
                "time_step":       ("FLOAT",  {"default": 0.05, "min": 0.001,"max": 1.0,   "step": 0.005,
                                               "tooltip": "How far the noise scrolls each hold — smaller = subtler animation"}),
                "step":            ("INT",    {"default": 3,    "min": 1,    "max": 60,
                                               "tooltip": "Frames per hold"}),
                "smoothness":      ("FLOAT",  {"default": 30.0, "min": 1.0,  "max": 200.0, "step": 1.0,
                                               "tooltip": "Fallback smoothness (used when no noise_texture connected)"}),
                "seed":            ("INT",    {"default": 42,   "min": 0,    "max": 9999}),
                "pencil_strength": ("FLOAT",  {"default": 0.25, "min": 0.0,  "max": 1.0,   "step": 0.01}),
                "pencil_scale":    ("FLOAT",  {"default": 1.5,  "min": 0.5,  "max": 10.0,  "step": 0.1}),
                "alpha_grain_strength": ("FLOAT", {"default": 0.6,  "min": 0.0, "max": 1.0, "step": 0.01,
                                                   "tooltip": "Grain that erodes stroke alpha — brush/bristle effect"}),
                "alpha_grain_scale":    ("FLOAT", {"default": 1.0,  "min": 0.3, "max": 8.0, "step": 0.1}),
                "fill_color":      ("STRING",  {"default": "#ED8AB6"}),
                "fill_image_color": ("STRING", {"default": "#ED8AB6",
                                               "tooltip": "Hex color actually present in fill_image — fill_image's hue/saturation/brightness are shifted so this color matches fill_color. Ignored if fill_image is not connected."}),
                "boil_fill":       ("BOOLEAN", {"default": True}),
                "erode":           ("INT",    {"default": 0,    "min": 0,    "max": 50,
                                               "tooltip": "Shrink the fill mask inward by N px with a hard, fully opaque edge (no transparency). Outlines/strokes are unaffected."}),
                "feather":         ("FLOAT",  {"default": 0.0,  "min": 0.0,  "max": 20.0,  "step": 0.5,
                                               "tooltip": "Gaussian blur sigma (px) applied to the fill mask edge before compositing — outlines/strokes are unaffected"}),
                "feather_gamma":   ("FLOAT",  {"default": 1.0,  "min": 0.1,  "max": 5.0,   "step": 0.05,
                                               "tooltip": "Transparency of the feathered edge — <1.0 more opaque, >1.0 more transparent. Only applies when feather > 0."}),
            },
            "optional": {
                "noise_texture": ("IMAGE",  {"tooltip": "Single noise image (e.g. Perlin/Simplex) — R=X displacement, G=Y displacement"}),
                "mask":          ("MASK",   {"tooltip": "Stroke mask from Blender GP Trace"}),
                "fill_mask":     ("MASK",   {"tooltip": "Pipo silhouette mask from SAM3"}),
                "fill_image":    ("IMAGE",  {"tooltip": "SAM3 frames used as fill"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "IMAGE", "IMAGE")
    RETURN_NAMES = ("images", "stroke_mask", "combined_mask", "outline_rgba", "fill_rgba")
    FUNCTION = "apply"
    CATEGORY = "fae/image"

    def apply(self, images, amplitude, uv_scale, time_step, step, smoothness, seed,
              pencil_strength, pencil_scale, alpha_grain_strength, alpha_grain_scale,
              fill_color, fill_image_color, boil_fill, erode, feather, feather_gamma,
              noise_texture=None, mask=None, fill_mask=None, fill_image=None):

        n, h, w, c = images.shape
        fill_rgb    = _hex_to_rgb(fill_color)
        fill_tensor = torch.tensor(fill_rgb, dtype=torch.float32)

        # Recolor fill_image so fill_image_color maps onto fill_color
        fill_image_matched = None
        if fill_image is not None:
            fill_image_matched = _match_fill_color(fill_image, fill_image_color, fill_color)

        # Prepare noise texture tensor [C, H_n, W_n]
        noise_tex = None
        if noise_texture is not None:
            # Use first frame if a batch was passed
            nt = noise_texture[0]        # [H_n, W_n, C_n]
            noise_tex = nt.permute(2, 0, 1).float()  # [C, H_n, W_n]

        out_images, out_masks, out_combined = [], [], []
        out_outline_rgba, out_fill_rgba = [], []
        pencil_cache = {}
        alpha_cache  = {}

        for i in range(n):
            hold      = i // step
            hold_seed = seed * 10000 + hold

            # ── Displacement ──────────────────────────────────────────────────
            if noise_tex is not None:
                time_offset = hold * time_step
                dx, dy = _squiggle_displacement(h, w, noise_tex, amplitude, uv_scale, time_offset)
            else:
                dx, dy = _fallback_displacement(h, w, amplitude, smoothness, hold_seed)

            # ── Warp outline image ────────────────────────────────────────────
            warped = _warp(images[i].permute(2, 0, 1), dx, dy).permute(1, 2, 0)

            # ── Warp stroke mask ──────────────────────────────────────────────
            if mask is not None and i < mask.shape[0]:
                stroke_mask = _warp(mask[i].unsqueeze(0), dx, dy).squeeze(0)
            else:
                stroke_mask = warped.mean(dim=-1).clamp(0, 1)

            # ── Alpha grain ───────────────────────────────────────────────────
            if alpha_grain_strength > 0:
                if hold not in alpha_cache:
                    rng = np.random.RandomState(hold_seed + 55555)
                    an = rng.randn(h, w).astype(np.float32)
                    an = gaussian_filter(an, sigma=alpha_grain_scale)
                    an = (an - an.min()) / (an.max() - an.min() + 1e-8)
                    alpha_cache[hold] = an
                alpha_noise  = torch.from_numpy(alpha_cache[hold])
                edge_weight  = 1.0 - stroke_mask.clamp(0, 1) ** 2
                stroke_mask  = (stroke_mask - alpha_noise * alpha_grain_strength * edge_weight).clamp(0, 1)

            # ── Pencil texture (applied to stroke color, not mask) ────────────
            stroke_color_rgb = warped  # rendered stroke color from Blender
            if pencil_strength > 0:
                if hold not in pencil_cache:
                    pencil_cache[hold] = _make_pencil_texture(h, w, pencil_strength, pencil_scale, hold_seed + 99999)
                texture = torch.from_numpy(pencil_cache[hold]).unsqueeze(-1)
                stroke_color_rgb = warped * texture

            sm = stroke_mask.clamp(0, 1).unsqueeze(-1)  # [H, W, 1]

            # ── Premultiplied outline: stroke color × clean mask ──────────────
            outline_premult = stroke_color_rgb * sm  # sharp edges from mask

            # ── Fill layer ────────────────────────────────────────────────────
            if fill_mask is not None and i < fill_mask.shape[0]:
                fm = fill_mask[i]
                if fill_image_matched is not None and i < fill_image_matched.shape[0]:
                    fi = fill_image_matched[i]
                    if boil_fill:
                        fi = _warp(fi.permute(2, 0, 1), dx, dy).permute(1, 2, 0)
                    fill_layer = fi
                else:
                    fill_layer = fill_tensor.view(1, 1, 3).expand(h, w, 3)
                if boil_fill:
                    fm = _warp(fm.unsqueeze(0), dx, dy).squeeze(0)
                fm = (fm.clamp(0, 1) > 0.5).float()
                if erode > 0 or feather > 0:
                    fm = _shape_fill_mask(fm, erode, feather, feather_gamma)
                fm = fm.unsqueeze(-1)
                # Premultiplied fill
                fill_premult = fill_layer * fm
                # Composite outline over fill (premultiplied A-over-B)
                result = outline_premult + fill_premult * (1.0 - sm)
                fill_rgba = torch.cat([fill_layer.clamp(0, 1), fm.clamp(0, 1)], dim=-1)
            else:
                result = outline_premult
                fill_rgba = torch.zeros((h, w, 4), dtype=torch.float32)

            # Separate layers, straight (unpremultiplied) alpha
            outline_rgba = torch.cat([stroke_color_rgb.clamp(0, 1), sm.clamp(0, 1)], dim=-1)

            # ── Combined mask ─────────────────────────────────────────────────
            if fill_mask is not None and i < fill_mask.shape[0]:
                wfm = _warp(fill_mask[i].unsqueeze(0), dx, dy).squeeze(0) if boil_fill else fill_mask[i]
                wfm = (wfm.clamp(0, 1) > 0.5).float()
                if erode > 0 or feather > 0:
                    wfm = _shape_fill_mask(wfm, erode, feather, feather_gamma)
                combined = (wfm + stroke_mask.clamp(0, 1)).clamp(0, 1)
            else:
                combined = stroke_mask.clamp(0, 1)

            out_images.append(result.clamp(0, 1))
            out_masks.append(stroke_mask.clamp(0, 1))
            out_combined.append(combined)
            out_outline_rgba.append(outline_rgba)
            out_fill_rgba.append(fill_rgba)

        return (torch.stack(out_images), torch.stack(out_masks), torch.stack(out_combined),
                torch.stack(out_outline_rgba), torch.stack(out_fill_rgba))
