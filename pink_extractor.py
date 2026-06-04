import numpy as np
import torch
from scipy.ndimage import binary_dilation, sobel


def _rgb_to_hsv(img):
    """img: [H, W, 3] float32 0-1 → h, s, v each [H, W] float32 in [0,1]."""
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc
    eps = 1e-10

    v = maxc
    s = np.where(maxc > eps, delta / maxc, 0.0)

    h = np.zeros_like(r)
    mr = (maxc == r) & (delta > eps)
    mg = (maxc == g) & ~mr & (delta > eps)
    mb = ~mr & ~mg & (delta > eps)
    h = np.where(mr, (g - b) / (delta + eps), h)
    h = np.where(mg, 2.0 + (b - r) / (delta + eps), h)
    h = np.where(mb, 4.0 + (r - g) / (delta + eps), h)
    h = (h / 6.0) % 1.0

    return h, s, v


def _detect_mask(img, hue_center, hue_tol, sat_min, sat_max, val_min, val_max, erosion):
    h, s, v = _rgb_to_hsv(img)

    # Wrap-aware hue distance (matches GLSL shader)
    hd = np.abs(h - hue_center)
    hd = np.minimum(hd, 1.0 - hd)

    mask = (hd <= hue_tol) & (s >= sat_min) & (s <= sat_max) & (v >= val_min) & (v <= val_max)

    # Erosion: keep pink pixel only if it has >= erosion pink 4-neighbors
    if erosion > 0:
        pf = mask.astype(np.float32)
        neighbors = (
            np.roll(pf, 1, axis=1) + np.roll(pf, -1, axis=1) +
            np.roll(pf, 1, axis=0) + np.roll(pf, -1, axis=0)
        )
        mask = mask & (neighbors >= float(erosion))

    return mask


def _build_outline(mask, outline_width, sharpness, outline_color_rgb):
    """
    Translates the GLSL pass-2 shader to numpy.
    Returns [H, W, 3] float32: background (white) everywhere,
    with outline_color drawn at mask boundaries.
    """
    H, W = mask.shape
    mf = mask.astype(np.float32)

    # Sobel gradient on the mask
    gx = sobel(mf, axis=1)
    gy = sobel(mf, axis=0)
    sobel_mag = np.clip(np.sqrt(gx ** 2 + gy ** 2) / 4.0 * sharpness, 0.0, 1.0)

    # Circle dilation — dilate mask by outline_width pixels, then subtract mask
    radius = max(1, int(round(outline_width)))
    dilated = binary_dilation(mask, iterations=radius)
    near_boundary = dilated & ~mask  # pixels outside mask but within outline_width of it

    # Inner circle at half width (for solid fill close to edge)
    inner_radius = max(1, radius // 2)
    inner_dilated = binary_dilation(mask, iterations=inner_radius)
    inner_near = inner_dilated & ~mask

    # Combine: inner region gets full opacity, outer region fades with Sobel
    circle_str = np.where(near_boundary,
                          np.where(inner_near, 1.0, np.minimum(1.0, sobel_mag * 3.0)),
                          0.0)
    outline_alpha = np.clip(np.maximum(circle_str, np.where(near_boundary, sobel_mag, 0.0)), 0.0, 1.0)

    # Composite: white background, blend outline color at boundary
    oc = np.array(outline_color_rgb, dtype=np.float32)
    result = np.ones((H, W, 3), dtype=np.float32)  # white background
    result[near_boundary] = (
        result[near_boundary] * (1.0 - outline_alpha[near_boundary, None]) +
        oc * outline_alpha[near_boundary, None]
    )

    return result


def _hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip("#")
    return (int(hex_str[0:2], 16) / 255.0,
            int(hex_str[2:4], 16) / 255.0,
            int(hex_str[4:6], 16) / 255.0)


class PinkExtractorNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "hue_center":    ("FLOAT", {"default": 0.93, "min": 0.0,  "max": 1.0,  "step": 0.01}),
                "hue_tolerance": ("FLOAT", {"default": 0.08, "min": 0.01, "max": 0.25, "step": 0.005}),
                "sat_min":       ("FLOAT", {"default": 0.30, "min": 0.0,  "max": 1.0,  "step": 0.01}),
                "sat_max":       ("FLOAT", {"default": 0.70, "min": 0.0,  "max": 1.0,  "step": 0.01}),
                "val_min":       ("FLOAT", {"default": 0.65, "min": 0.0,  "max": 1.0,  "step": 0.01}),
                "val_max":       ("FLOAT", {"default": 1.00, "min": 0.0,  "max": 1.0,  "step": 0.01}),
                "erosion":       ("INT",   {"default": 1,    "min": 0,    "max": 4,    "step": 1}),
                "outline_width": ("FLOAT", {"default": 3.0,  "min": 1.0,  "max": 12.0, "step": 0.5}),
                "sharpness":     ("FLOAT", {"default": 1.0,  "min": 0.0,  "max": 2.0,  "step": 0.05}),
                "outline_color": ("STRING", {"default": "#1a1a2e"}),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("mask", "outline")
    FUNCTION = "extract"
    CATEGORY = "fae/image"

    def extract(self, images, hue_center, hue_tolerance, sat_min, sat_max,
                val_min, val_max, erosion, outline_width, sharpness, outline_color):

        outline_rgb = _hex_to_rgb(outline_color)
        imgs_np = images.cpu().numpy()  # [N, H, W, C]

        masks, outlines = [], []
        for img in imgs_np:
            mask = _detect_mask(img[..., :3], hue_center, hue_tolerance,
                                sat_min, sat_max, val_min, val_max, erosion)
            outline = _build_outline(mask, outline_width, sharpness, outline_rgb)
            masks.append(torch.from_numpy(mask.astype(np.float32)))
            outlines.append(torch.from_numpy(outline))

        return (torch.stack(masks), torch.stack(outlines))
