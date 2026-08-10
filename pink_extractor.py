import numpy as np
import torch
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    label,
    sobel,
)


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


def _despeckle(mask, min_area):
    """Drop mask islands smaller than min_area pixels.

    Thresholding picks up isolated stray pixels, which are invisible in the mask
    itself but not downstream: the outline pass rings every connected component,
    so a single stray pixel dilates into a solid box of ink many times its size.
    Erosion can't stand in for this — it thins the whole silhouette rather than
    removing islands.
    """
    if min_area <= 1 or not mask.any():
        return mask
    lbl, n = label(mask)
    if n < 2:
        return mask
    areas = np.bincount(lbl.ravel())
    areas[0] = 0  # background is never kept on its own account
    return areas[lbl] >= min_area


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

    inner_radius = radius // 2
    if inner_radius > 0:
        inner_dilated = binary_dilation(mask, iterations=inner_radius)
        inner_near = inner_dilated & ~mask
    else:
        inner_near = np.zeros(mask.shape, dtype=bool)

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


def _sandwiched_gaps(mask, radius):
    """Non-pink pixels with pink on both opposing sides within radius — inner ink lines."""
    H, W = mask.shape
    mf = mask.astype(np.int32)

    # Padded cumsums so boundary arithmetic is clean
    cs_h = np.concatenate([np.zeros((H, 1), dtype=np.int32), np.cumsum(mf, axis=1)], axis=1)
    cs_v = np.concatenate([np.zeros((1, W), dtype=np.int32), np.cumsum(mf, axis=0)], axis=0)

    j = np.arange(W)
    pink_right = (cs_h[:, np.minimum(j + radius + 1, W)] - cs_h[:, j + 1]) > 0
    pink_left  = (cs_h[:, j] - cs_h[:, np.maximum(j - radius, 0)]) > 0

    i = np.arange(H)
    pink_down = (cs_v[np.minimum(i + radius + 1, H), :] - cs_v[i + 1, :]) > 0
    pink_up   = (cs_v[i, :] - cs_v[np.maximum(i - radius, 0), :]) > 0

    return ~mask & ((pink_left & pink_right) | (pink_up & pink_down))


def _build_outline_zones(mask, outline_width, inner_color_rgb, outer_color_rgb):
    """
    Inner zone: non-pink pixels sandwiched between pink on opposing sides (limb overlap lines).
    Outer zone: dilation ring outside the silhouette edge.
    """
    H, W = mask.shape
    radius = max(1, int(round(outline_width)))

    inner_near = _sandwiched_gaps(mask, radius)
    outer_near = binary_dilation(mask, iterations=radius) & ~mask & ~inner_near

    ic = np.array(inner_color_rgb, dtype=np.float32)
    oc = np.array(outer_color_rgb, dtype=np.float32)

    inner_img = np.ones((H, W, 3), dtype=np.float32)
    inner_img[inner_near] = ic

    outer_img = np.ones((H, W, 3), dtype=np.float32)
    outer_img[outer_near] = oc

    comp_img = np.ones((H, W, 3), dtype=np.float32)
    comp_img[outer_near] = oc
    comp_img[inner_near] = ic

    both = inner_near | outer_near

    return (inner_near.astype(np.float32), outer_near.astype(np.float32),
            both.astype(np.float32), inner_img, outer_img, comp_img)


def _hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip("#")
    return (int(hex_str[0:2], 16) / 255.0,
            int(hex_str[2:4], 16) / 255.0,
            int(hex_str[4:6], 16) / 255.0)


def _hex_to_hsv(hex_str):
    rgb = np.array(_hex_to_rgb(hex_str), dtype=np.float32).reshape(1, 1, 3)
    h, s, v = _rgb_to_hsv(rgb)
    return float(h[0, 0]), float(s[0, 0]), float(v[0, 0])


def _detect_mask_around_color(img, target_hex, hue_tol, sat_tol, val_tol, erosion):
    """Like _detect_mask, but centered on a target color's own HSV instead of
    hand-tuned per-channel ranges. Tolerances are ± around the target."""
    h0, s0, v0 = _hex_to_hsv(target_hex)
    h, s, v = _rgb_to_hsv(img)

    hd = np.abs(h - h0)
    hd = np.minimum(hd, 1.0 - hd)

    mask = (hd <= hue_tol) & (np.abs(s - s0) <= sat_tol) & (np.abs(v - v0) <= val_tol)

    if erosion > 0:
        pf = mask.astype(np.float32)
        neighbors = (
            np.roll(pf, 1, axis=1) + np.roll(pf, -1, axis=1) +
            np.roll(pf, 1, axis=0) + np.roll(pf, -1, axis=0)
        )
        mask = mask & (neighbors >= float(erosion))

    return mask


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
                "despeckle":     ("INT",   {"default": 16,   "min": 0,    "max": 512,  "step": 1,
                                            "tooltip": "Discard detected islands smaller than this many pixels. Stray pixels are invisible in the mask but the outline rings every island, turning one pixel into a box of ink. 0 = off."}),
                "outline_width": ("FLOAT", {"default": 3.0,  "min": 1.0,  "max": 50.0, "step": 0.5,
                                            "tooltip": "Outline width when Pipo is largest in the clip (closest to camera)"}),
                "sharpness":     ("FLOAT", {"default": 1.0,  "min": 0.0,  "max": 2.0,  "step": 0.05}),
                "outline_color": ("STRING", {"default": "#1a1a2e"}),
                "dynamic_outline": ("BOOLEAN", {"default": False,
                                                "tooltip": "Scale outline width by Pipo's mask size — thinner when farther from camera"}),
                "min_scale":       ("FLOAT", {"default": 0.3, "min": 0.05, "max": 1.0, "step": 0.05,
                                              "tooltip": "Smallest fraction of outline_width to use when Pipo is at its smallest"}),
                "smoothing":       ("FLOAT", {"default": 0.3, "min": 0.0,  "max": 0.95, "step": 0.05,
                                              "tooltip": "Temporal smoothing on the size-based scale (0 = none, higher = smoother)"}),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("mask", "outline", "inner_mask", "inner_lines")
    FUNCTION = "extract"
    CATEGORY = "fae/image"

    def extract(self, images, hue_center, hue_tolerance, sat_min, sat_max,
                val_min, val_max, erosion, outline_width, sharpness, outline_color,
                dynamic_outline, min_scale, smoothing, despeckle=16):

        outline_rgb = _hex_to_rgb(outline_color)
        imgs_np = images.cpu().numpy()  # [N, H, W, C]

        masks_np = [_despeckle(_detect_mask(img[..., :3], hue_center, hue_tolerance,
                                            sat_min, sat_max, val_min, val_max, erosion),
                               despeckle)
                    for img in imgs_np]

        widths = np.full(len(masks_np), outline_width, dtype=np.float32)
        if dynamic_outline:
            areas = np.array([m.sum() for m in masks_np], dtype=np.float32)
            max_area = areas.max()
            if max_area > 0:
                scale = np.clip(np.sqrt(areas / max_area), min_scale, 1.0)
                for i in range(1, len(scale)):
                    scale[i] = smoothing * scale[i - 1] + (1.0 - smoothing) * scale[i]
                widths = outline_width * scale

        masks, outlines, inner_masks, inner_lines = [], [], [], []
        oc = np.array(outline_rgb, dtype=np.float32)
        for mask, width in zip(masks_np, widths):
            outline = _build_outline(mask, float(width), sharpness, outline_rgb)
            masks.append(torch.from_numpy(mask.astype(np.float32)))
            outlines.append(torch.from_numpy(outline))

            # Interior linework only (no-outer-outline Pipo): non-pink pixels
            # sandwiched between pink — face, mouth, limb separation lines.
            inner = _sandwiched_gaps(mask, max(1, int(round(width))))
            inner_masks.append(torch.from_numpy(inner.astype(np.float32)))
            img = np.ones((*mask.shape, 3), dtype=np.float32)
            img[inner] = oc
            inner_lines.append(torch.from_numpy(img))

        return (torch.stack(masks), torch.stack(outlines),
                torch.stack(inner_masks), torch.stack(inner_lines))


class ColorExtractorNode:
    """PinkExtractor twin for arbitrary target colors (default: the #4D0F28
    ink maroon). The exact shade shifts slightly between projects but is
    constant within one, so detection is centered on a single hex color with
    tight ± tolerances instead of the pink node's wide hand-tuned ranges."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images":        ("IMAGE",),
                "target_color":  ("STRING", {"default": "#4D0F28",
                                             "tooltip": "The project's exact color — sample it from a real frame"}),
                "hue_tolerance": ("FLOAT", {"default": 0.03, "min": 0.005, "max": 0.25, "step": 0.005}),
                "sat_tolerance": ("FLOAT", {"default": 0.15, "min": 0.01,  "max": 0.5,  "step": 0.01}),
                "val_tolerance": ("FLOAT", {"default": 0.15, "min": 0.01,  "max": 0.5,  "step": 0.01}),
                "erosion":       ("INT",   {"default": 1,    "min": 0,     "max": 4,    "step": 1}),
                "despeckle":     ("INT",   {"default": 16,   "min": 0,     "max": 512,  "step": 1,
                                            "tooltip": "Discard detected islands smaller than this many pixels. Stray pixels are invisible in the mask but the outline rings every island, turning one pixel into a box of ink. 0 = off."}),
                "outline_width": ("FLOAT", {"default": 3.0,  "min": 1.0,   "max": 50.0, "step": 0.5}),
                "sharpness":     ("FLOAT", {"default": 1.0,  "min": 0.0,   "max": 2.0,  "step": 0.05}),
                "outline_color": ("STRING", {"default": "#1a1a2e"}),
                "dynamic_outline": ("BOOLEAN", {"default": False}),
                "min_scale":       ("FLOAT", {"default": 0.3, "min": 0.05, "max": 1.0,  "step": 0.05}),
                "smoothing":       ("FLOAT", {"default": 0.3, "min": 0.0,  "max": 0.95, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("mask", "outline", "inner_mask", "inner_lines")
    FUNCTION = "extract"
    CATEGORY = "fae/image"

    def extract(self, images, target_color, hue_tolerance, sat_tolerance, val_tolerance,
                erosion, outline_width, sharpness, outline_color,
                dynamic_outline, min_scale, smoothing, despeckle=16):

        outline_rgb = _hex_to_rgb(outline_color)
        imgs_np = images.cpu().numpy()

        masks_np = [_despeckle(_detect_mask_around_color(img[..., :3], target_color,
                                                         hue_tolerance, sat_tolerance,
                                                         val_tolerance, erosion),
                               despeckle)
                    for img in imgs_np]

        widths = np.full(len(masks_np), outline_width, dtype=np.float32)
        if dynamic_outline:
            areas = np.array([m.sum() for m in masks_np], dtype=np.float32)
            max_area = areas.max()
            if max_area > 0:
                scale = np.clip(np.sqrt(areas / max_area), min_scale, 1.0)
                for i in range(1, len(scale)):
                    scale[i] = smoothing * scale[i - 1] + (1.0 - smoothing) * scale[i]
                widths = outline_width * scale

        masks, outlines, inner_masks, inner_lines = [], [], [], []
        oc = np.array(outline_rgb, dtype=np.float32)
        for mask, width in zip(masks_np, widths):
            outline = _build_outline(mask, float(width), sharpness, outline_rgb)
            masks.append(torch.from_numpy(mask.astype(np.float32)))
            outlines.append(torch.from_numpy(outline))

            inner = _sandwiched_gaps(mask, max(1, int(round(width))))
            inner_masks.append(torch.from_numpy(inner.astype(np.float32)))
            img = np.ones((*mask.shape, 3), dtype=np.float32)
            img[inner] = oc
            inner_lines.append(torch.from_numpy(img))

        return (torch.stack(masks), torch.stack(outlines),
                torch.stack(inner_masks), torch.stack(inner_lines))


class PinkOutlineZonesNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images":          ("IMAGE",),
                "hue_center":      ("FLOAT", {"default": 0.93, "min": 0.0,  "max": 1.0,  "step": 0.01}),
                "hue_tolerance":   ("FLOAT", {"default": 0.08, "min": 0.01, "max": 0.25, "step": 0.005}),
                "sat_min":         ("FLOAT", {"default": 0.30, "min": 0.0,  "max": 1.0,  "step": 0.01}),
                "sat_max":         ("FLOAT", {"default": 0.70, "min": 0.0,  "max": 1.0,  "step": 0.01}),
                "val_min":         ("FLOAT", {"default": 0.65, "min": 0.0,  "max": 1.0,  "step": 0.01}),
                "val_max":         ("FLOAT", {"default": 1.00, "min": 0.0,  "max": 1.0,  "step": 0.01}),
                "erosion":         ("INT",   {"default": 1,    "min": 0,    "max": 4,    "step": 1}),
                "outline_width":   ("FLOAT", {"default": 3.0,  "min": 1.0,  "max": 50.0, "step": 0.5}),
                "inner_color":     ("STRING", {"default": "#1a1a2e"}),
                "outer_color":     ("STRING", {"default": "#4a4a6e"}),
                "dynamic_outline": ("BOOLEAN", {"default": False}),
                "min_scale":       ("FLOAT", {"default": 0.3, "min": 0.05, "max": 1.0, "step": 0.05}),
                "smoothing":       ("FLOAT", {"default": 0.3, "min": 0.0,  "max": 0.95, "step": 0.05}),
            }
        }

    RETURN_TYPES  = ("MASK", "MASK", "MASK", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES  = ("inner_mask", "outer_mask", "both_mask", "inner_image", "outer_image", "composited")
    FUNCTION = "extract"
    CATEGORY = "fae/image"

    def extract(self, images, hue_center, hue_tolerance, sat_min, sat_max,
                val_min, val_max, erosion, outline_width,
                inner_color, outer_color,
                dynamic_outline, min_scale, smoothing):

        inner_rgb = _hex_to_rgb(inner_color)
        outer_rgb = _hex_to_rgb(outer_color)
        imgs_np = images.cpu().numpy()

        masks_np = [_detect_mask(img[..., :3], hue_center, hue_tolerance,
                                  sat_min, sat_max, val_min, val_max, erosion)
                    for img in imgs_np]

        widths = np.full(len(masks_np), outline_width, dtype=np.float32)
        if dynamic_outline:
            areas = np.array([m.sum() for m in masks_np], dtype=np.float32)
            max_area = areas.max()
            if max_area > 0:
                scale = np.clip(np.sqrt(areas / max_area), min_scale, 1.0)
                for i in range(1, len(scale)):
                    scale[i] = smoothing * scale[i - 1] + (1.0 - smoothing) * scale[i]
                widths = outline_width * scale

        inner_masks, outer_masks, both_masks = [], [], []
        inner_imgs, outer_imgs, comp_imgs = [], [], []

        for mask, width in zip(masks_np, widths):
            im, om, bm, ii, oi, ci = _build_outline_zones(mask, float(width), inner_rgb, outer_rgb)
            inner_masks.append(torch.from_numpy(im))
            outer_masks.append(torch.from_numpy(om))
            both_masks.append(torch.from_numpy(bm))
            inner_imgs.append(torch.from_numpy(ii))
            outer_imgs.append(torch.from_numpy(oi))
            comp_imgs.append(torch.from_numpy(ci))

        return (
            torch.stack(inner_masks),
            torch.stack(outer_masks),
            torch.stack(both_masks),
            torch.stack(inner_imgs),
            torch.stack(outer_imgs),
            torch.stack(comp_imgs),
        )


class MaskOuterRingNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask":  ("MASK",),
                "width": ("INT",    {"default": 3,       "min": 1,   "max": 50,  "step": 1}),
                "color": ("STRING", {"default": "#1a1a2e"}),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("ring_mask", "ring_image")
    FUNCTION = "process"
    CATEGORY = "fae/image"

    def process(self, mask, width, color):
        color_rgb = _hex_to_rgb(color)
        c = np.array(color_rgb, dtype=np.float32)
        masks_np = mask.cpu().numpy()  # [N, H, W]

        ring_masks, ring_imgs = [], []
        for m in masks_np:
            bool_mask = m > 0.5
            eroded = binary_erosion(bool_mask, iterations=width)
            ring = bool_mask & ~eroded  # border pixels of the mask

            ring_masks.append(torch.from_numpy(ring.astype(np.float32)))

            H, W = m.shape
            img = np.ones((H, W, 3), dtype=np.float32)
            img[ring] = c
            ring_imgs.append(torch.from_numpy(img))

        return (torch.stack(ring_masks), torch.stack(ring_imgs))


class MaskCenteredStrokeNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask":  ("MASK",),
                "width": ("INT",    {"default": 6,       "min": 2,   "max": 100, "step": 2,
                                     "tooltip": "Total stroke width in pixels — half inside, half outside the mask edge"}),
                "color": ("STRING", {"default": "#1a1a2e"}),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("stroke_mask", "stroke_image")
    FUNCTION = "process"
    CATEGORY = "fae/image"

    def process(self, mask, width, color):
        color_rgb = _hex_to_rgb(color)
        c = np.array(color_rgb, dtype=np.float32)
        masks_np = mask.cpu().numpy()  # [N, H, W]
        half = max(1, width // 2)

        stroke_masks, stroke_imgs = [], []
        for m in masks_np:
            bool_mask = m > 0.5
            inner = bool_mask & ~binary_erosion(bool_mask, iterations=half)
            outer = binary_dilation(bool_mask, iterations=half) & ~bool_mask
            stroke = inner | outer

            stroke_masks.append(torch.from_numpy(stroke.astype(np.float32)))

            H, W = m.shape
            img = np.ones((H, W, 3), dtype=np.float32)
            img[stroke] = c
            stroke_imgs.append(torch.from_numpy(img))

        return (torch.stack(stroke_masks), torch.stack(stroke_imgs))


class MaskStrokeNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask":         ("MASK",),
                "inward_px":    ("INT", {"default": 2, "min": 0, "max": 100, "step": 1,
                                         "tooltip": "Pixels to draw inside the mask edge"}),
                "outward_px":   ("INT", {"default": 2, "min": 0, "max": 100, "step": 1,
                                         "tooltip": "Pixels to draw outside the mask edge"}),
                "color":        ("STRING", {"default": "#1a1a2e"}),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("stroke_mask", "stroke_image")
    FUNCTION = "process"
    CATEGORY = "fae/image"

    def process(self, mask, inward_px, outward_px, color):
        color_rgb = _hex_to_rgb(color)
        c = np.array(color_rgb, dtype=np.float32)
        masks_np = mask.cpu().numpy()

        stroke_masks, stroke_imgs = [], []
        for m in masks_np:
            bool_mask = m > 0.5

            inner = bool_mask & ~binary_erosion(bool_mask, iterations=inward_px) if inward_px > 0 else np.zeros_like(bool_mask)
            outer = binary_dilation(bool_mask, iterations=outward_px) & ~bool_mask if outward_px > 0 else np.zeros_like(bool_mask)
            stroke = inner | outer

            stroke_masks.append(torch.from_numpy(stroke.astype(np.float32)))

            H, W = m.shape
            img = np.ones((H, W, 3), dtype=np.float32)
            img[stroke] = c
            stroke_imgs.append(torch.from_numpy(img))

        return (torch.stack(stroke_masks), torch.stack(stroke_imgs))


class MaskDilateColorNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":       ("IMAGE",),
                "mask":        ("MASK",),
                "dilation_px": ("INT", {"default": 4, "min": 1, "max": 200, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "dilated_mask")
    FUNCTION = "process"
    CATEGORY = "fae/image"

    def process(self, image, mask, dilation_px):
        imgs_np = image.cpu().numpy()   # [N, H, W, C]
        masks_np = mask.cpu().numpy()   # [N, H, W]

        out_imgs, out_masks = [], []
        for img, m in zip(imgs_np, masks_np):
            bool_mask = m > 0.5

            if not bool_mask.any():
                out_imgs.append(torch.from_numpy(img))
                out_masks.append(torch.from_numpy(np.zeros_like(m)))
                continue

            _, (rows, cols) = distance_transform_edt(~bool_mask, return_indices=True)

            dilated = binary_dilation(bool_mask, iterations=dilation_px)
            ring = dilated & ~bool_mask

            out = img.copy()
            out[ring] = img[rows[ring], cols[ring]]

            out_imgs.append(torch.from_numpy(out))
            out_masks.append(torch.from_numpy(dilated.astype(np.float32)))

        return (torch.stack(out_imgs), torch.stack(out_masks))
