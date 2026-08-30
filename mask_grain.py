import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.special import erf

REGIONS = ["edge", "edge+inside", "inside", "outside", "whole"]


def _noise_field(shape, grain_size, seed):
    """Blurred white noise mapped back to a ~uniform [0,1] field.

    Blurring shrinks the spread by a factor that depends on grain_size, so the
    result is renormalised to unit variance and pushed through the gaussian CDF
    — that way `amount` means the same thing at every grain size.
    """
    rng = np.random.default_rng(seed)
    n = rng.standard_normal(shape).astype(np.float32)
    if grain_size > 0:
        n = gaussian_filter(n, sigma=grain_size)
    std = float(n.std())
    if std < 1e-8:
        return np.full(shape, 0.5, dtype=np.float32)
    return (0.5 * (1.0 + erf(n / (std * np.sqrt(2.0))))).astype(np.float32)


def _grain(u, density, hardness, bias):
    """Uniform [0,1] noise → a signed push in [-1,1].

    `density` is the fraction of the field that becomes grain at all — the rest
    sits at exactly zero and leaves the mask alone — and because the field is
    uniform, that fraction is exact. `bias` splits it between grain that eats
    into the mask and grain that adds specks outside it, and `hardness` turns
    the ramp from speck centre to nothing into a flat-topped dot.
    """
    f_neg = density * (1.0 - bias) / 2.0
    f_pos = density * (1.0 + bias) / 2.0
    g = np.zeros_like(u)
    if f_neg > 0:
        g -= np.clip((f_neg - u) / f_neg, 0.0, 1.0)
    if f_pos > 0:
        g += np.clip((u - (1.0 - f_pos)) / f_pos, 0.0, 1.0)
    if hardness > 0:
        g = np.clip(g / max(1.0 - hardness, 1e-3), -1.0, 1.0)
    return g


def _region_weight(m_u8, region, edge_width):
    """Where the grain is allowed to bite, as a [0,1] field."""
    if region == "whole":
        return np.ones(m_u8.shape, dtype=np.float32)
    inside = (m_u8 > 0).astype(np.float32)
    if region == "inside":
        return inside
    if region == "outside":
        return 1.0 - inside

    # a band of edge_width px either side of the boundary
    d_in = cv2.distanceTransform(m_u8, cv2.DIST_L2, 5)
    d_out = cv2.distanceTransform(255 - m_u8, cv2.DIST_L2, 5)
    dist = np.where(m_u8 > 0, d_in, d_out)
    t = np.clip(1.0 - dist / max(edge_width, 1e-3), 0.0, 1.0)
    band = t * t * (3.0 - 2.0 * t)          # smoothstep, so the band fades out

    if region == "edge+inside":
        return np.maximum(band, inside)     # solid inside, fading past the edge
    return band


def add_grain(mask, amount=0.5, density=1.0, grain_size=2.0, hardness=0.0,
              bias=0.0, region="edge", edge_width=8.0, seed=0,
              lock_to_screen=False):
    masks_np = mask.cpu().numpy()
    out = []
    for i, m in enumerate(masks_np):
        m_u8 = (m > 0.5).astype(np.uint8) * 255
        # Locked: one field in frame coordinates, reused for every frame, so the
        # specks sit still instead of re-randomising into a shimmer.
        u = _noise_field(m.shape, grain_size, seed if lock_to_screen else seed + i)
        g = _grain(u, density, hardness, bias)
        w = _region_weight(m_u8, region, edge_width)
        out.append(torch.from_numpy(np.clip(m + g * amount * w, 0.0, 1.0)))
    return torch.stack(out)


class MaskGrainNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask":       ("MASK",),
                "amount":     ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.01,
                                         "tooltip": "How hard each speck bites — not how many. 1.0 just reaches through; above 1 the specks saturate, so they come out solid black instead of grey."}),
                "density":    ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                         "tooltip": "How much of the area is grain at all. 0.1 = sparse scattered specks, 1.0 = grain everywhere."}),
                "grain_size": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 50.0, "step": 0.1,
                                         "tooltip": "Speck size in px. 0 = single-pixel noise, higher = coarser clumps."}),
                "hardness":   ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                         "tooltip": "0 = soft cloudy grain, 1 = hard on/off specks (stipple / dissolve)."}),
                "bias":       ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05,
                                         "tooltip": "-1 grain only eats into the mask, 0 both ways, +1 only adds specks outside."}),
                "region":     (REGIONS, {"default": "edge",
                                         "tooltip": "edge: only the boundary breaks up. edge+inside: the whole silhouette plus a fading band outside it. inside/outside: one side only. whole: everywhere."}),
                "edge_width": ("FLOAT", {"default": 8.0, "min": 0.5, "max": 200.0, "step": 0.5,
                                         "tooltip": "region=edge / edge+inside: how many px past the boundary the grain reaches"}),
                "seed":       ("INT",   {"default": 0, "min": 0, "max": 99999}),
                "lock_to_screen": ("BOOLEAN", {"default": False,
                                               "tooltip": "On: one grain pattern pinned to the frame — the specks never move or re-randomise. Off: new grain every frame (film-grain shimmer)."}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "apply"
    CATEGORY = "fae/mask"

    def apply(self, mask, amount, density, grain_size, hardness, bias, region,
              edge_width, seed, lock_to_screen):
        return (add_grain(mask, amount, density, grain_size, hardness, bias,
                          region, edge_width, seed, lock_to_screen),)
