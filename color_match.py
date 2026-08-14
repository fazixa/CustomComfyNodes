import torch

from .boil_effect import _hex_to_rgb, _rgb_to_hsv, _hsv_to_rgb

EPS = 1e-4


def _hue_delta(h_src, h_dst):
    """Shortest signed rotation around the hue wheel, in [-0.5, 0.5]."""
    return ((h_dst - h_src + 0.5) % 1.0) - 0.5


def _match_channel(x, src, dst):
    """Scale x so src lands exactly on dst, without crushing the top end.

    Below the anchor the mapping is the plain ratio. Brightening the anchor
    (ratio > 1) would push everything above it past 1.0 and flatten it to a
    single flat value, so that stretch is instead eased onto 1.0: the curve
    keeps the ratio's slope where it meets the anchor and rolls off from there,
    which leaves highlight separation intact. Darkening can't clip, so it stays
    linear throughout.
    """
    src = float(min(max(src, EPS), 1.0))
    dst = float(min(max(dst, 0.0), 1.0 - EPS))
    ratio = dst / src

    below = x * ratio
    if ratio <= 1.0:
        return below.clamp(0.0, 1.0)

    head = max(1.0 - src, EPS)
    # Slope the roll-off starts at, so it joins the ratio smoothly at the anchor.
    a = ratio * head / max(1.0 - dst, EPS)
    u = ((x - src) / head).clamp(0.0, 1.0)
    above = dst + (1.0 - dst) * (1.0 - (1.0 - u) ** a)

    return torch.where(x <= src, below, above).clamp(0.0, 1.0)


def match_color(images, source_hex, target_hex,
                hue_amount=1.0, saturation_amount=1.0, value_amount=1.0):
    """Shift images so source_hex would land on target_hex, in HSV.

    The whole frame moves by the same correction — the sampled colour is only
    how the shift is measured, not a region it's confined to.
    """
    src_hsv = _rgb_to_hsv(torch.tensor(_hex_to_rgb(source_hex), dtype=torch.float32))
    dst_hsv = _rgb_to_hsv(torch.tensor(_hex_to_rgb(target_hex), dtype=torch.float32))
    src_h, src_s, src_v = (float(c) for c in src_hsv)
    dst_h, dst_s, dst_v = (float(c) for c in dst_hsv)

    rgb, extra = images[..., :3], images[..., 3:]
    h, s, v = _rgb_to_hsv(rgb).unbind(-1)

    h = (h + _hue_delta(src_h, dst_h) * hue_amount) % 1.0
    # Easing the target toward the source is how amount dials a channel back:
    # at 0 the anchor doesn't move, so the ratio is 1 and nothing changes.
    s = _match_channel(s, src_s, src_s + (dst_s - src_s) * saturation_amount)
    v = _match_channel(v, src_v, src_v + (dst_v - src_v) * value_amount)

    out = _hsv_to_rgb(torch.stack([h, s, v], dim=-1))
    return torch.cat([out, extra], dim=-1) if extra.shape[-1] else out


class ColorMatchNode:
    """Grade one shot onto another using a colour they share.

    Pick the same real-world colour — Pipo's pink — as it appears in each shot,
    and the difference between those two samples describes the difference
    between the shots. Applying that correction to every pixel carries the
    whole frame across, rather than repainting the pink alone.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images":         ("IMAGE", {"tooltip": "The footage to correct — normally the generated shot"}),
                "source_color":   ("STRING", {"default": "#ED8AB6",
                                              "tooltip": "The shared colour as it appears in these images. Sample it off a real frame with the picker below"}),
                "target_color":   ("STRING", {"default": "#ED8AB6",
                                              "tooltip": "The same colour as it appears in the footage you're matching to — normally the original"}),
                "hue_amount":        ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                                "tooltip": "How much of the hue rotation to take. 0 leaves hue alone, 1 lands source_color's hue exactly on target_color's"}),
                "saturation_amount": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                                "tooltip": "How much of the saturation correction to take"}),
                "value_amount":      ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                                "tooltip": "How much of the brightness correction to take"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "match"
    CATEGORY = "fae/image"

    def match(self, images, source_color, target_color,
              hue_amount, saturation_amount, value_amount):
        return (match_color(images, source_color, target_color,
                            hue_amount, saturation_amount, value_amount),)
