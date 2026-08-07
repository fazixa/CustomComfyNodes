import cv2
import numpy as np
import torch
from scipy.ndimage import binary_fill_holes


def _both_sides_fill(mask_u8, max_dist, min_axes, iterate):
    """Fill background pixels that have mask on both sides (within max_dist)
    along at least min_axes of the 4 axes: horizontal, vertical, 2 diagonals.
    With iterate=True, filled pixels count as mask for the next pass, so the
    fill creeps deeper into gaps until stable.
    """
    d = max_dist
    eye = np.eye(d + 1, dtype=np.uint8)
    kerns = [
        (np.ones((1, d + 1), np.uint8), (0, 0), (d, 0)),
        (np.ones((d + 1, 1), np.uint8), (0, 0), (0, d)),
        (eye, (0, 0), (d, d)),
        (eye[::-1].copy(), (0, d), (d, 0)),
    ]
    cur = mask_u8.copy()
    for _ in range(20 if iterate else 1):
        count = np.zeros(cur.shape, np.uint8)
        for kern, a_fwd, a_bwd in kerns:
            fwd = cv2.dilate(cur, kern, anchor=a_fwd)
            bwd = cv2.dilate(cur, kern, anchor=a_bwd)
            count += ((fwd > 0) & (bwd > 0)).astype(np.uint8)
        fill = ((count >= min_axes) & (cur == 0)).astype(np.uint8) * 255
        if not fill.any():
            break
        cur = cv2.bitwise_or(cur, fill)
    return cur


def _fill_gaps(mask_u8, close_px, min_gap_width, min_gap_area, fill_holes):
    """Fill large pockets (e.g. between limbs and body) while leaving the
    original mask — including fine details like finger gaps — untouched.

    Morphological close at close_px on a scratch copy, then keep only the
    filled pockets that are both wide (survive erosion by min_gap_width/2)
    and large (>= min_gap_area); thin/small fills such as finger gaps and
    boundary slivers are discarded. Returns original mask OR'd with the
    surviving pockets.
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_px + 1, 2 * close_px + 1))
    closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k)
    pockets = cv2.bitwise_and(closed, cv2.bitwise_not(mask_u8))

    # Wide-pocket test: erode, then keep only components that survive
    r = max(1, min_gap_width // 2)
    ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    core = cv2.erode(pockets, ek)

    # Fully-enclosed holes (e.g. an open mouth) — optionally protected
    holes = binary_fill_holes(mask_u8 > 0) & (mask_u8 == 0)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pockets, connectivity=8)
    keep = np.zeros_like(mask_u8)
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] < min_gap_area:
            continue
        comp = (labels == lbl)
        if not core[comp].any():   # too thin everywhere — a sliver, not a pocket
            continue
        if not fill_holes and holes[comp].any():
            continue
        keep[comp] = 255

    return cv2.bitwise_or(mask_u8, keep)


class MaskFillGapsNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask":          ("MASK",),
                "method":        (["both_sides", "pocket_close"], {"default": "both_sides",
                                  "tooltip": "both_sides: fill pixels with mask on both sides along enough axes. pocket_close: seal pocket openings narrower than 2x close_px."}),
                "max_dist":      ("INT", {"default": 60, "min": 1, "max": 500, "step": 1,
                                          "tooltip": "both_sides: how far to look for mask on each side of a pixel"}),
                "min_axes":      ("INT", {"default": 4, "min": 1, "max": 4, "step": 1,
                                          "tooltip": "both_sides: axes (of horizontal/vertical/2 diagonals) that must have mask on both sides. 4 = strict, 3 = looser"}),
                "iterate":       ("BOOLEAN", {"default": True,
                                              "tooltip": "both_sides: repeat until stable so the fill creeps deeper into gaps"}),
                "close_px":      ("INT", {"default": 30, "min": 1, "max": 200, "step": 1,
                                          "tooltip": "Pocket openings narrower than ~2x this get sealed; wider ones stay open"}),
                "min_gap_width": ("INT", {"default": 10, "min": 1, "max": 200, "step": 1,
                                          "tooltip": "Filled pockets thinner than this are discarded (protects finger gaps, edge slivers)"}),
                "min_gap_area":  ("INT", {"default": 200, "min": 0, "max": 1000000, "step": 50,
                                          "tooltip": "Filled pockets smaller than this many pixels are discarded"}),
                "fill_holes":    ("BOOLEAN", {"default": False,
                                              "tooltip": "Also fill fully-enclosed holes (e.g. an open mouth). Off = only open gaps between limbs/body are filled."}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "fill"
    CATEGORY = "fae/mask"

    def fill(self, mask, method, max_dist, min_axes, iterate,
             close_px, min_gap_width, min_gap_area, fill_holes):
        out = []
        for i in range(mask.shape[0]):
            m_u8 = (mask[i].cpu().numpy() > 0.5).astype(np.uint8) * 255
            if method == "both_sides":
                filled = _both_sides_fill(m_u8, max_dist, min_axes, iterate)
            else:
                filled = _fill_gaps(m_u8, close_px, min_gap_width, min_gap_area, fill_holes)
            out.append(torch.from_numpy(filled.astype(np.float32) / 255.0))
        return (torch.stack(out),)
