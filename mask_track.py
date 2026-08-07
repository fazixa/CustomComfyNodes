import cv2
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


def _fill_from_nearest(field, invalid):
    """Replace `field` values where `invalid` is True with the nearest valid value."""
    if not invalid.any() or invalid.all():
        return field
    _, (rows, cols) = distance_transform_edt(invalid, return_indices=True)
    out = field.copy()
    out[invalid] = field[rows[invalid], cols[invalid]]
    return out


def _to_gray_u8(img_hw3):
    """float [H, W, 3] 0-1 -> uint8 grayscale."""
    return cv2.cvtColor((img_hw3 * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)


def _propagate_flow(frames_np, mask0, preset, excludes=None):
    """Warp mask forward frame-to-frame with dense DIS optical flow.

    For each new frame we compute backward flow (new -> previous) and sample
    the previous mask at p + flow(p). The mask stays float the whole way so
    soft edges survive; thresholding is left to the caller.
    """
    presets = {
        "fast":   cv2.DISOPTICAL_FLOW_PRESET_FAST,
        "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
        "quality": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,  # placeholder, replaced below
    }
    dis = cv2.DISOpticalFlow_create(presets.get(preset, cv2.DISOPTICAL_FLOW_PRESET_MEDIUM))
    if preset == "quality":
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        dis.setFinestScale(0)  # full-resolution finest level = highest quality

    h, w = mask0.shape
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    out = [mask0.astype(np.float32)]
    prev_gray = _to_gray_u8(frames_np[0])
    prev_mask = out[0]

    for i in range(1, len(frames_np)):
        gray = _to_gray_u8(frames_np[i])
        flow = dis.calc(gray, prev_gray, None)  # backward: current -> previous
        if excludes is not None:
            # Don't trust motion measured on excluded content (e.g. a composited
            # character) — replace its flow with the nearest surrounding flow.
            ex = excludes[min(i, len(excludes) - 1)] > 0.5
            flow[..., 0] = _fill_from_nearest(flow[..., 0], ex)
            flow[..., 1] = _fill_from_nearest(flow[..., 1], ex)
        map_x = xs + flow[..., 0]
        map_y = ys + flow[..., 1]
        warped = cv2.remap(prev_mask, map_x, map_y,
                           interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        out.append(warped)
        prev_gray, prev_mask = gray, warped

    return out


def _propagate_homography(frames_np, mask0, min_matches=12, excludes=None):
    """Warp mask0 into every frame via a SIFT homography computed directly
    against frame 0 — no frame-to-frame accumulation, so no drift. Meant for
    static scenes where only the camera moves. Falls back to the previous
    frame's result when matching fails."""
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()

    def _detect_mask_u8(idx):
        if excludes is None:
            return None
        ex = excludes[min(idx, len(excludes) - 1)]
        return ((ex < 0.5) * 255).astype(np.uint8)

    gray0 = _to_gray_u8(frames_np[0])
    kp0, des0 = sift.detectAndCompute(gray0, _detect_mask_u8(0))

    out = [mask0.astype(np.float32)]
    h, w = mask0.shape
    last = out[0]

    for i in range(1, len(frames_np)):
        warped = None
        gray = _to_gray_u8(frames_np[i])
        kp, des = sift.detectAndCompute(gray, _detect_mask_u8(i))

        if des is not None and des0 is not None and len(kp) >= min_matches:
            matches = bf.knnMatch(des0, des, k=2)
            good = [m for m, n in (p for p in matches if len(p) == 2)
                    if m.distance < 0.75 * n.distance]
            if len(good) >= min_matches:
                src = np.float32([kp0[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
                if H is not None and inliers is not None and inliers.sum() >= min_matches:
                    warped = cv2.warpPerspective(mask0.astype(np.float32), H, (w, h),
                                                 flags=cv2.INTER_LINEAR,
                                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

        if warped is None:
            warped = last  # hold last known position
        out.append(warped)
        last = warped

    return out


class MaskTrackNode:
    """Propagate a first-frame mask through a whole clip.

    The mask can contain multiple disjoint patches with unrelated content —
    propagation is per-pixel (flow) or scene-global (homography), never
    per-object, so nothing inside the mask needs to be a single trackable
    thing."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "mask":   ("MASK", {"tooltip": "Mask drawn on the FIRST frame (batch input: frame 0 is used)"}),
                "method": (["optical_flow", "global_homography"],
                           {"default": "optical_flow",
                            "tooltip": "optical_flow: follows whatever moves under the mask. "
                                       "global_homography: static scene + moving camera, drift-free (matches every frame to frame 1)."}),
                "flow_preset": (["fast", "medium", "quality"], {"default": "medium"}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                        "tooltip": "Binarize the propagated mask; 0 keeps it soft/float"}),
                "cleanup_px": ("INT", {"default": 0, "min": 0, "max": 20,
                                       "tooltip": "Morphological close+open radius to heal small holes/specks (applies when threshold > 0)"}),
            },
            "optional": {
                "exclude_mask": ("MASK", {"tooltip": "Pixels to ignore when measuring motion (e.g. a composited character). "
                                                     "Single mask = used for all frames; batch = per-frame. "
                                                     "Homography: no feature points there. Flow: motion there is replaced by surrounding motion."}),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("masks", "preview")
    FUNCTION = "track"
    CATEGORY = "fae/image"

    def track(self, images, mask, method, flow_preset, threshold, cleanup_px,
              exclude_mask=None):
        frames_np = images.cpu().numpy()  # [N, H, W, C]
        n, h, w, _ = frames_np.shape

        m0 = mask[0].cpu().numpy().astype(np.float32)
        if m0.shape != (h, w):
            m0 = cv2.resize(m0, (w, h), interpolation=cv2.INTER_LINEAR)

        excludes = None
        if exclude_mask is not None:
            excludes = []
            for j in range(exclude_mask.shape[0]):
                ex = exclude_mask[j].cpu().numpy().astype(np.float32)
                if ex.shape != (h, w):
                    ex = cv2.resize(ex, (w, h), interpolation=cv2.INTER_LINEAR)
                excludes.append(ex)

        if method == "global_homography":
            masks_np = _propagate_homography(frames_np, m0, excludes=excludes)
        else:
            masks_np = _propagate_flow(frames_np, m0, flow_preset, excludes=excludes)

        out_masks, out_previews = [], []
        kernel = None
        if cleanup_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * cleanup_px + 1, 2 * cleanup_px + 1))

        for i, m in enumerate(masks_np):
            m = np.clip(m, 0.0, 1.0)
            if threshold > 0:
                m = (m >= threshold).astype(np.float32)
                if kernel is not None:
                    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
                    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
            out_masks.append(torch.from_numpy(m))

            # Preview: mask tinted red over the frame
            frame = frames_np[i][..., :3].copy()
            frame[..., 0] = frame[..., 0] * (1 - m * 0.5) + m * 0.5
            frame[..., 1] = frame[..., 1] * (1 - m * 0.5)
            frame[..., 2] = frame[..., 2] * (1 - m * 0.5)
            out_previews.append(torch.from_numpy(frame.astype(np.float32)))

        return (torch.stack(out_masks), torch.stack(out_previews))
