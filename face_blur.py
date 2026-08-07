import cv2
import numpy as np
import torch
from deface.centerface import CenterFace


def _gradient_blur_region(frame, mask, x1, y1, x2, y2, strength):
    """Soft feathered oval blur — blurred center fades smoothly into original at edges."""
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    rx = (x2 - x1) // 2
    ry = (y2 - y1) // 2

    region_mask = np.zeros(frame.shape[:2], dtype=np.float32)
    cv2.ellipse(region_mask, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
    feather = max(rx, ry) // 3
    feather_k = (feather * 6 + 1) | 1
    region_mask = cv2.GaussianBlur(region_mask, (feather_k, feather_k), feather)

    k = strength | 1
    blurred = cv2.GaussianBlur(frame, (k, k), 0)

    mask_3ch = np.stack([region_mask] * 3, axis=-1)
    frame = (blurred * mask_3ch + frame * (1.0 - mask_3ch)).astype(np.uint8)
    mask = np.maximum(mask, region_mask)
    return frame, mask


def _patch_region(frame, mask, x1, y1, x2, y2, skin_color=(180, 155, 135)):
    """Solid skin-tone oval — for Seedance reference input prep."""
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    rx = (x2 - x1) // 2
    ry = (y2 - y1) // 2
    cv2.ellipse(frame, (cx, cy), (rx, ry), 0, 0, 360, skin_color, -1)
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
    return frame, mask


class FaceBlurNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "mode": (["gradient", "patch"], {"default": "gradient"}),
                "strength": ("INT", {"default": 31, "min": 1, "max": 199, "step": 2,
                                      "tooltip": "Blur kernel size (odd), gradient mode only"}),
                "padding": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01,
                                       "tooltip": "Expand each detected face box by this fraction"}),
                "confidence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                          "tooltip": "Face detection confidence threshold"}),
            },
            "optional": {
                "exclude_region": ("STRING", {"default": "",
                                               "tooltip": "x1,y1,x2,y2 — skip faces whose center falls inside this box"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "mask")
    FUNCTION = "process"
    CATEGORY = "fae/video"

    def process(self, images, mode, strength, padding, confidence, exclude_region=""):
        exclude = None
        if exclude_region.strip():
            exclude = tuple(int(v) for v in exclude_region.split(","))

        centerface = CenterFace()
        strength = strength | 1

        imgs_np = (images.cpu().numpy() * 255.0).astype(np.uint8)  # [N, H, W, 3] RGB
        height, width = imgs_np.shape[1], imgs_np.shape[2]

        out_frames = []
        out_masks = []

        for img in imgs_np:
            frame = np.ascontiguousarray(img[..., ::-1])  # RGB -> BGR for CenterFace
            mask = np.zeros((height, width), dtype=np.float32)
            dets, _ = centerface(frame, threshold=confidence)

            for det in dets:
                fx1, fy1, fx2, fy2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
                if exclude:
                    ex1, ey1, ex2, ey2 = exclude
                    cx, cy = (fx1 + fx2) // 2, (fy1 + fy2) // 2
                    if ex1 <= cx <= ex2 and ey1 <= cy <= ey2:
                        continue

                fw, fh = fx2 - fx1, fy2 - fy1
                x1 = max(0, fx1 - int(padding * fw))
                y1 = max(0, fy1 - int(padding * fh))
                x2 = min(width, fx2 + int(padding * fw))
                y2 = min(height, fy2 + int(padding * fh))

                if mode == "patch":
                    frame, mask = _patch_region(frame, mask, x1, y1, x2, y2)
                else:
                    frame, mask = _gradient_blur_region(frame, mask, x1, y1, x2, y2, strength)

            out_frames.append(frame[..., ::-1])  # BGR -> RGB
            out_masks.append(mask)

        images_out = torch.from_numpy(np.stack(out_frames).astype(np.float32) / 255.0)
        masks_out = torch.from_numpy(np.stack(out_masks))
        return (images_out, masks_out)
