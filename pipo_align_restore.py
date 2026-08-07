import cv2
import numpy as np
import torch

from .sift_align import to_u8, compute_homography


class PipoAlignRestoreNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_frames":  ("IMAGE",),
                "generated_frames": ("IMAGE",),
                "mask":             ("MASK", {"tooltip": "Region of generated_frames to replace with aligned original content. Excluded from SIFT features on generated_frames."}),
            },
            "optional": {
                "exclude_mask": ("MASK", {"tooltip": "Pixels excluded from SIFT feature extraction on original_frames."}),
                "max_features": ("INT", {"default": 2000, "min": 500,  "max": 5000, "step": 100}),
                "match_count":  ("INT", {"default": 50,   "min": 10,   "max": 1000, "step": 10}),
                "feather_px":   ("INT", {"default": 3,    "min": 0,    "max": 30,   "step": 1}),
            },
        }

    RETURN_TYPES  = ("IMAGE", "IMAGE")
    RETURN_NAMES  = ("composited", "restored_layer")
    FUNCTION      = "composite"
    CATEGORY      = "fae/video"

    def composite(self, original_frames, generated_frames, mask,
                  exclude_mask=None, max_features=2000, match_count=50, feather_px=3):

        n  = min(original_frames.shape[0], generated_frames.shape[0], mask.shape[0])
        gh, gw = generated_frames.shape[1], generated_frames.shape[2]
        results = []
        restored_layer_results = []

        for i in range(n):
            orig_u8 = to_u8(original_frames[i])
            gen_u8  = to_u8(generated_frames[i])
            mask_f  = mask[i].cpu().numpy().astype(np.float32)

            # Resize original + mask to generated resolution if needed
            oh, ow = orig_u8.shape[:2]
            if (oh, ow) != (gh, gw):
                orig_u8 = cv2.resize(orig_u8, (gw, gh), interpolation=cv2.INTER_LINEAR)

            mh, mw = mask_f.shape[:2]
            if (mh, mw) != (gh, gw):
                mask_f = cv2.resize(mask_f, (gw, gh), interpolation=cv2.INTER_LINEAR)

            # Exclude the replace region from generated's features (it doesn't match original there)
            gen_feature_mask = ((mask_f < 0.5) * 255).astype(np.uint8)

            # Exclude user-specified region from original's features
            orig_feature_mask = None
            if exclude_mask is not None and i < exclude_mask.shape[0]:
                excl_f = exclude_mask[i].cpu().numpy().astype(np.float32)
                eh, ew = excl_f.shape[:2]
                if (eh, ew) != (gh, gw):
                    excl_f = cv2.resize(excl_f, (gw, gh), interpolation=cv2.INTER_LINEAR)
                orig_feature_mask = ((excl_f < 0.5) * 255).astype(np.uint8)

            # Compute homography from original → generated using background features
            orig_gray = cv2.cvtColor(orig_u8, cv2.COLOR_RGB2GRAY)
            gen_gray  = cv2.cvtColor(gen_u8,  cv2.COLOR_RGB2GRAY)
            H = compute_homography(orig_gray, gen_gray, orig_feature_mask, gen_feature_mask,
                                    max_features, match_count)

            if H is not None:
                warped_orig = cv2.warpPerspective(orig_u8, H, (gw, gh), flags=cv2.INTER_LINEAR)
                warped_mask = cv2.warpPerspective(mask_f,  H, (gw, gh), flags=cv2.INTER_LINEAR)
            else:
                print(f"[PipoAlignRestore] frame {i}: not enough matches, skipping alignment")
                warped_orig = orig_u8
                warped_mask = mask_f

            if feather_px > 0:
                k = feather_px * 2 + 1
                warped_mask = cv2.GaussianBlur(warped_mask, (k, k), 0)

            # Repositioned original layer on its own, as RGBA (alpha = warped mask)
            restored_rgba = np.dstack([
                warped_orig,
                np.clip(warped_mask * 255, 0, 255).astype(np.uint8),
            ])
            restored_layer_results.append(
                torch.from_numpy(restored_rgba).float() / 255.0
            )

            # Composite aligned original over generated
            m = warped_mask[:, :, np.newaxis]
            composited = (warped_orig.astype(np.float32) * m
                          + gen_u8.astype(np.float32) * (1.0 - m))

            results.append(torch.from_numpy(
                np.clip(composited, 0, 255).astype(np.uint8)
            ).float() / 255.0)

        return (torch.stack(results), torch.stack(restored_layer_results))
