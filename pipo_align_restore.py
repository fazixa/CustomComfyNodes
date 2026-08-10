import cv2
import numpy as np
import torch

from .sift_align import (
    ALIGN_MODELS,
    to_u8,
    compute_transform,
    smooth_transforms,
    lock_transforms,
)


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
                "max_features": ("INT", {"default": 2000, "min": 500,  "max": 5000, "step": 100, "tooltip": "SIFT keeps this many keypoints, strongest first. Past ~2000 it starts admitting weak, poorly localised ones that make alignment less steady, not more."}),
                "match_count":  ("INT", {"default": 0,    "min": 0,    "max": 1000, "step": 10, "tooltip": "Cap on matches kept after the ratio test. 0 = keep all, which is normally best."}),
                "feather_px":   ("INT", {"default": 3,    "min": 0,    "max": 30,   "step": 1}),
                "align_model":  (ALIGN_MODELS, {"default": "similarity", "tooltip": "Motion allowed between the two shots. similarity = pan/rotate/zoom, affine = adds shear and stretch, homography = adds perspective. Pick the least that covers the footage; surplus freedom becomes flicker."}),
                "smoothing":    ("INT", {"default": 5,    "min": 0,    "max": 31,   "step": 2, "tooltip": "Frames averaged over when smoothing the alignment. 0 or 1 disables it. Raise it for steadier tracking, lower it if alignment lags fast camera motion."}),
                "lock_alignment": ("BOOLEAN", {"default": False, "tooltip": "Use one transform for the whole clip. Correct only when neither the original nor the generated background moves — e.g. a still photo as the original. Removes alignment flicker entirely."}),
            },
        }

    RETURN_TYPES  = ("IMAGE", "IMAGE")
    RETURN_NAMES  = ("composited", "restored_layer")
    FUNCTION      = "composite"
    CATEGORY      = "fae/video"

    def _frame_arrays(self, original_frames, generated_frames, mask, exclude_mask,
                      i, gh, gw):
        """Per-frame arrays at generated resolution, shared by both passes."""
        orig_u8 = to_u8(original_frames[i])
        gen_u8  = to_u8(generated_frames[i])
        mask_f  = mask[i].cpu().numpy().astype(np.float32)

        oh, ow = orig_u8.shape[:2]
        if (oh, ow) != (gh, gw):
            orig_u8 = cv2.resize(orig_u8, (gw, gh), interpolation=cv2.INTER_LINEAR)

        mh, mw = mask_f.shape[:2]
        if (mh, mw) != (gh, gw):
            mask_f = cv2.resize(mask_f, (gw, gh), interpolation=cv2.INTER_LINEAR)

        orig_feature_mask = None
        if exclude_mask is not None and i < exclude_mask.shape[0]:
            excl_f = exclude_mask[i].cpu().numpy().astype(np.float32)
            eh, ew = excl_f.shape[:2]
            if (eh, ew) != (gh, gw):
                excl_f = cv2.resize(excl_f, (gw, gh), interpolation=cv2.INTER_LINEAR)
            orig_feature_mask = ((excl_f < 0.5) * 255).astype(np.uint8)

        return orig_u8, gen_u8, mask_f, orig_feature_mask

    def composite(self, original_frames, generated_frames, mask,
                  exclude_mask=None, max_features=2000, match_count=0, feather_px=3,
                  align_model="similarity", smoothing=5, lock_alignment=False):

        n  = min(original_frames.shape[0], generated_frames.shape[0], mask.shape[0])
        gh, gw = generated_frames.shape[1], generated_frames.shape[2]
        results = []
        restored_layer_results = []

        # Pass 1 — estimate the original → generated transform for every frame,
        # so the sequence can be conditioned before anything is warped.
        transforms = []
        for i in range(n):
            orig_u8, gen_u8, mask_f, orig_feature_mask = self._frame_arrays(
                original_frames, generated_frames, mask, exclude_mask, i, gh, gw)

            # The replace region doesn't correspond to the original, by definition
            gen_feature_mask = ((mask_f < 0.5) * 255).astype(np.uint8)
            orig_gray = cv2.cvtColor(orig_u8, cv2.COLOR_RGB2GRAY)
            gen_gray  = cv2.cvtColor(gen_u8,  cv2.COLOR_RGB2GRAY)
            transforms.append(compute_transform(orig_gray, gen_gray, orig_feature_mask,
                                                gen_feature_mask, max_features,
                                                match_count, align_model))

        solved = sum(t is not None for t in transforms)
        if solved < n:
            print(f"[PipoAlignRestore] {n - solved}/{n} frames had too few matches")

        if lock_alignment:
            transforms = lock_transforms(transforms, (gw, gh))
        elif smoothing > 1:
            transforms = smooth_transforms(transforms, smoothing, (gw, gh))

        # Pass 2 — warp and composite with the conditioned transforms.
        for i in range(n):
            orig_u8, gen_u8, mask_f, _ = self._frame_arrays(
                original_frames, generated_frames, mask, exclude_mask, i, gh, gw)
            H = transforms[i]

            if H is not None:
                warped_orig = cv2.warpPerspective(orig_u8, H, (gw, gh), flags=cv2.INTER_LINEAR)
                warped_mask = cv2.warpPerspective(mask_f,  H, (gw, gh), flags=cv2.INTER_LINEAR)
            else:
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
