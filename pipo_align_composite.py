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


class PipoAlignCompositeNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_frames":  ("IMAGE",),
                "generated_frames": ("IMAGE",),
                "pipo_mask":        ("MASK",),
            },
            "optional": {
                "boiled_frames": ("IMAGE", {"tooltip": "Boil Effect output — used as composite source instead of generated_frames. generated_frames still used for SIFT alignment."}),
                "exclude_mask":  ("MASK", {"tooltip": "Pixels excluded from SIFT feature extraction on original_frames — e.g. a pipo already composited into the footage that doesn't match the generated frames."}),
                "gp_frames":     ("IMAGE",),
                "gp_mask":       ("MASK",),
                "max_features":  ("INT", {"default": 2000, "min": 500,  "max": 5000, "step": 100, "tooltip": "SIFT keeps this many keypoints, strongest first. Past ~2000 it starts admitting weak, poorly localised ones that make alignment less steady, not more."}),
                "match_count":   ("INT", {"default": 0,    "min": 0,    "max": 1000, "step": 10, "tooltip": "Cap on matches kept after the ratio test. 0 = keep all, which is normally best."}),
                "feather_px":    ("INT", {"default": 3,    "min": 0,    "max": 30,   "step": 1}),
                "align_model":   (ALIGN_MODELS, {"default": "similarity", "tooltip": "Motion allowed between the two shots. similarity = pan/rotate/zoom, affine = adds shear and stretch, homography = adds perspective. Pick the least that covers the footage; surplus freedom becomes flicker."}),
                "smoothing":     ("INT", {"default": 5,    "min": 0,    "max": 31,   "step": 2, "tooltip": "Frames averaged over when smoothing the alignment. 0 or 1 disables it. Raise it for steadier tracking, lower it if alignment lags fast camera motion."}),
                "lock_alignment": ("BOOLEAN", {"default": False, "tooltip": "Use one transform for the whole clip. Correct only when neither the original nor the generated background moves — e.g. a still photo as the original. Removes alignment flicker entirely."}),
            },
        }

    RETURN_TYPES  = ("IMAGE", "IMAGE")
    RETURN_NAMES  = ("composited", "pipo_layer")
    FUNCTION      = "composite"
    CATEGORY      = "fae/video"

    def _frame_arrays(self, original_frames, generated_frames, pipo_mask, exclude_mask,
                      i, oh, ow):
        """Per-frame arrays at original resolution, shared by both passes."""
        orig_u8 = to_u8(original_frames[i])
        gen_u8  = to_u8(generated_frames[i])
        mask_f  = pipo_mask[i].cpu().numpy().astype(np.float32)

        gh, gw = gen_u8.shape[:2]
        if (gh, gw) != (oh, ow):
            gen_u8 = cv2.resize(gen_u8, (ow, oh), interpolation=cv2.INTER_LINEAR)
            mask_f = cv2.resize(mask_f, (ow, oh), interpolation=cv2.INTER_LINEAR)

        orig_feature_mask = None
        if exclude_mask is not None and i < exclude_mask.shape[0]:
            excl_f = exclude_mask[i].cpu().numpy().astype(np.float32)
            eh, ew = excl_f.shape[:2]
            if (eh, ew) != (oh, ow):
                excl_f = cv2.resize(excl_f, (ow, oh), interpolation=cv2.INTER_LINEAR)
            orig_feature_mask = ((excl_f < 0.5) * 255).astype(np.uint8)

        return orig_u8, gen_u8, mask_f, orig_feature_mask

    def composite(self, original_frames, generated_frames, pipo_mask,
                  boiled_frames=None, exclude_mask=None, gp_frames=None, gp_mask=None,
                  max_features=2000, match_count=0, feather_px=3,
                  align_model="similarity", smoothing=5, lock_alignment=False):

        n  = min(original_frames.shape[0], generated_frames.shape[0], pipo_mask.shape[0])
        oh, ow = original_frames.shape[1], original_frames.shape[2]
        has_gp = gp_frames is not None and gp_mask is not None
        results = []
        pipo_layer_results = []

        # Pass 1 — estimate the generated → original transform for every frame.
        # Conditioning them as a sequence is what steadies the composite, so
        # they're all solved before anything is warped.
        transforms = []
        for i in range(n):
            orig_u8, gen_u8, mask_f, orig_feature_mask = self._frame_arrays(
                original_frames, generated_frames, pipo_mask, exclude_mask, i, oh, ow)

            bg_mask   = ((mask_f < 0.5) * 255).astype(np.uint8)
            orig_gray = cv2.cvtColor(orig_u8, cv2.COLOR_RGB2GRAY)
            gen_gray  = cv2.cvtColor(gen_u8,  cv2.COLOR_RGB2GRAY)
            transforms.append(compute_transform(gen_gray, orig_gray, bg_mask,
                                                orig_feature_mask, max_features,
                                                match_count, align_model))

        solved = sum(t is not None for t in transforms)
        if solved < n:
            print(f"[PipoAlignComposite] {n - solved}/{n} frames had too few matches")

        if lock_alignment:
            transforms = lock_transforms(transforms, (ow, oh))
        elif smoothing > 1:
            transforms = smooth_transforms(transforms, smoothing, (ow, oh))

        # Pass 2 — warp and composite with the conditioned transforms.
        for i in range(n):
            orig_u8, gen_u8, mask_f, _ = self._frame_arrays(
                original_frames, generated_frames, pipo_mask, exclude_mask, i, oh, ow)
            H = transforms[i]

            # Use boiled_frames as composite source if provided, else fall back to generated_frames
            if boiled_frames is not None and i < boiled_frames.shape[0]:
                src_u8 = to_u8(boiled_frames[i])
                bh, bw = src_u8.shape[:2]
                if (bh, bw) != (oh, ow):
                    src_u8 = cv2.resize(src_u8, (ow, oh), interpolation=cv2.INTER_LINEAR)
            else:
                src_u8 = gen_u8

            if H is not None:
                warped_gen  = cv2.warpPerspective(src_u8, H, (ow, oh), flags=cv2.INTER_LINEAR)
                warped_mask = cv2.warpPerspective(mask_f, H, (ow, oh), flags=cv2.INTER_LINEAR)
            else:
                warped_gen  = src_u8
                warped_mask = mask_f

            if feather_px > 0:
                k = feather_px * 2 + 1
                warped_mask = cv2.GaussianBlur(warped_mask, (k, k), 0)

            # Repositioned pipo layer on its own, as RGBA (alpha = warped pipo mask)
            pipo_rgba = np.dstack([
                warped_gen,
                np.clip(warped_mask * 255, 0, 255).astype(np.uint8),
            ])
            pipo_layer_results.append(
                torch.from_numpy(pipo_rgba).float() / 255.0
            )

            # Composite pipo fill over original
            m = warped_mask[:, :, np.newaxis]
            composited = (warped_gen.astype(np.float32) * m
                          + orig_u8.astype(np.float32) * (1.0 - m))

            # Composite GP outline on top using same homography
            if has_gp and i < gp_frames.shape[0] and i < gp_mask.shape[0]:
                gp_u8  = to_u8(gp_frames[i])
                gp_f   = gp_mask[i].cpu().numpy().astype(np.float32)

                gh2, gw2 = gp_u8.shape[:2]
                if (gh2, gw2) != (oh, ow):
                    gp_u8 = cv2.resize(gp_u8, (ow, oh), interpolation=cv2.INTER_LINEAR)
                    gp_f  = cv2.resize(gp_f,  (ow, oh), interpolation=cv2.INTER_LINEAR)

                if H is not None:
                    warped_gp_u8  = cv2.warpPerspective(gp_u8, H, (ow, oh), flags=cv2.INTER_LINEAR)
                    warped_gp_mask = cv2.warpPerspective(gp_f,  H, (ow, oh), flags=cv2.INTER_LINEAR)
                else:
                    warped_gp_u8  = gp_u8
                    warped_gp_mask = gp_f

                if feather_px > 0:
                    warped_gp_mask = cv2.GaussianBlur(warped_gp_mask, (k, k), 0)

                gm = warped_gp_mask[:, :, np.newaxis]
                composited = (warped_gp_u8.astype(np.float32) * gm
                              + composited * (1.0 - gm))

            results.append(torch.from_numpy(
                np.clip(composited, 0, 255).astype(np.uint8)
            ).float() / 255.0)

        return (torch.stack(results), torch.stack(pipo_layer_results))
