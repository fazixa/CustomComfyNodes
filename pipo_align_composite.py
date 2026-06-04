import cv2
import numpy as np
import torch


def _to_u8(t):
    """(H, W, 3) float32 [0,1] → uint8 RGB"""
    return (t.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


def _homography(gen_gray, orig_gray, bg_mask, max_features, match_count):
    """
    Find homography mapping gen → orig coordinate space.
    bg_mask: uint8 (H,W), 255 = regions to use for feature detection (background only).
    Returns H matrix or None if not enough matches.
    """
    detector = cv2.SIFT_create(nfeatures=max_features)
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    kp1, des1 = detector.detectAndCompute(gen_gray, bg_mask)
    kp2, des2 = detector.detectAndCompute(orig_gray, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None

    matches = sorted(matcher.match(des1, des2), key=lambda m: m.distance)[:match_count]
    if len(matches) < 4:
        return None

    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    return H


class PipoAlignCompositeNode:
    """
    Corrects the coordinate-space offset between a Seedance-generated video
    and the original source video before compositing a segmented Pipo mask.

    Workflow:
      SAM2 → pipo_mask  ──┐
      generated_frames ───┼─→ [this node] → composited IMAGE batch
      original_frames  ───┘

    Per frame:
      1. Detect SIFT features on the background of the generated frame
         (Pipo region is masked out so he doesn't confuse the matcher).
      2. Match against the original frame and compute a homography
         (generated → original coordinate space).
      3. Warp both the generated pixels and the mask using that homography.
      4. Composite warped Pipo onto the original with optional edge feathering.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_frames": ("IMAGE",),
                "generated_frames": ("IMAGE",),
                "pipo_mask":        ("MASK",),
            },
            "optional": {
                "max_features": ("INT",   {"default": 2000, "min": 500,  "max": 5000, "step": 100}),
                "match_count":  ("INT",   {"default": 50,   "min": 10,   "max": 200,  "step": 10}),
                "feather_px":   ("INT",   {"default": 3,    "min": 0,    "max": 30,   "step": 1}),
            },
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("composited",)
    FUNCTION      = "composite"
    CATEGORY      = "fae/video"

    def composite(self, original_frames, generated_frames, pipo_mask,
                  max_features=2000, match_count=50, feather_px=3):

        n  = min(original_frames.shape[0], generated_frames.shape[0], pipo_mask.shape[0])
        oh, ow = original_frames.shape[1], original_frames.shape[2]
        results = []

        for i in range(n):
            orig_u8 = _to_u8(original_frames[i])
            gen_u8  = _to_u8(generated_frames[i])
            mask_f  = pipo_mask[i].cpu().numpy().astype(np.float32)  # (H, W) [0,1]

            # Resize generated + mask to match original resolution if needed
            gh, gw = gen_u8.shape[:2]
            if (gh, gw) != (oh, ow):
                gen_u8 = cv2.resize(gen_u8, (ow, oh), interpolation=cv2.INTER_LINEAR)
                mask_f = cv2.resize(mask_f, (ow, oh), interpolation=cv2.INTER_LINEAR)

            # Detect features only in the background (exclude Pipo)
            bg_mask  = ((mask_f < 0.5) * 255).astype(np.uint8)
            orig_gray = cv2.cvtColor(orig_u8, cv2.COLOR_RGB2GRAY)
            gen_gray  = cv2.cvtColor(gen_u8,  cv2.COLOR_RGB2GRAY)

            H = _homography(gen_gray, orig_gray, bg_mask, max_features, match_count)

            if H is not None:
                warped_gen  = cv2.warpPerspective(gen_u8, H, (ow, oh), flags=cv2.INTER_LINEAR)
                warped_mask = cv2.warpPerspective(mask_f, H, (ow, oh), flags=cv2.INTER_LINEAR)
            else:
                print(f"[PipoAlignComposite] frame {i}: not enough matches, skipping alignment")
                warped_gen  = gen_u8
                warped_mask = mask_f

            # Feather mask edges
            if feather_px > 0:
                k = feather_px * 2 + 1
                warped_mask = cv2.GaussianBlur(warped_mask, (k, k), 0)

            # Alpha composite: Pipo over original
            m = warped_mask[:, :, np.newaxis]
            composited = (warped_gen.astype(np.float32) * m
                          + orig_u8.astype(np.float32) * (1.0 - m))
            composited = np.clip(composited, 0, 255).astype(np.uint8)

            results.append(torch.from_numpy(composited).float() / 255.0)

        return (torch.stack(results),)
