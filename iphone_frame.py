import numpy as np
import torch


# ---------- HLG → SDR colour science ----------

def _hlg_inverse_oetf(e_prime):
    """HLG encoded signal → scene-linear light (ITU-R BT.2100-2)."""
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    return np.where(
        e_prime <= 0.5,
        e_prime ** 2 / 3.0,
        (np.exp((e_prime - c) / a) + b) / 12.0,
    )


_BT2020_TO_BT709 = np.array([
    [ 1.6605, -0.5876, -0.0728],
    [-0.1246,  1.1329, -0.0083],
    [-0.0182, -0.1006,  1.1187],
], dtype=np.float32)


def _bt709_oetf(linear):
    """Linear display light → BT.709 gamma-encoded signal."""
    return np.where(
        linear < 0.018,
        4.5 * linear,
        1.0993 * np.power(np.maximum(linear, 0.0), 0.45) - 0.0993,
    )


def _hlg_to_sdr(img_float, system_gamma=1.2):
    """
    Full HLG HDR → SDR pipeline.
    img_float: (H, W, 3) float32 in [0, 1], HLG-encoded BT.2020.
    Returns: (H, W, 3) float32 in [0, 1], BT.709 SDR.
    """
    scene_linear = _hlg_inverse_oetf(img_float)
    # System gamma maps scene-linear → display-linear
    display_linear_2020 = np.power(np.maximum(scene_linear, 0.0), 1.0 / system_gamma)
    # BT.2020 primaries → BT.709 primaries
    h, w = display_linear_2020.shape[:2]
    display_linear_709 = (display_linear_2020.reshape(-1, 3) @ _BT2020_TO_BT709.T).reshape(h, w, 3)
    display_linear_709 = np.clip(display_linear_709, 0.0, 1.0)
    return np.clip(_bt709_oetf(display_linear_709), 0.0, 1.0)


def _hlg_to_sdr_batch(batch, system_gamma=1.2):
    """
    (N, H, W, 3) float32 [0,1] HLG-encoded → BT.709 SDR in one pass.
    """
    scene = _hlg_inverse_oetf(batch)
    display_2020 = np.power(np.maximum(scene, 0.0), 1.0 / system_gamma)
    n, h, w = display_2020.shape[:3]
    display_709 = np.clip((display_2020.reshape(-1, 3) @ _BT2020_TO_BT709.T).reshape(n, h, w, 3), 0.0, 1.0)
    return np.clip(_bt709_oetf(display_709), 0.0, 1.0)


# ---------- ComfyUI nodes ----------

class iPhoneVideoFrameNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_number": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1}),
                "apply_hlg_correction": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "get_frame"
    CATEGORY = "fae/video"

    def get_frame(self, images, frame_number, apply_hlg_correction):
        idx = min(frame_number, images.shape[0] - 1)
        frame = images[idx:idx + 1]  # (1, H, W, 3)

        if apply_hlg_correction:
            arr = frame.cpu().numpy()[0]  # (H, W, 3) float32 [0,1]
            arr = _hlg_to_sdr(arr)
            frame = torch.from_numpy(arr).unsqueeze(0)

        return (frame,)


class iPhoneHLGCorrectNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "apply_hlg_correction": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "correct"
    CATEGORY = "fae/video"

    def correct(self, images, apply_hlg_correction):
        if not apply_hlg_correction:
            return (images,)

        arr = images.cpu().numpy()  # (N, H, W, 3) float32 [0,1]
        arr = _hlg_to_sdr_batch(arr)
        return (torch.from_numpy(arr),)
