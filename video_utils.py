import os
import subprocess
import tempfile
import numpy as np
import torch
from PIL import Image

FFMPEG = "/opt/homebrew/bin/ffmpeg"


class VideoChangeFramerateNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "input_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01}),
                "output_fps": ("FLOAT", {"default": 12.0, "min": 1.0, "max": 240.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("images", "fps")
    FUNCTION = "change_fps"
    CATEGORY = "fae/video"

    def change_fps(self, images, input_fps, output_fps):
        n, h, w, c = images.shape

        # Write input frames to a temp video
        with tempfile.NamedTemporaryFile(suffix="_in.mp4", delete=False) as f:
            in_path = f.name
        with tempfile.NamedTemporaryFile(suffix="_out.mp4", delete=False) as f:
            out_path = f.name

        try:
            # Encode input frames → temp video at input_fps
            frames_np = (images.cpu().numpy() * 255).astype(np.uint8)
            proc = subprocess.Popen(
                [FFMPEG, "-y",
                 "-f", "rawvideo", "-vcodec", "rawvideo",
                 "-s", f"{w}x{h}", "-pix_fmt", "rgb24",
                 "-r", str(input_fps),
                 "-i", "pipe:0",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 in_path],
                stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
            proc.stdin.write(frames_np.tobytes())
            proc.stdin.close()
            proc.wait()

            # Re-encode at output_fps using ffmpeg fps filter
            subprocess.run(
                [FFMPEG, "-y",
                 "-i", in_path,
                 "-vf", f"fps={output_fps}",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 out_path],
                check=True, stderr=subprocess.DEVNULL,
            )

            # Read output frames back
            result = subprocess.run(
                [FFMPEG, "-i", out_path,
                 "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                capture_output=True, check=True,
            )
            raw = np.frombuffer(result.stdout, dtype=np.uint8)
            out_n = len(raw) // (h * w * 3)
            out_frames = raw[: out_n * h * w * 3].reshape(out_n, h, w, 3)
            out_tensor = torch.from_numpy(out_frames).float() / 255.0

        finally:
            for p in (in_path, out_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass

        return (out_tensor, float(output_fps))
