import os
import tempfile
from io import BytesIO

import numpy as np
import replicate
from PIL import Image
from comfy_api.latest import InputImpl, VideoContainer

MODEL = "bytedance/seedance-2.0:0542b07b95add8fdc6d760bc76c0ab4304dd92260bcfa09acb4faa8601aadf66"

MAX_REFERENCE_IMAGES = 9
MAX_REFERENCE_VIDEOS = 3


# Returned as file objects (not base64 data URIs) so the replicate client uploads
# them via the Files API and passes a URL instead of inlining large payloads into
# the prediction request body, which can time out for video-sized data.
def _image_to_file(image_tensor, name):
    img = (image_tensor[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    buffer = BytesIO()
    Image.fromarray(img).save(buffer, format="PNG")
    buffer.seek(0)
    buffer.name = name
    return buffer


def _images_to_files(image_batch, limit):
    files = []
    for i in range(min(image_batch.shape[0], limit)):
        img = (image_batch[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        buffer = BytesIO()
        Image.fromarray(img).save(buffer, format="PNG")
        buffer.seek(0)
        buffer.name = f"reference_image_{i}.png"
        files.append(buffer)
    return files


def _video_to_file(video, name):
    buffer = BytesIO()
    video.save_to(buffer, format=VideoContainer.MP4)
    buffer.seek(0)
    buffer.name = name
    return buffer


class SeedanceGenerateNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "duration": ("INT", {"default": 5, "min": -1, "max": 15}),
                "resolution": (["480p", "720p", "1080p"], {"default": "720p"}),
                "aspect_ratio": (["16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "9:21", "adaptive"], {"default": "16:9"}),
                "generate_audio": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
            },
            "optional": {
                "image": ("IMAGE",),
                "last_frame_image": ("IMAGE",),
                "reference_images": ("IMAGE",),
                "reference_video_1": ("VIDEO",),
                "reference_video_2": ("VIDEO",),
                "reference_video_3": ("VIDEO",),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "generate"
    CATEGORY = "fae/replicate"

    def generate(self, prompt, duration, resolution, aspect_ratio, generate_audio, seed,
                  image=None, last_frame_image=None, reference_images=None,
                  reference_video_1=None, reference_video_2=None, reference_video_3=None):
        if not os.environ.get("REPLICATE_API_TOKEN"):
            raise RuntimeError("REPLICATE_API_TOKEN environment variable is not set.")

        payload = {
            "prompt": prompt,
            "duration": duration,
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "generate_audio": generate_audio,
            "seed": seed,
        }
        if image is not None:
            payload["image"] = _image_to_file(image, "image.png")
        if last_frame_image is not None:
            payload["last_frame_image"] = _image_to_file(last_frame_image, "last_frame_image.png")
        if reference_images is not None:
            payload["reference_images"] = _images_to_files(reference_images, MAX_REFERENCE_IMAGES)

        reference_videos = [v for v in (reference_video_1, reference_video_2, reference_video_3) if v is not None]
        if reference_videos:
            payload["reference_videos"] = [
                _video_to_file(v, f"reference_video_{i}.mp4")
                for i, v in enumerate(reference_videos[:MAX_REFERENCE_VIDEOS])
            ]

        output = replicate.run(MODEL, input=payload)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(output.read())
            video_path = f.name

        return (InputImpl.VideoFromFile(video_path),)


NODE_CLASS_MAPPINGS = {
    "SeedanceGenerate": SeedanceGenerateNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedanceGenerate": "Seedance 2.0 Generate",
}
