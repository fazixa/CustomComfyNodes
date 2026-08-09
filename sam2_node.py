import os
import sys
import json
import logging
import tempfile
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

SAM2_ROOT = "/Users/fae/Documents/Projects/sam2"
if SAM2_ROOT not in sys.path:
    sys.path.insert(0, SAM2_ROOT)

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

_predictor = None
_video_predictor = None
_frames_cache = {}  # video_key -> frames_dir


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_predictor():
    global _predictor
    if _predictor is not None:
        return _predictor
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    checkpoint = os.path.join(SAM2_ROOT, "checkpoints/sam2.1_hiera_tiny.pt")
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    device = _get_device()
    logger.info(f"[sam2] loading image predictor on {device}")
    _predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device=device))
    return _predictor


def get_video_predictor():
    global _video_predictor
    if _video_predictor is not None:
        return _video_predictor
    from sam2.build_sam import build_sam2_video_predictor
    checkpoint = os.path.join(SAM2_ROOT, "checkpoints/sam2.1_hiera_tiny.pt")
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    device = _get_device()
    logger.info(f"[sam2] loading video predictor on {device}")
    _video_predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
    return _video_predictor


def run_sam2(img_np: np.ndarray, points: list, labels: list):
    """Single-image SAM2 inference. Returns (mask HxW bool, score float)."""
    predictor = get_predictor()
    with torch.inference_mode():
        predictor.set_image(img_np)
        masks, scores, _ = predictor.predict(
            point_coords=np.array(points, dtype=np.float32),
            point_labels=np.array(labels, dtype=np.int32),
            multimask_output=True,
        )
    best = int(np.argmax(scores))
    return masks[best], float(scores[best])


def extract_video_frames(video_path: str) -> tuple[str, int, float]:
    """
    Extract all frames from video into a temp dir (cached).
    Returns (frames_dir, frame_count, fps).
    """
    stat = os.stat(video_path)
    cache_key = f"{video_path}:{stat.st_mtime}:{stat.st_size}"

    if cache_key in _frames_cache:
        frames_dir = _frames_cache[cache_key]
        if os.path.isdir(frames_dir):
            files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
            if files:
                return frames_dir, len(files), _frames_cache.get(cache_key + ":fps", 24.0)

    frames_dir = tempfile.mkdtemp(prefix="fae_sam2_video_")
    import av
    fps = 24.0
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.guessed_rate or 24.0)
        for i, frame in enumerate(container.decode(stream)):
            img = frame.to_image().convert("RGB")
            img.save(os.path.join(frames_dir, f"{i:06d}.jpg"), quality=95)

    files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    _frames_cache[cache_key] = frames_dir
    _frames_cache[cache_key + ":fps"] = fps
    logger.info(f"[sam2] extracted {len(files)} frames from {os.path.basename(video_path)}")
    return frames_dir, len(files), fps


def get_video_frame_image(video_path: str, frame_index: int) -> Image.Image:
    """Return a specific frame as PIL Image.

    Decoded with PyAV rather than the ffmpeg binary — ComfyUI Desktop is
    launched from Finder and doesn't inherit a shell PATH, so /opt/homebrew/bin
    isn't visible to it and shelling out to ffmpeg fails.
    """
    import av
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        fps = float(stream.average_rate or stream.guessed_rate or 24.0)

        if frame_index > 0 and stream.time_base:
            # Seek to the keyframe at or before the target, then decode forward.
            offset = int((frame_index / fps) / float(stream.time_base))
            try:
                container.seek(offset, stream=stream, backward=True)
            except Exception:
                container.seek(0)

        last = None
        for frame in container.decode(stream):
            last = frame
            if frame.pts is None:
                break
            if round(float(frame.pts * stream.time_base) * fps) >= frame_index:
                break
        if last is not None:
            return last.to_image().convert("RGB")

    return Image.new("RGB", (512, 512), 0)


def get_video_info(video_path: str) -> dict:
    """Return frame count, fps, width, height."""
    import av
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.guessed_rate or 24.0)
        w = stream.width
        h = stream.height
        duration = float(container.duration or 0) / 1_000_000
        frame_count = stream.frames or int(duration * fps)
    return {"frame_count": frame_count, "fps": fps, "width": w, "height": h}


import folder_paths

# node id -> the video path that node last resolved, so the widget can preview
# frames from a VIDEO input whose source isn't a file the frontend can name
# (generated video, trimmed clips). Populated on execution.
_last_video_paths = {}


def get_last_video_path(node_id: str):
    return _last_video_paths.get(str(node_id))


def resolve_video_input(video) -> str:
    """Resolve a ComfyUI VIDEO object to a readable path on disk.

    VideoFromFile hands back its own path when it has one, which we can decode
    directly. Anything else — BytesIO-backed, or carrying a trim window that the
    raw file wouldn't reflect — gets written out to a temp file first.
    """
    try:
        src = video.get_stream_source()
    except Exception:
        src = None

    start, duration = getattr(video, "get_active_trim_window", lambda: (0.0, 0.0))()
    if isinstance(src, str) and os.path.exists(src) and not (start or duration):
        return src

    fd, path = tempfile.mkstemp(prefix="fae_sam2_input_", suffix=".mp4")
    os.close(fd)
    video.save_to(path)
    return path


class SAM2SegmentNode:

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = sorted([
            f for f in os.listdir(input_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".gif"))
        ])
        return {
            "required": {
                "image": (files, {"image_upload": True}),
                "points_json": ("STRING", {"default": "[]"}),
                "labels_json": ("STRING", {"default": "[]"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("image", "mask", "masked_image")
    FUNCTION = "segment"
    CATEGORY = "fae/segmentation"

    def segment(self, image, points_json, labels_json):
        image_path = folder_paths.get_annotated_filepath(image)
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)

        img_tensor = torch.from_numpy(img_np).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # [1, H, W, C]

        points = json.loads(points_json)
        labels = json.loads(labels_json)

        if not points:
            h, w = img_np.shape[:2]
            return (img_tensor, torch.zeros(1, h, w, dtype=torch.float32), torch.zeros_like(img_tensor))

        mask, _ = run_sam2(img_np, points, labels)
        mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        masked_image = img_tensor * mask_tensor.unsqueeze(-1)
        return (img_tensor, mask_tensor, masked_image)

    @classmethod
    def IS_CHANGED(cls, image, points_json, labels_json):
        return f"{image}:{points_json}:{labels_json}"


class SAM2SegmentVideoNode:

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = sorted([
            f for f in os.listdir(input_dir)
            if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm"))
        ])
        return {
            "required": {
                "annotate_frame": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1}),
                "points_json": ("STRING", {"default": "[]"}),
                "labels_json": ("STRING", {"default": "[]"}),
            },
            "optional": {
                "video": ("VIDEO",),
                "video_file": (files, {"video_upload": True}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("frames", "masks", "masked_frames")
    FUNCTION = "segment_video"
    CATEGORY = "fae/segmentation"

    # Partial execution only targets output nodes (see validate_prompt), and the
    # widget's Load video button queues this node on its own to pull the
    # incoming video through whatever generates it.
    OUTPUT_NODE = True

    @classmethod
    def VALIDATE_INPUTS(cls, video_file=None):
        # Opting out of ComfyUI's combo check: video_file is only a fallback for
        # the VIDEO socket, so a stale or empty value must not block the graph.
        # segment_video decides whether there's a usable video and says so.
        return True

    def segment_video(self, annotate_frame, points_json, labels_json,
                      video=None, video_file=None, unique_id=None):
        video_path = None
        if video is not None:
            video_path = resolve_video_input(video)
        elif video_file:
            candidate = folder_paths.get_annotated_filepath(video_file)
            if os.path.exists(candidate):
                video_path = candidate

        if video_path is None:
            raise ValueError(
                "SAM2 Segment Video: connect a VIDEO input, or pick a video file "
                f"that exists in the input directory (got {video_file!r})."
            )

        # Let the widget preview frames from this source on the next interaction.
        if unique_id is not None:
            _last_video_paths[str(unique_id)] = video_path

        points = json.loads(points_json)
        labels = json.loads(labels_json)

        logger.info(f"[sam2] extracting frames from {os.path.basename(video_path)}")
        frames_dir, frame_count, fps = extract_video_frames(video_path)
        frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))

        # Load all frames as a batch tensor
        frame_tensors = []
        for f in frame_files:
            img_np = np.array(Image.open(os.path.join(frames_dir, f)).convert("RGB"))
            frame_tensors.append(torch.from_numpy(img_np).float() / 255.0)
        frames_batch = torch.stack(frame_tensors)  # [N, H, W, C]

        n, h, w, _ = frames_batch.shape

        if not points:
            return (frames_batch, torch.zeros(n, h, w), torch.zeros_like(frames_batch))

        predictor = get_video_predictor()
        offload = _get_device().type == "mps"
        inference_state = predictor.init_state(frames_dir, offload_video_to_cpu=offload)

        try:
            ann_frame = min(annotate_frame, frame_count - 1)
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame,
                obj_id=1,
                points=np.array(points, dtype=np.float32),
                labels=np.array(labels, dtype=np.int32),
                clear_old_points=True,
                # Points arrive as raw pixel coords in the frame's own resolution;
                # normalize_coords=True divides by video_W/H before SAM2 scales to
                # its 1024px input. With False, SAM2 skips the divide and scales the
                # raw pixels by 1024, putting every point far off-canvas.
                normalize_coords=True,
            )

            frame_masks = {}
            with torch.inference_mode():
                for frame_idx, obj_ids, masks in predictor.propagate_in_video(inference_state):
                    frame_masks[frame_idx] = (masks > 0)[0, 0].cpu().numpy().astype(np.float32)
        finally:
            predictor.reset_state(inference_state)

        mask_tensors = [
            torch.from_numpy(frame_masks.get(i, np.zeros((h, w), dtype=np.float32)))
            for i in range(n)
        ]
        masks_batch = torch.stack(mask_tensors)  # [N, H, W]
        masked_frames = frames_batch * masks_batch.unsqueeze(-1)

        return (frames_batch, masks_batch, masked_frames)

    @classmethod
    def IS_CHANGED(cls, annotate_frame, points_json, labels_json,
                   video=None, video_file=None, unique_id=None):
        # A connected VIDEO input is tracked by ComfyUI through its upstream node.
        return f"{video_file}:{annotate_frame}:{points_json}:{labels_json}"
