"""API routes backing the SAM2 click-to-annotate widget.

The widget needs masks and video frames while you're editing the graph, long
before the node executes, so it talks to these endpoints directly rather than
going through ComfyUI's execution flow.
"""

import os
import base64
import logging
from io import BytesIO

import numpy as np
from PIL import Image

from .sam2_node import (
    run_sam2,
    get_video_frame_image,
    get_video_info,
    get_last_video_path,
)

logger = logging.getLogger(__name__)


def register_routes():
    import server
    import folder_paths
    from aiohttp import web

    def resolve_source(filename="", subfolder="", node_id=None):
        """Locate the video a widget is asking about.

        Either an input-directory filename (the local picker, or an upstream
        loader the frontend could read a filename off), or a node id we look up
        in our own cache of last-executed paths. The client never supplies a
        path directly, so there's nothing to traverse out of.
        """
        if node_id:
            path = get_last_video_path(node_id)
            return path if path and os.path.exists(path) else None
        if filename:
            path = folder_paths.get_annotated_filepath(
                f"{subfolder}/{filename}" if subfolder else filename
            )
            return path if os.path.exists(path) else None
        return None

    def png_response(mask, score):
        buf = BytesIO()
        Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(buf, format="PNG")
        return web.json_response(
            {"mask": base64.b64encode(buf.getvalue()).decode(), "score": score}
        )

    @server.PromptServer.instance.routes.post("/fae/sam2/predict")
    async def sam2_predict(request):
        try:
            data = await request.json()
            filename = data.get("filename", "")
            subfolder = data.get("subfolder", "")
            points = data.get("points", [])
            labels = data.get("labels", [])

            if not filename:
                return web.json_response({"error": "missing filename"}, status=400)
            if not points:
                return web.json_response({"mask": None})

            image_path = folder_paths.get_annotated_filepath(
                f"{subfolder}/{filename}" if subfolder else filename
            )
            if not os.path.exists(image_path):
                return web.json_response({"error": "image not found"}, status=404)

            img_np = np.array(Image.open(image_path).convert("RGB"))
            return png_response(*run_sam2(img_np, points, labels))

        except Exception as e:
            logger.exception("[sam2] predict error")
            return web.json_response({"error": str(e)}, status=500)

    @server.PromptServer.instance.routes.get("/fae/sam2/video_info")
    async def sam2_video_info(request):
        try:
            video_path = resolve_source(
                request.rel_url.query.get("filename", ""),
                request.rel_url.query.get("subfolder", ""),
                request.rel_url.query.get("node_id"),
            )
            if not video_path:
                return web.json_response({"error": "video not available"}, status=404)

            return web.json_response(get_video_info(video_path))

        except Exception as e:
            logger.exception("[sam2] video_info error")
            return web.json_response({"error": str(e)}, status=500)

    @server.PromptServer.instance.routes.get("/fae/sam2/video_frame")
    async def sam2_video_frame(request):
        try:
            frame_index = int(request.rel_url.query.get("frame", "0"))
            video_path = resolve_source(
                request.rel_url.query.get("filename", ""),
                request.rel_url.query.get("subfolder", ""),
                request.rel_url.query.get("node_id"),
            )
            if not video_path:
                return web.json_response({"error": "video not available"}, status=404)

            img = get_video_frame_image(video_path, frame_index)
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=90)
            return web.json_response({"frame": base64.b64encode(buf.getvalue()).decode(),
                                      "width": img.width, "height": img.height})

        except Exception as e:
            logger.exception("[sam2] video_frame error")
            return web.json_response({"error": str(e)}, status=500)

    @server.PromptServer.instance.routes.post("/fae/sam2/video_predict")
    async def sam2_video_predict(request):
        """Preview: run the image predictor on a single video frame."""
        try:
            data = await request.json()
            frame_index = int(data.get("frame", 0))
            points = data.get("points", [])
            labels = data.get("labels", [])

            if not points:
                return web.json_response({"mask": None})

            video_path = resolve_source(
                data.get("filename", ""), data.get("subfolder", ""), data.get("node_id")
            )
            if not video_path:
                return web.json_response({"mask": None})

            img_np = np.array(get_video_frame_image(video_path, frame_index))
            return png_response(*run_sam2(img_np, points, labels))

        except Exception as e:
            logger.exception("[sam2] video_predict error")
            return web.json_response({"error": str(e)}, status=500)

    logger.info("[sam2] API routes registered")


try:
    register_routes()
except Exception as e:
    logger.warning(f"[sam2] could not register API routes: {e}")
