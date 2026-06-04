import os
import json
import subprocess
import tempfile
import shutil
import time
import logging
import numpy as np
import torch
from PIL import Image

BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"

BLENDER_SCRIPT = '''
import bpy
import sys
import json
import os

argv  = sys.argv
args  = argv[argv.index("--") + 1:]
frames_dir    = args[0]
output_dir    = args[1]
blend_file    = args[2]
img_w         = int(args[3])
img_h         = int(args[4])
fps           = float(args[5])
n_frames      = int(args[6])
threshold     = float(args[7])
fill_color    = json.loads(args[8])
stroke_radius = float(args[9])
stroke_color  = json.loads(args[10])

# ── Scene ─────────────────────────────────────────────────────────────────────
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

scene = bpy.context.scene
scene.render.resolution_x = img_w
scene.render.resolution_y = img_h
scene.render.fps           = max(1, round(fps))
scene.frame_start          = 1
scene.frame_end            = n_frames
scene.render.film_transparent = True
scene.render.image_settings.color_mode = "RGBA"
scene.render.image_settings.file_format = "PNG"
scene.render.engine = "BLENDER_WORKBENCH"
scene.display.shading.light = "FLAT"
scene.display.shading.color_type = "MATERIAL"

# ── Create one image empty, reuse for all frames ──────────────────────────────
bpy.ops.object.empty_add(type="IMAGE", location=(0, 0, 0))
empty = bpy.context.active_object
empty.name = "ImageRef"
empty.empty_display_size = 1.0
empty.empty_image_offset = (-0.5, -0.5)
empty.hide_render = True

frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))

gp_obj = None

for fi, fname in enumerate(frame_files):
    bpy_frame = fi + 1
    scene.frame_set(bpy_frame)

    img = bpy.data.images.load(os.path.join(frames_dir, fname))
    empty.data = img

    if gp_obj is None:
        bpy.context.view_layer.objects.active = empty
        empty.select_set(True)
        bpy.ops.grease_pencil.trace_image(target="NEW", threshold=threshold, mode="SINGLE")
        gp_obj = next(o for o in bpy.data.objects if o.type == "GREASEPENCIL")
    else:
        bpy.ops.object.select_all(action="DESELECT")
        empty.select_set(True)
        gp_obj.select_set(True)
        bpy.context.view_layer.objects.active = empty
        bpy.ops.grease_pencil.trace_image(
            target="SELECTED", threshold=threshold, mode="SINGLE", use_current_frame=True
        )
    print(f"[fae_blender] traced frame {bpy_frame}/{n_frames}", flush=True)
    sys.stdout.flush()

# ── Configure GP ─────────────────────────────────────────────────────────────
gp_obj = next(o for o in bpy.data.objects if o.type == "GREASEPENCIL")
gp_obj.use_grease_pencil_lights = False

for layer in gp_obj.data.layers:
    for frame in layer.frames:
        for stroke in frame.drawing.strokes:
            for pt in stroke.points:
                pt.radius = stroke_radius

for mat in gp_obj.data.materials:
    if mat and mat.is_grease_pencil:
        gp_mat = mat.grease_pencil
        gp_mat.fill_color  = tuple(fill_color)
        gp_mat.show_fill   = True
        gp_mat.color       = tuple(stroke_color)
        gp_mat.show_stroke = True

# ── Camera ────────────────────────────────────────────────────────────────────
cam_data = bpy.data.cameras.new("Camera")
cam_data.type        = "ORTHO"
cam_data.ortho_scale = 1.0
cam_obj = bpy.data.objects.new("Camera", cam_data)
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj
cam_obj.location       = (0, 0, 10)
cam_obj.rotation_euler = (0, 0, 0)

# ── Save blend ────────────────────────────────────────────────────────────────
scene.render.filepath = os.path.join(output_dir, "frame_")
print("[fae_blender] saving blend...", flush=True)
bpy.ops.wm.save_as_mainfile(filepath=blend_file)
print("[fae_blender] setup done", flush=True)
'''


def _srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

def _hex_to_rgba(hex_str):
    h = hex_str.strip().lstrip("#").ljust(6, "0")
    r = _srgb_to_linear(int(h[0:2], 16) / 255.0)
    g = _srgb_to_linear(int(h[2:4], 16) / 255.0)
    b = _srgb_to_linear(int(h[4:6], 16) / 255.0)
    a = int(h[6:8], 16) / 255.0 if len(h) >= 8 else 1.0
    return [r, g, b, a]


class BlenderGPTraceNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images":          ("IMAGE",),
                "fps":             ("FLOAT",  {"default": 12.0, "min": 1.0, "max": 120.0, "step": 0.5}),
                "threshold":       ("FLOAT",  {"default": 0.5,  "min": 0.0, "max": 1.0,   "step": 0.01}),
                "fill_color":      ("STRING", {"default": "#601A3B"}),
                "stroke_color":    ("STRING", {"default": "#FFFFFF"}),
                "stroke_radius":   ("FLOAT",  {"default": 0.05, "min": 0.001, "max": 1.0, "step": 0.001}),
                "blend_save_path": ("STRING", {"default": os.path.expanduser("~/Documents/ComfyUI_Blender")}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "mask")
    FUNCTION = "trace"
    CATEGORY = "fae/blender"

    def trace(self, images, fps, threshold, fill_color, stroke_color, stroke_radius, blend_save_path):
        n, h, w, c = images.shape
        imgs_np = images.cpu().numpy()
        _log = logging.getLogger(__name__)

        tmp         = tempfile.mkdtemp(prefix="fae_blender_")
        frames_dir  = os.path.join(tmp, "frames")
        output_dir  = os.path.join(tmp, "output")
        script_path = os.path.join(tmp, "trace.py")
        os.makedirs(frames_dir)
        os.makedirs(output_dir)

        try:
            for i, img in enumerate(imgs_np):
                Image.fromarray((img[..., :3] * 255).astype(np.uint8)).save(
                    os.path.join(frames_dir, f"frame_{i+1:06d}.png")
                )

            fill_rgba   = _hex_to_rgba(fill_color)
            stroke_rgba = _hex_to_rgba(stroke_color)
            os.makedirs(blend_save_path, exist_ok=True)
            blend_file = os.path.join(blend_save_path, f"gp_trace_{int(time.time())}.blend")

            with open(script_path, "w") as f:
                f.write(BLENDER_SCRIPT)

            def _run(cmd):
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                lines = []
                for line in proc.stdout:
                    line = line.rstrip()
                    lines.append(line)
                    if line.startswith("[fae_blender]"):
                        _log.info(line)
                proc.wait(timeout=600)
                return proc.returncode, "\n".join(lines[-60:])

            # Step 1: build scene and save .blend
            setup_cmd = [
                BLENDER, "--background", "--factory-startup",
                "--python", script_path,
                "--",
                frames_dir, output_dir, blend_file,
                str(w), str(h), str(fps), str(n),
                str(threshold),
                json.dumps(fill_rgba),
                str(stroke_radius),
                json.dumps(stroke_rgba),
            ]
            rc, log = _run(setup_cmd)
            if rc != 0:
                raise RuntimeError(f"Blender setup failed:\n{log}")

            # Step 2: render via CLI
            render_cmd = [
                BLENDER, "--background", blend_file,
                "--render-output", os.path.join(output_dir, "frame_"),
                "--render-format", "PNG",
                "--render-anim",
            ]
            rc2, log2 = _run(render_cmd)
            if rc2 != 0:
                raise RuntimeError(f"Blender render failed:\n{log2}")

            rendered = sorted(f for f in os.listdir(output_dir) if f.endswith(".png"))
            if not rendered:
                raise RuntimeError(f"Blender produced no output frames.\n{log2}")

            out_frames, out_masks = [], []
            for fname in rendered:
                rgba = np.array(Image.open(os.path.join(output_dir, fname)).convert("RGBA"))
                out_frames.append(torch.from_numpy(rgba[..., :3]).float() / 255.0)
                out_masks.append(torch.from_numpy(rgba[..., 3]).float() / 255.0)

            return (torch.stack(out_frames), torch.stack(out_masks))

        finally:
            shutil.rmtree(tmp, ignore_errors=True)
