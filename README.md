# fae Custom ComfyUI Nodes

A personal collection of ComfyUI nodes. All nodes are under `fae/` in the node menu.

---

## Pink Extractor
**Category:** `fae/image`

Detects and extracts a pink-colored subject from an image or video batch using HSV color thresholding — tuned by default for Pipo's pink (`#ED8AB6`).

**Inputs**
| Input | Description |
|---|---|
| `images` | IMAGE batch |
| `hue_center` | Center of the hue range to detect (0–1, default 0.93) |
| `hue_tolerance` | Hue range width around the center |
| `sat_min / sat_max` | Saturation bounds |
| `val_min / val_max` | Value (brightness) bounds |
| `erosion` | Noise removal passes (0–4) |
| `outline_width` | Width of the drawn outline in pixels |
| `sharpness` | Sobel edge sharpness for the outline |
| `outline_color` | Hex color for the outline stroke |

**Outputs**
- `mask` — Binary mask, white where pink was detected
- `outline` — Clean outline drawing on white background

---

## Blender GP Trace
**Category:** `fae/blender`

Runs Blender silently in the background, traces an image or video batch to Grease Pencil using Blender's built-in potrace integration, renders the result as RGBA frames, and saves a `.blend` file.

Feed a **dark-on-white** image (invert a mask before connecting if needed — potrace traces dark areas).

**Inputs**
| Input | Description |
|---|---|
| `images` | IMAGE batch to trace |
| `fps` | Frame rate for video input/output |
| `threshold` | Brightness cutoff for tracing (0–1) |
| `fill_color` | Hex color for the GP fill (sRGB, converted to linear internally) |
| `stroke_color` | Hex color for the GP stroke outline |
| `stroke_radius` | Thickness of the GP strokes |
| `blend_save_path` | Directory where the `.blend` file is saved (timestamped) |

**Outputs**
- `images` — Rendered RGB frames
- `mask` — Alpha channel from the render, usable as a mask downstream

---

## Video Change Framerate
**Category:** `fae/video`

Changes the framerate of an IMAGE batch using FFmpeg's `fps` filter. Resamples frames to match the new rate — useful for slowing down, speeding up, or standardizing fps before export.

**Inputs**
| Input | Description |
|---|---|
| `images` | IMAGE batch |
| `input_fps` | Original frame rate of the batch |
| `output_fps` | Target frame rate |

**Outputs**
- `images` — Resampled IMAGE batch
- `fps` — The output fps value (can be wired directly into VHS Video Combine)

---

## iPhone Video Frame
**Category:** `fae/video`

Extracts a single frame from an IMAGE batch by index, with optional HLG → SDR color correction for iPhone HDR footage (BT.2100 HLG → BT.709).

**Inputs**
| Input | Description |
|---|---|
| `images` | IMAGE batch |
| `frame_number` | Index of the frame to extract |
| `apply_hlg_correction` | Apply HLG HDR → SDR tone mapping |

**Outputs**
- `image` — Single extracted frame

---

## iPhone HLG Correct
**Category:** `fae/video`

Applies HLG HDR → SDR color correction to an entire IMAGE batch. Converts iPhone HLG-encoded BT.2100 footage to standard BT.709 SDR for use in SDR workflows.

**Inputs**
| Input | Description |
|---|---|
| `images` | IMAGE batch |
| `apply_hlg_correction` | Toggle correction on/off |

**Outputs**
- `images` — Color-corrected IMAGE batch

---

## Pipo Align Composite
**Category:** `fae/video`

Aligns a Seedance-generated video to the original source video using SIFT feature matching, then composites a segmented Pipo mask over the original frames. Fixes the coordinate-space offset that occurs when Seedance shifts or scales the scene relative to the reference footage.

**Inputs**
| Input | Description |
|---|---|
| `original_frames` | Source video frames (the real background) |
| `generated_frames` | Seedance-generated frames containing Pipo |
| `pipo_mask` | Segmentation mask isolating Pipo (from SAM2/SAM3) |
| `max_features` | Max SIFT features to detect |
| `match_count` | Number of feature matches to use for homography |
| `feather_px` | Edge feathering on the mask for a smoother composite |

**Outputs**
- `composited` — Pipo composited onto the original background, per frame
