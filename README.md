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
| `outline_width` | Outline width in pixels (when `dynamic_outline` is on, this is the width at Pipo's largest/closest frame) |
| `sharpness` | Sobel edge sharpness for the outline |
| `outline_color` | Hex color for the outline stroke |
| `dynamic_outline` | Scale outline width by Pipo's mask size — thinner when farther from camera |
| `min_scale` | *(shown when `dynamic_outline` is on)* Smallest fraction of `outline_width` used at Pipo's smallest size |
| `smoothing` | *(shown when `dynamic_outline` is on)* Temporal smoothing on the size-based scale, 0–0.95 |

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

Aligns a Seedance-generated video to the original source video using SIFT feature matching, then composites Pipo over the original frames. Computes the homography once from `generated_frames` → `original_frames` and optionally reuses it for a second GP outline layer — so fill and outline are aligned in a single pass without running feature matching twice.

**Inputs**
| Input | Description |
|---|---|
| `original_frames` | Source video frames (the real background) |
| `generated_frames` | Seedance-generated frames containing Pipo (used for alignment) |
| `pipo_mask` | Segmentation mask isolating Pipo (from SAM2/SAM3) |
| `boiled_frames` | *(optional)* Boil Effect output — used as composite source instead of `generated_frames`. `generated_frames` still used for SIFT alignment |
| `gp_frames` | *(optional)* Blender GP Trace rendered frames (outline layer) |
| `gp_mask` | *(optional)* Alpha mask from Blender GP Trace |
| `max_features` | Max SIFT features to detect |
| `match_count` | Number of feature matches to use for homography |
| `feather_px` | Edge feathering on the mask, shared by both outputs |

**Outputs**
- `composited` — Pipo fill (and GP outline if connected) composited onto the original background, per frame
- `pipo_layer` — Pipo's source layer (`boiled_frames`/`generated_frames`) repositioned by the same SIFT homography, as RGBA with alpha = the warped & feathered `pipo_mask`. Connect to Save Image to export transparent PNGs

---

## Pipo Align Restore
**Category:** `fae/video`

The opposite of Pipo Align Composite: instead of placing generated content onto the original, it patches part of the **generated** video back with content from the **original** video. Computes the homography from `original_frames` → `generated_frames` using SIFT, warps the original into the generated frame's perspective, then composites it into the region defined by `mask`.

`mask` is excluded from feature matching on `generated_frames` (its content doesn't correspond to the original there, by definition). `exclude_mask` is excluded from feature matching on `original_frames`, for any regions in the source video that shouldn't be used for alignment.

**Inputs**
| Input | Description |
|---|---|
| `original_frames` | Source video frames to restore from |
| `generated_frames` | Generated frames to patch (used for alignment) |
| `mask` | Region of `generated_frames` to replace with aligned original content. Excluded from SIFT features on `generated_frames` |
| `exclude_mask` | *(optional)* Region of `original_frames` excluded from SIFT features |
| `max_features` | Max SIFT features to detect |
| `match_count` | Number of feature matches to use for homography |
| `feather_px` | Edge feathering on the mask |

**Outputs**
- `composited` — `generated_frames` with `mask` region replaced by aligned `original_frames` content, per frame
- `restored_layer` — The aligned original content on its own, as RGBA with alpha = the warped & feathered `mask`. Connect to Save Image to export transparent PNGs
