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
| `despeckle` | Discard detected islands smaller than this many pixels (0 = off). Stray thresholded pixels are invisible in the mask, but the outline pass rings *every* island, so one pixel becomes a box of ink several times its size — the little squares that show up along arms and edges. `erosion` can't do this job; it thins the whole silhouette instead of removing islands |
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

---

### Alignment settings (both Pipo Align nodes)

| Input | Description |
|---|---|
| `align_model` | How much motion the fit is allowed. `similarity` (default) = pan, rotate, uniform zoom. `affine` adds shear and non-uniform stretch. `homography` adds perspective |
| `smoothing` | Frames averaged over when smoothing the alignment across the clip. `0`/`1` disables it |
| `lock_alignment` | Collapse to one transform for the whole clip |
| `max_features` | SIFT keypoints kept, strongest first |
| `match_count` | Cap on matches kept after the ratio test. `0` = keep all |

**Which model for which footage.** Pick the least freedom that covers the shot — every unused degree of freedom gets spent fitting noise, which comes out as per-frame wobble.

- **`similarity`** — the right default. Correct whenever the two shots differ by a pan, a roll, or a zoom: locked-off originals, tripod shots, handheld with no real parallax, and anything generated from a still.
- **`affine`** — when the generated video is stretched relative to the original, e.g. a non-uniform resize or an aspect change somewhere in the chain.
- **`homography`** — only when the camera genuinely moves through the scene and flat surfaces visibly keystone: a dolly past a wall, a strong angle change, a tilt across a floor. Needs well-spread matches across depth to be worth the extra freedom.

**Smoothing.** Raise it when the background is static or barely moving and the composite still swims. Lower it (or set `0`) when the camera moves fast, since a wide window makes the alignment lag real motion. `lock_alignment` is the end of that scale — use it only when *neither* side moves, such as a still photo as the original; it removes alignment flicker completely, and will drift visibly if the footage does move.

Measured on a 97-frame shot with a still photo as the original, so the correct transform is identical on every frame and all movement is error:

| Setting | Frame-to-frame flicker | Worst |
|---|---|---|
| `homography`, `match_count=50`, no smoothing (old behaviour) | 2.55 px | 8.10 px |
| `similarity`, ratio test, no smoothing | 0.50 px | 1.57 px |
| `similarity`, ratio test, `smoothing=5` | 0.14 px | 0.38 px |
| `similarity`, ratio test, `lock_alignment` | 0.00 px | 0.00 px |

Note `max_features` above ~2000 makes alignment *less* steady, not more: SIFT keeps keypoints strongest-first, so raising the cap admits weak, poorly localised ones. `match_count` is now a cap on matches that already passed a ratio test, and `0` (keep all) is normally best.

---

## SAM2 Segment
**Category:** `fae/segmentation`

Segments a still image with SAM2. The node draws the image on a canvas widget — left-click to add a foreground point, right-click for background — and previews the resulting mask live as you click. `Backspace` undoes the last point, `Esc` clears them.

Requires a local SAM2 checkout at `/Users/fae/Documents/Projects/sam2` (see `SAM2_ROOT` in `sam2_node.py`) with the `sam2.1_hiera_tiny` checkpoint downloaded. Runs on CUDA, MPS, or CPU, whichever is available.

**Inputs**
| Input | Description |
|---|---|
| `image` | Image from the input directory, with upload |
| `points_json` | Click points, written by the canvas widget |
| `labels_json` | Per-point label (1 = foreground, 0 = background), written by the widget |

**Outputs**
- `image` — The source image
- `mask` — Binary mask for the highest-scoring SAM2 proposal
- `masked_image` — Source image with the mask applied

---

## SAM2 Segment Video
**Category:** `fae/segmentation`

Tracks a subject through a video with SAM2's video predictor. Scrub to a frame, click the subject, and the node propagates that selection across every frame when the graph runs.

Propagation runs in both directions, so you can annotate on whichever frame shows the subject most clearly and the whole clip is covered — not just the frames after it.

Takes a `VIDEO` input, so it can sit downstream of Load Video or of a generator like Seedance 2.0 Generate. When nothing is connected it falls back to the `video_file` picker.

Frame previews resolve in two ways: if the chain traces back to a node that names a file, the widget reads frames immediately; if the video doesn't exist yet (generated video), queue the graph once and the widget previews from the path the node resolved on that run. Note that the click preview runs the single-frame *image* predictor — propagation across frames happens when the graph executes.

**⟳ Load video** pulls the incoming video in. With the `video` socket connected it queues this node the way Save Video's run button does — partial execution runs everything the input depends on, so a generator upstream actually produces its video — then previews frames from what this node resolved. With no socket connected it just reads the picked file.

Because previews after a run come from the resolved video rather than a filename read off an upstream loader, they reflect anything done to the video in between, such as a Video Slice trim.

**Inputs**
| Input | Description |
|---|---|
| `video` | *(optional)* `VIDEO` input to segment |
| `video_file` | *(optional)* Video from the input directory, used when `video` isn't connected |
| `annotate_frame` | Frame the click points refer to, set by the scrubber |
| `points_json` | Click points, written by the canvas widget |
| `labels_json` | Per-point label (1 = foreground, 0 = background), written by the widget |

**Outputs**
- `frames` — Every frame of the video as an image batch
- `masks` — Propagated mask per frame
- `masked_frames` — Frames with their mask applied
