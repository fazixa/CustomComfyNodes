import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_filter1d


# ── Potrace helpers ───────────────────────────────────────────────────────────

def _to_pbm(arr_uint8: np.ndarray, path: str):
    """Write a binary numpy array (0/1) as P4 PBM."""
    h, w = arr_uint8.shape
    pad = (8 - w % 8) % 8
    padded = np.pad(arr_uint8, ((0, 0), (0, pad)))
    packed = np.packbits(padded, axis=1)
    with open(path, "wb") as f:
        f.write(f"P4\n{w} {h}\n".encode())
        f.write(packed.tobytes())


def _vectorize(img_pil: Image.Image, threshold: int) -> str:
    """PIL image → potrace → SVG string."""
    arr = np.array(img_pil.convert("L"), dtype=np.uint8)
    if arr.mean() < 128:
        arr = 255 - arr
    binary = (arr < threshold).astype(np.uint8)

    with tempfile.TemporaryDirectory() as tmp:
        pbm = str(Path(tmp) / "in.pbm")
        svg = str(Path(tmp) / "out.svg")
        _to_pbm(binary, pbm)
        r = subprocess.run(
            ["/opt/homebrew/bin/potrace", pbm, "-s", "--turdsize", "4", "--alphamax", "1.0", "-o", svg],
            capture_output=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"potrace failed: {r.stderr.decode()}")
        return Path(svg).read_text()


# ── SVG path parsing ──────────────────────────────────────────────────────────

def _cubic(p0, p1, p2, p3, n=12):
    pts = []
    for i in range(n + 1):
        t = i / n
        x = (1-t)**3*p0[0]+3*(1-t)**2*t*p1[0]+3*(1-t)*t**2*p2[0]+t**3*p3[0]
        y = (1-t)**3*p0[1]+3*(1-t)**2*t*p1[1]+3*(1-t)*t**2*p2[1]+t**3*p3[1]
        pts.append((x, y))
    return pts


def _parse_d(d: str) -> list:
    tokens = re.findall(
        r'[MmCcLlZzHhVv]|[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', d
    )
    segs, cur, start, cmd, nums = [], [0., 0.], [0., 0.], 'M', []

    def flush():
        nonlocal cmd
        while True:
            if cmd in ('M', 'm') and len(nums) >= 2:
                x, y = nums.pop(0), nums.pop(0)
                if cmd == 'm': x += cur[0]; y += cur[1]
                cur[0], cur[1] = x, y; start[0], start[1] = x, y
                segs.append(('M', x, y)); cmd = 'L' if cmd == 'M' else 'l'
            elif cmd in ('L', 'l') and len(nums) >= 2:
                x, y = nums.pop(0), nums.pop(0)
                if cmd == 'l': x += cur[0]; y += cur[1]
                cur[0], cur[1] = x, y; segs.append(('L', x, y))
            elif cmd in ('C', 'c') and len(nums) >= 6:
                x1,y1,x2,y2,x,y = [nums.pop(0) for _ in range(6)]
                if cmd == 'c':
                    x1+=cur[0]; y1+=cur[1]; x2+=cur[0]; y2+=cur[1]
                    x+=cur[0];  y+=cur[1]
                cur[0], cur[1] = x, y; segs.append(('C', x1,y1, x2,y2, x,y))
            elif cmd in ('H', 'h') and len(nums) >= 1:
                x = nums.pop(0) + (cur[0] if cmd == 'h' else 0)
                cur[0] = x; segs.append(('L', x, cur[1]))
            elif cmd in ('V', 'v') and len(nums) >= 1:
                y = nums.pop(0) + (cur[1] if cmd == 'v' else 0)
                cur[1] = y; segs.append(('L', cur[0], y))
            elif cmd in ('Z', 'z'):
                cur[0], cur[1] = start[0], start[1]; segs.append(('Z',)); break
            else:
                break

    for t in tokens:
        if re.match(r'[MmCcLlZzHhVv]', t):
            if nums: flush()
            cmd = t; nums = []
        else:
            nums.append(float(t))
    if nums: flush()
    return segs


def _get_transform(svg_text: str):
    m = re.search(r'translate\(([^,)]+),([^)]+)\).*?scale\(([^,)]+),([^)]+)\)', svg_text)
    if m:
        return float(m[1]), float(m[2]), float(m[3]), float(m[4])
    return 0., 0., 1., 1.


def _apply_tf(segs, tx, ty, sx, sy):
    def tp(x, y): return x*sx+tx, y*sy+ty
    out = []
    for s in segs:
        if s[0] == 'M': out.append(('M', *tp(s[1], s[2])))
        elif s[0] == 'L': out.append(('L', *tp(s[1], s[2])))
        elif s[0] == 'C':
            out.append(('C', *tp(s[1],s[2]), *tp(s[3],s[4]), *tp(s[5],s[6])))
        else: out.append(s)
    return out


def _parse_svg(svg_text: str):
    tx, ty, sx, sy = _get_transform(svg_text)
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    root = ET.fromstring(svg_text)
    ns = "http://www.w3.org/2000/svg"
    paths = []
    for el in root.iter(f"{{{ns}}}path"):
        segs = _parse_d(el.get("d", ""))
        segs = _apply_tf(segs, tx, ty, sx, sy)
        paths.append(segs)
    w = float(re.sub(r'[^\d.]', '', root.get("width", "512")))
    h = float(re.sub(r'[^\d.]', '', root.get("height", "512")))
    return paths, w, h


# ── Roughen ───────────────────────────────────────────────────────────────────

def _to_polyline(segs, max_seg):
    out, cur = [], (0., 0.)
    for s in segs:
        if s[0] == 'M':
            cur = (s[1], s[2]); out.append(s)
        elif s[0] == 'L':
            dx, dy = s[1]-cur[0], s[2]-cur[1]
            ln = (dx**2+dy**2)**.5
            n = max(1, int(ln/max_seg))
            for k in range(1, n+1):
                t = k/n; out.append(('L', cur[0]+t*dx, cur[1]+t*dy))
            cur = (s[1], s[2])
        elif s[0] == 'C':
            p0=cur; p1=(s[1],s[2]); p2=(s[3],s[4]); p3=(s[5],s[6])
            chord = ((p3[0]-p0[0])**2+(p3[1]-p0[1])**2)**.5
            n = max(4, int(chord/max_seg*2))
            for k in range(1, n+1):
                t = k/n
                x=(1-t)**3*p0[0]+3*(1-t)**2*t*p1[0]+3*(1-t)*t**2*p2[0]+t**3*p3[0]
                y=(1-t)**3*p0[1]+3*(1-t)**2*t*p1[1]+3*(1-t)*t**2*p2[1]+t**3*p3[1]
                out.append(('L', x, y))
            cur = p3
        else:
            out.append(s)
    return out


def _roughen_paths(paths, size_px, detail_per_inch, smoothing_sigma, seed, dpi=72):
    rng = np.random.default_rng(seed)
    max_seg = dpi / detail_per_inch
    out_paths = []
    for segs in paths:
        segs = _to_polyline(segs, max_seg)
        pts_idx = [i for i,s in enumerate(segs) if s[0] in ('M','L')]
        n = len(pts_idx)
        if n < 2:
            out_paths.append(segs); continue
        nx = gaussian_filter1d(rng.uniform(-1,1,n), sigma=max(0.01, smoothing_sigma)) * size_px
        ny = gaussian_filter1d(rng.uniform(-1,1,n), sigma=max(0.01, smoothing_sigma)) * size_px
        disp = {pts_idx[k]: (nx[k], ny[k]) for k in range(n)}
        new = []
        for i, s in enumerate(segs):
            dx, dy = disp.get(i, (0., 0.))
            if s[0] in ('M','L'): new.append((s[0], s[1]+dx, s[2]+dy))
            else: new.append(s)
        out_paths.append(new)
    return out_paths


# ── Render paths → PIL ────────────────────────────────────────────────────────

def _segs_to_d(segs) -> str:
    parts = []
    for s in segs:
        if s[0] == 'M': parts.append(f"M {s[1]:.3f},{s[2]:.3f}")
        elif s[0] == 'L': parts.append(f"L {s[1]:.3f},{s[2]:.3f}")
        elif s[0] == 'C':
            parts.append(f"C {s[1]:.3f},{s[2]:.3f} {s[3]:.3f},{s[4]:.3f} {s[5]:.3f},{s[6]:.3f}")
        elif s[0] == 'Z': parts.append("Z")
    return " ".join(parts)


def _render(paths, w, h, ink_color="#000000") -> Image.Image:
    """Render filled paths to RGBA PIL image via rsvg-convert (transparent bg)."""
    svg_el = ET.Element("svg", {
        "xmlns": "http://www.w3.org/2000/svg",
        "width": f"{w:.0f}pt", "height": f"{h:.0f}pt",
        "viewBox": f"0 0 {w:.0f} {h:.0f}",
    })
    g = ET.SubElement(svg_el, "g", {"fill": ink_color, "stroke": "none"})
    for segs in paths:
        ET.SubElement(g, "path", {"d": _segs_to_d(segs)})
    ET.indent(svg_el, space="  ")
    svg_text = ET.tostring(svg_el, encoding="unicode", xml_declaration=True)

    with tempfile.TemporaryDirectory() as tmp:
        svg_path = str(Path(tmp) / "in.svg")
        png_path = str(Path(tmp) / "out.png")
        Path(svg_path).write_text(svg_text)
        r = subprocess.run(
            ["/opt/homebrew/bin/rsvg-convert", "-w", str(int(w)), svg_path, "-o", png_path],
            capture_output=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"rsvg-convert failed: {r.stderr.decode()}")
        return Image.open(png_path).convert("RGBA").copy()


# ── ComfyUI Node ──────────────────────────────────────────────────────────────

class OutlineRoughenNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":     ("IMAGE",),
                "roughen":   ("BOOLEAN", {"default": True,
                                          "tooltip": "Toggle the roughen effect on/off"}),
                "size":      ("FLOAT",  {"default": 1.0, "min": 0.1, "max": 20.0, "step": 0.1,
                                         "tooltip": "Max displacement per point in px"}),
                "detail":    ("FLOAT",  {"default": 52.0, "min": 1.0, "max": 300.0, "step": 1.0,
                                         "tooltip": "Points per inch — higher = finer grain"}),
                "smoothing": ("FLOAT",  {"default": 1.2, "min": 0.0, "max": 10.0, "step": 0.1,
                                         "tooltip": "0 = jagged corners, higher = smoother waves"}),
                "threshold": ("INT",    {"default": 180, "min": 0, "max": 255,
                                         "tooltip": "Ink detection threshold"}),
                "seed":        ("INT",    {"default": 42, "min": 0, "max": 99999}),
                "ink_color":   ("STRING", {"default": "#000000",
                                           "tooltip": "Fill color of the vector shapes"}),
                "feather":     ("FLOAT",  {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.5,
                                           "tooltip": "Gaussian blur sigma (px) applied to the alpha/mask only — RGB is untouched"}),
                "opacity":     ("FLOAT",  {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                           "tooltip": "Multiplies the alpha/mask — 1.0 fully opaque, 0.0 invisible"}),
            }
        }

    RETURN_TYPES  = ("IMAGE", "MASK")
    RETURN_NAMES  = ("image", "mask")
    FUNCTION      = "apply"
    CATEGORY      = "fae/image"

    def apply(self, image, roughen, size, detail, smoothing, threshold, seed,
              ink_color="#000000", feather=0.0, opacity=1.0):
        out_images, out_masks = [], []

        for i in range(image.shape[0]):
            # Tensor [H,W,C] → PIL
            frame_np = (image[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            pil = Image.fromarray(frame_np)

            # Vectorize with potrace
            svg_text = _vectorize(pil, threshold)

            # Parse SVG paths + dimensions
            paths, w, h = _parse_svg(svg_text)

            # Apply roughen (or skip if toggled off)
            if roughen:
                paths = _roughen_paths(paths, size_px=size, detail_per_inch=detail,
                                        smoothing_sigma=smoothing, seed=seed + i)

            # Render to RGBA (transparent background)
            result_pil = _render(paths, w, h, ink_color=ink_color)

            # Split RGBA → RGB image + alpha mask
            rgba = np.array(result_pil, dtype=np.float32) / 255.0  # [H,W,4]
            rgb  = rgba[:, :, :3]                                    # [H,W,3]
            mask = rgba[:, :, 3]                                     # [H,W]

            if feather > 0:
                mask = gaussian_filter(mask, sigma=feather)
            if opacity < 1.0:
                mask = mask * opacity

            out_images.append(torch.from_numpy(rgb))
            out_masks.append(torch.from_numpy(mask))

        return (torch.stack(out_images), torch.stack(out_masks))
