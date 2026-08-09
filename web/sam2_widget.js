import { app } from "../../scripts/app.js";

const FG_COLOR = "#00e676";
const BG_COLOR = "#ff1744";
const PT_R = 7;

// ── Shared rendering helpers ─────────────────────────────────────────────────

function drawMaskOverlay(ctx, maskImg, w, h) {
  const tmp = document.createElement("canvas");
  tmp.width = w;
  tmp.height = h;
  const tc = tmp.getContext("2d");
  tc.drawImage(maskImg, 0, 0, w, h);
  const id = tc.getImageData(0, 0, w, h);
  const d = id.data;
  for (let i = 0; i < d.length; i += 4) {
    const v = d[i];
    d[i] = 0; d[i + 1] = 230; d[i + 2] = 118;
    d[i + 3] = v > 128 ? 130 : 0;
  }
  tc.putImageData(id, 0, 0);
  ctx.drawImage(tmp, 0, 0);
}

function drawPoint(ctx, cx, cy, label) {
  ctx.beginPath();
  ctx.arc(cx, cy, PT_R, 0, Math.PI * 2);
  ctx.fillStyle = label === 1 ? FG_COLOR : BG_COLOR;
  ctx.fill();
  ctx.strokeStyle = "rgba(255,255,255,0.9)";
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.fillStyle = "white";
  ctx.font = `bold ${PT_R + 3}px sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(label === 1 ? "+" : "−", cx, cy + 0.5);
}

function drawPlaceholder(ctx, w, h, text) {
  ctx.fillStyle = "#12121e";
  ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = "#484870";
  ctx.strokeStyle = "#484870";
  ctx.setLineDash([6, 4]);
  ctx.strokeRect(4, 4, w - 8, h - 8);
  ctx.setLineDash([]);
  ctx.font = "13px sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, w / 2, h / 2);
}

// ── Build the shared canvas DOM widget ───────────────────────────────────────

function buildCanvasWidget(node, state, options = {}) {
  const { hintText = "L-click = foreground · R-click = background · Backspace = undo · Esc = clear",
          placeholderText = "Select a file above to start segmenting",
          predictFn } = options;

  const container = document.createElement("div");
  container.style.cssText = "position:relative;width:100%;background:#12121e;";

  const canvas = document.createElement("canvas");
  canvas.style.cssText = "display:block;width:100%;cursor:crosshair;";
  canvas.width = 512;
  canvas.height = 320;
  container.appendChild(canvas);

  const hint = document.createElement("div");
  hint.style.cssText =
    "position:absolute;bottom:6px;left:50%;transform:translateX(-50%);" +
    "font:11px sans-serif;color:rgba(255,255,255,0.45);pointer-events:none;white-space:nowrap;";
  hint.textContent = hintText;
  container.appendChild(hint);

  node.addDOMWidget("sam2_canvas", "sam2_canvas", container, {
    getValue() { return null; },
    setValue() {},
    serialize: false,
  });

  let placeholder = placeholderText;
  function setPlaceholder(text) {
    placeholder = text;
    if (!state.img) redraw();
  }

  function redraw() {
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!state.img) { drawPlaceholder(ctx, canvas.width, canvas.height, placeholder); return; }
    ctx.drawImage(state.img, 0, 0, canvas.width, canvas.height);
    if (state.maskImg) drawMaskOverlay(ctx, state.maskImg, canvas.width, canvas.height);
    for (let i = 0; i < state.points.length; i++) {
      const [px, py] = state.points[i];
      drawPoint(ctx, (px / state.imgW) * canvas.width, (py / state.imgH) * canvas.height, state.labels[i]);
    }
  }

  function setFrame(img, imgW, imgH) {
    state.img = img;
    state.imgW = imgW;
    state.imgH = imgH;
    canvas.width = 512;
    canvas.height = Math.round(512 * (imgH / imgW));
    state.points = [];
    state.labels = [];
    state.maskImg = null;
    redraw();
  }

  canvas.addEventListener("mousedown", (e) => {
    e.preventDefault(); e.stopPropagation();
    if (!state.img) return;
    const rect = canvas.getBoundingClientRect();
    const imgX = ((e.clientX - rect.left) / rect.width) * state.imgW;
    const imgY = ((e.clientY - rect.top) / rect.height) * state.imgH;
    state.points.push([imgX, imgY]);
    state.labels.push(e.button === 2 ? 0 : 1);
    state.syncWidgets();
    clearTimeout(state.debounce);
    state.debounce = setTimeout(async () => {
      const maskB64 = await predictFn(state);
      if (maskB64) {
        const img = new Image();
        img.onload = () => { state.maskImg = img; redraw(); };
        img.src = `data:image/png;base64,${maskB64}`;
      } else {
        state.maskImg = null; redraw();
      }
    }, 60);
    redraw();
  });

  canvas.addEventListener("contextmenu", (e) => { e.preventDefault(); e.stopPropagation(); });

  canvas.setAttribute("tabindex", "0");
  canvas.addEventListener("mouseenter", () => canvas.focus());
  canvas.addEventListener("keydown", (e) => {
    if (e.key === "Backspace" && state.points.length) {
      state.points.pop(); state.labels.pop();
      if (!state.points.length) state.maskImg = null;
      state.syncWidgets();
      redraw();
    } else if (e.key === "Escape") {
      state.points = []; state.labels = []; state.maskImg = null;
      state.syncWidgets(); redraw();
    }
  });

  return { canvas, redraw, setFrame, setPlaceholder };
}

// ── Image node ───────────────────────────────────────────────────────────────

app.registerExtension({
  name: "fae.SAM2Segment",
  async nodeCreated(node) {
    if (node.comfyClass !== "SAM2Segment") return;

    const pointsW = node.widgets?.find((w) => w.name === "points_json");
    const labelsW = node.widgets?.find((w) => w.name === "labels_json");
    if (pointsW) pointsW.type = "converted-widget";
    if (labelsW) labelsW.type = "converted-widget";

    const state = {
      img: null, imgW: 0, imgH: 0,
      filename: "", subfolder: "",
      points: [], labels: [], maskImg: null, debounce: null,
      syncWidgets() {
        if (pointsW) pointsW.value = JSON.stringify(this.points);
        if (labelsW) labelsW.value = JSON.stringify(this.labels);
      },
    };

    const { setFrame } = buildCanvasWidget(node, state, {
      placeholderText: "Select an image above to start segmenting",
      async predictFn(s) {
        if (!s.filename || !s.points.length) return null;
        try {
          const res = await fetch("/fae/sam2/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ filename: s.filename, subfolder: s.subfolder, points: s.points, labels: s.labels }),
          });
          const data = await res.json();
          return data.mask || null;
        } catch { return null; }
      },
    });

    const imgWidget = node.widgets?.find((w) => w.name === "image");
    if (imgWidget) {
      const origCb = imgWidget.callback?.bind(imgWidget);
      imgWidget.callback = function (v) {
        origCb?.(v);
        loadImage(v, state, setFrame);
      };
      if (imgWidget.value) setTimeout(() => loadImage(imgWidget.value, state, setFrame), 100);
    }
  },
});

function loadImage(value, state, setFrame) {
  const filename = typeof value === "string" ? value : (value?.filename ?? value?.name ?? "");
  const subfolder = typeof value === "object" ? (value?.subfolder ?? "") : "";
  if (!filename) return;
  state.filename = filename;
  state.subfolder = subfolder;
  const params = new URLSearchParams({ filename, type: "input" });
  if (subfolder) params.set("subfolder", subfolder);
  const img = new Image();
  img.onload = () => setFrame(img, img.naturalWidth, img.naturalHeight);
  img.src = `/view?${params}`;
}

// ── Video node ───────────────────────────────────────────────────────────────

const VIDEO_EXT_RE = /\.(mp4|mov|avi|mkv|webm)$/i;
const LOAD_LABEL = "⟳ Load video";

// The frontend attaches its own media preview for video_upload combos, which
// lands on top of our frame canvas. We draw the frame ourselves, so refuse it.
const PREVIEW_WIDGETS = ["$$canvas-image-preview", "$$comfy_animation_preview"];

function suppressMediaPreview(node) {
  Object.defineProperty(node, "imgs", {
    get() { return undefined; },
    set() {},
    configurable: true,
  });

  const addDOMWidget = node.addDOMWidget.bind(node);
  node.addDOMWidget = (name, type, el, opts) =>
    PREVIEW_WIDGETS.includes(name)
      ? { name, type, value: null, onRemove() {} }
      : addDOMWidget(name, type, el, opts);

  return function strip() {
    if (!node.widgets) return;
    for (let i = node.widgets.length - 1; i >= 0; i--) {
      if (PREVIEW_WIDGETS.includes(node.widgets[i].name)) {
        node.widgets[i].onRemove?.();
        node.widgets.splice(i, 1);
      }
    }
  };
}

// Walk back along the VIDEO link looking for a node that names a file we can
// read frames from — a LoadVideo, or a passthrough like Video Slice in front of
// one. Generated video (SeedanceGenerate) has no file until the graph runs.
function upstreamVideoFile(node, depth = 0) {
  if (!node || depth > 8) return null;
  const slot = node.inputs?.findIndex((i) => i.name === "video" || i.type === "VIDEO");
  if (slot == null || slot < 0) return null;
  const src = node.getInputNode?.(slot);
  if (!src) return null;

  for (const w of src.widgets ?? []) {
    const v = w?.value;
    const name = typeof v === "string" ? v : (v?.filename ?? v?.name ?? "");
    if (typeof name === "string" && VIDEO_EXT_RE.test(name)) {
      return { filename: name, subfolder: (typeof v === "object" ? v?.subfolder ?? "" : "") };
    }
  }
  return upstreamVideoFile(src, depth + 1);
}

// The endpoints take either an input-dir filename or this node's id, which the
// backend maps to whatever video the node resolved on its last execution.
function sourceFields(state) {
  if (state.filename) return { filename: state.filename, subfolder: state.subfolder };
  if (state.nodeId != null) return { node_id: String(state.nodeId) };
  return null;
}

// `video` used to be the first widget; it's now a socket, with the picker moved
// to `video_file` at the end. Workflows saved before that deserialize one slot
// out of phase — annotate_frame receives the filename, video_file receives the
// old labels_json — which fails validation. Rotate the values back into place.
function migrateWidgetOrder(node) {
  const get = (n) => node.widgets?.find((w) => w.name === n);
  const [file, frame, points, labels] =
    ["video_file", "annotate_frame", "points_json", "labels_json"].map(get);
  if (!file || !frame || !points || !labels) return false;
  if (typeof frame.value !== "string" || !VIDEO_EXT_RE.test(frame.value)) return false;

  const stale = file.value;
  file.value = frame.value;
  frame.value = points.value;
  points.value = labels.value;
  labels.value = stale;
  console.log(`[sam2] migrated widget order on node ${node.id} -> ${file.value}`);
  return true;
}

app.registerExtension({
  name: "fae.SAM2SegmentVideo",

  async loadedGraphNode(node) {
    if (node.comfyClass !== "SAM2SegmentVideo") return;
    if (migrateWidgetOrder(node)) node.__faeReloadSource?.();
  },

  async nodeCreated(node) {
    if (node.comfyClass !== "SAM2SegmentVideo") return;

    const stripPreview = suppressMediaPreview(node);

    const pointsW = node.widgets?.find((w) => w.name === "points_json");
    const labelsW = node.widgets?.find((w) => w.name === "labels_json");
    const frameW  = node.widgets?.find((w) => w.name === "annotate_frame");
    if (pointsW) pointsW.type = "converted-widget";
    if (labelsW) labelsW.type = "converted-widget";

    const state = {
      img: null, imgW: 0, imgH: 0,
      filename: "", subfolder: "", nodeId: null,
      points: [], labels: [], maskImg: null, debounce: null,
      frameIndex: 0, frameCount: 0,
      syncWidgets() {
        if (pointsW) pointsW.value = JSON.stringify(this.points);
        if (labelsW) labelsW.value = JSON.stringify(this.labels);
        if (frameW) frameW.value = this.frameIndex;
      },
    };

    // Added before the canvas so it sits above the frame preview. Assigned for
    // real further down, once setPlaceholder exists; the closure reads the
    // current binding when clicked. serialize=false keeps it out of
    // widgets_values (the slot is still held, so nothing shifts).
    let loadSource = async () => {};
    const loadBtn = node.addWidget("button", LOAD_LABEL, null, () => loadSource());
    loadBtn.serialize = false;

    const { canvas, redraw, setPlaceholder } = buildCanvasWidget(node, state, {
      placeholderText: "Connect a video or pick a file above",
      async predictFn(s) {
        const src = sourceFields(s);
        if (!src || !s.points.length) return null;
        try {
          const res = await fetch("/fae/sam2/video_predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ...src, frame: s.frameIndex, points: s.points, labels: s.labels }),
          });
          const data = await res.json();
          return data.mask || null;
        } catch { return null; }
      },
    });

    // ── Frame scrubber ────────────────────────────────────────────────────
    const scrubber = document.createElement("div");
    scrubber.style.cssText =
      "display:flex;align-items:center;gap:6px;padding:4px 6px;" +
      "background:#1a1a2e;font:12px sans-serif;color:#aaa;";

    const prevBtn = document.createElement("button");
    prevBtn.textContent = "◀";
    prevBtn.style.cssText = "background:#2a2a4a;border:none;color:#aaa;padding:2px 7px;cursor:pointer;border-radius:3px;";

    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = "0"; slider.max = "100"; slider.value = "0";
    slider.style.cssText = "flex:1;accent-color:#7c7cff;";

    const nextBtn = document.createElement("button");
    nextBtn.textContent = "▶";
    nextBtn.style.cssText = prevBtn.style.cssText;

    const frameLabel = document.createElement("span");
    frameLabel.textContent = "0 / 0";
    frameLabel.style.minWidth = "60px";

    scrubber.append(prevBtn, slider, nextBtn, frameLabel);
    canvas.parentElement.appendChild(scrubber);

    let frameFetchDebounce = null;

    async function goToFrame(idx) {
      const src = sourceFields(state);
      if (!src) return;
      idx = Math.max(0, Math.min(idx, state.frameCount - 1));
      state.frameIndex = idx;
      slider.value = idx;
      frameLabel.textContent = `${idx} / ${state.frameCount - 1}`;
      if (frameW) frameW.value = idx;

      clearTimeout(frameFetchDebounce);
      frameFetchDebounce = setTimeout(async () => {
        try {
          const params = new URLSearchParams({ ...src, frame: String(idx) });
          const res = await fetch(`/fae/sam2/video_frame?${params}`);
          const data = await res.json();
          if (data.frame) {
            const img = new Image();
            img.onload = () => {
              // New frame means the old points no longer describe what's shown.
              state.img = img;
              state.imgW = data.width;
              state.imgH = data.height;
              canvas.width = 512;
              canvas.height = Math.round(512 * (data.height / data.width));
              state.points = [];
              state.labels = [];
              state.maskImg = null;
              state.syncWidgets();
              redraw();
            };
            img.src = `data:image/jpeg;base64,${data.frame}`;
          }
        } catch (err) { console.error("[sam2] video_frame error:", err); }
      }, 80);
    }

    prevBtn.addEventListener("click", () => goToFrame(state.frameIndex - 1));
    nextBtn.addEventListener("click", () => goToFrame(state.frameIndex + 1));
    slider.addEventListener("input", () => goToFrame(parseInt(slider.value)));

    // ── Source resolution ─────────────────────────────────────────────────
    function resolveSource() {
      const slot = node.inputs?.findIndex((i) => i.name === "video");
      if (slot >= 0 && node.inputs[slot].link != null) {
        const up = upstreamVideoFile(node);
        if (up) return { ...up, nodeId: null, note: "" };
        return { filename: "", subfolder: "", nodeId: node.id,
                 note: "Queue once, then hit Load video" };
      }
      const w = node.widgets?.find((x) => x.name === "video_file");
      const v = w?.value;
      const filename = typeof v === "string" ? v : (v?.filename ?? v?.name ?? "");
      if (!filename) {
        return { filename: "", subfolder: "", nodeId: null,
                 note: "Connect a video or pick a file, then hit Load video" };
      }
      return { filename, subfolder: typeof v === "object" ? v?.subfolder ?? "" : "",
               nodeId: null, note: "" };
    }

    loadSource = async function () {
      stripPreview();

      const next = resolveSource();
      state.filename = next.filename;
      state.subfolder = next.subfolder;
      state.nodeId = next.nodeId;
      state.frameIndex = 0;
      state.points = [];
      state.labels = [];
      state.maskImg = null;

      const src = sourceFields(state);
      if (!src) {
        state.img = null;
        setPlaceholder(next.note);
        return;
      }

      loadBtn.name = "Loading…";
      node.setDirtyCanvas(true);
      try {
        const res = await fetch(`/fae/sam2/video_info?${new URLSearchParams(src)}`);
        if (!res.ok) throw new Error(`video_info ${res.status}`);
        const info = await res.json();
        state.frameCount = info.frame_count || 1;
        slider.max = state.frameCount - 1;
        slider.value = 0;
        frameLabel.textContent = `0 / ${state.frameCount - 1}`;
      } catch {
        // No file behind the socket yet — it appears after one execution.
        state.img = null;
        setPlaceholder(next.note || "Queue once, then hit Load video");
        return;
      } finally {
        loadBtn.name = LOAD_LABEL;
        node.setDirtyCanvas(true);
      }

      goToFrame(0);
    };

    const fileWidget = node.widgets?.find((w) => w.name === "video_file");
    if (fileWidget) {
      const origCb = fileWidget.callback?.bind(fileWidget);
      fileWidget.callback = function (v) {
        origCb?.(v);
        loadSource();
      };
    }

    const origOnConnectionsChange = node.onConnectionsChange;
    node.onConnectionsChange = function (type, index, connected, linkInfo, ioSlot) {
      const r = origOnConnectionsChange?.apply(this, arguments);
      if (ioSlot?.name === "video" || this.inputs?.[index]?.name === "video") {
        setTimeout(loadSource, 50);
      }
      return r;
    };

    // loadedGraphNode calls this after repairing a pre-socket workflow.
    node.__faeReloadSource = loadSource;

    setTimeout(loadSource, 100);
  },
});
