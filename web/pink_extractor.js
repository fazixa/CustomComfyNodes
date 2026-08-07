import { app } from "../../../scripts/app.js";

// Show min_scale / smoothing only when dynamic_outline is enabled.
app.registerExtension({
    name: "fae.PinkExtractor.DynamicOutline",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!["PinkExtractor", "ColorExtractor"].includes(nodeData.name)) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);

            const toggle = this.widgets?.find(w => w.name === "dynamic_outline");
            const dependents = ["min_scale", "smoothing"]
                .map(name => this.widgets.find(w => w.name === name))
                .filter(Boolean);

            if (!toggle || dependents.length === 0) return r;

            const setVisible = (visible) => {
                for (const w of dependents) {
                    if (w.hidden === !visible) continue;
                    w.hidden = !visible;
                    if (!visible) {
                        w._origComputeSize = w.computeSize?.bind(w);
                        w.computeSize = () => [0, -4];
                    } else if (w._origComputeSize) {
                        w.computeSize = w._origComputeSize;
                        delete w._origComputeSize;
                    }
                }
                this.setSize(this.computeSize());
                app.graph.setDirtyCanvas(true, true);
            };

            const origCallback = toggle.callback;
            toggle.callback = function (...args) {
                const ret = origCallback?.apply(this, args);
                setVisible(toggle.value);
                return ret;
            };

            setVisible(toggle.value);
            return r;
        };
    },
});

// ── Eyedropper: pick a pink shade from the screen and set hue/sat/val ranges ──

function hexToHsv(hex) {
    hex = hex.replace("#", "");
    const r = parseInt(hex.substr(0, 2), 16) / 255;
    const g = parseInt(hex.substr(2, 2), 16) / 255;
    const b = parseInt(hex.substr(4, 2), 16) / 255;
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    const delta = max - min;

    let h = 0;
    if (delta !== 0) {
        if (max === r) h = ((g - b) / delta) % 6;
        else if (max === g) h = (b - r) / delta + 2;
        else h = (r - g) / delta + 4;
        h = (h / 6 + 1) % 1;
    }
    const s = max === 0 ? 0 : delta / max;
    return [h, s, max];
}

// Center a window of fixed width on `center`, sliding to stay in [0, 1].
function centeredRange(center, width) {
    let lo = center - width / 2;
    let hi = center + width / 2;
    if (lo < 0) { hi -= lo; lo = 0; }
    if (hi > 1) { lo -= (hi - 1); hi = 1; }
    return [Math.max(0, lo), Math.min(1, hi)];
}

app.registerExtension({
    name: "fae.PinkExtractor.Eyedropper",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "PinkExtractor") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);

            const hueW    = this.widgets?.find(w => w.name === "hue_center");
            const satMinW = this.widgets?.find(w => w.name === "sat_min");
            const satMaxW = this.widgets?.find(w => w.name === "sat_max");
            const valMinW = this.widgets?.find(w => w.name === "val_min");
            const valMaxW = this.widgets?.find(w => w.name === "val_max");
            if (!hueW) return r;

            this.addWidget("button", "Pick Pink Shade", null, async () => {
                if (!window.EyeDropper) {
                    alert("EyeDropper API not available in this app/browser version.");
                    return;
                }
                try {
                    const result = await new window.EyeDropper().open();
                    const [h, s, v] = hexToHsv(result.sRGBHex);

                    hueW.value = h;
                    hueW.callback?.(h);

                    if (satMinW && satMaxW) {
                        const [lo, hi] = centeredRange(s, satMaxW.value - satMinW.value);
                        satMinW.value = lo;
                        satMaxW.value = hi;
                        satMinW.callback?.(lo);
                        satMaxW.callback?.(hi);
                    }
                    if (valMinW && valMaxW) {
                        const [lo, hi] = centeredRange(v, valMaxW.value - valMinW.value);
                        valMinW.value = lo;
                        valMaxW.value = hi;
                        valMinW.callback?.(lo);
                        valMaxW.callback?.(hi);
                    }

                    app.graph.setDirtyCanvas(true, true);
                } catch (err) {
                    // User cancelled (Escape) — ignore.
                }
            });

            this.setSize(this.computeSize());
            return r;
        };
    },
});

// ── ColorExtractor: eyedrop the exact target color straight into target_color ──

app.registerExtension({
    name: "fae.ColorExtractor.Eyedropper",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "ColorExtractor") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);

            const targetW = this.widgets?.find(w => w.name === "target_color");
            if (!targetW) return r;

            const button = this.addWidget("button", "Pick Target Color", null, async () => {
                if (!window.EyeDropper) {
                    alert("EyeDropper API not available in this app/browser version.");
                    return;
                }
                try {
                    const result = await new window.EyeDropper().open();
                    targetW.value = result.sRGBHex.toUpperCase();
                    targetW.callback?.(targetW.value);
                    app.graph.setDirtyCanvas(true, true);
                } catch (err) {
                    // User cancelled (Escape) — ignore.
                }
            });

            // Sit directly under the target_color field.
            const targetIdx = this.widgets.indexOf(targetW);
            const buttonIdx = this.widgets.indexOf(button);
            if (targetIdx >= 0 && buttonIdx > targetIdx) {
                this.widgets.splice(buttonIdx, 1);
                this.widgets.splice(targetIdx + 1, 0, button);
            }

            this.setSize(this.computeSize());
            return r;
        };
    },
});
