import { app } from "../../../scripts/app.js";

// Add an eyedropper button under fill_color / fill_image_color so colors
// can be sampled directly from anywhere on screen (Chromium EyeDropper API —
// works in ComfyUI Desktop and Chrome/Edge).
function addEyedropper(node, widgetName, label) {
    const targetWidget = node.widgets?.find(w => w.name === widgetName);
    if (!targetWidget) return;

    const button = node.addWidget("button", `Pick ${label}`, null, async () => {
        if (!window.EyeDropper) {
            alert("EyeDropper API not available in this app/browser version.");
            return;
        }
        try {
            const result = await new window.EyeDropper().open();
            targetWidget.value = result.sRGBHex;
            targetWidget.callback?.(targetWidget.value);
            app.graph.setDirtyCanvas(true, true);
        } catch (err) {
            // User cancelled (Escape) — ignore.
        }
    });

    // Move the button to sit directly under the widget it controls.
    const targetIdx = node.widgets.indexOf(targetWidget);
    const buttonIdx = node.widgets.indexOf(button);
    if (targetIdx >= 0 && buttonIdx > targetIdx) {
        node.widgets.splice(buttonIdx, 1);
        node.widgets.splice(targetIdx + 1, 0, button);
    }
}

app.registerExtension({
    name: "fae.BoilEffect.Eyedropper",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "BoilEffect") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);

            addEyedropper(this, "fill_color", "fill_color");
            addEyedropper(this, "fill_image_color", "fill_image_color");

            this.setSize(this.computeSize());
            return r;
        };
    },
});
