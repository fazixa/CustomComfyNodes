import { app } from "../../scripts/app.js";

// Adds a "Pick <label>" button under a hex-string widget, sampling from
// anywhere on screen via the Chromium EyeDropper API — so a colour can be
// taken straight off a preview image in the graph. Works in ComfyUI Desktop
// and Chrome/Edge.
export function addEyedropper(node, widgetName, label) {
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
    button.serialize = false;

    // Move the button to sit directly under the widget it controls.
    const targetIdx = node.widgets.indexOf(targetWidget);
    const buttonIdx = node.widgets.indexOf(button);
    if (targetIdx >= 0 && buttonIdx > targetIdx) {
        node.widgets.splice(buttonIdx, 1);
        node.widgets.splice(targetIdx + 1, 0, button);
    }
}

// Attach eyedroppers to a node type as it registers. Pairs are [widgetName, label].
export function registerEyedroppers(extensionName, nodeName, pairs) {
    app.registerExtension({
        name: extensionName,
        async beforeRegisterNodeDef(nodeType, nodeData) {
            if (nodeData.name !== nodeName) return;

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated?.apply(this, arguments);
                for (const [widgetName, label] of pairs) {
                    addEyedropper(this, widgetName, label);
                }
                this.setSize(this.computeSize());
                return r;
            };
        },
    });
}
