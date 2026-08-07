import { app } from "../../../scripts/app.js";

// Give the prompt textarea more room by default.
app.registerExtension({
    name: "fae.SeedanceGenerate.PromptSize",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "SeedanceGenerate") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            this.setSize([Math.max(this.size[0], 350), Math.max(this.size[1], 500)]);
            return r;
        };
    },
});
