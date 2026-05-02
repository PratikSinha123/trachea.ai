/**
 * TracheaAI — 2D CT Slice Viewer
 *
 * Displays axial, coronal, and sagittal slices from the API.
 * Also renders the cross-section diameter profile chart.
 */

export class SliceViewer {
    constructor(canvasId, chartCanvasId, apiBase = window.location.origin) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext("2d");
        this.chartCanvas = document.getElementById(chartCanvasId);
        this.chartCtx = this.chartCanvas.getContext("2d");
        this.apiBase = apiBase.replace(/\/$/, "");

        this.scanId = null;
        this.axis = "axial";
        this.dimensions = { axial: 0, coronal: 0, sagittal: 0 };
        this.currentIndex = 0;
        this.crossSections = [];
    }

    async loadScan(scanId, dimensions, crossSections) {
        this.scanId = scanId;
        this.dimensions = dimensions;
        this.crossSections = crossSections || [];
        this.currentIndex = Math.floor((dimensions[this.axis] || 100) / 2);

        this._drawChart();
        await this.loadSlice();
    }

    setAxis(axis) {
        this.axis = axis;
        this.currentIndex = Math.floor((this.dimensions[axis] || 100) / 2);
        this.loadSlice();
    }

    setIndex(index) {
        this.currentIndex = index;
        this.loadSlice();
    }

    async loadSlice() {
        if (!this.scanId) return;

        const url = `${this.apiBase}/api/scan/${this.scanId}/slice/${this.axis}/${this.currentIndex}`;

        try {
            const response = await fetch(url);
            if (!response.ok) {
                this._drawPlaceholder("Slice not available");
                return;
            }

            const blob = await response.blob();
            const img = new Image();
            img.onload = () => {
                // Fit image to canvas maintaining aspect ratio
                const cw = this.canvas.width;
                const ch = this.canvas.height;
                const scale = Math.min(cw / img.width, ch / img.height);
                const w = img.width * scale;
                const h = img.height * scale;

                this.ctx.fillStyle = "#000";
                this.ctx.fillRect(0, 0, cw, ch);
                this.ctx.drawImage(img, (cw - w) / 2, (ch - h) / 2, w, h);

                // Draw crosshair
                this.ctx.strokeStyle = "rgba(34, 211, 238, 0.3)";
                this.ctx.lineWidth = 0.5;
                this.ctx.beginPath();
                this.ctx.moveTo(cw / 2, 0);
                this.ctx.lineTo(cw / 2, ch);
                this.ctx.moveTo(0, ch / 2);
                this.ctx.lineTo(cw, ch / 2);
                this.ctx.stroke();
            };
            img.src = URL.createObjectURL(blob);
        } catch (err) {
            this._drawPlaceholder("Error loading slice");
        }
    }

    _drawPlaceholder(text) {
        const cw = this.canvas.width;
        const ch = this.canvas.height;
        this.ctx.fillStyle = "#111620";
        this.ctx.fillRect(0, 0, cw, ch);
        this.ctx.fillStyle = "#5a6882";
        this.ctx.font = "12px Inter";
        this.ctx.textAlign = "center";
        this.ctx.fillText(text, cw / 2, ch / 2);
    }

    _drawChart() {
        const ctx = this.chartCtx;
        const w = this.chartCanvas.width;
        const h = this.chartCanvas.height;
        const pad = { top: 20, right: 15, bottom: 25, left: 40 };

        // Clear
        ctx.fillStyle = "#1a2233";
        ctx.fillRect(0, 0, w, h);

        if (!this.crossSections || this.crossSections.length < 2) {
            ctx.fillStyle = "#5a6882";
            ctx.font = "11px Inter";
            ctx.textAlign = "center";
            ctx.fillText("No profile data", w / 2, h / 2);
            return;
        }

        const data = this.crossSections;
        const zValues = data.map((d) => d.z_physical);
        const diameters = data.map((d) => d.equiv_diameter_mm);

        const zMin = Math.min(...zValues);
        const zMax = Math.max(...zValues);
        const dMin = 0;
        const dMax = Math.max(...diameters) * 1.2;

        const plotW = w - pad.left - pad.right;
        const plotH = h - pad.top - pad.bottom;

        const scaleX = (z) => pad.left + ((z - zMin) / (zMax - zMin || 1)) * plotW;
        const scaleY = (d) => pad.top + plotH - ((d - dMin) / (dMax - dMin || 1)) * plotH;

        // Grid lines
        ctx.strokeStyle = "rgba(255,255,255,0.05)";
        ctx.lineWidth = 0.5;
        for (let i = 0; i <= 4; i++) {
            const y = pad.top + (plotH / 4) * i;
            ctx.beginPath();
            ctx.moveTo(pad.left, y);
            ctx.lineTo(w - pad.right, y);
            ctx.stroke();
        }

        // Draw diameter profile
        ctx.beginPath();
        ctx.strokeStyle = "#22d3ee";
        ctx.lineWidth = 1.5;
        for (let i = 0; i < data.length; i++) {
            const x = scaleX(zValues[i]);
            const y = scaleY(diameters[i]);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Fill under curve
        ctx.lineTo(scaleX(zValues[zValues.length - 1]), pad.top + plotH);
        ctx.lineTo(scaleX(zValues[0]), pad.top + plotH);
        ctx.closePath();
        const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + plotH);
        grad.addColorStop(0, "rgba(34, 211, 238, 0.15)");
        grad.addColorStop(1, "rgba(34, 211, 238, 0)");
        ctx.fillStyle = grad;
        ctx.fill();

        // Axis labels
        ctx.fillStyle = "#5a6882";
        ctx.font = "9px JetBrains Mono";
        ctx.textAlign = "center";
        ctx.fillText("Z position (mm)", w / 2, h - 3);

        ctx.save();
        ctx.translate(10, h / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText("Diameter (mm)", 0, 0);
        ctx.restore();

        // Y-axis tick labels
        ctx.textAlign = "right";
        for (let i = 0; i <= 4; i++) {
            const val = dMin + ((dMax - dMin) / 4) * (4 - i);
            const y = pad.top + (plotH / 4) * i;
            ctx.fillText(val.toFixed(1), pad.left - 4, y + 3);
        }
    }
}
