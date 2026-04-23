/**
 * TracheaAI — Main Application Controller
 *
 * Connects the UI controls to the 3D viewer, slice viewer, and backend API.
 */

import { Viewer3D } from "./viewer3d.js";
import { SliceViewer } from "./sliceViewer.js";

const API_BASE = window.location.origin;

// ─── State ──────────────────────────────────────────────────
let viewer3d = null;
let sliceViewer = null;
let currentScan = null;
let morphFrameCount = 0;

// ─── DOM Elements ───────────────────────────────────────────
const loadingScreen = document.getElementById("loading-screen");
const loaderBar = document.getElementById("loader-bar");
const appEl = document.getElementById("app");
const scanSelect = document.getElementById("scan-select");
const emptyState = document.getElementById("empty-state");
const deviceBadge = document.getElementById("device-badge");

// Display mode
const displayBtns = document.querySelectorAll("#display-mode .toggle-btn");
const morphControls = document.getElementById("morph-controls");
const morphSlider = document.getElementById("morph-slider");
const morphFrameLabel = document.getElementById("morph-frame-label");
const btnPlayMorph = document.getElementById("btn-play-morph");

// Controls
const opacitySlider = document.getElementById("opacity-slider");
const wireframeToggle = document.getElementById("wireframe-toggle");

// Stats
const statElements = {
    volDiseased: document.getElementById("stat-vol-diseased"),
    volHealthy: document.getElementById("stat-vol-healthy"),
    volChange: document.getElementById("stat-vol-change"),
    avgDiam: document.getElementById("stat-avg-diam"),
    minDiam: document.getElementById("stat-min-diam"),
    stenosis: document.getElementById("stat-stenosis"),
    anomalies: document.getElementById("stat-anomalies"),
};

// Anomalies
const anomalyList = document.getElementById("anomaly-list");

// Slice viewer
const sliceTabs = document.querySelectorAll(".slice-tab");
const sliceSlider = document.getElementById("slice-slider");
const sliceLabel = document.getElementById("slice-label");

// Process modal
const processModal = document.getElementById("process-modal");
const btnProcess = document.getElementById("btn-process");
const btnCancel = document.getElementById("btn-cancel-process");
const btnStart = document.getElementById("btn-start-process");
const inputPath = document.getElementById("input-path");
const inputScanId = document.getElementById("input-scan-id");
const processStatus = document.getElementById("process-status");
const processStatusText = document.getElementById("process-status-text");

// ─── Initialization ────────────────────────────────────────
async function init() {
    // Animate loading bar
    loaderBar.style.width = "30%";

    // Init 3D viewer
    viewer3d = new Viewer3D("viewer-canvas");
    loaderBar.style.width = "60%";

    // Init slice viewer
    sliceViewer = new SliceViewer("slice-canvas", "profile-chart");
    loaderBar.style.width = "80%";

    // Load scan list
    await loadScanList();
    loaderBar.style.width = "100%";

    // Check server status
    checkServer();

    // Setup event listeners
    setupEvents();

    // Transition to app
    setTimeout(() => {
        loadingScreen.classList.add("fade-out");
        appEl.classList.remove("hidden");
        // Fix WebGL camera projection after unhiding container
        if (viewer3d) {
            viewer3d._onResize();
        }
    }, 600);
}

// ─── API Calls ──────────────────────────────────────────────
async function loadScanList() {
    try {
        const res = await fetch(`${API_BASE}/api/scans`);
        const scans = await res.json();

        scanSelect.innerHTML = '<option value="">Select a scan...</option>';
        for (const scan of scans) {
            const opt = document.createElement("option");
            opt.value = scan.scan_id;
            opt.textContent = scan.scan_id;
            scanSelect.appendChild(opt);
        }
    } catch (err) {
        console.warn("Could not load scan list:", err);
    }
}

async function loadScan(scanId) {
    if (!scanId) {
        viewer3d.clearAll();
        emptyState.classList.remove("hidden");
        return;
    }

    emptyState.classList.add("hidden");
    currentScan = scanId;

    try {
        // Load metadata
        const metaRes = await fetch(`${API_BASE}/api/scan/${scanId}`);
        const meta = await metaRes.json();

        // Load 3D meshes
        try {
            await viewer3d.loadMesh(`${API_BASE}/api/scan/${scanId}/mesh/diseased`, "diseased");
        } catch (e) {
            console.warn("Could not load diseased mesh:", e);
        }

        try {
            await viewer3d.loadMesh(`${API_BASE}/api/scan/${scanId}/mesh/healthy`, "healthy");
        } catch (e) {
            console.warn("Could not load healthy mesh:", e);
        }

        // Load morph frames
        const morphRes = await fetch(`${API_BASE}/api/scan/${scanId}/morph_count`);
        const { count } = await morphRes.json();
        morphFrameCount = count;
        morphSlider.max = Math.max(0, count - 1);

        viewer3d.morphMeshes = new Array(count).fill(null);
        for (let i = 0; i < count; i++) {
            try {
                await viewer3d.loadMorphFrame(
                    `${API_BASE}/api/scan/${scanId}/morph/${i}`, i
                );
            } catch (e) {
                console.warn(`Morph frame ${i} failed:`, e);
            }
        }

        // Update stats
        updateStats(meta.stats || {});
        updateAnomalies(meta.anomalies || []);

        // Load slice viewer
        try {
            const dimRes = await fetch(`${API_BASE}/api/scan/${scanId}/dimensions`);
            const dims = await dimRes.json();
            await sliceViewer.loadScan(scanId, dims, meta.cross_sections);
            sliceSlider.max = dims.axial - 1;
            sliceSlider.value = Math.floor(dims.axial / 2);
            sliceLabel.textContent = `Slice ${sliceSlider.value} / ${dims.axial}`;
        } catch (e) {
            console.warn("Could not load slice viewer:", e);
        }

    } catch (err) {
        console.error("Failed to load scan:", err);
    }
}

async function checkServer() {
    try {
        const res = await fetch(`${API_BASE}/api/scans`);
        if (res.ok) {
            deviceBadge.querySelector(".badge-text").textContent = "Server Connected";
            deviceBadge.querySelector(".badge-dot").style.background = "#34d399";
        }
    } catch {
        deviceBadge.querySelector(".badge-text").textContent = "Server Offline";
        deviceBadge.querySelector(".badge-dot").style.background = "#f87171";
    }
}

// ─── UI Updates ─────────────────────────────────────────────
function updateStats(stats) {
    if (!stats) stats = {};
    const fmt = (v, unit = "") => v !== undefined && v !== null && !isNaN(v) ? `${Number(v).toFixed(1)}${unit}` : "—";

    statElements.volDiseased.textContent = fmt(stats.diseased_volume_mm3 ? stats.diseased_volume_mm3 / 1000 : undefined, " cm³");
    statElements.volHealthy.textContent = fmt(stats.healthy_volume_mm3 ? stats.healthy_volume_mm3 / 1000 : undefined, " cm³");

    const change = stats.volume_change_pct;
    if (change !== undefined && !isNaN(change)) {
        statElements.volChange.textContent = `${change > 0 ? "+" : ""}${change.toFixed(1)}%`;
        statElements.volChange.style.color = change > 0 ? "#34d399" : "#f87171";
    } else {
        statElements.volChange.textContent = "—";
    }

    statElements.avgDiam.textContent = fmt(stats.mean_diseased_diameter_mm, " mm");
    statElements.minDiam.textContent = fmt(stats.min_diseased_diameter_mm, " mm");
    statElements.stenosis.textContent = fmt(stats.max_stenosis_pct, "%");
    statElements.anomalies.textContent = stats.num_anomalies !== undefined && !isNaN(stats.num_anomalies) ? stats.num_anomalies : "—";
}

function updateAnomalies(anomalies) {
    if (!anomalies.length) {
        anomalyList.innerHTML = '<p class="placeholder-text">No anomalies detected</p>';
        return;
    }

    anomalyList.innerHTML = "";
    for (const a of anomalies) {
        const div = document.createElement("div");
        div.className = `anomaly-item ${a.type}`;
        div.innerHTML = `
            <div class="anomaly-type ${a.type}">${a.type.toUpperCase()}</div>
            <div class="anomaly-detail">
                Z: ${a.z_mm?.toFixed(1)} mm · ${a.observed_mm?.toFixed(1)} mm
                (expected ${a.expected_mm?.toFixed(1)} mm) · ${a.deviation_pct?.toFixed(0)}% deviation
            </div>
        `;
        anomalyList.appendChild(div);
    }
}

// ─── Event Handlers ─────────────────────────────────────────
function setupEvents() {
    // Scan selection
    scanSelect.addEventListener("change", (e) => loadScan(e.target.value));

    // Display mode
    displayBtns.forEach((btn) => {
        btn.addEventListener("click", () => {
            displayBtns.forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");
            const mode = btn.dataset.mode;
            viewer3d.setDisplayMode(mode);
            morphControls.style.display = mode === "morph" ? "block" : "none";
        });
    });

    // Morph slider
    morphSlider.addEventListener("input", (e) => {
        const frame = parseInt(e.target.value);
        viewer3d.setMorphFrame(frame);
        morphFrameLabel.textContent = `${frame} / ${morphFrameCount - 1}`;
    });

    // Play morph
    let isPlaying = false;
    btnPlayMorph.addEventListener("click", () => {
        if (isPlaying) {
            viewer3d.stopMorphAnimation();
            btnPlayMorph.textContent = "▶ Play";
            isPlaying = false;
        } else {
            isPlaying = true;
            btnPlayMorph.textContent = "⏸ Pause";
            viewer3d.playMorphAnimation((frame, total) => {
                morphSlider.value = frame;
                morphFrameLabel.textContent = `${frame} / ${total - 1}`;
                if (frame >= total - 1) {
                    btnPlayMorph.textContent = "▶ Play";
                    isPlaying = false;
                }
            }, 200);
        }
    });

    // Opacity
    opacitySlider.addEventListener("input", (e) => {
        viewer3d.setOpacity(parseInt(e.target.value) / 100);
    });

    // Wireframe
    wireframeToggle.addEventListener("change", (e) => {
        viewer3d.setWireframe(e.target.checked);
    });

    // Slice tabs
    sliceTabs.forEach((tab) => {
        tab.addEventListener("click", () => {
            sliceTabs.forEach((t) => t.classList.remove("active"));
            tab.classList.add("active");
            const axis = tab.dataset.axis;
            sliceViewer.setAxis(axis);
            if (sliceViewer.dimensions[axis]) {
                sliceSlider.max = sliceViewer.dimensions[axis] - 1;
                sliceSlider.value = sliceViewer.currentIndex;
                sliceLabel.textContent = `Slice ${sliceViewer.currentIndex} / ${sliceViewer.dimensions[axis]}`;
            }
        });
    });

    // Slice slider
    sliceSlider.addEventListener("input", (e) => {
        const idx = parseInt(e.target.value);
        sliceViewer.setIndex(idx);
        const max = sliceViewer.dimensions[sliceViewer.axis] || 100;
        sliceLabel.textContent = `Slice ${idx} / ${max}`;
    });

    // Process modal
    btnProcess.addEventListener("click", () => processModal.classList.remove("hidden"));
    btnCancel.addEventListener("click", () => processModal.classList.add("hidden"));
    document.querySelector(".modal-backdrop")?.addEventListener("click", () => {
        processModal.classList.add("hidden");
    });

    btnStart.addEventListener("click", async () => {
        const path = inputPath.value.trim();
        if (!path) return;

        processStatus.classList.remove("hidden");
        processStatusText.textContent = "Processing... This may take a few minutes.";

        try {
            const res = await fetch(`${API_BASE}/api/process`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    path: path,
                    scan_id: inputScanId.value.trim() || null,
                }),
            });
            const data = await res.json();
            const sid = data.scan_id;

            // Poll status
            const poll = setInterval(async () => {
                const statusRes = await fetch(`${API_BASE}/api/status/${sid}`);
                const status = await statusRes.json();

                if (status.status === "done") {
                    clearInterval(poll);
                    processStatusText.textContent = "✅ Done! Loading scan...";
                    await loadScanList();
                    scanSelect.value = sid;
                    await loadScan(sid);
                    setTimeout(() => processModal.classList.add("hidden"), 1000);
                } else if (status.status === "error") {
                    clearInterval(poll);
                    processStatusText.textContent = `❌ Error: ${status.error}`;
                }
            }, 3000);
        } catch (err) {
            processStatusText.textContent = `❌ Error: ${err.message}`;
        }
    });
}

// ─── Start ──────────────────────────────────────────────────
init();
