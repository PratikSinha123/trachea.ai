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

// Anomalies — now handled inside updateAnomalies via getElementById

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

        // Load context meshes
        const contextLayers = ["body", "heart", "aorta", "pulmonary_artery"];
        for (const layer of contextLayers) {
            try {
                await viewer3d.loadMesh(`${API_BASE}/api/scan/${scanId}/mesh/${layer}`, layer);
            } catch (e) {
                console.warn(`Could not load ${layer} mesh:`, e);
            }
        }

        // Load morph frames — non-blocking background load
        const morphRes = await fetch(`${API_BASE}/api/scan/${scanId}/morph_count`);
        const { count } = await morphRes.json();
        morphFrameCount = count;
        morphSlider.max = Math.max(0, count - 1);
        morphSlider.value = 0;
        morphFrameLabel.textContent = `Loading… 0 / ${count - 1}`;

        viewer3d.morphMeshes = new Array(count).fill(null);

        // Load first frame synchronously so morph mode shows something immediately
        if (count > 0) {
            try {
                await viewer3d.loadMorphFrame(`${API_BASE}/api/scan/${scanId}/morph/0`, 0);
                morphFrameLabel.textContent = `0 / ${count - 1}`;
            } catch (e) {
                console.warn("Morph frame 0 failed:", e);
            }
        }

        // Load remaining frames in background — don't await
        (async () => {
            let loaded = 1;
            for (let i = 1; i < count; i++) {
                try {
                    await viewer3d.loadMorphFrame(`${API_BASE}/api/scan/${scanId}/morph/${i}`, i);
                    loaded++;
                    // Update label only if still in morph mode
                    if (viewer3d.displayMode === "morph") {
                        morphFrameLabel.textContent = `${viewer3d.activeMorphFrame} / ${count - 1} (${loaded}/${count} loaded)`;
                    }
                } catch (e) {
                    console.warn(`Morph frame ${i} failed:`, e);
                }
            }
            morphFrameLabel.textContent = `${viewer3d.activeMorphFrame} / ${count - 1}`;
            console.log(`✅ All ${count} morph frames loaded`);
        })();

        // Update stats
        updateStats(meta.stats || {});
        updateAnomalies(meta.anomalies || []);

        // Store cross-sections for annotation system
        window._lastCrossSections = meta.cross_sections || [];

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

        // Auto-add stenosis annotations if toggle is on
        const annotToggle = document.getElementById("annotation-toggle");
        if (viewer3d && annotToggle?.checked && window._lastCrossSections?.length > 0) {
            viewer3d.addStenosisAnnotations(window._lastCrossSections);
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
    const fmt = (v, unit = "") => (v !== undefined && v !== null && !isNaN(Number(v)))
        ? `${Number(v).toFixed(1)}${unit}` : "—";

    // API field names (from curl output):
    // volume_diseased_cm3, volume_healthy_cm3
    // avg_diameter_diseased_mm, avg_diameter_healthy_mm
    // min_diameter_mm, max_stenosis_pct, anomalies_found

    const dVol = stats.volume_diseased_cm3;
    const hVol = stats.volume_healthy_cm3;
    const dAvg = stats.avg_diameter_diseased_mm;
    const hAvg = stats.avg_diameter_healthy_mm;
    const minD = stats.min_diameter_mm;
    const maxS = stats.max_stenosis_pct;
    const numA = stats.anomalies_found;

    document.getElementById("stat-vol-diseased").textContent = fmt(dVol, " cm³");
    document.getElementById("stat-vol-healthy").textContent  = fmt(hVol, " cm³");

    // Volume reduction (positive = airway got smaller = bad)
    const volChangeEl = document.getElementById("stat-vol-change");
    if (dVol !== undefined && hVol !== undefined && hVol > 0) {
        const reduction = ((hVol - dVol) / hVol * 100).toFixed(1);
        volChangeEl.textContent = `${reduction}%`;
        volChangeEl.style.color = reduction > 20 ? "#f87171" : reduction > 10 ? "#facc15" : "#34d399";
    } else {
        volChangeEl.textContent = "—";
    }

    document.getElementById("stat-avg-diam").textContent   = fmt(dAvg, " mm");
    document.getElementById("stat-avg-diam-h").textContent = fmt(hAvg, " mm");

    const minDEl = document.getElementById("stat-min-diam");
    minDEl.textContent = fmt(minD, " mm");
    // Clinical thresholds: < 10mm = critical, < 15mm = severe
    minDEl.style.color = minD !== undefined
        ? (minD < 10 ? "#ef4444" : minD < 15 ? "#f87171" : minD < 20 ? "#facc15" : "#34d399")
        : "inherit";

    const stenosisEl = document.getElementById("stat-stenosis");
    stenosisEl.textContent = fmt(maxS, "%");
    stenosisEl.style.color = maxS !== undefined
        ? (maxS > 50 ? "#ef4444" : maxS > 30 ? "#f87171" : maxS > 15 ? "#facc15" : "#34d399")
        : "inherit";

    // Stenosis progress bar
    const bar = document.getElementById("stenosis-bar");
    if (bar && maxS !== undefined) {
        bar.style.width = `${Math.min(maxS, 100)}%`;
    }

    document.getElementById("stat-anomalies").textContent = (numA !== undefined && !isNaN(numA)) ? numA : "—";

    // Clinical severity badge
    const badge = document.getElementById("severity-badge");
    const sevLabel = document.getElementById("severity-label");
    const sevSub = document.getElementById("severity-sub");
    const sevIcon = document.getElementById("severity-icon");
    if (badge && maxS !== undefined) {
        badge.style.display = "block";
        if (maxS >= 50) {
            badge.style.background = "rgba(239,68,68,0.15)";
            badge.style.borderColor = "rgba(239,68,68,0.4)";
            sevIcon.textContent = "🚨";
            sevLabel.textContent = "CRITICAL Stenosis ≥ 50%";
            sevLabel.style.color = "#ef4444";
            sevSub.textContent = "Immediate intervention required";
        } else if (maxS >= 30) {
            badge.style.background = "rgba(248,113,113,0.12)";
            badge.style.borderColor = "rgba(248,113,113,0.3)";
            sevIcon.textContent = "⚠️";
            sevLabel.textContent = `Severe Stenosis — ${maxS.toFixed(0)}% narrowing`;
            sevLabel.style.color = "#f87171";
            sevSub.textContent = "Clinical evaluation recommended";
        } else if (maxS >= 15) {
            badge.style.background = "rgba(250,204,21,0.10)";
            badge.style.borderColor = "rgba(250,204,21,0.3)";
            sevIcon.textContent = "⚠️";
            sevLabel.textContent = `Moderate Stenosis — ${maxS.toFixed(0)}% narrowing`;
            sevLabel.style.color = "#facc15";
            sevSub.textContent = "Monitor and follow-up";
        } else {
            badge.style.background = "rgba(52,211,153,0.10)";
            badge.style.borderColor = "rgba(52,211,153,0.3)";
            sevIcon.textContent = "✅";
            sevLabel.textContent = "Airway within normal limits";
            sevLabel.style.color = "#34d399";
            sevSub.textContent = "No significant stenosis detected";
        }
    }
}

function updateAnomalies(anomalies) {
    const anomalyList = document.getElementById("anomaly-list");
    if (!anomalyList) return;

    if (!anomalies || !anomalies.length) {
        anomalyList.innerHTML = '<p class="placeholder-text">No anomalies detected</p>';
        return;
    }

    // Sort by severity (highest deviation first), show top 10
    const sorted = [...anomalies].sort((a, b) => (b.deviation_pct || 0) - (a.deviation_pct || 0)).slice(0, 10);
    anomalyList.innerHTML = "";

    for (const a of sorted) {
        const pct = a.deviation_pct?.toFixed(0) ?? "?";
        const diam = a.diseased_diameter_mm?.toFixed(1) ?? "?";
        const expected = a.expected_diameter_mm?.toFixed(1) ?? "?";
        const z = a.z_physical?.toFixed(1) ?? "?";

        const severity = (a.deviation_pct >= 50) ? "critical" : (a.deviation_pct >= 30) ? "severe" : "moderate";
        const severityColor = severity === "critical" ? "#ef4444" : severity === "severe" ? "#f87171" : "#facc15";
        const borderColor = severityColor + "44";

        const div = document.createElement("div");
        div.className = `anomaly-item stenosis`;
        div.style.cssText = `border-left: 3px solid ${severityColor}; margin-bottom: 8px; padding: 8px 10px; background: rgba(255,255,255,0.03); border-radius: 0 6px 6px 0;`;
        div.innerHTML = `
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                <span style="font-size:11px; font-weight:700; color:${severityColor}; text-transform:uppercase; letter-spacing:0.05em;">${severity} ${a.type || "Stenosis"}</span>
                <span style="font-size:13px; font-weight:700; color:${severityColor};">${pct}%</span>
            </div>
            <div style="font-size:11px; color:#9ca3af; line-height:1.5;">
                <span>📍 Z: ${z} mm</span>&nbsp;&nbsp;
                <span>⌀ ${diam} mm</span>&nbsp;&nbsp;
                <span style="color:#6b7280;">→ expected ${expected} mm</span>
            </div>
            <div style="margin-top:5px; background:rgba(255,255,255,0.07); border-radius:3px; height:4px;">
                <div style="height:100%; width:${Math.min(a.deviation_pct||0,100)}%; background:${severityColor}; border-radius:3px;"></div>
            </div>
        `;
        anomalyList.appendChild(div);
    }

    if (anomalies.length > 10) {
        const more = document.createElement("p");
        more.style.cssText = "text-align:center; font-size:11px; color:#6b7280; margin-top:8px;";
        more.textContent = `+ ${anomalies.length - 10} more stenotic zones`;
        anomalyList.appendChild(more);
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

            // When switching to morph mode, jump to frame 0 immediately
            if (mode === "morph") {
                const frame = parseInt(morphSlider.value) || 0;
                viewer3d.setMorphFrame(frame);
                morphFrameLabel.textContent = `${viewer3d.activeMorphFrame} / ${morphFrameCount - 1}`;
            }
        });
    });

    // Morph slider
    morphSlider.addEventListener("input", (e) => {
        const frame = parseInt(e.target.value);
        viewer3d.setMorphFrame(frame);
        // Show actual frame being displayed (may differ if not yet loaded)
        const actual = viewer3d.activeMorphFrame;
        const total = morphFrameCount - 1;
        morphFrameLabel.textContent = actual !== frame
            ? `${actual} / ${total} (nearest to ${frame})`
            : `${actual} / ${total}`;
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
    const wireframeToggle = document.getElementById("wireframe-toggle");
    wireframeToggle.addEventListener("change", (e) => {
        if (viewer3d) viewer3d.setWireframe(e.target.checked);
    });

    // Context Toggles
    const contextToggles = document.querySelectorAll(".context-toggle");
    contextToggles.forEach(toggle => {
        toggle.addEventListener("change", (e) => {
            if (viewer3d) {
                viewer3d.setContextVisibility(e.target.dataset.layer, e.target.checked);
            }
        });
    });

    // Cross-section toggle
    const crossSectionToggle = document.getElementById("cross-section-toggle");
    const crossSectionControls = document.getElementById("cross-section-controls");
    const crossSectionSlider = document.getElementById("cross-section-slider");
    crossSectionToggle?.addEventListener("change", (e) => {
        const enabled = e.target.checked;
        crossSectionControls.style.display = enabled ? "block" : "none";
        if (viewer3d) viewer3d.setCrossSectionEnabled(enabled);
    });
    crossSectionSlider?.addEventListener("input", (e) => {
        if (viewer3d && viewer3d.crossSectionEnabled) {
            // Map slider -100..100 to actual mesh coordinates
            const base = viewer3d.crossSectionPlane.constant;
            const mesh = viewer3d.diseasedMesh;
            if (mesh) {
                const box = new THREE.Box3().setFromObject(mesh);
                const size = box.getSize(new THREE.Vector3());
                const center = box.getCenter(new THREE.Vector3());
                const offset = (parseInt(e.target.value) / 100) * size.x * 0.5;
                viewer3d.crossSectionPlane.constant = center.x + offset;
            }
        }
    });

    // Breathing animation toggle
    const breathingToggle = document.getElementById("breathing-toggle");
    breathingToggle?.addEventListener("change", (e) => {
        if (viewer3d) viewer3d.setBreathingEnabled(e.target.checked);
    });

    // Stenosis annotation toggle
    const annotationToggle = document.getElementById("annotation-toggle");
    annotationToggle?.addEventListener("change", (e) => {
        if (viewer3d) {
            if (e.target.checked) {
                // Re-add annotations from last loaded metadata
                if (window._lastCrossSections) {
                    viewer3d.addStenosisAnnotations(window._lastCrossSections);
                }
            } else {
                viewer3d.addStenosisAnnotations([]); // clears all
            }
        }
    });

    // Morph Controls
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
