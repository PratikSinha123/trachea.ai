/**
 * TracheaAI — Main Application Controller
 *
 * Connects the UI controls to the 3D viewer, slice viewer, and backend API.
 */

import { Viewer3D } from "./viewer3d.js";
import { SliceViewer } from "./sliceViewer.js";

const API_BASE = (window.TRACHEA_API_BASE_URL || window.location.origin).replace(/\/$/, "");

// ─── State ──────────────────────────────────────────────────
let viewer3d = null;
let sliceViewer = null;
let currentScan = null;
let morphFrameCount = 0;
let isVercelMode = false;  // true when running on Vercel (read-only viewer)

// ─── DOM Elements ───────────────────────────────────────────
const loadingScreen = document.getElementById("loading-screen");
const loaderBar = document.getElementById("loader-bar");
const appEl = document.getElementById("app");
const scanSelect = document.getElementById("scan-select");
const emptyState = document.getElementById("empty-state");
const deviceBadge = document.getElementById("device-badge");

// Dashboard
const dashboardOverlay = document.getElementById("dashboard-overlay");
const patientGrid = document.getElementById("patient-grid");
const patientSearch = document.getElementById("patient-search");
const btnDashboard = document.getElementById("btn-dashboard");
const dashTotalCount = document.getElementById("dash-total-count");
const dashStenosisCount = document.getElementById("dash-stenosis-count");

let allScans = []; // Store full scan list for searching

// Display mode
const displayBtns = document.querySelectorAll("#display-mode .toggle-btn");
const morphControls = document.getElementById("morph-controls");
const morphSlider = document.getElementById("morph-slider");
const morphFrameLabel = document.getElementById("morph-frame-label");
const btnPlayMorph = document.getElementById("btn-play-morph");

// Controls
const opacitySlider = document.getElementById("opacity-slider");

// Slice viewer
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
    loaderBar.style.width = "30%";
    viewer3d = new Viewer3D("viewer-canvas");
    loaderBar.style.width = "60%";
    sliceViewer = new SliceViewer("slice-canvas", "profile-chart", API_BASE);
    loaderBar.style.width = "80%";
    await loadScanList();
    loaderBar.style.width = "100%";
    checkServer();
    setupEvents();

    setTimeout(() => {
        loadingScreen.classList.add("fade-out");
        appEl.classList.remove("hidden");
        dashboardOverlay.classList.remove("hidden");
        if (viewer3d) viewer3d._onResize();
    }, 600);
}

// ─── API Calls ──────────────────────────────────────────────
async function loadScanList() {
    try {
        const res = await fetch(`${API_BASE}/api/scans`);
        allScans = await res.json();
        renderScanList(allScans);
        renderPatientGrid(allScans);
        dashTotalCount.textContent = allScans.length;
        dashStenosisCount.textContent = allScans.filter(s => (s.stats?.max_stenosis_pct || 0) > 20).length;
    } catch (err) {
        console.warn("Could not load scan list:", err);
    }
}

function renderScanList(scans) {
    const currentVal = scanSelect.value;
    scanSelect.innerHTML = '<option value="">Select a scan...</option>';
    for (const scan of scans) {
        const opt = document.createElement("option");
        opt.value = scan.scan_id;
        opt.textContent = scan.scan_id;
        scanSelect.appendChild(opt);
    }
    scanSelect.value = currentVal;
}

function renderPatientGrid(scans) {
    patientGrid.innerHTML = "";
    if (scans.length === 0) {
        patientGrid.innerHTML = '<p style="text-align:center; color:var(--text-dim); padding:40px; grid-column: 1/-1;">No patient records match your search criteria.</p>';
        return;
    }

    for (const scan of scans) {
        const stenosis = scan.stats?.max_stenosis_pct || 0;
        const severity = stenosis > 40 ? "critical" : stenosis > 20 ? "warning" : "normal";
        const statusColor = severity === "critical" ? "var(--clinical-red)" : severity === "warning" ? "var(--clinical-yellow)" : "var(--clinical-green)";

        const card = document.createElement("div");
        card.className = "patient-card";
        card.innerHTML = `
            <div class="card-header">
                <span class="patient-id">${scan.scan_id.split('__')[0]}</span>
                <span class="status-indicator status-${severity}"></span>
            </div>
            <div class="card-metrics">
                <div class="card-metric">
                    <span class="metric-label">Stenosis</span>
                    <span class="metric-value" style="color:${statusColor}">${stenosis.toFixed(1)}%</span>
                </div>
                <div class="card-metric">
                    <span class="metric-label">Min Diam</span>
                    <span class="metric-value">${(scan.stats?.min_diameter_mm || 0).toFixed(1)}mm</span>
                </div>
            </div>
            <div style="margin-top:20px; display:flex; gap:10px;">
                <button class="btn-primary" style="flex:1; font-size:11px; padding:8px;" onclick="window._loadScan('${scan.scan_id}')">ANALYZE 3D</button>
            </div>
        `;
        patientGrid.appendChild(card);
    }
}

window._loadScan = (id) => {
    scanSelect.value = id;
    loadScan(id);
};

async function loadScan(scanId) {
    if (!scanId) {
        viewer3d.clearAll();
        dashboardOverlay.classList.remove("hidden");
        emptyState.classList.remove("hidden");
        return;
    }

    dashboardOverlay.classList.add("hidden");
    emptyState.classList.add("hidden");
    currentScan = scanId;

    try {
        const metaRes = await fetch(`${API_BASE}/api/scan/${scanId}`);
        const meta = await metaRes.json();

        // 1. Critical Path: Load only the trachea meshes initially
        await viewer3d.loadMesh(`${API_BASE}/api/scan/${scanId}/mesh/diseased`, "diseased");
        await viewer3d.loadMesh(`${API_BASE}/api/scan/${scanId}/mesh/healthy`, "healthy");

        // 2. Non-critical: Delay context meshes and morph frames
        // These will be loaded on-demand or with a slight delay
        updateStats(meta.stats || {});
        updateAnomalies(meta.anomalies || []);
        window._lastCrossSections = meta.cross_sections || [];

        const dimRes = await fetch(`${API_BASE}/api/scan/${scanId}/dimensions`);
        const dims = await dimRes.json();
        await sliceViewer.loadScan(scanId, dims, meta.cross_sections);
        sliceSlider.max = dims.axial - 1;
        sliceSlider.value = Math.floor(dims.axial / 2);
        sliceLabel.textContent = `Slice ${sliceSlider.value} / ${dims.axial}`;

    } catch (err) { console.error(err); }
}

// Helper to load context meshes on-demand
async function ensureContextLoaded(layer) {
    if (!currentScan || viewer3d.contextMeshes[layer]) return;
    try {
        await viewer3d.loadMesh(`${API_BASE}/api/scan/${currentScan}/mesh/${layer}`, layer);
    } catch (e) { console.warn(`Failed to lazy-load ${layer}`); }
}

// Helper to load morph frames on-demand
async function ensureMorphLoaded() {
    if (!currentScan || viewer3d.morphMeshes.length > 0) return;
    try {
        const morphRes = await fetch(`${API_BASE}/api/scan/${currentScan}/morph_count`);
        const { count } = await morphRes.json();
        viewer3d.morphMeshes = new Array(count).fill(null);
        for (let i = 0; i < count; i++) {
            await viewer3d.loadMorphFrame(`${API_BASE}/api/scan/${currentScan}/morph/${i}`, i);
        }
    } catch (e) { console.warn("Failed to load morph frames"); }
}

async function checkServer() {
    try {
        const res = await fetch(`${API_BASE}/api/scans`);
        if (res.ok) {
            const isCloud = window.location.hostname.includes('vercel.app');
            isVercelMode = isCloud;
            deviceBadge.querySelector(".badge-text").textContent = isCloud ? "☁ Cloud Mode" : "Local Server";
            deviceBadge.querySelector(".badge-dot").style.background = isCloud ? "#818cf8" : "#34d399";
        }
    } catch {
        deviceBadge.querySelector(".badge-text").textContent = "Offline";
        deviceBadge.querySelector(".badge-dot").style.background = "#f87171";
    }
}

// ─── UI Updates ─────────────────────────────────────────────
function updateStats(stats) {
    if (!stats) return;
    const maxS = stats.max_stenosis_pct || 0;
    
    document.getElementById("stat-vol-diseased").textContent = `${(stats.volume_diseased_cm3 || 0).toFixed(1)} cm³`;
    document.getElementById("stat-vol-healthy").textContent = `${(stats.volume_healthy_cm3 || 0).toFixed(1)} cm³`;
    document.getElementById("stat-avg-diam").textContent = `${(stats.avg_diameter_diseased_mm || 0).toFixed(1)} mm`;
    document.getElementById("stat-min-diam").textContent = `${(stats.min_diameter_mm || 0).toFixed(1)} mm`;
    document.getElementById("stat-stenosis").textContent = `${maxS.toFixed(1)}%`;
    document.getElementById("stenosis-bar").style.width = `${Math.min(maxS, 100)}%`;
    document.getElementById("stat-anomalies").textContent = stats.anomalies_found || 0;

    updateClinicalReport(stats);

    const badge = document.getElementById("severity-badge");
    if (badge) {
        badge.style.display = "block";
        const label = document.getElementById("severity-label");
        if (maxS > 40) { label.textContent = "Critical Stenosis"; badge.style.borderColor = "#f87171"; }
        else if (maxS > 15) { label.textContent = "Moderate Stenosis"; badge.style.borderColor = "#fb923c"; }
        else { label.textContent = "Normal Airway"; badge.style.borderColor = "#34d399"; }
    }
}

function updateClinicalReport(stats) {
    const reportContent = document.getElementById("report-content");
    if (!reportContent) return;
    const maxS = stats.max_stenosis_pct || 0;
    const diagnosis = maxS > 40 ? "Severe Tracheal Stenosis" : maxS > 15 ? "Moderate Stenosis" : "Normal Anatomical Variant";
    reportContent.innerHTML = `
        <p><strong>Diagnosis:</strong> ${diagnosis}</p>
        <p>Volumetric analysis shows <strong>${maxS.toFixed(1)}%</strong> narrowing at the narrowest point.</p>
    `;
}

function updateAnomalies(anomalies) {
    const list = document.getElementById("anomaly-list");
    list.innerHTML = anomalies.length ? "" : "No anomalies found.";
    anomalies.sort((a,b) => b.deviation_pct - a.deviation_pct).slice(0, 5).forEach(a => {
        const div = document.createElement("div");
        div.className = "anomaly-item";
        div.innerHTML = `<span>Stenosis at Z=${a.z_physical.toFixed(0)}mm: ${a.deviation_pct.toFixed(1)}%</span>`;
        list.appendChild(div);
    });
}

// ─── Event Handlers ─────────────────────────────────────────
function setupEvents() {
    btnDashboard.addEventListener("click", () => loadScan(""));
    patientSearch.addEventListener("input", (e) => {
        const query = e.target.value.toLowerCase();
        renderPatientGrid(allScans.filter(s => s.scan_id.toLowerCase().includes(query)));
    });
    scanSelect.addEventListener("change", (e) => loadScan(e.target.value));

    displayBtns.forEach((btn) => {
        btn.addEventListener("click", async () => {
            displayBtns.forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");
            const mode = btn.dataset.mode;
            
            if (mode === "morph") {
                await ensureMorphLoaded();
            }
            
            viewer3d.setDisplayMode(mode);
        });
    });

    opacitySlider.addEventListener("input", (e) => viewer3d.setOpacity(e.target.value / 100));
    
    document.querySelectorAll(".context-toggle").forEach(t => {
        t.addEventListener("change", async (e) => {
            const layer = e.target.dataset.layer;
            if (e.target.checked) {
                // If it's vessels, we might need to load both aorta and pulmonary_artery
                if (layer === "vessels") {
                    await ensureContextLoaded("aorta");
                    await ensureContextLoaded("pulmonary_artery");
                } else {
                    await ensureContextLoaded(layer);
                }
            }
            viewer3d.setContextVisibility(layer, e.target.checked);
        });
    });

    // View switching
    document.querySelectorAll(".slice-tab").forEach(tab => {
        tab.addEventListener("click", () => {
            document.querySelectorAll(".slice-tab").forEach(t => t.classList.remove("active"));
            tab.classList.add("active");
            if (tab.dataset.view) {
                document.getElementById("ct-view-container").classList.toggle("hidden", tab.dataset.view !== "ct");
                document.getElementById("report-view-container").classList.toggle("hidden", tab.dataset.view !== "report");
            } else {
                sliceViewer.setAxis(tab.dataset.axis);
            }
        });
    });

    sliceSlider.addEventListener("input", (e) => {
        sliceViewer.setIndex(parseInt(e.target.value));
        sliceLabel.textContent = `Slice ${e.target.value}`;
    });

    btnProcess.addEventListener("click", () => processModal.classList.remove("hidden"));
    btnCancel.addEventListener("click", () => processModal.classList.add("hidden"));
    btnStart.addEventListener("click", async () => {
        // ... (existing process logic)
    });
}

init();
