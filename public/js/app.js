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
let isVercelMode = false;  // true when running on Vercel (read-only viewer)

// ─── DOM Elements ───────────────────────────────────────────
const scanSelect = document.getElementById("scan-select");
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

// ─── Initialization ────────────────────────────────────────
async function init() {
    console.log("Initializing TracheaAI Core...");
    
    // Init 3D viewer
    viewer3d = new Viewer3D("viewer-canvas");
    
    // Init slice viewer
    sliceViewer = new SliceViewer("slice-canvas", "profile-chart", API_BASE);
    
    // Load scan list
    await loadScanList();
    
    // Check server status
    checkServer();

    // Setup event listeners
    setupEvents();

    // Fix WebGL camera projection
    if (viewer3d) {
        viewer3d._onResize();
    }
    
    // Ensure dashboard is visible
    if (dashboardOverlay) {
        dashboardOverlay.classList.remove("hidden");
    }
}

// ─── API Calls ──────────────────────────────────────────────
async function loadScanList() {
    try {
        const res = await fetch(`${API_BASE}/api/scans`);
        allScans = await res.json();

        renderPatientGrid(allScans);
        
        if (dashTotalCount) dashTotalCount.textContent = allScans.length;
        if (dashStenosisCount) dashStenosisCount.textContent = allScans.filter(s => (s.stats?.max_stenosis_pct || 0) > 20).length;

    } catch (err) {
        console.warn("Could not load scan list:", err);
    }
}

function renderPatientGrid(scans) {
    if (!patientGrid) return;
    patientGrid.innerHTML = "";
    
    if (scans.length === 0) {
        patientGrid.innerHTML = '<p style="text-align:center; color:var(--text-dim); padding:40px; grid-column: 1/-1;">No patient records match your search criteria.</p>';
        return;
    }

    for (const scan of scans) {
        const stenosis = scan.stats?.max_stenosis_pct || 0;
        const severity = stenosis > 40 ? "critical" : stenosis > 15 ? "warning" : "normal";
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

// Global accessor for cards
window._loadScan = (id) => {
    loadScan(id);
};

async function loadScan(scanId) {
    if (!scanId) {
        if (viewer3d) viewer3d.clearAll();
        if (dashboardOverlay) dashboardOverlay.classList.remove("hidden");
        return;
    }

    if (dashboardOverlay) dashboardOverlay.classList.add("hidden");
    currentScan = scanId;

    try {
        // Load metadata
        const metaRes = await fetch(`${API_BASE}/api/scan/${scanId}`);
        const meta = await metaRes.json();

        // Load 3D meshes
        await viewer3d.loadMesh(`${API_BASE}/api/scan/${scanId}/mesh/diseased`, "diseased");
        await viewer3d.loadMesh(`${API_BASE}/api/scan/${scanId}/mesh/healthy`, "healthy");

        // Update stats
        updateStats(meta.stats || {});
        window._lastCrossSections = meta.cross_sections || [];

        // Load slice viewer
        try {
            const dimRes = await fetch(`${API_BASE}/api/scan/${scanId}/dimensions`);
            const dims = await dimRes.json();
            await sliceViewer.loadScan(scanId, dims, meta.cross_sections);
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
            const isCloud = window.location.hostname.includes('vercel.app');
            isVercelMode = isCloud;
            const badgeText = deviceBadge?.querySelector(".badge-text");
            if (badgeText) badgeText.textContent = isCloud ? "Cloud Mode" : "Local Server";
        }
    } catch {
        const badgeText = deviceBadge?.querySelector(".badge-text");
        if (badgeText) badgeText.textContent = "Offline";
    }
}

// ─── UI Updates ─────────────────────────────────────────────
function updateStats(stats) {
    if (!stats) return;
    const maxS = stats.max_stenosis_pct || 0;
    
    const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };

    set("stat-vol-diseased", `${(stats.volume_diseased_cm3 || 0).toFixed(1)} cm³`);
    set("stat-stenosis", `${maxS.toFixed(1)}%`);
    set("stat-min-diam", `${(stats.min_diameter_mm || 0).toFixed(1)} mm`);
    set("stat-anomalies", stats.anomalies_found || 0);

    const sevLabel = document.getElementById("severity-label");
    if (sevLabel) {
        if (maxS > 40) { sevLabel.textContent = "CRITICAL"; sevLabel.style.color = "var(--clinical-red)"; }
        else if (maxS > 15) { sevLabel.textContent = "MODERATE"; sevLabel.style.color = "var(--clinical-yellow)"; }
        else { sevLabel.textContent = "NORMAL"; sevLabel.style.color = "var(--clinical-green)"; }
    }

    updateClinicalReport(stats);
}

function updateClinicalReport(stats) {
    const reportContent = document.getElementById("report-content");
    if (!reportContent) return;
    const maxS = stats.max_stenosis_pct || 0;
    const diagnosis = maxS > 40 ? "Severe Tracheal Stenosis" : maxS > 15 ? "Moderate Stenosis" : "Normal Anatomical Variant";
    reportContent.innerHTML = `
        <p><strong>Primary Diagnosis:</strong> ${diagnosis}</p>
        <p>AI Volumetric analysis indicates a peak airway narrowing of <strong>${maxS.toFixed(1)}%</strong>.</p>
        <p style="margin-top:10px; font-size:10px; color:var(--text-dim);">* This report is automated and requires clinical validation.</p>
    `;
}

// ─── Event Handlers ─────────────────────────────────────────
function setupEvents() {
    if (btnDashboard) {
        btnDashboard.addEventListener("click", () => loadScan(""));
    }

    if (patientSearch) {
        patientSearch.addEventListener("input", (e) => {
            const query = e.target.value.toLowerCase();
            const filtered = allScans.filter(s => s.scan_id.toLowerCase().includes(query));
            renderPatientGrid(filtered);
        });
    }

    displayBtns.forEach((btn) => {
        btn.addEventListener("click", async () => {
            displayBtns.forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");
            if (viewer3d) viewer3d.setDisplayMode(btn.dataset.mode);
        });
    });

    if (opacitySlider) {
        opacitySlider.addEventListener("input", (e) => {
            if (viewer3d) viewer3d.setOpacity(e.target.value / 100);
        });
    }

    // View switching
    document.querySelectorAll(".btn-toggle").forEach(tab => {
        if (tab.dataset.view) {
            tab.addEventListener("click", () => {
                document.querySelectorAll("[data-view]").forEach(t => t.classList.remove("active"));
                tab.classList.add("active");
                const ctView = document.getElementById("ct-view-container");
                const reportView = document.getElementById("report-view-container");
                if (tab.dataset.view === "ct") {
                    ctView?.classList.remove("hidden");
                    reportView?.classList.add("hidden");
                } else {
                    ctView?.classList.add("hidden");
                    reportView?.classList.remove("hidden");
                }
            });
        }
    });

    if (btnProcess) {
        btnProcess.addEventListener("click", () => processModal?.classList.remove("hidden"));
    }
    if (btnCancel) {
        btnCancel.addEventListener("click", () => processModal?.classList.add("hidden"));
    }
}

// ─── Start ──────────────────────────────────────────────────
init();
