/**
 * TracheaAI — Elite Main Controller
 * Professional Clinical Version (v12)
 */

import { Viewer3D } from "./viewer3d.js";
import { SliceViewer } from "./sliceViewer.js";

const API_BASE = (window.TRACHEA_API_BASE_URL || window.location.origin).replace(/\/$/, "");

// ─── Global State ──────────────────────────────────────────
let viewer3d = null;
let sliceViewer = null;
let currentScan = null;
let allScans = [];

// ─── Initialization ────────────────────────────────────────
async function init() {
    console.log("🚀 TracheaAI: Initializing Professional Clinical Suite...");
    
    try {
        // 1. Initialize Viewers
        const canvas3d = document.getElementById("viewer-canvas");
        if (canvas3d) {
            viewer3d = new Viewer3D("viewer-canvas");
            console.log("✅ 3D Viewport Engine Ready");
        }

        const canvasSlice = document.getElementById("slice-canvas");
        if (canvasSlice) {
            sliceViewer = new SliceViewer("slice-canvas", "profile-chart", API_BASE);
            console.log("✅ Diagnostic Slice Engine Ready");
        }
        
        // 2. Load Patient Data
        await loadScanList();
        
        // 3. Bind UI Interactivity
        setupEvents();

        // 4. Final Layout Adjustment
        if (viewer3d) viewer3d._onResize();
        
        // 5. Default View: Dashboard
        const dashboard = document.getElementById("dashboard-overlay");
        if (dashboard) dashboard.classList.remove("hidden");

        console.log("✅ System Startup Complete");

    } catch (err) {
        console.error("❌ Critical Startup Error:", err);
    }
}

// ─── Data Management ────────────────────────────────────────
async function loadScanList() {
    try {
        const res = await fetch(`${API_BASE}/api/scans`);
        if (!res.ok) throw new Error("Failed to fetch scan manifest");
        allScans = await res.json();

        renderPatientGrid(allScans);
        
        const totalEl = document.getElementById("dash-total-count");
        const stenEl = document.getElementById("dash-stenosis-count");
        
        if (totalEl) totalEl.textContent = allScans.length;
        if (stenEl) stenEl.textContent = allScans.filter(s => (s.stats?.max_stenosis_pct || 0) > 20).length;

    } catch (err) {
        console.warn("⚠️ Data Sync Warning:", err);
    }
}

function renderPatientGrid(scans) {
    const grid = document.getElementById("patient-grid");
    if (!grid) return;
    grid.innerHTML = "";
    
    if (scans.length === 0) {
        grid.innerHTML = '<div style="grid-column:1/-1; padding:100px; text-align:center; color:var(--text-dim);">NO CLINICAL RECORDS FOUND</div>';
        return;
    }

    scans.forEach(scan => {
        const stenosis = scan.stats?.max_stenosis_pct || 0;
        const severity = stenosis > 40 ? "critical" : stenosis > 15 ? "warning" : "normal";
        const statusColor = severity === "critical" ? "var(--clinical-red)" : severity === "warning" ? "var(--clinical-yellow)" : "var(--clinical-green)";

        const card = document.createElement("div");
        card.className = "patient-card";
        card.onclick = () => loadScan(scan.scan_id);
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
                    <span class="metric-label">Min Area</span>
                    <span class="metric-value">${(scan.stats?.min_diameter_mm || 0).toFixed(1)}mm</span>
                </div>
            </div>
            <div style="margin-top:20px;">
                <button class="btn-primary" style="width:100%; font-size:10px; padding:6px;">OPEN CASE</button>
            </div>
        `;
        grid.appendChild(card);
    });
}

async function loadScan(scanId) {
    const dashboard = document.getElementById("dashboard-overlay");
    if (!scanId) {
        if (viewer3d) viewer3d.clearAll();
        if (dashboard) dashboard.classList.remove("hidden");
        return;
    }

    if (dashboard) dashboard.classList.add("hidden");
    currentScan = scanId;

    try {
        const metaRes = await fetch(`${API_BASE}/api/scan/${scanId}`);
        const meta = await metaRes.json();

        if (viewer3d) {
            await viewer3d.loadMesh(`${API_BASE}/api/scan/${scanId}/mesh/diseased`, "diseased");
            await viewer3d.loadMesh(`${API_BASE}/api/scan/${scanId}/mesh/healthy`, "healthy");
        }

        updateStats(meta.stats || {});
        window._lastCrossSections = meta.cross_sections || [];

        if (sliceViewer) {
            const dimRes = await fetch(`${API_BASE}/api/scan/${scanId}/dimensions`);
            const dims = await dimRes.json();
            await sliceViewer.loadScan(scanId, dims, meta.cross_sections);
        }

    } catch (err) {
        console.error("Failed to load patient record:", err);
    }
}

// ─── UI Orchestration ───────────────────────────────────────
function updateStats(stats) {
    if (!stats) return;
    const maxS = stats.max_stenosis_pct || 0;
    
    const safeSet = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.textContent = val;
    };

    safeSet("stat-vol-diseased", `${(stats.volume_diseased_cm3 || 0).toFixed(1)} cm³`);
    safeSet("stat-stenosis", `${maxS.toFixed(1)}%`);
    safeSet("stat-min-diam", `${(stats.min_diameter_mm || 0).toFixed(1)} mm`);
    safeSet("stat-anomalies", stats.anomalies_found || 0);

    const sevLabel = document.getElementById("severity-label");
    if (sevLabel) {
        if (maxS > 40) { sevLabel.textContent = "CRITICAL"; sevLabel.style.color = "var(--clinical-red)"; }
        else if (maxS > 15) { sevLabel.textContent = "WARNING"; sevLabel.style.color = "var(--clinical-yellow)"; }
        else { sevLabel.textContent = "NORMAL"; sevLabel.style.color = "var(--clinical-green)"; }
    }

    updateClinicalReport(stats);
}

function updateClinicalReport(stats) {
    const reportContent = document.getElementById("report-content");
    if (!reportContent) return;
    const maxS = stats.max_stenosis_pct || 0;
    const diag = maxS > 40 ? "SEVERE STENOSIS" : maxS > 15 ? "MODERATE OBSTRUCTION" : "NORMAL LIMITS";
    
    reportContent.innerHTML = `
        <div style="font-family:var(--font-data); font-size:11px;">
            <p style="margin-bottom:8px;"><strong>DIAGNOSIS:</strong> ${diag}</p>
            <p><strong>NARROWING:</strong> ${maxS.toFixed(1)}%</p>
            <p style="margin-top:12px; color:var(--text-dim); line-height:1.4;">Automated AI analysis complete. Morphological deviations detected in ${stats.anomalies_found || 0} zones.</p>
        </div>
    `;
}

function setupEvents() {
    const dashBtn = document.getElementById("btn-dashboard");
    if (dashBtn) dashBtn.onclick = () => loadScan("");

    const search = document.getElementById("patient-search");
    if (search) {
        search.oninput = (e) => {
            const query = e.target.value.toLowerCase();
            renderPatientGrid(allScans.filter(s => s.scan_id.toLowerCase().includes(query)));
        };
    }

    document.querySelectorAll("#display-mode .btn-toggle").forEach(btn => {
        btn.onclick = () => {
            document.querySelectorAll("#display-mode .btn-toggle").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            if (viewer3d) viewer3d.setDisplayMode(btn.dataset.mode);
        };
    });

    const opacity = document.getElementById("opacity-slider");
    if (opacity) {
        opacity.oninput = (e) => { if (viewer3d) viewer3d.setOpacity(e.target.value / 100); };
    }

    document.querySelectorAll(".panel-right .btn-toggle").forEach(tab => {
        tab.onclick = () => {
            document.querySelectorAll(".panel-right .btn-toggle").forEach(t => t.classList.remove("active"));
            tab.classList.add("active");
            const view = tab.dataset.view;
            const ct = document.getElementById("ct-view-container");
            const rep = document.getElementById("report-view-container");
            if (view === "ct") { ct?.classList.remove("hidden"); rep?.classList.add("hidden"); }
            else { ct?.classList.add("hidden"); rep?.classList.remove("hidden"); }
        };
    });
}

// ─── Boot ──────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", init);
