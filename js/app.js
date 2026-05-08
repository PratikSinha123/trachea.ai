/**
 * TracheaAI — Clinical Application Controller
 */

import { Viewer3D } from "./viewer3d.js";
import { SliceViewer } from "./sliceViewer.js";

const API_BASE = (window.TRACHEA_API_BASE_URL || window.location.origin).replace(/\/$/, "");

// ─── Global State ──────────────────────────────────────────
let viewer3d = null;
let sliceViewer = null;
let currentScanId = null;
let allScans = [];

// ─── Initialization ────────────────────────────────────────
async function init() {
    console.log("🫁 TracheaAI System Booting...");
    
    try {
        // 1. Initialize Viewers
        viewer3d = new Viewer3D("viewer-canvas");
        sliceViewer = new SliceViewer("slice-canvas", "profile-chart", API_BASE);
        
        // 2. Load Patient Data
        await loadScanList();
        
        // 3. Bind UI Events
        setupEvents();

        // 4. Initial Resize
        viewer3d._onResize();
        
        console.log("✅ System Startup Complete");

    } catch (err) {
        console.error("❌ Critical Startup Error:", err);
    }
}

// ─── Data Management ────────────────────────────────────────
async function loadScanList() {
    // Priority: Manifest (Static) -> API (Live)
    if (window.TRACHEA_SCAN_DATA && window.TRACHEA_SCAN_DATA.length > 0) {
        allScans = window.TRACHEA_SCAN_DATA;
    } else {
        try {
            const res = await fetch(`${API_BASE}/api/scans`);
            if (res.ok) allScans = await res.json();
        } catch (err) {
            console.warn("Could not fetch scans from API, using empty list.");
        }
    }

    renderPatientList(allScans);
}

function renderPatientList(scans) {
    const list = document.getElementById("patient-list");
    if (!list) return;
    list.innerHTML = "";
    
    if (scans.length === 0) {
        list.innerHTML = '<div class="loading-state">No clinical records found.</div>';
        return;
    }

    scans.forEach(scan => {
        const stenosis = scan.stats?.max_stenosis_pct || 0;
        const severity = stenosis > 40 ? "critical" : stenosis > 15 ? "warning" : "normal";
        const dotColor = severity === "critical" ? "var(--accent-red)" : severity === "warning" ? "var(--accent-yellow)" : "var(--accent-green)";

        const item = document.createElement("div");
        item.className = `patient-item ${currentScanId === scan.scan_id ? 'active' : ''}`;
        item.onclick = () => loadScan(scan.scan_id);
        
        // Shorten ID for display
        const displayId = scan.scan_id.split('__')[0].replace('LIDC-IDRI-', 'LIDC-');

        item.innerHTML = `
            <div class="patient-item-header">
                <span class="patient-item-id">${displayId}</span>
                <span class="severity-dot" style="background:${dotColor}"></span>
            </div>
            <div class="patient-item-meta">
                <span>Stenosis: ${stenosis.toFixed(1)}%</span>
            </div>
        `;
        list.appendChild(item);
    });
}

async function loadScan(scanId) {
    if (currentScanId === scanId) return;
    
    const welcomeView = document.getElementById("welcome-view");
    const analysisPanel = document.getElementById("analysis-panel");
    const statusText = document.getElementById("viewer-info");

    if (!scanId) {
        currentScanId = null;
        viewer3d.clearAll();
        welcomeView?.classList.remove("hidden");
        analysisPanel?.classList.add("hidden");
        statusText.textContent = "SYSTEM READY // SELECT A CASE";
        renderPatientList(allScans);
        return;
    }

    currentScanId = scanId;
    renderPatientList(allScans); // Update active state
    
    welcomeView?.classList.add("hidden");
    analysisPanel?.classList.remove("hidden");
    statusText.textContent = `ANALYZING: ${scanId.split('__')[0]}`;

    try {
        // Try to find the scan in our local allScans list first (data manifest)
        const localMeta = allScans.find(s => s.scan_id === scanId);
        let meta = localMeta;

        if (!meta) {
            // Fallback: Fetch metadata via API or static path
            const paths = [
                `${API_BASE}/api/scan/${scanId}`,
                `${API_BASE}/data/${scanId}/metadata.json`,
                `data/${scanId}/metadata.json`
            ];
            
            for (const path of paths) {
                try {
                    const res = await fetch(path);
                    if (res.ok) {
                        meta = await res.json();
                        break;
                    }
                } catch(e) {}
            }
        }

        if (!meta) throw new Error("Metadata not found");

        // Load 3D Meshes
        viewer3d.clearAll();
        
        // Define mesh paths
        const meshBase = meta.meshes?.diseased ? 
            (meta.meshes.diseased.startsWith('http') ? '' : `${API_BASE}/data/${scanId}/`) : 
            `${API_BASE}/api/scan/${scanId}/mesh/`;

        await viewer3d.loadMesh(meta.meshes?.diseased ? `${meshBase}${meta.meshes.diseased}` : `${meshBase}diseased`, "diseased");
        await viewer3d.loadMesh(meta.meshes?.healthy ? `${meshBase}${meta.meshes.healthy}` : `${meshBase}healthy`, "healthy");
        
        // Load context meshes
        if (meta.meshes?.context) {
            for (const [layer, path] of Object.entries(meta.meshes.context)) {
                try { await viewer3d.loadMesh(`${meshBase}${path}`, layer); } catch(e) {}
            }
        }

        updateMetrics(meta.stats || {});
        
        // Update 2D Slice Viewer
        if (sliceViewer) {
            const dims = meta.dimensions || { axial: 200, coronal: 200, sagittal: 200 };
            await sliceViewer.loadScan(scanId, dims, meta.cross_sections);
        }

    } catch (err) {
        console.error("Failed to load patient record:", err);
        statusText.textContent = "ERROR LOADING RECORD";
    }
}

// ─── UI Orchestration ───────────────────────────────────────
function updateMetrics(stats) {
    if (!stats) return;
    const maxS = stats.max_stenosis_pct || 0;
    
    const setVal = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.textContent = val;
    };

    setVal("stat-vol-diseased", `${(stats.volume_diseased_cm3 || 0).toFixed(1)} cm³`);
    setVal("stat-stenosis", `${maxS.toFixed(1)}%`);
    setVal("stat-min-diam", `${(stats.min_diameter_mm || 0).toFixed(1)} mm`);

    const badge = document.getElementById("severity-badge");
    const label = document.getElementById("severity-label");
    if (badge && label) {
        if (maxS > 40) {
            label.textContent = "CRITICAL STENOSIS";
            badge.style.background = "rgba(239, 68, 68, 0.1)";
            badge.style.color = "var(--accent-red)";
            badge.style.borderColor = "rgba(239, 68, 68, 0.2)";
        } else if (maxS > 15) {
            label.textContent = "MODERATE OBSTRUCTION";
            badge.style.background = "rgba(245, 158, 11, 0.1)";
            badge.style.color = "var(--accent-yellow)";
            badge.style.borderColor = "rgba(245, 158, 11, 0.2)";
        } else {
            label.textContent = "NORMAL ANATOMY";
            badge.style.background = "rgba(16, 185, 129, 0.1)";
            badge.style.color = "var(--accent-green)";
            badge.style.borderColor = "rgba(16, 185, 129, 0.2)";
        }
    }

    renderClinicalReport(stats);
}

function renderClinicalReport(stats) {
    const container = document.getElementById("report-content");
    if (!container) return;
    
    const maxS = stats.max_stenosis_pct || 0;
    const date = new Date().toLocaleDateString();

    container.innerHTML = `
        <div style="font-family:var(--font-mono); font-size:11px; margin-bottom:12px; color:var(--text-dim);">
            REPORT DATE: ${date}
        </div>
        <p style="margin-bottom:12px;"><strong>Automated Morphological Assessment:</strong></p>
        <p style="margin-bottom:8px;">The AI pipeline detected a maximal luminal narrowing of <strong>${maxS.toFixed(1)}%</strong>.</p>
        <p style="margin-bottom:8px;">Total airway volume is measured at <strong>${(stats.volume_diseased_cm3 || 0).toFixed(1)} cm³</strong> compared to a predicted healthy volume of <strong>${(stats.volume_healthy_cm3 || 0).toFixed(1)} cm³</strong>.</p>
        <p style="margin-top:16px; border-top:1px solid var(--border); padding-top:12px; color:var(--text-muted); font-style:italic;">
            Note: This report is generated by an experimental AI system and should be verified by a board-certified radiologist.
        </p>
    `;
}

function setupEvents() {
    // Search
    const search = document.getElementById("patient-search");
    if (search) {
        search.oninput = (e) => {
            const query = e.target.value.toLowerCase();
            renderPatientList(allScans.filter(s => s.scan_id.toLowerCase().includes(query)));
        };
    }

    // Display Modes
    document.querySelectorAll("#display-mode .btn-toggle").forEach(btn => {
        btn.onclick = () => {
            document.querySelectorAll("#display-mode .btn-toggle").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            viewer3d?.setDisplayMode(btn.dataset.mode);
        };
    });

    // Opacity
    const opacity = document.getElementById("opacity-slider");
    if (opacity) {
        opacity.oninput = (e) => viewer3d?.setOpacity(e.target.value / 100);
    }

    // Context Layers
    document.querySelectorAll(".context-toggle").forEach(toggle => {
        toggle.onchange = (e) => {
            viewer3d?.setContextVisibility(e.target.dataset.layer, e.target.checked);
        };
    });

    // Tabs
    document.querySelectorAll(".tab-btn").forEach(btn => {
        btn.onclick = () => {
            document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            
            const view = btn.dataset.view;
            document.getElementById("ct-view-container")?.classList.toggle("hidden", view !== "ct");
            document.getElementById("report-view-container")?.classList.toggle("hidden", view !== "report");
        };
    });

    // Modal
    const modal = document.getElementById("process-modal");
    document.getElementById("btn-process").onclick = () => modal?.classList.remove("hidden");
    document.getElementById("btn-cancel-process").onclick = () => modal?.classList.add("hidden");
    
    document.getElementById("btn-start-process").onclick = async () => {
        const path = document.getElementById("input-path").value;
        if (!path) return alert("Please enter a DICOM path.");
        
        try {
            const res = await fetch(`${API_BASE}/api/process`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ input_path: path })
            });
            if (res.ok) {
                alert("Processing started successfully.");
                modal?.classList.add("hidden");
            } else {
                alert("Failed to start processing.");
            }
        } catch (err) {
            alert("Network error.");
        }
    };
}

// ─── Boot ──────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", init);
