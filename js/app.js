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
    statusText.textContent = `LOADING: ${scanId.split('__')[0]}...`;

    try {
        // Try to find the scan in our local allScans list first (data manifest)
        const localMeta = allScans.find(s => s.scan_id === scanId);
        let meta = localMeta;

        if (!meta) {
            // Fallback: Fetch metadata via API or static path
            const paths = [
                `${API_BASE}/api/scan/${scanId}`,
                `${API_BASE}/data/${scanId}/metadata.json`,
                `${API_BASE}/public/data/${scanId}/metadata.json`,
                `data/${scanId}/metadata.json`,
                `public/data/${scanId}/metadata.json`,
                `/data/${scanId}/metadata.json`
            ];
            
            for (const path of paths) {
                try {
                    const res = await fetch(path);
                    if (res.ok) {
                        meta = await res.json();
                        console.log(`✅ Metadata loaded from: ${path}`);
                        break;
                    }
                } catch(e) {}
            }
        }

        if (!meta) throw new Error("Metadata record not found on server.");

        // Load 3D Meshes
        viewer3d.clearAll();
        
        // Try multiple bases for meshes
        const meshBases = [
            `${API_BASE}/data/${scanId}/`,
            `${API_BASE}/data/${scanId}/meshes/`,
            `${API_BASE}/public/data/${scanId}/`,
            `${API_BASE}/public/data/${scanId}/meshes/`,
            `${API_BASE}/api/scan/${scanId}/mesh/`
        ];

        const loadMeshWithRetry = async (type, fallbackName) => {
            const fileName = (meta.meshes?.[type] || fallbackName).split('/').pop();
            let loaded = false;
            for (const base of meshBases) {
                try {
                    const url = `${base}${fileName}`;
                    console.log(`📡 Trying mesh: ${url}`);
                    await viewer3d.loadMesh(url, type);
                    loaded = true;
                    console.log(`✅ Loaded ${type} from ${base}`);
                    break;
                } catch(e) {}
            }
            if (!loaded) console.warn(`Could not load ${type} mesh from any path.`);
        };

        statusText.textContent = `RENDERING: ${scanId.split('__')[0]}...`;
        await loadMeshWithRetry("diseased", "diseased.glb");
        await loadMeshWithRetry("healthy", "healthy.glb");
        
        // Load context meshes
        if (meta.meshes?.context) {
            for (const [layer, path] of Object.entries(meta.meshes.context)) {
                try { await loadMeshWithRetry(layer, path); } catch(e) {}
            }
        }

        updateMetrics(meta.stats || {}, meta.anomalies || []);
        statusText.textContent = `ANALYSIS COMPLETE: ${scanId.split('__')[0]}`;
        
        // Update 2D Slice Viewer
        if (sliceViewer) {
            const dims = meta.dimensions || { axial: 200, coronal: 200, sagittal: 200 };
            await sliceViewer.loadScan(scanId, dims, meta.cross_sections);
        }

    } catch (err) {
        console.error("❌ Failed to load patient record:", err);
        statusText.textContent = `ERROR: COULD NOT LOAD RECORD [${scanId.split('__')[0]}]`;
        alert(`Error loading data. Path tried might be incorrect on Vercel.`);
    }
}

// ─── UI Orchestration ───────────────────────────────────────
function updateMetrics(stats, anomalies = []) {
    if (!stats) return;
    const maxS = stats.max_stenosis_pct || 0;
    
    // 1. Calculate Cotton-Myer Grade (Pediatric & Adult Standard)
    let cottonMyer = "N/A";
    if (maxS > 0 && maxS <= 50) cottonMyer = "Grade I";
    else if (maxS > 50 && maxS <= 70) cottonMyer = "Grade II";
    else if (maxS > 70 && maxS <= 99) cottonMyer = "Grade III";
    else if (maxS > 99) cottonMyer = "Grade IV";
    else if (maxS === 0) cottonMyer = "Normal";

    // 2. Calculate Obstruction Length (Z-range of significant deviation)
    let obsLength = 0;
    const stenoticPoints = (anomalies || []).filter(a => a.type === "stenosis" && a.deviation_pct > 15);
    if (stenoticPoints.length > 1) {
        const zValues = stenoticPoints.map(p => p.z_mm);
        obsLength = Math.max(...zValues) - Math.min(...zValues);
    } else if (stenoticPoints.length === 1) {
        obsLength = 1.0; // Minimal focal point
    }

    // 3. Estimate Reference Diameter (Expected normal anatomy at lesion site)
    let refDiam = stats.mean_healthy_diameter_mm || 0;
    if (anomalies && anomalies.length > 0) {
        const maxStenPoint = anomalies.reduce((prev, curr) => (curr.deviation_pct > prev.deviation_pct) ? curr : prev, { deviation_pct: -1 });
        if (maxStenPoint.expected_mm) refDiam = maxStenPoint.expected_mm;
    }

    const setVal = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.textContent = val;
    };

    setVal("stat-vol-diseased", `${(stats.volume_diseased_cm3 || stats.diseased_volume_mm3 / 1000 || 0).toFixed(1)} cm³`);
    setVal("stat-stenosis", `${maxS.toFixed(1)}%`);
    setVal("stat-min-diam", `${(stats.min_diameter_mm || stats.min_diseased_diameter_mm || 0).toFixed(1)} mm`);
    setVal("stat-cotton-myer", cottonMyer);
    setVal("stat-obs-length", `${obsLength.toFixed(1)} mm`);
    setVal("stat-ref-diam", `${refDiam.toFixed(1)} mm`);

    const badge = document.getElementById("severity-badge");
    const label = document.getElementById("severity-label");
    if (badge && label) {
        if (maxS > 70) {
            label.textContent = "CRITICAL OBSTRUCTION";
            badge.style.background = "rgba(239, 68, 68, 0.15)";
            badge.style.color = "var(--accent-red)";
            badge.style.borderColor = "rgba(239, 68, 68, 0.4)";
        } else if (maxS > 30) {
            label.textContent = "MODERATE STENOSIS";
            badge.style.background = "rgba(245, 158, 11, 0.15)";
            badge.style.color = "var(--accent-yellow)";
            badge.style.borderColor = "rgba(245, 158, 11, 0.4)";
        } else if (maxS > 10) {
            label.textContent = "MILD DEVIATION";
            badge.style.background = "rgba(59, 130, 246, 0.15)";
            badge.style.color = "var(--accent-blue)";
            badge.style.borderColor = "rgba(59, 130, 246, 0.4)";
        } else {
            label.textContent = "NORMAL ANATOMY";
            badge.style.background = "rgba(16, 185, 129, 0.15)";
            badge.style.color = "var(--accent-green)";
            badge.style.borderColor = "rgba(16, 185, 129, 0.4)";
        }
    }

    renderClinicalReport(stats, cottonMyer, obsLength);
}

function renderClinicalReport(stats, cottonMyer, obsLength) {
    const container = document.getElementById("report-content");
    if (!container) return;
    
    const maxS = stats.max_stenosis_pct || 0;
    const date = new Date().toLocaleDateString();

    container.innerHTML = `
        <div style="font-family:var(--font-mono); font-size:10px; margin-bottom:16px; color:var(--text-dim); text-align:right;">
            DOCUMENT ID: ${Math.random().toString(36).substr(2, 9).toUpperCase()} // DATE: ${date}
        </div>
        <div style="border-left: 3px solid var(--accent-blue); padding-left: 16px; margin-bottom: 24px;">
            <p style="font-weight: 800; font-size: 11px; text-transform: uppercase; color: var(--accent-blue); margin-bottom: 4px;">Assessment Summary</p>
            <p>Automated analysis of the tracheal lumen indicates a <strong>${maxS.toFixed(1)}%</strong> reduction in cross-sectional area at the point of maximal narrowing.</p>
        </div>

        <div style="margin-bottom: 20px;">
            <p style="font-weight: 700; margin-bottom: 8px;">Clinical Classification:</p>
            <ul style="padding-left: 20px; list-style-type: square;">
                <li><strong>Cotton-Myer Scale:</strong> ${cottonMyer}</li>
                <li><strong>Morphological Type:</strong> ${obsLength > 20 ? 'Diffuse' : 'Focal'} Narrowing</li>
                <li><strong>Craniocaudal Extent:</strong> ${obsLength.toFixed(1)} mm</li>
            </ul>
        </div>

        <div style="margin-bottom: 20px;">
            <p style="font-weight: 700; margin-bottom: 8px;">Volumetric Findings:</p>
            <p>Total airway volume: <strong>${(stats.volume_diseased_cm3 || stats.diseased_volume_mm3 / 1000 || 0).toFixed(1)} cm³</strong></p>
            <p>Predicted physiological volume: <strong>${(stats.volume_healthy_cm3 || stats.healthy_volume_mm3 / 1000 || 0).toFixed(1)} cm³</strong></p>
            <p>Net volume deficit: <strong>${(stats.volume_change_pct || 0).toFixed(1)}%</strong></p>
        </div>
        
        <p style="margin-top:24px; border-top:1px solid rgba(255,255,255,0.05); padding-top:16px; color:var(--text-dim); font-style:italic; font-size:11px;">
            DISCLAIMER: This report is an AI-generated pedagogical tool for medical students. It does not constitute a certified medical diagnosis and must be reviewed against raw DICOM data by qualified personnel.
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
