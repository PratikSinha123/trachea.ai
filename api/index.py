"""Vercel Serverless API for TracheaAI.

Serves pre-processed scan data from the public/data/ directory.
This is a lightweight serverless function — no heavy ML dependencies needed.

Vercel routes all /api/* requests to this handler.
"""

import json
import os
from http.server import BaseHTTPRequestHandler

# On Vercel, static files in public/ are served at the root.
# But the API function can read them from the filesystem too.
# Vercel project root is at /var/task (or wherever the function runs).
# Try to find the public/data directory robustly
_possible_roots = [
    os.getcwd(),
    os.path.dirname(os.path.abspath(__file__)),
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "/var/task"
]
DATA_ROOT = None
for r in _possible_roots:
    candidate = os.path.join(r, "public", "data")
    if os.path.isdir(candidate):
        DATA_ROOT = candidate
        break
if not DATA_ROOT:
    DATA_ROOT = os.path.join(os.getcwd(), "public", "data")  # default fallback


def _read_json(path):
    """Read and parse a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def _json_response(handler, data, status=200):
    """Send a JSON response."""
    body = json.dumps(data).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Cache-Control", "public, max-age=3600")
    handler.end_headers()
    handler.wfile.write(body)


def _file_response(handler, path, content_type, cache_max_age=86400):
    """Send a file response."""
    if not os.path.isfile(path):
        _json_response(handler, {"error": "Not found"}, 404)
        return

    with open(path, "rb") as f:
        data = f.read()

    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Cache-Control", f"public, max-age={cache_max_age}")
    handler.end_headers()
    handler.wfile.write(data)


def _error_response(handler, message, status=400):
    """Send an error response."""
    _json_response(handler, {"error": message}, status)


class handler(BaseHTTPRequestHandler):
    """Vercel serverless function handler for TracheaAI API."""

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        """Route GET requests to the appropriate handler."""
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        
        # If Vercel rewrites /api/foo to /api/index.py?route=foo
        if "route" in query:
            route_val = query["route"][0].strip("/")
            path = "/api/" + route_val if route_val else "/api"
        else:
            path = parsed.path.rstrip("/")

        # /api/scans — list all scans
        if path == "/api/scans":
            return self._handle_scans()

        # /api/scan/{scan_id} — scan metadata
        parts = path.split("/")

        # /api/scan/{scan_id}
        if len(parts) == 4 and parts[1] == "api" and parts[2] == "scan":
            return self._handle_scan_meta(parts[3])

        # /api/scan/{scan_id}/mesh/{mesh_type}
        if (len(parts) == 6 and parts[1] == "api" and parts[2] == "scan"
                and parts[4] == "mesh"):
            return self._handle_mesh(parts[3], parts[5])

        # /api/scan/{scan_id}/morph/{frame_index}
        if (len(parts) == 6 and parts[1] == "api" and parts[2] == "scan"
                and parts[4] == "morph"):
            return self._handle_morph(parts[3], parts[5])

        # /api/scan/{scan_id}/morph_count
        if (len(parts) == 5 and parts[1] == "api" and parts[2] == "scan"
                and parts[4] == "morph_count"):
            return self._handle_morph_count(parts[3])

        # /api/scan/{scan_id}/dimensions
        if (len(parts) == 5 and parts[1] == "api" and parts[2] == "scan"
                and parts[4] == "dimensions"):
            return self._handle_dimensions(parts[3])

        # /api/scan/{scan_id}/slice/{axis}/{index}
        if (len(parts) == 7 and parts[1] == "api" and parts[2] == "scan"
                and parts[4] == "slice"):
            return self._handle_slice(parts[3], parts[5], parts[6])

        # /api/config — serve config JS
        if path == "/api/config":
            return self._handle_config()

        _error_response(self, f"Unknown API route: {path}", 404)

    def do_POST(self):
        """Handle POST requests — limited functionality on Vercel."""
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if "route" in query:
            route_val = query["route"][0].strip("/")
            path = "/api/" + route_val if route_val else "/api"
        else:
            path = parsed.path.rstrip("/")

        if path in ("/api/process", "/api/nnunet/predict"):
            _json_response(self, {
                "error": "Processing is only available when running locally. "
                         "This Vercel deployment serves pre-processed scans only.",
                "hint": "Run the pipeline locally: python auto_pipeline.py --ai /path/to/dicom"
            }, 501)
            return

        _error_response(self, f"Unknown API route: {path}", 404)

    # ── Route Handlers ──────────────────────────────────────────

    def _handle_scans(self):
        """Return list of all processed scans."""
        scans_path = os.path.join(DATA_ROOT, "scans.json")
        if os.path.isfile(scans_path):
            _json_response(self, _read_json(scans_path))
        else:
            _json_response(self, [])

    def _handle_scan_meta(self, scan_id):
        """Return metadata for a specific scan."""
        meta_path = os.path.join(DATA_ROOT, scan_id, "metadata.json")
        if not os.path.isfile(meta_path):
            _error_response(self, f"Scan '{scan_id}' not found", 404)
            return
        _json_response(self, _read_json(meta_path))

    def _handle_mesh(self, scan_id, mesh_type):
        """Serve a GLB mesh file."""
        allowed = {"diseased", "healthy", "aorta", "pulmonary_artery", "heart", "body"}
        if mesh_type not in allowed:
            _error_response(self, f"mesh_type must be one of {allowed}")
            return

        path = os.path.join(DATA_ROOT, scan_id, "meshes", f"{mesh_type}.glb")
        _file_response(self, path, "model/gltf-binary")

    def _handle_morph(self, scan_id, frame_index_str):
        """Serve a morph animation frame GLB."""
        try:
            frame_index = int(frame_index_str)
        except ValueError:
            _error_response(self, "frame_index must be an integer")
            return

        path = os.path.join(DATA_ROOT, scan_id, "meshes", "morph", f"morph_{frame_index:03d}.glb")
        _file_response(self, path, "model/gltf-binary")

    def _handle_morph_count(self, scan_id):
        """Return the number of morph frames."""
        morph_dir = os.path.join(DATA_ROOT, scan_id, "meshes", "morph")
        if not os.path.isdir(morph_dir):
            _json_response(self, {"count": 0})
            return

        count = len([f for f in os.listdir(morph_dir) if f.endswith(".glb")])
        _json_response(self, {"count": count})

    def _handle_dimensions(self, scan_id):
        """Return CT scan dimensions from the export manifest."""
        manifest_path = os.path.join(DATA_ROOT, scan_id, "export_manifest.json")
        if not os.path.isfile(manifest_path):
            # Fallback: try metadata
            meta_path = os.path.join(DATA_ROOT, scan_id, "metadata.json")
            if os.path.isfile(meta_path):
                # Estimate from cross-sections
                meta = _read_json(meta_path)
                cs = meta.get("cross_sections", [])
                if cs:
                    max_z = max(c.get("z_index", 0) for c in cs) + 1
                    _json_response(self, {
                        "axial": max_z,
                        "coronal": 512,  # Reasonable default
                        "sagittal": 512,
                    })
                    return
            _error_response(self, "Dimensions not available", 404)
            return

        manifest = _read_json(manifest_path)
        dims = manifest.get("dimensions", {})
        _json_response(self, dims)

    def _handle_slice(self, scan_id, axis, index_str):
        """Serve a pre-rendered slice PNG."""
        if axis not in ("axial", "coronal", "sagittal"):
            _error_response(self, "axis must be 'axial', 'coronal', or 'sagittal'")
            return

        try:
            index = int(index_str)
        except ValueError:
            _error_response(self, "index must be an integer")
            return

        # Find the closest pre-rendered slice
        manifest_path = os.path.join(DATA_ROOT, scan_id, "export_manifest.json")
        if os.path.isfile(manifest_path):
            manifest = _read_json(manifest_path)
            slices = manifest.get("slice_manifest", {}).get(axis, [])

            if slices:
                # Find nearest pre-rendered slice
                closest = min(slices, key=lambda s: abs(s["index"] - index))
                slice_path = os.path.join(DATA_ROOT, scan_id, closest["file"])
                _file_response(self, slice_path, "image/png")
                return

        # Direct file lookup (if slices were exported with index-based names)
        direct_path = os.path.join(DATA_ROOT, scan_id, "slices", axis, f"{index:04d}.png")
        if os.path.isfile(direct_path):
            _file_response(self, direct_path, "image/png")
            return

        _error_response(self, f"Slice not found: {axis}/{index}", 404)

    def _handle_config(self):
        """Serve the config JavaScript."""
        api_base = os.environ.get("TRACHEA_API_BASE_URL", "")
        js = f'window.TRACHEA_API_BASE_URL = {json.dumps(api_base)};'
        body = js.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/javascript; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)
