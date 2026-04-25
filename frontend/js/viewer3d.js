/**
 * TracheaAI — Interactive 3D Trachea Viewer
 *
 * Features:
 *  - GLB mesh loading (diseased + healthy)
 *  - Morph animation with frame interpolation
 *  - OrbitControls for rotate/zoom/pan
 *  - Wireframe toggle
 *  - Opacity control
 *  - Premium lighting setup
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

export class Viewer3D {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.loader = new GLTFLoader();

        // Meshes
        this.diseasedMesh = null;
        this.healthyMesh = null;
        this.morphMeshes = [];
        this.contextMeshes = {
            body: null,
            heart: null,
            aorta: null,
            pulmonary_artery: null
        };
        this.contextVisible = {
            body: false,
            heart: false,
            vessels: false
        };
        this.activeMorphFrame = -1;

        // State
        this.displayMode = "diseased";
        this.opacity = 0.85;
        this.wireframe = false;
        this.isAnimating = false;
        this.animationId = null;
        this.crossSectionEnabled = false;
        this.crossSectionPlane = new THREE.Plane(new THREE.Vector3(1, 0, 0), 0); // sagittal cut
        this.annotations = [];
        this._breathT = 0;
        this._breathEnabled = true;

        this._init();
    }

    _init() {
        // Renderer
        this.renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            antialias: true,
            alpha: true,
        });
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.setClearColor(0x0a0d14, 1);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.5;
        this.renderer.localClippingEnabled = true; // needed for cross-section

        // Scene
        this.scene = new THREE.Scene();

        // Camera
        const rect = this.canvas.parentElement.getBoundingClientRect();
        this.camera = new THREE.PerspectiveCamera(
            45, rect.width / rect.height, 0.1, 5000
        );
        this.camera.position.set(0, 0, 250);

        // Controls
        this.controls = new OrbitControls(this.camera, this.canvas);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.08;
        this.controls.rotateSpeed = 0.8;
        this.controls.zoomSpeed = 1.2;
        this.controls.panSpeed = 0.8;
        this.controls.minDistance = 20;
        this.controls.maxDistance = 1000;

        // Lighting
        this._setupLighting();

        // Grid helper
        this._setupGrid();

        // Handle resize
        this._onResize();
        window.addEventListener("resize", () => this._onResize());

        // Start render loop
        this._animate();
    }

    _setupLighting() {
        // Warm ambient — like an operating theatre, not a dark cave
        const ambient = new THREE.AmbientLight(0xffe5d0, 0.9);
        this.scene.add(ambient);

        // Primary surgical key light — bright, warm white
        const key = new THREE.DirectionalLight(0xfff8f0, 2.2);
        key.position.set(80, 180, 200);
        key.castShadow = true;
        key.shadow.mapSize.width = 2048;
        key.shadow.mapSize.height = 2048;
        this.scene.add(key);

        // Soft fill from the left (reduces harsh shadows)
        const fill = new THREE.DirectionalLight(0xffe0d0, 0.9);
        fill.position.set(-150, 80, 100);
        this.scene.add(fill);

        // Warm back-light (simulates tissue translucency / subsurface scatter)
        const back = new THREE.DirectionalLight(0xff9060, 0.7);
        back.position.set(0, -100, -200);
        this.scene.add(back);

        // Top-down light for realistic depth
        const top = new THREE.DirectionalLight(0xffffff, 0.5);
        top.position.set(0, 300, 0);
        this.scene.add(top);

        // Hemisphere for gentle sky/ground gradient
        const hemi = new THREE.HemisphereLight(0xfff5ee, 0x3a1a10, 0.4);
        this.scene.add(hemi);
    }

    _setupGrid() {
        const grid = new THREE.GridHelper(400, 40, 0x1a2233, 0x111620);
        grid.position.y = -100;
        grid.material.opacity = 0.3;
        grid.material.transparent = true;
        this.scene.add(grid);
    }

    _onResize() {
        const rect = this.canvas.parentElement.getBoundingClientRect();
        this.camera.aspect = rect.width / rect.height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(rect.width, rect.height);
    }

    _animate() {
        requestAnimationFrame(() => this._animate());
        this.controls.update();

        // Subtle breathing animation — slow pulsation
        if (this._breathEnabled && this.diseasedMesh) {
            this._breathT += 0.012;
            const scale = 1.0 + 0.012 * Math.sin(this._breathT);
            this.diseasedMesh.scale.setScalar(scale);
            if (this.healthyMesh) this.healthyMesh.scale.setScalar(scale);
        }

        this.renderer.render(this.scene, this.camera);
    }

    // ─── Mesh Loading ──────────────────────────────────────────

    async loadMesh(url, type = "diseased") {
        return new Promise((resolve, reject) => {
            this.loader.load(
                url,
                (gltf) => {
                    const mesh = gltf.scene;

                    // Apply material overrides + force smooth shading
                    mesh.traverse((child) => {
                        if (child.isMesh) {
                            // CRITICAL: recompute smooth per-vertex normals client-side
                            // This eliminates the flat-faceted triangulated appearance
                            child.geometry.computeVertexNormals();
                            child.material = this._createMaterial(type);
                            child.material.flatShading = false;
                            child.castShadow = true;
                            child.receiveShadow = true;
                        }
                    });

                    if (type === "diseased") {
                        if (this.diseasedMesh) this.scene.remove(this.diseasedMesh);
                        this.diseasedMesh = mesh;
                    } else if (type === "healthy") {
                        if (this.healthyMesh) this.scene.remove(this.healthyMesh);
                        this.healthyMesh = mesh;
                    } else if (this.contextMeshes.hasOwnProperty(type)) {
                        if (this.contextMeshes[type]) this.scene.remove(this.contextMeshes[type]);
                        this.contextMeshes[type] = mesh;
                    }

                    this.scene.add(mesh);
                    this._centerCamera(mesh);
                    this._updateVisibility();

                    resolve(mesh);
                },
                undefined,
                (error) => {
                    console.error(`Failed to load mesh: ${url}`, error);
                    reject(error);
                }
            );
        });
    }

    async loadMorphFrame(url, index) {
        return new Promise((resolve, reject) => {
            this.loader.load(
                url,
                (gltf) => {
                    const mesh = gltf.scene;
                    mesh.traverse((child) => {
                        if (child.isMesh) {
                            const t = index / Math.max(this.morphMeshes.length, 1);
                            child.material = this._createMorphMaterial(t);
                            child.castShadow = true;
                        }
                    });
                    mesh.visible = false;
                    this.morphMeshes[index] = mesh;
                    this.scene.add(mesh);
                    resolve(mesh);
                },
                undefined,
                reject
            );
        });
    }

    _createMaterial(type) {
        // Real tissue-like materials — stenosis heatmap for diseased
        const configs = {
            // Diseased trachea: vertex-colored stenosis heatmap (green=normal, red=severe)
            // Semi-transparent so you can see the HOLLOW airway lumen inside
            diseased: {
                color: 0xffffff,        // vertex colors override this
                vertexColors: true,     // use per-vertex stenosis colors
                emissive: 0x0a0000,
                emissiveIntensity: 0.05,
                roughness: 0.55,
                metalness: 0.0,
                clearcoat: 0.5,
                clearcoatRoughness: 0.3,
                opacityMult: 0.72,      // semi-transparent → hollow lumen visible inside
            },
            // Healthy trachea: warm pink tissue, opaque
            healthy: {
                color: 0xe87070,
                emissive: 0x1a0808,
                emissiveIntensity: 0.05,
                roughness: 0.55,
                metalness: 0.0,
                clearcoat: 0.7,
                clearcoatRoughness: 0.2,
                sheen: 0.5,
                sheenColor: 0xf0a0a0,
                sheenRoughness: 0.4,
                opacityMult: 0.78,
            },
            // Ghost overlay when showing both
            healthy_ghost: {
                color: 0x34d399,
                emissive: 0x0a2a1a,
                emissiveIntensity: 0.2,
                roughness: 0.3,
                metalness: 0.0,
                clearcoat: 0.2,
                clearcoatRoughness: 0.5,
                opacityMult: 0.22,
            },
            // Context layers
            body: { color: 0xf0e8e0, emissive: 0x000000, emissiveIntensity: 0, roughness: 0.1, metalness: 0.0, transmission: 0.92, ior: 1.3, clearcoat: 0.0, opacityMult: 0.12, depthWrite: false },
            heart: { color: 0x8b2020, emissive: 0x200505, emissiveIntensity: 0.1, roughness: 0.65, metalness: 0.0, clearcoat: 0.4, clearcoatRoughness: 0.3, opacityMult: 0.45 },
            aorta: { color: 0xcc2222, emissive: 0x220000, emissiveIntensity: 0.1, roughness: 0.4, metalness: 0.0, clearcoat: 0.5, clearcoatRoughness: 0.2, opacityMult: 0.55 },
            pulmonary_artery: { color: 0x3355bb, emissive: 0x000822, emissiveIntensity: 0.1, roughness: 0.4, metalness: 0.0, clearcoat: 0.5, clearcoatRoughness: 0.2, opacityMult: 0.55 },
        };

        const c = configs[type] || configs.diseased;
        const targetOpacity = this.opacity * (c.opacityMult ?? 1.0);

        return new THREE.MeshPhysicalMaterial({
            color: c.color,
            vertexColors: c.vertexColors === true,
            emissive: c.emissive ?? 0x000000,
            emissiveIntensity: c.emissiveIntensity ?? 0.1,
            metalness: c.metalness ?? 0.0,
            roughness: c.roughness ?? 0.6,
            transmission: c.transmission ?? 0.0,
            ior: c.ior ?? 1.5,
            clearcoat: c.clearcoat ?? 0.0,
            clearcoatRoughness: c.clearcoatRoughness ?? 0.5,
            sheen: c.sheen ?? 0.0,
            sheenColor: c.sheenColor ? new THREE.Color(c.sheenColor) : new THREE.Color(0xffffff),
            sheenRoughness: c.sheenRoughness ?? 0.5,
            transparent: true,
            opacity: targetOpacity,
            wireframe: this.wireframe,
            side: THREE.DoubleSide,
            clippingPlanes: this.crossSectionEnabled ? [this.crossSectionPlane] : [],
            clipShadows: true,
            depthWrite: c.depthWrite !== undefined ? c.depthWrite : (targetOpacity > 0.6),
        });
    }

    _createMorphMaterial(t) {
        // Interpolate color from red (diseased) to green (healthy)
        const colorA = new THREE.Color(0xe84040);
        const colorB = new THREE.Color(0x34d399);
        const color = colorA.clone().lerp(colorB, t);

        const emissiveA = new THREE.Color(0x3a0a0a);
        const emissiveB = new THREE.Color(0x0a2a1a);
        const emissive = emissiveA.clone().lerp(emissiveB, t);

        return new THREE.MeshPhysicalMaterial({
            color: color,
            emissive: emissive,
            emissiveIntensity: 0.15,
            metalness: 0.1,
            roughness: 0.55,
            transparent: true,
            opacity: this.opacity,
            wireframe: this.wireframe,
            side: THREE.DoubleSide,
            clearcoat: 0.3,
            clearcoatRoughness: 0.4,
        });
    }

    _centerCamera(mesh) {
        const box = new THREE.Box3().setFromObject(mesh);
        if (box.isEmpty()) {
            console.warn("Mesh bounding box is empty, skipping camera centering.");
            return;
        }

        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());

        const maxDim = Math.max(size.x, size.y, size.z);
        if (maxDim === 0 || isNaN(maxDim)) return;

        const fov = this.camera.fov * (Math.PI / 180);
        const dist = maxDim / (2 * Math.tan(fov / 2)) * 1.8;

        // Position camera slightly front-right for a clinical 3/4 view
        this.camera.position.set(
            center.x + dist * 0.3,
            center.y + dist * 0.1,
            center.z + dist * 0.95
        );
        this.controls.target.copy(center);
        this.controls.update();

        // Set cross-section plane through center of mesh
        this.crossSectionPlane.constant = center.x;
    }

    // ─── Display Controls ──────────────────────────────────────

    setDisplayMode(mode) {
        this.displayMode = mode;
        this._updateVisibility();
    }

    _updateVisibility() {
        const m = this.displayMode;

        if (this.diseasedMesh) {
            this.diseasedMesh.visible = (m === "diseased" || m === "both");
        }
        if (this.healthyMesh) {
            const showHealthy = (m === "healthy" || m === "both");
            this.healthyMesh.visible = showHealthy;
            // In 'both' mode, show healthy as transparent ghost so diseased is clear
            if (showHealthy) {
                this.healthyMesh.traverse((child) => {
                    if (child.isMesh) {
                        child.material = this._createMaterial(m === "both" ? "healthy_ghost" : "healthy");
                    }
                });
            }
        }

        // Context layers
        if (this.contextMeshes.body) {
            this.contextMeshes.body.visible = this.contextVisible.body;
        }
        if (this.contextMeshes.heart) {
            this.contextMeshes.heart.visible = this.contextVisible.heart;
        }
        if (this.contextMeshes.aorta) {
            this.contextMeshes.aorta.visible = this.contextVisible.vessels;
        }
        if (this.contextMeshes.pulmonary_artery) {
            this.contextMeshes.pulmonary_artery.visible = this.contextVisible.vessels;
        }

        // Hide morph meshes unless in morph mode
        for (const mesh of this.morphMeshes) {
            if (mesh) mesh.visible = false;
        }

        if (m === "morph") {
            if (this.diseasedMesh) this.diseasedMesh.visible = false;
            if (this.healthyMesh) this.healthyMesh.visible = false;

            // Auto-select first loaded frame if none active yet
            if (this.activeMorphFrame < 0) {
                const firstLoaded = this.morphMeshes.findIndex(x => x !== null);
                if (firstLoaded >= 0) this.activeMorphFrame = firstLoaded;
            }

            if (this.activeMorphFrame >= 0 && this.morphMeshes[this.activeMorphFrame]) {
                this.morphMeshes[this.activeMorphFrame].visible = true;
            }
        }
    }

    setMorphFrame(index) {
        // Hide previous
        if (this.activeMorphFrame >= 0 && this.morphMeshes[this.activeMorphFrame]) {
            this.morphMeshes[this.activeMorphFrame].visible = false;
        }
        this.activeMorphFrame = index;
        // Show this frame if in morph mode
        if (this.displayMode === "morph") {
            if (this.diseasedMesh) this.diseasedMesh.visible = false;
            if (this.healthyMesh) this.healthyMesh.visible = false;
            // Show nearest available frame if exact not loaded yet
            let target = index;
            while (target >= 0 && !this.morphMeshes[target]) target--;
            if (target < 0) {
                target = index;
                while (target < this.morphMeshes.length && !this.morphMeshes[target]) target++;
            }
            if (target >= 0 && target < this.morphMeshes.length && this.morphMeshes[target]) {
                this.morphMeshes[target].visible = true;
                this.activeMorphFrame = target;
            }
        }
    }

    setContextVisibility(layer, isVisible) {
        if (this.contextVisible.hasOwnProperty(layer)) {
            this.contextVisible[layer] = isVisible;
            this._updateVisibility();
        }
    }

    setCrossSectionEnabled(enabled) {
        this.crossSectionEnabled = enabled;
        // Rebuild materials with/without clipping plane
        const applyClip = (mesh, type) => {
            if (!mesh) return;
            mesh.traverse((child) => {
                if (child.isMesh) {
                    child.material = this._createMaterial(type);
                }
            });
        };
        applyClip(this.diseasedMesh, 'diseased');
        applyClip(this.healthyMesh, this.displayMode === 'both' ? 'healthy_ghost' : 'healthy');
    }

    setBreathingEnabled(enabled) {
        this._breathEnabled = enabled;
        if (!enabled && this.diseasedMesh) {
            this.diseasedMesh.scale.setScalar(1.0);
            if (this.healthyMesh) this.healthyMesh.scale.setScalar(1.0);
        }
    }

    addStenosisAnnotations(crossSections) {
        // Remove existing annotations
        this.annotations.forEach(a => this.scene.remove(a));
        this.annotations = [];

        if (!crossSections || crossSections.length === 0) return;

        // Find top 3 worst stenosis points
        const sorted = [...crossSections]
            .filter(cs => cs.deviation_pct > 15)
            .sort((a, b) => b.deviation_pct - a.deviation_pct)
            .slice(0, 3);

        // We need the mesh bounding box to position annotations
        if (!this.diseasedMesh) return;
        const box = new THREE.Box3().setFromObject(this.diseasedMesh);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());

        sorted.forEach((cs, i) => {
            // Estimate Y position from z_physical (0 = bottom of trachea)
            const zRange = crossSections[crossSections.length - 1].z_physical - crossSections[0].z_physical;
            const zFrac = (cs.z_physical - crossSections[0].z_physical) / Math.max(zRange, 1);
            const yPos = box.min.y + zFrac * size.y;

            // Glowing ring at the stenosis position
            const ringGeo = new THREE.TorusGeometry(size.x * 0.6, 0.8, 8, 32);
            const ringMat = new THREE.MeshBasicMaterial({
                color: cs.deviation_pct > 40 ? 0xff2222 : cs.deviation_pct > 25 ? 0xffaa00 : 0xffff00,
                transparent: true,
                opacity: 0.85,
            });
            const ring = new THREE.Mesh(ringGeo, ringMat);
            ring.position.set(center.x, yPos, center.z);
            ring.rotation.x = Math.PI / 2;
            this.scene.add(ring);
            this.annotations.push(ring);
        });
    }

    setOpacity(value) {
        this.opacity = value;
        const updateMat = (mesh) => {
            if (!mesh) return;
            mesh.traverse((child) => {
                if (child.isMesh && child.material) {
                    child.material.opacity = value;
                }
            });
        };
        updateMat(this.diseasedMesh);
        updateMat(this.healthyMesh);
        this.morphMeshes.forEach(updateMat);
    }

    setWireframe(enabled) {
        this.wireframe = enabled;
        const updateMat = (mesh) => {
            if (!mesh) return;
            mesh.traverse((child) => {
                if (child.isMesh && child.material) {
                    child.material.wireframe = enabled;
                }
            });
        };
        updateMat(this.diseasedMesh);
        updateMat(this.healthyMesh);
        this.morphMeshes.forEach(updateMat);
    }

    // ─── Morph Animation ───────────────────────────────────────

    playMorphAnimation(onFrame, speed = 150) {
        if (this.isAnimating) return;
        this.isAnimating = true;

        let frame = 0;
        const total = this.morphMeshes.length;

        const step = () => {
            if (!this.isAnimating || frame >= total) {
                this.isAnimating = false;
                return;
            }
            this.setMorphFrame(frame);
            if (onFrame) onFrame(frame, total);
            frame++;
            this.animationId = setTimeout(step, speed);
        };

        step();
    }

    stopMorphAnimation() {
        this.isAnimating = false;
        if (this.animationId) {
            clearTimeout(this.animationId);
            this.animationId = null;
        }
    }

    // ─── Cleanup ───────────────────────────────────────────────

    clearAll() {
        const removeMesh = (mesh) => {
            if (mesh) {
                this.scene.remove(mesh);
                mesh.traverse((child) => {
                    if (child.isMesh) {
                        child.geometry?.dispose();
                        child.material?.dispose();
                    }
                });
            }
        };

        removeMesh(this.diseasedMesh);
        removeMesh(this.healthyMesh);
        this.morphMeshes.forEach(removeMesh);

        this.diseasedMesh = null;
        this.healthyMesh = null;
        this.morphMeshes = [];
        this.activeMorphFrame = -1;
    }
}
