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
        this.renderer.setClearColor(0x0a0d12, 1);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.2;

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
        // Ambient
        const ambient = new THREE.AmbientLight(0x404060, 0.6);
        this.scene.add(ambient);

        // Key light
        const key = new THREE.DirectionalLight(0xffeedd, 1.2);
        key.position.set(100, 200, 150);
        key.castShadow = true;
        this.scene.add(key);

        // Fill light
        const fill = new THREE.DirectionalLight(0x88aacc, 0.5);
        fill.position.set(-100, 50, -100);
        this.scene.add(fill);

        // Rim light (cyan accent)
        const rim = new THREE.DirectionalLight(0x22d3ee, 0.4);
        rim.position.set(0, -100, -200);
        this.scene.add(rim);

        // Hemisphere light for natural feel
        const hemi = new THREE.HemisphereLight(0xffffff, 0x444466, 0.3);
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
        this.renderer.render(this.scene, this.camera);
    }

    // ─── Mesh Loading ──────────────────────────────────────────

    async loadMesh(url, type = "diseased") {
        return new Promise((resolve, reject) => {
            this.loader.load(
                url,
                (gltf) => {
                    const mesh = gltf.scene;

                    // Apply material overrides
                    mesh.traverse((child) => {
                        if (child.isMesh) {
                            child.material = this._createMaterial(type);
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
        const colors = {
            diseased: { color: 0xe84040, emissive: 0x3a0a0a, opacityMult: 1.0, rough: 0.55, metal: 0.1 },
            healthy: { color: 0x34d399, emissive: 0x0a2a1a, opacityMult: 1.0, rough: 0.55, metal: 0.1 },
            body: { color: 0xeeece0, emissive: 0x000000, opacityMult: 0.15, rough: 0.1, metal: 0.0, transmission: 0.9, ior: 1.3 },
            heart: { color: 0x992222, emissive: 0x110000, opacityMult: 0.4, rough: 0.6, metal: 0.1 },
            aorta: { color: 0xcc2222, emissive: 0x220000, opacityMult: 0.5, rough: 0.4, metal: 0.1 },
            pulmonary_artery: { color: 0x2222cc, emissive: 0x000022, opacityMult: 0.5, rough: 0.4, metal: 0.1 }
        };

        const c = colors[type] || colors.diseased;
        const targetOpacity = this.opacity * (c.opacityMult || 1.0);

        return new THREE.MeshPhysicalMaterial({
            color: c.color,
            emissive: c.emissive,
            emissiveIntensity: 0.15,
            metalness: c.metal !== undefined ? c.metal : 0.1,
            roughness: c.rough !== undefined ? c.rough : 0.55,
            transmission: c.transmission || 0.0,
            ior: c.ior || 1.5,
            transparent: true,
            opacity: targetOpacity,
            wireframe: this.wireframe,
            side: THREE.DoubleSide,
            clearcoat: 0.3,
            clearcoatRoughness: 0.4,
            depthWrite: type !== "body" // Don't write depth for the body so internal organs render correctly
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
        if (maxDim === 0 || isNaN(maxDim)) {
            return;
        }

        const fov = this.camera.fov * (Math.PI / 180);
        const dist = maxDim / (2 * Math.tan(fov / 2)) * 1.5;

        this.camera.position.set(center.x, center.y, center.z + dist);
        this.controls.target.copy(center);
        this.controls.update();
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
            this.healthyMesh.visible = (m === "healthy" || m === "both");
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

        if (m === "morph" && this.activeMorphFrame >= 0 && this.morphMeshes[this.activeMorphFrame]) {
            if (this.diseasedMesh) this.diseasedMesh.visible = false;
            if (this.healthyMesh) this.healthyMesh.visible = false;
            this.morphMeshes[this.activeMorphFrame].visible = true;
        }
    }

    setMorphFrame(index) {
        // Hide previous
        if (this.activeMorphFrame >= 0 && this.morphMeshes[this.activeMorphFrame]) {
            this.morphMeshes[this.activeMorphFrame].visible = false;
        }
        this.activeMorphFrame = index;
        if (this.displayMode === "morph") {
            this._updateVisibility();
        }
    }

    setContextVisibility(layer, isVisible) {
        if (this.contextVisible.hasOwnProperty(layer)) {
            this.contextVisible[layer] = isVisible;
            this._updateVisibility();
        }
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
