// Viewer for tiles written by scripts/build_game_tiles.py. Loads <data>/scene.json (default ./data, or ?scene=<url of the folder>).
// Level of detail: a tile is replaced by its children when the camera is closer than K x its geometric error (screen-space-error style).
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const base = new URLSearchParams(location.search).get("scene") || "./data";
const K = 260;                         // refine when distance < K * geometric error (metres)
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2)); renderer.setSize(innerWidth, innerHeight); document.body.appendChild(renderer.domElement);
renderer.outputColorSpace = THREE.SRGBColorSpace;
const scene = new THREE.Scene(); scene.background = new THREE.Color(0xc9a27e); scene.fog = new THREE.Fog(0xc9a27e, 1500, 9000);
const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.05, 50000);
scene.add(new THREE.HemisphereLight(0xffe9d2, 0x5a3b28, 0.9)); const sun = new THREE.DirectionalLight(0xffffff, 1.6); sun.position.set(-400, 600, 300); scene.add(sun);
const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true;
const loader = new GLTFLoader(); const texLoader = new THREE.TextureLoader();
const tiles = new Map(); let manifest = null; let showProv = false, wire = false;

function key(t) { return `${t.level}/${t.x}/${t.y}`; }
async function loadTile(t) {
  if (t._loading || t._obj) return; t._loading = true;
  const g = await loader.loadAsync(`${base}/${t.uri}`); const obj = g.scene; t._obj = obj;
  obj.traverse(o => { if (o.isMesh) { o.userData.colorMap = o.material.map; o.material.side = THREE.FrontSide; } });
  obj.visible = false; scene.add(obj); t._loading = false; applyStyle(t);
}
function applyStyle(t) {
  if (!t._obj) return;
  t._obj.traverse(o => {
    if (!o.isMesh) return; o.material.wireframe = wire;
    if (showProv) { if (!t._prov) { t._prov = texLoader.load(`${base}/${t.provenance_uri}`); t._prov.flipY = false; t._prov.colorSpace = THREE.SRGBColorSpace; } o.material.map = t._prov; }
    else o.material.map = o.userData.colorMap; o.material.needsUpdate = true;
  });
}
function centerGltf(t) { const c = t.center_enu; return new THREE.Vector3(c[0], c[2], -c[1]); }   // ENU -> glTF axes
function select(t, out) {                     // depth-first: keep a tile if it is close enough or has no children
  const d = camera.position.distanceTo(centerGltf(t)) - Math.hypot(...t.half_size_enu);
  const kids = t.children.map(u => byUri.get(u)).filter(Boolean);
  if (kids.length && d < K * t.geometric_error_m) {
    const ready = kids.every(k => k._obj); kids.forEach(k => loadTile(k));
    if (ready) { kids.forEach(k => select(k, out)); return; }
  }
  out.push(t); loadTile(t);
}
let byUri = new Map(), root = null;
fetch(`${base}/scene.json`).then(r => r.json()).then(m => {
  manifest = m; m.tiles.forEach(t => { tiles.set(key(t), t); byUri.set(t.uri, t); }); root = tiles.get("0/0/0");
  reset(); animate();
});
function reset() {
  const q = new URLSearchParams(location.search), c = q.get("cam"), t = q.get("target");
  if (c && t) { camera.position.fromArray(c.split(",").map(Number)); controls.target.fromArray(t.split(",").map(Number)); }
  else { camera.position.set(0, 260, 340); controls.target.set(0, 0, 0); }
  controls.update();
}
function groundView() {                        // drop to rover eye height (2 m) above the terrain at the orbit target, looking north-east
  const t = controls.target.clone(); const ray = new THREE.Raycaster(new THREE.Vector3(t.x, t.y + 5000, t.z), new THREE.Vector3(0, -1, 0));
  const meshes = []; tiles.forEach(x => { if (x._obj && x._obj.visible) x._obj.traverse(o => { if (o.isMesh) meshes.push(o); }); });
  scene.updateMatrixWorld(true); const hits = ray.intersectObjects(meshes, false);
  if (hits.length) { const p = hits[0].point; camera.position.set(p.x, p.y + 2.0, p.z); controls.target.set(p.x + 40, p.y + 1.0, p.z - 40); controls.update(); }
  return hits.length;
}
window.__groundView = groundView;
addEventListener("keydown", e => {
  if (e.key === "r" || e.key === "R") reset();
  if (e.key === "g" || e.key === "G") groundView();
});
document.getElementById("prov").onchange = e => { showProv = e.target.checked; tiles.forEach(applyStyle); };
document.getElementById("wire").onchange = e => { wire = e.target.checked; tiles.forEach(applyStyle); };
addEventListener("resize", () => { camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); });
function animate() {
  requestAnimationFrame(animate); controls.update();
  if (root) { const sel = []; select(root, sel); const vis = new Set(sel.map(key)); tiles.forEach((t, k) => { if (t._obj) t._obj.visible = vis.has(k); });
    const lv = sel.map(t => t.level); document.getElementById("stats").textContent = `tiles shown ${sel.length}, levels ${Math.min(...lv)}–${Math.max(...lv)}, camera height ${camera.position.y.toFixed(1)} m`; }
  renderer.render(scene, camera);
}
window.__ready = () => root && [...tiles.values()].filter(t => t._obj && t._obj.visible).length > 0;
