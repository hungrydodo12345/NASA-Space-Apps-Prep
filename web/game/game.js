// Jezero Explorer: open-world and story modes over the streamed terrain tiles (scripts/build_game_tiles.py + scripts/build_story.py).
// Data folder: ./data (or ?data=<url>) holding scene.json, story.json, traverse.json, overview.jpg and tiles/.
// Structure: Env (sky, sun, fog: the hook for weather and lighting), Terrain (LOD streaming with unloading), Rover (placeholder driving on the
// terrain surface), Story (chapters and objectives), Minimap. Scene axes are glTF: x = east, y = up, z = -north.
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const DATA = new URLSearchParams(location.search).get("data") || "./data";
const $ = id => document.getElementById(id);
const enu2v = (e, n, u = 0) => new THREE.Vector3(e, u, -n);

// ---------------- renderer, Env
const renderer = new THREE.WebGLRenderer({ canvas: $("gl"), antialias: true, preserveDrawingBuffer: true, logarithmicDepthBuffer: true });   // log depth: no z-fighting over 0.3 m - 60 km
renderer.setPixelRatio(Math.min(devicePixelRatio, 2)); renderer.setSize(innerWidth, innerHeight); renderer.outputColorSpace = THREE.SRGBColorSpace;
const scene = new THREE.Scene(); const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.3, 60000);
export const Env = {                         // weather and lighting plug in here: change these values (or animate them) and call Env.apply()
  sunAzimuthDeg: 250, sunElevationDeg: 35, skyColor: 0xcfa27c, dust: 0.25,
  hemi: new THREE.HemisphereLight(0xffe7cf, 0x4a2f1f, 0.85), sun: new THREE.DirectionalLight(0xfff2e0, 1.7),
  apply() {
    const az = THREE.MathUtils.degToRad(this.sunAzimuthDeg), el = THREE.MathUtils.degToRad(this.sunElevationDeg);
    this.sun.position.set(Math.sin(az) * Math.cos(el), Math.sin(el), -Math.cos(az) * Math.cos(el)).multiplyScalar(1000);
    scene.background = new THREE.Color(this.skyColor); scene.fog = new THREE.Fog(this.skyColor, 600 + 3000 * (1 - this.dust), 4000 + 14000 * (1 - this.dust));
  },
};
scene.add(Env.hemi, Env.sun, Env.sun.target); Env.apply();

// ---------------- Terrain streaming
const Terrain = {
  tiles: new Map(), byUri: new Map(), root: null, loader: new GLTFLoader(), inflight: 0, maxInflight: 6, K: 240, frame: 0, showProv: false, budget: 900,
  async init() {
    const sc = await (await fetch(`${DATA}/scene.json`)).json(); this.scene = sc;
    sc.tiles.forEach(t => { t.key = `${t.level}/${t.x}/${t.y}`; this.tiles.set(t.key, t); this.byUri.set(t.uri, t); t.c = enu2v(t.center_enu[0], t.center_enu[1], t.center_enu[2]); t.r = Math.hypot(...t.half_size_enu); });
    this.root = this.tiles.get("0/0/0"); return sc;
  },
  load(t) {
    if (t.obj || t.loading || this.inflight >= this.maxInflight) return; t.loading = true; this.inflight++;
    this.loader.load(`${DATA}/${t.uri}`, g => { t.obj = g.scene; t.obj.visible = false; t.obj.traverse(o => { if (o.isMesh) { o.userData.map = o.material.map; o.material.side = THREE.FrontSide; } });
      scene.add(t.obj); t.loading = false; this.inflight--; this.style(t); }, undefined, () => { t.loading = false; this.inflight--; t.failed = true; });
  },
  style(t) {
    if (!t.obj) return;
    t.obj.traverse(o => { if (!o.isMesh) return;
      if (this.showProv) { if (!t.prov) { t.prov = new THREE.TextureLoader().load(`${DATA}/${t.provenance_uri}`); t.prov.flipY = false; t.prov.colorSpace = THREE.SRGBColorSpace; } o.material.map = t.prov; }
      else o.material.map = o.userData.map; o.material.needsUpdate = true; });
  },
  select(t, eye, out) {
    const d = Math.max(0, eye.distanceTo(t.c) - t.r); const kids = t.children.map(u => this.byUri.get(u)).filter(k => k && !k.failed);
    // hysteresis: refine below K x error, un-refine only above 1.3 K x error, so tiles do not swap back and forth at the boundary
    const want = kids.length && (d < this.K * t.geometric_error_m || (t.refined && d < 1.3 * this.K * t.geometric_error_m));
    if (want) {
      kids.forEach(k => { k.wanted = this.frame; this.load(k); });
      if (kids.every(k => k.obj)) { t.refined = true; kids.forEach(k => this.select(k, eye, out)); return; }
    } else t.refined = false;
    out.push(t); this.load(t);
  },
  update(eye) {
    if (!this.root) return []; this.frame++; const sel = []; this.select(this.root, eye, sel); const vis = new Set(sel);
    this.tiles.forEach(t => { if (!t.obj) return; const v = vis.has(t); t.obj.visible = v; if (v) t.seen = this.frame; });
    if (this.frame % 60 === 0) this.evict(eye);
    this.visible = sel; return sel;
  },
  evict(eye) {                                     // keep at most `budget` tiles in memory: drop the least recently shown, farthest first
    const loaded = [...this.tiles.values()].filter(t => t.obj && t.level > 2); if (loaded.length <= this.budget) return;
    loaded.sort((a, b) => (a.seen || 0) - (b.seen || 0) || b.c.distanceTo(eye) - a.c.distanceTo(eye));
    for (const t of loaded.slice(0, loaded.length - this.budget)) { if (t.obj.visible || this.frame - (t.wanted || 0) < 120 || this.frame - (t.seen || 0) < 300) continue; scene.remove(t.obj);
      t.obj.traverse(o => { if (o.isMesh) { o.geometry.dispose(); o.userData.map && o.userData.map.dispose(); o.material.dispose(); } }); if (t.prov) { t.prov.dispose(); t.prov = null; } t.obj = null; }
  },
  ray: new THREE.Raycaster(),
  heightAt(x, z) {                                 // terrain height under (x, z) from the loaded, visible tiles nearest the point
    const p = new THREE.Vector3(x, 0, z); const near = (this.visible || []).filter(t => t.obj && Math.hypot(t.c.x - x, t.c.z - z) < t.r + 2);
    const meshes = []; near.sort((a, b) => b.level - a.level).forEach(t => t.obj.traverse(o => { if (o.isMesh) meshes.push(o); }));
    if (!meshes.length) return null; this.ray.set(new THREE.Vector3(x, 8000, z), new THREE.Vector3(0, -1, 0)); this.ray.far = 20000;
    const h = this.ray.intersectObjects(meshes, false); return h.length ? h[0].point.y : null;
  },
};

// ---------------- Rover (placeholder: a body, a mast and six wheels; real kinematics and suspension plug in here)
const Rover = {
  g: new THREE.Group(), heading: 0, speed: 0, pos: new THREE.Vector3(), odo: 0, keys: {},
  build() {
    const m = c => new THREE.MeshStandardMaterial({ color: c, roughness: 0.7, metalness: 0.1 });
    const body = new THREE.Mesh(new THREE.BoxGeometry(2.0, 0.6, 2.9), m(0xe8e4dc)); body.position.y = 1.0; this.g.add(body);
    const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 1.2), m(0xcfc9be)); mast.position.set(0.55, 1.9, -1.1); this.g.add(mast);
    const head = new THREE.Mesh(new THREE.BoxGeometry(0.45, 0.22, 0.25), m(0x999186)); head.position.set(0.55, 2.55, -1.1); this.g.add(head);
    for (const x of [-1.1, 1.1]) for (const z of [-1.1, 0, 1.1]) { const w = new THREE.Mesh(new THREE.CylinderGeometry(0.26, 0.26, 0.4, 16), m(0x3a3632)); w.rotation.z = Math.PI / 2; w.position.set(x, 0.26, z); this.g.add(w); }
    scene.add(this.g); this.placeholder = [...this.g.children];
    // NASA/JPL-Caltech Perseverance model (science.nasa.gov 3D resources), scaled to the real rover's 3.0 m length; replaces the placeholder once loaded
    new GLTFLoader().load("./assets/perseverance.glb", gltf => {
      const m = gltf.scene; const box = new THREE.Box3().setFromObject(m), size = box.getSize(new THREE.Vector3());
      const s = 3.0 / Math.max(size.x, size.z); m.scale.setScalar(s); m.rotation.y = Math.PI;              // model front (hazcams) is +z; the game's forward is -z
      const b2 = new THREE.Box3().setFromObject(m), c = b2.getCenter(new THREE.Vector3()); m.position.set(-c.x, -b2.min.y, -c.z);
      m.traverse(o => { if (o.isMesh) { o.castShadow = true; if (o.material) o.material.side = THREE.FrontSide; } });
      this.placeholder.forEach(o => this.g.remove(o)); this.g.add(m); this.model = m;
    }, undefined, e => console.warn("Perseverance model not loaded, keeping the placeholder", e));
  },
  place(e, n, headingDeg) { this.pos.copy(enu2v(e, n, 0)); this.heading = THREE.MathUtils.degToRad(headingDeg || 0); this.speed = 0; this.ground(true); },
  ground(force) {
    const y = Terrain.heightAt(this.pos.x, this.pos.z); if (y !== null) this.pos.y = y; else if (force) this.pos.y = 0;
    const f = new THREE.Vector3(Math.sin(this.heading), 0, -Math.cos(this.heading));          // forward (heading 0 = north)
    const yf = Terrain.heightAt(this.pos.x + f.x * 1.4, this.pos.z + f.z * 1.4), yb = Terrain.heightAt(this.pos.x - f.x * 1.4, this.pos.z - f.z * 1.4);
    const pitch = (yf !== null && yb !== null) ? Math.atan2(yf - yb, 2.8) : 0;
    this.g.position.copy(this.pos); this.g.rotation.set(0, 0, 0); this.g.rotateY(-this.heading); this.g.rotateX(pitch);
  },
  update(dt) {
    const k = this.keys, vmax = k.ShiftLeft || k.ShiftRight ? 25 : 5;            // game speeds; Perseverance's real top speed is about 0.04 m/s
    const thr = (k.KeyW || k.ArrowUp ? 1 : 0) - (k.KeyS || k.ArrowDown ? 1 : 0), steer = (k.KeyD || k.ArrowRight ? 1 : 0) - (k.KeyA || k.ArrowLeft ? 1 : 0);
    this.speed += (thr * vmax - this.speed) * Math.min(1, dt * (thr ? 1.5 : 3)); this.heading += steer * dt * 0.9 * (this.speed >= 0 ? 1 : -1) * Math.min(1, Math.abs(this.speed) / 1.5 + 0.3);
    const step = this.speed * dt; this.pos.x += Math.sin(this.heading) * step; this.pos.z -= Math.cos(this.heading) * step; this.odo += Math.abs(step); this.ground(false);
  },
};

// ---------------- camera rig
let camMode = 0;                                  // 0 chase, 1 mast camera (2 m), 2 high orbit
function updateCamera() {
  const p = Rover.pos, h = Rover.heading, f = new THREE.Vector3(Math.sin(h), 0, -Math.cos(h));
  if (camMode === 0) { camera.position.copy(p).addScaledVector(f, -12).add(new THREE.Vector3(0, 6, 0)); camera.lookAt(p.clone().addScaledVector(f, 8).add(new THREE.Vector3(0, 1.5, 0))); }
  else if (camMode === 1) { camera.position.copy(p).add(new THREE.Vector3(0, 2.2, 0)).addScaledVector(f, 1.2); camera.lookAt(p.clone().addScaledVector(f, 40).add(new THREE.Vector3(0, 0.8, 0))); }
  else if (camMode === 2) { camera.position.copy(p).addScaledVector(f, -250).add(new THREE.Vector3(0, 220, 0)); camera.lookAt(p); }
  else { camera.position.copy(p).add(new THREE.Vector3(0, 2.0, 0)); camera.lookAt(p.clone().addScaledVector(f, 100).add(new THREE.Vector3(0, 2.0 - 0.35, 0))); }   // 3: level mast view for panoramas
}

// ---------------- Story and points of interest
const Story = {
  data: null, mode: "open", idx: 0, open: false, markers: [],
  async init() { this.data = await (await fetch(`${DATA}/story.json`)).json(); this.traverse = await (await fetch(`${DATA}/traverse.json`)).json(); },
  start(mode) {
    // idx = the next chapter to reach. Chapter 0 happens at the start point and opens as the introduction; saved progress resumes at the last chapter reached.
    const N = this.data.chapters.length; this.mode = mode; const saved = Math.min(Number(localStorage.getItem("jezero.chapter") || 0), N - 1);
    const s = this.data.start; Rover.place(s.pos[0], s.pos[1], s.heading_deg); this.makeMarkers();
    if (mode === "story") { const c = this.data.chapters[saved]; if (saved > 0) Rover.place(c.target[0], c.target[1], 0); this.idx = saved + 1; this.show(c, saved === 0); }
    else this.idx = N;
    this.hud();
  },
  makeMarkers() {
    this.markers.forEach(m => scene.remove(m)); this.markers = [];
    const pts = this.mode === "story" ? [] : this.data.pois;
    for (const p of pts) { const m = new THREE.Mesh(new THREE.CylinderGeometry(0.3, 0.3, 30, 8), new THREE.MeshBasicMaterial({ color: p.kind === "story" ? 0xe8934a : p.kind === "traverse" ? 0x6fb3ff : 0xffffff, transparent: true, opacity: 0.55 }));
      m.userData = p; m.position.copy(enu2v(p.pos[0], p.pos[1], 0)); scene.add(m); this.markers.push(m); }
    this.beacon = new THREE.Mesh(new THREE.CylinderGeometry(0.6, 0.6, 60, 12), new THREE.MeshBasicMaterial({ color: 0xe8934a, transparent: true, opacity: 0.7 })); scene.add(this.beacon); this.markers.push(this.beacon);
  },
  current() { return this.mode === "story" && this.idx < this.data.chapters.length ? this.data.chapters[this.idx] : null; },
  show(c, intro) { this.open = true; $("story").style.display = "block"; const i = this.data.chapters.indexOf(c) + 1; $("storySol").textContent = `Chapter ${i} of ${this.data.chapters.length} · sol ${c.sol}`;
    $("storyTitle").textContent = c.title; $("storyText").textContent = intro ? c.text : c.text; $("storyTask").textContent = "Next: " + c.task; },
  close() { this.open = false; $("story").style.display = "none"; },
  update() {
    for (const m of this.markers) { if (m === this.beacon) continue; const y = Terrain.heightAt(m.position.x, m.position.z); if (y !== null) m.position.y = y + 15; }
    if (!this.beacon) return;
    const c = this.current(); this.beacon.visible = !!c;
    if (c) { const t = enu2v(c.target[0], c.target[1], 0); const y = Terrain.heightAt(t.x, t.z); this.beacon.position.set(t.x, (y ?? Rover.pos.y) + 30, t.z);
      const d = Math.hypot(t.x - Rover.pos.x, t.z - Rover.pos.z);
      if (!this.open && d < c.radius) { localStorage.setItem("jezero.chapter", this.idx); this.show(c, false); this.idx++; } }
    this.hud();
  },
  hud() {
    $("hudMode").textContent = this.mode === "story" ? "Story mode" : "Open world";
    const c = this.current(), p = Rover.pos;
    if (c) { const t = enu2v(c.target[0], c.target[1]); const d = Math.hypot(t.x - p.x, t.z - p.z), b = Math.atan2(t.x - p.x, -(t.z - p.z));
      const prev = this.data.chapters[this.idx - 1]; $("hudObj").textContent = `Next: ${c.title}. ${prev ? prev.task : ""}`; $("hudSub").textContent = `${d.toFixed(0)} m to objective · driven ${(Rover.odo / 1000).toFixed(2)} km`;
      const rel = THREE.MathUtils.radToDeg(b - Rover.heading); $("arrow").style.display = "block"; $("arrow").style.transform = `translateX(-50%) rotate(${rel}deg)`; }
    else { let best = null; for (const q of this.data.pois) { const v = enu2v(q.pos[0], q.pos[1]); const d = Math.hypot(v.x - p.x, v.z - p.z); if (!best || d < best.d) best = { q, d }; }
      $("hudObj").textContent = best ? `Nearest: ${best.q.title}` : ""; $("hudSub").textContent = `${best ? best.d.toFixed(0) + " m away · " : ""}E ${p.x.toFixed(0)} m, N ${(-p.z).toFixed(0)} m · driven ${(Rover.odo / 1000).toFixed(2)} km`; $("arrow").style.display = "none"; }
  },
};

// ---------------- minimap
const Mini = {
  img: new Image(), ctx: $("miniCanvas").getContext("2d"),
  init(sc) { this.ov = sc.overview; this.img.src = `${DATA}/${this.ov.uri}`; },
  toPx(e, n) { const [w, nn] = this.ov.enu_west_north, [ee, s] = this.ov.enu_east_south, W = this.ctx.canvas.width; return [(e - w) / (ee - w) * W, (nn - n) / (nn - s) * W]; },
  draw() {
    const c = this.ctx, W = c.canvas.width; c.fillStyle = "#000"; c.fillRect(0, 0, W, W); if (this.img.complete) c.drawImage(this.img, 0, 0, W, W);
    if (Story.traverse) { c.strokeStyle = "rgba(111,179,255,0.9)"; c.lineWidth = 2; c.beginPath(); Story.traverse.points.forEach((t, i) => { const [x, y] = this.toPx(t.enu[0], t.enu[1]); i ? c.lineTo(x, y) : c.moveTo(x, y); }); c.stroke(); }
    const ch = Story.current(); if (ch) { const [x, y] = this.toPx(ch.target[0], ch.target[1]); c.fillStyle = "#e8934a"; c.beginPath(); c.arc(x, y, 6, 0, 7); c.fill(); }
    const [x, y] = this.toPx(Rover.pos.x, -Rover.pos.z); c.save(); c.translate(x, y); c.rotate(Rover.heading); c.fillStyle = "#fff"; c.beginPath(); c.moveTo(0, -9); c.lineTo(6, 7); c.lineTo(-6, 7); c.closePath(); c.fill(); c.restore();
  },
};

// ---------------- input, loop
addEventListener("keydown", e => { Rover.keys[e.code] = true;
  if (e.code === "KeyC") camMode = (camMode + 1) % 3;
  if (e.code === "KeyP") { Terrain.showProv = !Terrain.showProv; Terrain.tiles.forEach(t => Terrain.style(t)); }
  if (e.code === "KeyM") { $("menu").style.display = "grid"; }
  if ((e.code === "Enter" || e.code === "Space") && Story.open) { e.preventDefault(); Story.close(); } });
addEventListener("keyup", e => { Rover.keys[e.code] = false; });
addEventListener("resize", () => { camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); });
$("btnContinue").onclick = () => Story.close();
function begin(mode) { $("menu").style.display = "none"; ["hud", "mini", "help"].forEach(i => $(i).style.display = "block"); Story.start(mode); }
$("btnStory").onclick = () => begin("story"); $("btnOpen").onclick = () => begin("open");

let last = performance.now();
function loop(t) {
  requestAnimationFrame(loop); const dt = Math.min(0.1, (t - last) / 1000); last = t;
  if (!Story.open) Rover.update(dt); else Rover.ground(false);
  updateCamera(); Terrain.update(camera.position); Story.update(); Mini.draw(); renderer.render(scene, camera);
}
(async () => {
  const sc = await Terrain.init(); await Story.init(); Mini.init(sc); Rover.build();
  const s = Story.data.start; Rover.place(s.pos[0], s.pos[1], 0); $("loadmsg").textContent = "Ready.";
  requestAnimationFrame(loop);
  window.__game = { begin, Rover, Terrain, Story, Env, camera, setCam: m => { camMode = m; }, setFov: v => { camera.fov = v; camera.updateProjectionMatrix(); } };      // hooks for tests and future modules (weather, lighting, rover physics)
})();
