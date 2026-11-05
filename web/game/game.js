// Jezero Explorer: open-world and story modes over streamed HiRISE terrain tiles
// Integrates official NASA 3D Models, Rocker-Bogie Kinematics, 5-DOF Arm & Ingenuity Flight Dynamics
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { RoverKinematics, ROVER_SPECS } from "./rover_sim.js";
import { MarsAudio } from "./audio.js";

const $ = id => document.getElementById(id);
const enu2v = (e, n, u = 0) => new THREE.Vector3(e, u, -n);

const toast = (msg, ms = 3000) => {
  const t = $("toast");
  if (!t) return;
  t.textContent = msg;
  t.style.display = "block";
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.style.display = "none", ms);
};

// ---------------- renderer, Env
const renderer = new THREE.WebGLRenderer({ canvas: $("gl"), antialias: true, preserveDrawingBuffer: true, logarithmicDepthBuffer: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.3, 60000);

export const Env = {
  sunAzimuthDeg: 250, sunElevationDeg: 35, skyColor: 0xcfa27c, dust: 0.25,
  hemi: new THREE.HemisphereLight(0xffe7cf, 0x4a2f1f, 0.85),
  sun: new THREE.DirectionalLight(0xfff2e0, 1.7),
  apply() {
    const az = THREE.MathUtils.degToRad(this.sunAzimuthDeg), el = THREE.MathUtils.degToRad(this.sunElevationDeg);
    this.sun.position.set(Math.sin(az) * Math.cos(el), Math.sin(el), -Math.cos(az) * Math.cos(el)).multiplyScalar(1000);
    scene.background = new THREE.Color(this.skyColor);
    scene.fog = new THREE.Fog(this.skyColor, 600 + 3000 * (1 - this.dust), 4000 + 14000 * (1 - this.dust));
  },
};
Env.sun.castShadow = true;
scene.add(Env.hemi, Env.sun, Env.sun.target);
Env.apply();

// ---------------- Terrain streaming
const Terrain = {
  tiles: new Map(), byUri: new Map(), root: null, loader: new GLTFLoader(), inflight: 0, maxInflight: 6, K: 240, frame: 0, showProv: false, budget: 900,
  dataUrl: "./data",
  async init(dataUrl) {
    this.dataUrl = dataUrl;
    const sc = await (await fetch(`${this.dataUrl}/scene.json`)).json();
    this.scene = sc;
    sc.tiles.forEach(t => {
      t.key = `${t.level}/${t.x}/${t.y}`;
      this.tiles.set(t.key, t);
      this.byUri.set(t.uri, t);
      t.c = enu2v(t.center_enu[0], t.center_enu[1], t.center_enu[2]);
      t.r = Math.hypot(...t.half_size_enu);
    });
    this.root = this.tiles.get("0/0/0");
    return sc;
  },
  load(t) {
    if (t.obj || t.loading || this.inflight >= this.maxInflight) return;
    t.loading = true; this.inflight++;
    this.loader.load(`${this.dataUrl}/${t.uri}`, g => {
      t.obj = g.scene;
      t.obj.visible = false;
      t.obj.traverse(o => {
        if (o.isMesh) {
          o.userData.map = o.material.map;
          o.material.side = THREE.FrontSide;
          o.receiveShadow = true;
        }
      });
      scene.add(t.obj);
      t.loading = false;
      this.inflight--;
      this.style(t);
    }, undefined, () => {
      t.loading = false;
      this.inflight--;
      t.failed = true;
    });
  },
  style(t) {
    if (!t.obj) return;
    t.obj.traverse(o => {
      if (!o.isMesh) return;
      if (this.showProv) {
        if (!t.prov) {
          t.prov = new THREE.TextureLoader().load(`${this.dataUrl}/${t.provenance_uri}`);
          t.prov.flipY = false;
          t.prov.colorSpace = THREE.SRGBColorSpace;
        }
        o.material.map = t.prov;
      } else {
        o.material.map = o.userData.map;
      }
      o.material.needsUpdate = true;
    });
  },
  select(t, eye, out) {
    const d = Math.max(0, eye.distanceTo(t.c) - t.r);
    const kids = t.children.map(u => this.byUri.get(u)).filter(k => k && !k.failed);
    const want = kids.length && (d < this.K * t.geometric_error_m || (t.refined && d < 1.3 * this.K * t.geometric_error_m));
    if (want) {
      kids.forEach(k => { k.wanted = this.frame; this.load(k); });
      if (kids.every(k => k.obj)) {
        t.refined = true;
        kids.forEach(k => this.select(k, eye, out));
        return;
      }
    } else {
      t.refined = false;
    }
    out.push(t);
    this.load(t);
  },
  update(eye) {
    if (!this.root) return [];
    this.frame++;
    const sel = [];
    this.select(this.root, eye, sel);
    const vis = new Set(sel);
    this.tiles.forEach(t => {
      if (!t.obj) return;
      const v = vis.has(t);
      t.obj.visible = v;
      if (v) t.seen = this.frame;
    });
    if (this.frame % 60 === 0) this.evict(eye);
    this.visible = sel;
    return sel;
  },
  evict(eye) {
    const loaded = [...this.tiles.values()].filter(t => t.obj && t.level > 2);
    if (loaded.length <= this.budget) return;
    loaded.sort((a, b) => (a.seen || 0) - (b.seen || 0) || b.c.distanceTo(eye) - a.c.distanceTo(eye));
    for (const t of loaded.slice(0, loaded.length - this.budget)) {
      if (t.obj.visible || this.frame - (t.wanted || 0) < 120 || this.frame - (t.seen || 0) < 300) continue;
      scene.remove(t.obj);
      t.obj.traverse(o => {
        if (o.isMesh) {
          o.geometry.dispose();
          if (o.userData.map) o.userData.map.dispose();
          o.material.dispose();
        }
      });
      if (t.prov) { t.prov.dispose(); t.prov = null; }
      t.obj = null;
    }
  },
  ray: new THREE.Raycaster(),
  heightAt(x, z) {
    const near = (this.visible || []).filter(t => t.obj && Math.hypot(t.c.x - x, t.c.z - z) < t.r + 2);
    const meshes = [];
    near.sort((a, b) => b.level - a.level).forEach(t => t.obj.traverse(o => { if (o.isMesh) meshes.push(o); }));
    if (!meshes.length) return null;
    this.ray.set(new THREE.Vector3(x, 8000, z), new THREE.Vector3(0, -1, 0));
    this.ray.far = 20000;
    const h = this.ray.intersectObjects(meshes, false);
    return h.length ? h[0].point.y : null;
  },
};

// ---------------- Rover Kinematics Engine
const Rover = new RoverKinematics(scene, Terrain);

// ---------------- Multi-Camera Suite
const CAM_MODES = [
  { id: 0, name: 'Chase View', desc: 'Trailing survey camera' },
  { id: 1, name: 'Mastcam-Z Stereo', desc: 'Remote Sensing Mast (2.2 m)' },
  { id: 2, name: 'Front Hazcam', desc: 'Low-chassis hazard detection' },
  { id: 3, name: 'High Orbit View', desc: 'Satellite reconnaissance' }
];
let camMode = 0;

function updateCamera() {
  const p = Rover.pos, h = Rover.heading, f = new THREE.Vector3(Math.sin(h), 0, -Math.cos(h));
  if (camMode === 0) {
    // Smooth trailing chase view
    const targetPos = p.clone().addScaledVector(f, -10).add(new THREE.Vector3(0, 4.8, 0));
    camera.position.lerp(targetPos, 0.15);
    camera.lookAt(p.clone().addScaledVector(f, 6).add(new THREE.Vector3(0, 1.3, 0)));
  } else if (camMode === 1) {
    // 2.2 m Mastcam-Z
    camera.position.copy(p).add(new THREE.Vector3(0, 2.2, 0)).addScaledVector(f, 0.8);
    camera.lookAt(p.clone().addScaledVector(f, 40).add(new THREE.Vector3(0, 0.8, 0)));
  } else if (camMode === 2) {
    // Low Front Hazcam
    camera.position.copy(p).add(new THREE.Vector3(0, 0.9, 0)).addScaledVector(f, 1.4);
    camera.lookAt(p.clone().addScaledVector(f, 8).add(new THREE.Vector3(0, 0.2, 0)));
  } else {
    // High orbit view
    camera.position.copy(p).addScaledVector(f, -250).add(new THREE.Vector3(0, 220, 0));
    camera.lookAt(p);
  }
}

function cycleCamera() {
  camMode = (camMode + 1) % CAM_MODES.length;
  const c = CAM_MODES[camMode];
  $("camName").textContent = c.name;
  $("bCam").textContent = `📷 ${c.name}`;
  toast(`Camera: ${c.name} — ${c.desc}`);
}

// ---------------- Solar Diurnal Time of Day
function setSolTime(solHr) {
  const hourAngle = ((solHr - 12) / 12) * Math.PI;
  const elevation = Math.sin(Math.PI / 2 - Math.abs(hourAngle)) * 1.25;
  const azDeg = ((hourAngle + Math.PI) * (180 / Math.PI)) % 360;
  const elDeg = elevation * (180 / Math.PI);

  Env.sunAzimuthDeg = azDeg;
  Env.sunElevationDeg = Math.max(-5, elDeg);

  if (elDeg > 20) {
    Env.skyColor = 0xcfa27c; // Warm midday tan
  } else if (elDeg > 4) {
    Env.skyColor = 0xb87348; // Amber golden hour
  } else if (elDeg > -2) {
    Env.skyColor = 0x483a54; // Twilight Martian blue/violet sunset
  } else {
    Env.skyColor = 0x0a0c14; // Night
  }
  Env.apply();

  const totalMin = Math.round(solHr * 60);
  const h = Math.floor(totalMin / 60) % 24;
  const m = totalMin % 60;
  $("timeVal").textContent = `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')} LST`;
}

// ---------------- Story Mode & Objectives
const Story = {
  data: null, mode: "open", idx: 0, open: false, markers: [],
  async init(dataUrl) {
    this.data = await (await fetch(`${dataUrl}/story.json`)).json();
    this.traverse = await (await fetch(`${dataUrl}/traverse.json`)).json();
  },
  start(mode) {
    const N = this.data.chapters.length;
    this.mode = mode;
    const saved = Math.min(Number(localStorage.getItem("jezero.chapter") || 0), N - 1);
    const s = this.data.start;
    Rover.place(s.pos[0], s.pos[1], s.heading_deg);
    this.makeMarkers();
    if (mode === "story") {
      const c = this.data.chapters[saved];
      if (saved > 0) Rover.place(c.target[0], c.target[1], 0);
      this.idx = saved + 1;
      this.show(c, saved === 0);
    } else {
      this.idx = N;
    }
    this.hud();
  },
  makeMarkers() {
    this.markers.forEach(m => scene.remove(m));
    this.markers = [];
    const pts = this.mode === "story" ? [] : this.data.pois;
    for (const p of pts) {
      const m = new THREE.Mesh(new THREE.CylinderGeometry(0.3, 0.3, 30, 8), new THREE.MeshBasicMaterial({ color: p.kind === "story" ? 0xe8934a : p.kind === "traverse" ? 0x6fb3ff : 0xffffff, transparent: true, opacity: 0.55 }));
      m.userData = p;
      m.position.copy(enu2v(p.pos[0], p.pos[1], 0));
      scene.add(m);
      this.markers.push(m);
    }
    this.beacon = new THREE.Mesh(new THREE.CylinderGeometry(0.6, 0.6, 60, 12), new THREE.MeshBasicMaterial({ color: 0xe8934a, transparent: true, opacity: 0.7 }));
    scene.add(this.beacon);
    this.markers.push(this.beacon);
  },
  current() {
    return this.mode === "story" && this.idx < this.data.chapters.length ? this.data.chapters[this.idx] : null;
  },
  show(c, intro) {
    this.open = true;
    $("story").style.display = "block";
    const i = this.data.chapters.indexOf(c) + 1;
    $("storySol").textContent = `Chapter ${i} of ${this.data.chapters.length} · sol ${c.sol}`;
    $("storyTitle").textContent = c.title;
    $("storyText").textContent = c.text;
    $("storyTask").textContent = "Next: " + c.task;

    // Show drill action button on coring chapters (Roubion and Montdenier)
    const isDrillChapter = (c.sol === 164 || c.sol === 196 || (c.title && c.title.toLowerCase().includes('sample')));
    const btnAction = $("btnStoryAction");
    if (btnAction) {
      btnAction.style.display = isDrillChapter ? 'inline-block' : 'none';
      btnAction.onclick = () => {
        toast("Deploying 5-DOF arm and spinning coring drill bit…");
        Rover.triggerDrill(() => {
          toast("Core Sample extracted and safely sealed into ultra-clean titanium tube!", 4500);
        });
      };
    }
  },
  close() {
    this.open = false;
    $("story").style.display = "none";
  },
  update() {
    for (const m of this.markers) {
      if (m === this.beacon) continue;
      const y = Terrain.heightAt(m.position.x, m.position.z);
      if (y !== null) m.position.y = y + 15;
    }
    if (!this.beacon) return;
    const c = this.current();
    this.beacon.visible = !!c;
    if (c) {
      const t = enu2v(c.target[0], c.target[1], 0);
      const y = Terrain.heightAt(t.x, t.z);
      this.beacon.position.set(t.x, (y ?? Rover.pos.y) + 30, t.z);
      const d = Math.hypot(t.x - Rover.pos.x, t.z - Rover.pos.z);
      if (!this.open && d < c.radius) {
        localStorage.setItem("jezero.chapter", this.idx);
        this.show(c, false);
        this.idx++;
        MarsAudio.playBeep(980, 0.2);
      }
    }
    this.hud();
  },
  hud() {
    $("hudMode").textContent = this.mode === "story" ? "Story mode" : "Open world";
    const c = this.current(), p = Rover.pos;
    if (c) {
      const t = enu2v(c.target[0], c.target[1]);
      const d = Math.hypot(t.x - p.x, t.z - p.z), b = Math.atan2(t.x - p.x, -(t.z - p.z));
      const prev = this.data.chapters[this.idx - 1];
      $("hudObj").textContent = `Next: ${c.title}. ${prev ? prev.task : ""}`;
      $("hudSub").textContent = `${d.toFixed(0)} m to objective · driven ${(Rover.odo / 1000).toFixed(2)} km`;
      const rel = THREE.MathUtils.radToDeg(b - Rover.heading);
      $("arrow").style.display = "block";
      $("arrow").style.transform = `translateX(-50%) rotate(${rel}deg)`;
    } else {
      let best = null;
      for (const q of this.data.pois) {
        const v = enu2v(q.pos[0], q.pos[1]);
        const d = Math.hypot(v.x - p.x, v.z - p.z);
        if (!best || d < best.d) best = { q, d };
      }
      $("hudObj").textContent = best ? `Nearest: ${best.q.title}` : "";
      $("hudSub").textContent = `${best ? best.d.toFixed(0) + " m away · " : ""}E ${p.x.toFixed(0)} m, N ${(-p.z).toFixed(0)} m · driven ${(Rover.odo / 1000).toFixed(2)} km`;
      $("arrow").style.display = "none";
    }

    // Update Top Telemetry HUD
    $("craftName").textContent = Rover.currentSpec.name;
    $("speedVal").textContent = `${Math.abs(Rover.speed).toFixed(1)} m/s`;
    $("pitchRollVal").textContent = `${(Rover.pitch * 180 / Math.PI).toFixed(0)}° / ${(Rover.roll * 180 / Math.PI).toFixed(0)}°`;
  },
};

// ---------------- Minimap
const Mini = {
  img: new Image(),
  ctx: $("miniCanvas").getContext("2d"),
  init(sc, dataUrl) {
    this.ov = sc.overview;
    this.img.src = `${dataUrl}/${this.ov.uri}`;
  },
  toPx(e, n) {
    const [w, nn] = this.ov.enu_west_north, [ee, s] = this.ov.enu_east_south, W = this.ctx.canvas.width;
    return [(e - w) / (ee - w) * W, (nn - n) / (nn - s) * W];
  },
  draw() {
    const c = this.ctx, W = c.canvas.width;
    c.fillStyle = "#000";
    c.fillRect(0, 0, W, W);
    if (this.img.complete) c.drawImage(this.img, 0, 0, W, W);
    if (Story.traverse) {
      c.strokeStyle = "rgba(111,179,255,0.9)";
      c.lineWidth = 2;
      c.beginPath();
      Story.traverse.points.forEach((t, i) => {
        const [x, y] = this.toPx(t.enu[0], t.enu[1]);
        i ? c.lineTo(x, y) : c.moveTo(x, y);
      });
      c.stroke();
    }
    const ch = Story.current();
    if (ch) {
      const [x, y] = this.toPx(ch.target[0], ch.target[1]);
      c.fillStyle = "#e8934a";
      c.beginPath();
      c.arc(x, y, 6, 0, 7);
      c.fill();
    }
    const [x, y] = this.toPx(Rover.pos.x, -Rover.pos.z);
    c.save();
    c.translate(x, y);
    c.rotate(Rover.heading);
    c.fillStyle = "#fff";
    c.beginPath();
    c.moveTo(0, -9); c.lineTo(6, 7); c.lineTo(-6, 7);
    c.closePath();
    c.fill();
    c.restore();
  },
};

// ---------------- Input & UI Listeners
addEventListener("keydown", e => {
  Rover.keys[e.code] = true;
  if (e.code === "KeyC") cycleCamera();
  if (e.code === "KeyL") toggleLights();
  if (e.code === "KeyM") toggleAudio();
  if (e.code === "KeyP") {
    Terrain.showProv = !Terrain.showProv;
    Terrain.tiles.forEach(t => Terrain.style(t));
    toast(`Provenance overlay: ${Terrain.showProv ? 'ON (Red=measured, Green=simulated)' : 'OFF'}`);
  }
  if ((e.code === "Enter" || e.code === "Space") && Story.open) {
    e.preventDefault();
    Story.close();
  }
});
addEventListener("keyup", e => { Rover.keys[e.code] = false; });
addEventListener("resize", () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

$("btnContinue").onclick = () => Story.close();

function toggleLights() {
  const on = Rover.toggleHeadlights();
  $("bLights").textContent = `💡 Lights: ${on ? 'On' : 'Off'}`;
  $("bLights").classList.toggle('active', on);
  toast(`Rover headlights: ${on ? 'ON' : 'OFF'}`);
}
$("bLights").onclick = toggleLights;

function toggleAudio() {
  const on = MarsAudio.toggle();
  $("bAudio").textContent = `🔊 Audio: ${on ? 'On' : 'Off'}`;
  $("bAudio").classList.toggle('active', on);
  toast(`Synthesized Mars Audio: ${on ? 'ON' : 'OFF'}`);
}
$("bAudio").onclick = toggleAudio;

$("bCam").onclick = cycleCamera;

$("bDrill").onclick = () => {
  toast("Deploying 5-DOF arm and spinning coring drill…");
  Rover.triggerDrill(() => {
    toast("Core bedrock sample extracted and sealed!", 4000);
  });
};

$("solSlider").oninput = (e) => {
  setSolTime(+e.target.value);
};

// Craft switcher modal
let selectedCraft = 'perseverance';
$("bCraft").onclick = () => { $("charModal").style.display = "grid"; };
$("bCloseChar").onclick = () => { $("charModal").style.display = "none"; };
document.querySelectorAll(".char-card").forEach(card => {
  card.onclick = () => {
    document.querySelectorAll(".char-card").forEach(c => c.classList.remove("selected"));
    card.classList.add("selected");
    selectedCraft = card.dataset.id;
    MarsAudio.playBeep(880, 0.05);
  };
});
$("bDeployChar").onclick = () => {
  $("charModal").style.display = "none";
  Rover.loadCraft(selectedCraft, (spec) => {
    $("craftName").textContent = spec.name;
    $("bDrill").style.display = spec.isHelicopter ? 'none' : 'inline-block';
    toast(`Deployed ${spec.name} to Jezero Crater`);
  });
};

function begin(mode) {
  $("menu").style.display = "none";
  ["topBar", "hud", "mini", "toolbar", "help"].forEach(i => $(i).style.display = "flex");
  Story.start(mode);
}
$("btnStory").onclick = () => begin("story");
$("btnOpen").onclick = () => begin("open");

// ---------------- Master Animation Loop
let last = performance.now();
function loop(t) {
  requestAnimationFrame(loop);
  const dt = Math.min(0.08, (t - last) / 1000);
  last = t;

  if (!Story.open) {
    Rover.update(dt);
  } else {
    Rover.ground(false);
  }

  updateCamera();
  Terrain.update(camera.position);
  Story.update();
  Mini.draw();
  renderer.render(scene, camera);
}

// ---------------- Bootstrap
(async () => {
  const DEFAULT_DATA = "https://hungrydodo12345.github.io/NASA-Space-Apps-Prep/data";
  let dataUrl = new URLSearchParams(location.search).get("data") || "./data";

  try {
    const headCheck = await fetch(`${dataUrl}/scene.json`, { method: "HEAD" });
    if (!headCheck.ok) dataUrl = DEFAULT_DATA;
  } catch (e) {
    dataUrl = DEFAULT_DATA;
  }

  const sc = await Terrain.init(dataUrl);
  await Story.init(dataUrl);
  Mini.init(sc, dataUrl);

  Rover.loadCraft('perseverance', () => {
    $("craftName").textContent = Rover.currentSpec.name;
  });

  const s = Story.data.start;
  Rover.place(s.pos[0], s.pos[1], s.heading_deg || 0);

  setSolTime(11.5);
  $("loadmsg").textContent = "Ready. (LOD Terrain & NASA 3D Models Loaded)";
  
  const autoMode = new URLSearchParams(location.search).get("mode");
  if (autoMode === "open" || autoMode === "story") {
    begin(autoMode);
  }
  
  requestAnimationFrame(loop);

  window.__game = {
    begin, Rover, Terrain, Story, Env, MarsAudio, camera,
    setCam: m => { camMode = m; },
    setFov: v => { camera.fov = v; camera.updateProjectionMatrix(); }
  };
})();
