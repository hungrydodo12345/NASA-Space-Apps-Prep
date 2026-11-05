/* Jezero Ground Truth Explorer — Mars Rover Mechanics & Free-Roam Simulation
 * Integrates Three.js r128, GLTF NASA Models, Kinematics, Solar Time-of-Day,
 * Multi-Camera Scientific Imaging, and Historic Sol Campaigns.
 */
(() => {
'use strict';
const D = 'data/';
const $ = (id) => document.getElementById(id);
const toast = (msg, ms = 3000) => {
  const t = $('toast');
  t.textContent = msg;
  t.style.display = 'block';
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.style.display = 'none', ms);
};
const loadImg = (src) => new Promise((ok, err) => {
  const i = new Image();
  i.onload = () => ok(i);
  i.onerror = () => err(new Error('load ' + src));
  i.src = src;
});
const pixels = (img) => {
  const c = document.createElement('canvas');
  c.width = img.width; c.height = img.height;
  const g = c.getContext('2d', { willReadFrequently: true });
  g.drawImage(img, 0, 0);
  return g.getImageData(0, 0, img.width, img.height).data;
};

/* ---------- Tag Ledger (stub / persistent local storage) ---------- */
const Ledger = {
  key: 'jezero_tags_v2',
  list() { try { return JSON.parse(localStorage.getItem(this.key) || '[]'); } catch (e) { return []; } },
  add(tag) {
    const all = this.list();
    all.push(tag);
    try { localStorage.setItem(this.key, JSON.stringify(all)); } catch (e) {}
    return all;
  },
};

async function main() {
  const man = await (await fetch(D + 'manifest.json')).json();
  let scores = null;
  try { scores = await (await fetch(D + 'scores.json')).json(); } catch (e) {}
  const [hImg, cImg, corImg] = await Promise.all([
    loadImg(D + man.height.file),
    loadImg(D + man.classes.file),
    loadImg(D + man.corridor.file)
  ]);
  const SIZE = man.size_m, HALF = SIZE / 2;

  /* ---------- Heights (12-bit packed DTM) ---------- */
  const hp = pixels(hImg), HN = man.height.n, H = new Float32Array(HN * HN);
  for (let i = 0; i < HN * HN; i++) {
    H[i] = man.height.emin + (((hp[i * 4] << 4) | (hp[i * 4 + 1] >> 4)) / 4095) * (man.height.emax - man.height.emin) - man.height.datum;
  }
  const heightAt = (x, z) => {
    const fx = Math.min(HN - 1.001, Math.max(0, (x / SIZE + 0.5) * HN - 0.5));
    const fz = Math.min(HN - 1.001, Math.max(0, (z / SIZE + 0.5) * HN - 0.5));
    const x0 = fx | 0, z0 = fz | 0, tx = fx - x0, tz = fz - z0;
    const a = H[z0 * HN + x0], b = H[z0 * HN + x0 + 1], c = H[(z0 + 1) * HN + x0], d = H[(z0 + 1) * HN + x0 + 1];
    return (a * (1 - tx) + b * tx) * (1 - tz) + (c * (1 - tx) + d * tx) * tz;
  };
  const corPx = pixels(corImg), corW = corImg.width;

  /* ---------- WebGL Renderer & Adaptive Upscaler Pass ---------- */
  const canvas = $('view');
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance' });
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.setPixelRatio(1);

  let renderScale = Math.min(1, 1.6 / (window.devicePixelRatio || 1) + 0.25);
  const rt = new THREE.WebGLRenderTarget(4, 4, { minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter });
  const postScene = new THREE.Scene(), postCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  const post = new THREE.ShaderMaterial({
    uniforms: { src: { value: rt.texture }, texel: { value: new THREE.Vector2() }, sharp: { value: 0.35 } },
    vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }',
    fragmentShader: `uniform sampler2D src; uniform vec2 texel; uniform float sharp; varying vec2 vUv;
      void main(){ vec3 c = texture2D(src, vUv).rgb; vec3 n = texture2D(src, vUv + vec2(0., texel.y)).rgb, s = texture2D(src, vUv - vec2(0., texel.y)).rgb;
        vec3 e = texture2D(src, vUv + vec2(texel.x, 0.)).rgb, w = texture2D(src, vUv - vec2(texel.x, 0.)).rgb;
        vec3 mn = min(c, min(min(n, s), min(e, w))), mx = max(c, max(max(n, s), max(e, w)));
        vec3 o = c + sharp * (4.0 * c - n - s - e - w) * 0.25; gl_FragColor = vec4(clamp(o, mn, mx), 1.0); }`,
  });
  postScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), post));

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(60, 1, 0.05, 3500);

  const resize = () => {
    const w = window.innerWidth, h = window.innerHeight;
    renderer.setSize(w, h, false);
    canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
    camera.aspect = w / h; camera.updateProjectionMatrix();
    const rw = Math.max(64, Math.round(w * renderScale)), rh = Math.max(64, Math.round(h * renderScale));
    rt.setSize(rw, rh); post.uniforms.texel.value.set(1 / rw, 1 / rh);
  };
  window.addEventListener('resize', resize);
  resize();

  /* ---------- HiRISE Measured Terrain & Detail Synthesis ---------- */
  const tl = new THREE.TextureLoader();
  const tex = (f, nearest) => {
    const t = tl.load(D + f);
    t.flipY = false;
    t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
    if (nearest) {
      t.minFilter = t.magFilter = THREE.NearestFilter;
    } else {
      t.minFilter = THREE.LinearMipmapLinearFilter;
      t.anisotropy = Math.min(+(new URLSearchParams(location.search).get('aniso') || 1), renderer.capabilities.getMaxAnisotropy());
    }
    return t;
  };

  const baseTex = tex(man.base.file), clsTex = tex(man.classes.file), conTex = tex(man.contrast.file), corTex = tex(man.corridor.file, true);
  const atlasTex = tex(man.atlas.file);
  atlasTex.minFilter = THREE.LinearFilter;
  atlasTex.generateMipmaps = false;

  const MAXS = Math.max(1, man.corridor.stops.length);
  const stops = new Array(MAXS).fill(0).map(() => new THREE.Vector4(1e6, 1e6, 0, 0));
  man.corridor.stops.slice(0, MAXS).forEach((s, i) => stops[i].set(s.x, s.z, s.ratio, s.tile));

  const NS = man.corridor.stops.length, IM = Math.ceil(SIZE), idxData = new Uint8Array(IM * IM * 4);
  for (let gz = 0; gz < IM; gz++) for (let gx = 0; gx < IM; gx++) {
    const x = gx + 0.5 - SIZE / 2, z = gz + 0.5 - SIZE / 2;
    let best = 0, bd = 1e9;
    man.corridor.stops.forEach((st, i) => {
      const dx = x - st.x, dz = z - st.z, d = dx * dx + dz * dz;
      if (Math.abs(dx) < man.corridor.half_m && Math.abs(dz) < man.corridor.half_m && d < bd) {
        bd = d; best = i + 1;
      }
    });
    idxData[(gz * IM + gx) * 4] = best;
  }
  const corIdx = new THREE.DataTexture(idxData, IM, IM, THREE.RGBAFormat);
  corIdx.minFilter = corIdx.magFilter = THREE.NearestFilter;
  corIdx.needsUpdate = true;

  const spData = new Uint8Array(Math.max(1, NS) * 2 * 4);
  man.corridor.stops.forEach((st, i) => {
    const ex = Math.round((st.x / SIZE + 0.5) * 65535), ez = Math.round((st.z / SIZE + 0.5) * 65535);
    spData.set([ex >> 8, ex & 255, ez >> 8, ez & 255], i * 4);
    spData.set([Math.round(Math.min(1, st.ratio / 4) * 255), st.tile, 0, 255], (NS + i) * 4);
  });
  const stopTex = new THREE.DataTexture(spData, Math.max(1, NS), 2, THREE.RGBAFormat);
  stopTex.minFilter = stopTex.magFilter = THREE.NearestFilter;
  stopTex.needsUpdate = true;

  const hc = man.corridor.half_m, xs = man.corridor.stops.map((s) => s.x), zs = man.corridor.stops.map((s) => s.z);
  const corBox = new THREE.Vector4(Math.min(...xs) - hc, Math.min(...zs) - hc, Math.max(...xs) + hc, Math.max(...zs) + hc);

  const URLP = new URLSearchParams(location.search);
  const seg = Math.min(+(URLP.get('seg') || 96), man.height.n - 1);
  const geo = new THREE.PlaneGeometry(SIZE, SIZE, seg, seg);
  geo.rotateX(-Math.PI / 2);
  const pos = geo.attributes.position;
  for (let i = 0; i < pos.count; i++) pos.setY(i, heightAt(pos.getX(i), pos.getZ(i)));
  geo.computeVertexNormals();

  const sun = new THREE.Vector3(-0.35, 0.72, 0.6).normalize();

  const terrainMat = new THREE.ShaderMaterial({
    extensions: { derivatives: true }, defines: { QUALITY: 2 },
    uniforms: {
      baseTex: { value: baseTex }, clsTex: { value: clsTex }, conTex: { value: conTex }, atlasTex: { value: atlasTex }, corTex: { value: corTex },
      ampRatio: { value: new THREE.Vector4(...man.amp_ratio) }, conScale: { value: man.contrast.scale }, size: { value: SIZE },
      tileM: { value: man.atlas.tile_m }, tilesPerClass: { value: man.atlas.tiles_per_class },
      stops: { value: stops }, nStops: { value: new URLSearchParams(location.search).has('nocor') ? 0 : Math.min(MAXS, man.corridor.stops.length) },
      stopHalf: { value: man.corridor.half_m }, corBox: { value: corBox }, corIdx: { value: corIdx }, stopTex: { value: stopTex },
      nStopsF: { value: Math.max(1, NS) }, corCols: { value: man.corridor.cols }, corTile: { value: man.corridor.tile_px / corW },
      sunDir: { value: sun }, tau: { value: 0.6 }, showProv: { value: 0 }, camPos: { value: new THREE.Vector3() }, quality: { value: 2 },
      cutBox: { value: new THREE.Vector4(1e6, 1e6, -1e6, -1e6) },
      bakeTex: { value: null }, bakeOrigin: { value: new THREE.Vector2(1e6, 1e6) }, bakeSize: { value: 64 },
    },
    vertexShader: `varying vec3 vPos; varying vec3 vN;
      void main(){ vPos = position; vN = normal; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: `
      uniform sampler2D baseTex, bakeTex; uniform float size; uniform vec2 bakeOrigin; uniform float bakeSize;
      uniform vec3 sunDir, camPos; uniform float tau, showProv; uniform vec4 cutBox;
      varying vec3 vPos; varying vec3 vN;
      void main(){
        vec2 w = vPos.xz; vec2 uv = w / size + 0.5;
        if (w.x > cutBox.x && w.x < cutBox.z && w.y > cutBox.y && w.y < cutBox.w) discard;
        float g = texture2D(baseTex, uv).r;
        float mpp = max(fwidth(w.x), fwidth(w.y));
        float detail = 0.0, real = 0.0, fade = 0.0;
        #if QUALITY > 0
        fade = 1.0 - smoothstep(0.05, 0.22, mpp);
        vec2 bt = (w - bakeOrigin) / bakeSize;
        if (fade > 0.01 && bt.x > 0.0 && bt.x < 1.0 && bt.y > 0.0 && bt.y < 1.0) {
          vec4 bk = texture2D(bakeTex, bt); detail = (bk.r - 0.5) / 1.5 * fade; real = bk.g;
        }
        #endif
        float v = clamp(g + detail, 0.0, 1.0);
        vec3 albedo = mix(vec3(0.30, 0.18, 0.11), vec3(0.88, 0.64, 0.46), v);
        vec3 n = normalize(vN);
        float light = 0.72 + 0.38 * max(dot(n, sunDir), 0.0) - 0.06 * (tau - 0.6);
        vec3 col = albedo * light;
        if (showProv > 0.5 && fade > 0.05) col = mix(col, real > 0.5 ? vec3(0.2, 0.9, 0.35) : vec3(0.95, 0.25, 0.2), 0.30 * fade);
        float dist = length(vPos - camPos);
        float fogK = 1.0 - exp(-dist * 0.0009 * tau * tau);
        vec3 dust = mix(vec3(0.78, 0.6, 0.45), vec3(0.62, 0.44, 0.3), clamp(tau / 4.0, 0.0, 1.0));
        gl_FragColor = vec4(mix(col, dust, clamp(fogK, 0.0, 0.92)), 1.0);
      }`,
  });
  const terrain = new THREE.Mesh(geo, terrainMat);
  terrain.receiveShadow = true;
  scene.add(terrain);

  // Detail bake pass
  const BAKE_M = 64, BAKE_PX = 1024, STRIPS = 8;
  const bakeRT = [0, 1].map(() => new THREE.WebGLRenderTarget(BAKE_PX, BAKE_PX, { minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, depthBuffer: false }));
  const bakeUniforms = Object.assign({}, terrainMat.uniforms, { bakeOrigin: { value: new THREE.Vector2(1e6, 1e6) } });
  const bakeMat = new THREE.ShaderMaterial({
    defines: { QUALITY: 2, MAXS: MAXS }, uniforms: bakeUniforms, depthTest: false, depthWrite: false,
    vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }',
    fragmentShader: `
      uniform sampler2D clsTex, conTex, atlasTex, corTex, corIdx, stopTex; uniform vec4 ampRatio; uniform float conScale, size, tileM, tilesPerClass;
      uniform int nStops; uniform float stopHalf, corCols, corTile, nStopsF, bakeSize; uniform vec4 corBox; uniform vec2 bakeOrigin;
      varying vec2 vUv;
      float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
      float tileSample(float cls, vec2 w){
        vec2 q = w / tileM; vec2 cell = floor(q); vec2 f = fract(q);
        float h = hash(cell + cls * 17.13); float rot = floor(hash(cell + 3.7) * 4.0);
        if (rot == 1.0) f = vec2(1.0 - f.y, f.x); else if (rot == 2.0) f = 1.0 - f; else if (rot == 3.0) f = vec2(f.y, 1.0 - f.x);
        float variant = floor(h * tilesPerClass);
        vec2 auv = (vec2(variant, cls) + clamp(f, 0.004, 0.996)) / vec2(tilesPerClass, 4.0);
        return (texture2D(atlasTex, auv).r - 0.5) * 6.0;
      }
      float bomb(float cls, vec2 w){
        #if QUALITY < 2
        return tileSample(cls, w);
        #else
        vec2 fa = fract(w / tileM), fb = fract(w / tileM + 0.5);
        float wa = sin(3.14159 * fa.x) * sin(3.14159 * fa.y), wb = sin(3.14159 * fb.x) * sin(3.14159 * fb.y);
        return (wa * tileSample(cls, w) + wb * tileSample(cls + 0.0, w + 0.5 * tileM + 7.3)) / sqrt(wa * wa + wb * wb + 1e-4);
        #endif
      }
      void main(){
        vec2 w = bakeOrigin + vUv * bakeSize; vec2 uv = w / size + 0.5;
        vec4 cls = texture2D(clsTex, uv); cls /= max(dot(cls, vec4(1.0)), 1e-3);
        float contrast = texture2D(conTex, uv).r * conScale;
        float detail = 0.0, real = 0.0;
        if (cls.x > 0.03) detail += cls.x * ampRatio.x * bomb(0.0, w);
        if (cls.y > 0.03) detail += cls.y * ampRatio.y * bomb(1.0, w);
        if (cls.z > 0.03) detail += cls.z * ampRatio.z * bomb(2.0, w);
        if (cls.w > 0.03) detail += cls.w * ampRatio.w * bomb(3.0, w);
        vec4 a2 = ampRatio * ampRatio;
        detail *= sqrt(dot(cls, a2) / max(dot(cls * cls, a2), 1e-6));
        if (nStops > 0 && w.x > corBox.x && w.x < corBox.z && w.y > corBox.y && w.y < corBox.w) {
          float id = floor(texture2D(corIdx, uv).r * 255.0 + 0.5);
          if (id > 0.5) {
            float u = (id - 0.5) / nStopsF;
            vec4 a = texture2D(stopTex, vec2(u, 0.25)), b = texture2D(stopTex, vec2(u, 0.75));
            vec2 ctr = (vec2(a.r * 65280.0 + a.g * 255.0, a.b * 65280.0 + a.a * 255.0) / 65535.0 - 0.5) * size;
            vec2 d = w - ctr;
            if (abs(d.x) < stopHalf && abs(d.y) < stopHalf) {
              vec2 t = (d + stopHalf) / (2.0 * stopHalf);
              float tile = floor(b.g * 255.0 + 0.5); vec2 tc = vec2(mod(tile, corCols), floor(tile / corCols));
              vec4 c = texture2D(corTex, (tc + t) * corTile);
              if (c.g > 0.5) { detail = (c.r - 0.5) * 6.0 * (b.r * 4.0); real = 1.0; }
            }
          }
        }
        gl_FragColor = vec4(clamp(detail * contrast * 1.5 + 0.5, 0.0, 1.0), real, 0.0, 1.0);
      }`,
  });
  const bakeScene = new THREE.Scene();
  bakeScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), bakeMat));
  let bakeFront = 0, bakeStrip = -1, bakePending = null;
  const requestBake = (ox, oz) => { bakePending = [ox, oz]; bakeStrip = 0; };
  const bakeStep = () => {
    if (bakeStrip < 0 || terrainMat.defines.QUALITY === 0) return;
    const back = bakeRT[1 - bakeFront], h = BAKE_PX / STRIPS;
    bakeUniforms.bakeOrigin.value.set(bakePending[0], bakePending[1]);
    back.scissor.set(0, bakeStrip * h, BAKE_PX, h);
    back.scissorTest = true;
    renderer.setRenderTarget(back);
    renderer.render(bakeScene, postCam);
    renderer.setRenderTarget(null);
    if (++bakeStrip >= STRIPS) {
      bakeFront = 1 - bakeFront; bakeStrip = -1;
      terrainMat.uniforms.bakeTex.value = bakeRT[bakeFront].texture;
      terrainMat.uniforms.bakeOrigin.value.set(bakePending[0], bakePending[1]);
    }
  };

  // Near patch: full 1 m DTM detail
  const NEAR = 96, NSEG = 96;
  const nearUniforms = Object.assign({}, terrainMat.uniforms, { cutBox: { value: new THREE.Vector4(1e6, 1e6, -1e6, -1e6) } });
  const nearMat = new THREE.ShaderMaterial({
    extensions: { derivatives: true }, defines: { QUALITY: 2 },
    uniforms: nearUniforms, vertexShader: terrainMat.vertexShader, fragmentShader: terrainMat.fragmentShader
  });
  const ngeo = new THREE.PlaneGeometry(NEAR, NEAR, NSEG, NSEG);
  ngeo.rotateX(-Math.PI / 2);
  const nbase = ngeo.attributes.position.array.slice();
  const near = new THREE.Mesh(ngeo, nearMat);
  near.receiveShadow = true;
  near.frustumCulled = false;
  scene.add(near);

  let nearC = [1e9, 1e9];
  const updateNear = (x, z) => {
    const cx = Math.round(x / 8) * 8, cz = Math.round(z / 8) * 8;
    if (cx === nearC[0] && cz === nearC[1]) return;
    nearC = [cx, cz];
    const a = ngeo.attributes.position;
    for (let i = 0; i < a.count; i++) {
      const px = nbase[i * 3] + cx, pz = nbase[i * 3 + 2] + cz;
      a.setXYZ(i, px, heightAt(px, pz), pz);
    }
    a.needsUpdate = true;
    ngeo.computeVertexNormals();
    ngeo.computeBoundingSphere();
    const m = NEAR / 2 - 2;
    terrainMat.uniforms.cutBox.value.set(cx - m, cz - m, cx + m, cz + m);
    requestBake(cx - BAKE_M / 2, cz - BAKE_M / 2);
  };

  /* ---------- Lighting, Solar Time & Dust ---------- */
  MarsLighting.init(scene, sun);
  const sky = () => {
    const t = +$('tau').value;
    terrainMat.uniforms.tau.value = t;
    nearMat.uniforms.tau.value = t;
  };
  $('tau').addEventListener('input', sky);
  $('solSlider').addEventListener('input', (e) => {
    MarsLighting.setSolTime(+e.target.value, scene, [terrainMat, nearMat]);
    $('solClock').textContent = MarsLighting.getFormattedTime();
  });

  /* ---------- Campaign & Beacon Setup ---------- */
  MarsCampaign.init(scene);

  /* Gyro look-around for mobile */
  let gyro = null;
  function onOri(e) { if (e.alpha != null) gyro = { a: e.alpha, b: e.beta, g: e.gamma }; }
  window.addEventListener('deviceorientation', onOri);
  const zee = new THREE.Vector3(0, 0, 1), q1 = new THREE.Quaternion(-Math.sqrt(0.5), 0, 0, Math.sqrt(0.5));
  const eul = new THREE.Euler(), qd = new THREE.Quaternion(), q0 = new THREE.Quaternion();
  const deviceQuat = () => {
    if (!gyro) return null;
    const d = THREE.MathUtils.degToRad;
    eul.set(d(gyro.b), d(gyro.a), -d(gyro.g), 'YXZ');
    qd.setFromEuler(eul);
    qd.multiply(q1);
    qd.multiply(q0.setFromAxisAngle(zee, -d(window.orientation || 0)));
    return qd;
  };

  /* ---------- Vehicles & Craft Loading Engine ---------- */
  let currentRig = null;
  let currentCraftSpec = RoverModels.SPECS.perseverance;
  const currentPos = { x: man.start.x, y: heightAt(man.start.x, man.start.z), z: man.start.z };
  MarsMechanics.state.yaw = man.start.yaw || 0;

  function setVehicle(roverId, spawnCoord) {
    const spec = RoverModels.SPECS[roverId] || RoverModels.SPECS.perseverance;
    currentCraftSpec = spec;

    if (spawnCoord) {
      currentPos.x = spawnCoord.x;
      currentPos.z = spawnCoord.z;
      currentPos.y = (typeof spawnCoord.y === 'number') ? spawnCoord.y : heightAt(spawnCoord.x, spawnCoord.z);
      MarsMechanics.state.yaw = spawnCoord.yaw || 0;
    } else {
      currentPos.y = heightAt(currentPos.x, currentPos.z);
    }

    if (currentRig && currentRig.root) {
      scene.remove(currentRig.root);
    }

    MarsMechanics.resetState();
    MarsMechanics.state.isHelicopter = !!spec.isHelicopter;

    // Immediately attach the procedural rig as an instant zero-latency craft
    const instantRig = spec.isHelicopter ? RoverModels.createProceduralIngenuity() : RoverModels.createProceduralRover(spec);
    currentRig = instantRig;
    instantRig.root.position.set(currentPos.x, currentPos.y, currentPos.z);
    instantRig.root.rotation.set(0, MarsMechanics.state.yaw, 0, 'YXZ');
    scene.add(instantRig.root);
    MarsLighting.registerRig(instantRig);

    // Initial safe camera placement
    MarsCameras.update(0.05, camera, currentPos, MarsMechanics.state.yaw, 0, 0, currentRig, gyro, deviceQuat);

    $('craftName').textContent = spec.name;

    // Toggle UI buttons based on whether craft is rover or helicopter
    if (spec.isHelicopter) {
      $('armControls').style.display = 'none';
      $('heliControls').style.display = 'inline-flex';
    } else {
      $('armControls').style.display = 'inline-flex';
      $('heliControls').style.display = 'none';
    }

    RoverModels.loadGLBModel(spec, scene, (rig) => {
      // If still matching active craft, replace with loaded NASA GLB model
      if (currentCraftSpec.id === spec.id) {
        if (currentRig && currentRig.root) scene.remove(currentRig.root);
        currentRig = rig;
        rig.root.position.set(currentPos.x, currentPos.y, currentPos.z);
        rig.root.rotation.set(MarsMechanics.state.pitch, MarsMechanics.state.yaw, MarsMechanics.state.roll, 'YXZ');
        scene.add(rig.root);
        MarsLighting.registerRig(rig);
      }
      toast(`Deployed ${spec.name} to Jezero Crater`);
    });
  }

  // Initial load: support ?craft= query parameter or default to Perseverance
  const initCraft = URLP.get('craft') || 'perseverance';
  const initialCamp = MarsCampaign.selectRover(initCraft);
  setVehicle(initCraft, (initCraft === 'perseverance' && !URLP.has('craft')) ? { x: man.start.x, z: man.start.z, yaw: man.start.yaw } : initialCamp.startCoord);

  /* ---------- Tag ledger & raycasting ---------- */
  const tagGroup = new THREE.Group();
  scene.add(tagGroup);
  const drawTags = () => {
    tagGroup.clear();
    for (const t of Ledger.list()) {
      const m = new THREE.Mesh(new THREE.ConeGeometry(0.25, 0.8, 8), new THREE.MeshLambertMaterial({ color: t.provenance.startsWith('measured (rover') ? 0x4fd27a : 0xe0b040 }));
      m.position.set(t.x, t.y + 0.5, t.z);
      tagGroup.add(m);
    }
  };
  drawTags();

  const corridorReal = (x, z) => {
    for (const s of man.corridor.stops) {
      const dx = x - s.x, dz = z - s.z, h = man.corridor.half_m;
      if (Math.abs(dx) < h && Math.abs(dz) < h) {
        const tx = (dx + h) / (2 * h), tz = (dz + h) / (2 * h), tc = s.tile % man.corridor.cols, tr = Math.floor(s.tile / man.corridor.cols);
        const px = Math.floor((tc + tx) * man.corridor.tile_px), py = Math.floor((tr + tz) * man.corridor.tile_px);
        if (corPx[(py * corW + px) * 4 + 1] > 127) return true;
      }
    }
    return false;
  };
  const provenanceAt = (p, metresPerPixel) => {
    if (corridorReal(p.x, p.z)) return 'measured (rover photo, ~cm)';
    if (metresPerPixel < 0.12) return 'simulated (rover-texture detail)';
    return 'measured (orbital, ~60 cm real detail)';
  };

  const ray = new THREE.Raycaster();
  const reticleHit = () => {
    ray.setFromCamera(new THREE.Vector2(0, 0), camera);
    const h = ray.intersectObjects([near, terrain]);
    return h.length ? h[0] : null;
  };
  const mppAt = (dist) => 2 * dist * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) / (window.innerHeight * renderScale);
  const toLatLon = (x, z) => {
    const Rm = 3394839.8133163, k = 180 / Math.PI;
    return [man.center.lat - (z / Rm) * k, man.center.lon + (x / (Rm * Math.cos(15 * Math.PI / 180))) * k];
  };

  /* ---------- User Input & Key Bindings ---------- */
  const keys = {};
  addEventListener('keydown', (e) => {
    const k = e.key.toLowerCase();
    keys[k] = true;
    if (k === 'c') cycleCamera();
    if (k === 'l') toggleLights();
    if (k === 'm') toggleAudio();
    if (k === 'p') toggleProv();
    if (k === 'z') adjustZoom(0.5);
    if (k === 'x') adjustZoom(-0.5);
  });
  addEventListener('keyup', (e) => { keys[e.key.toLowerCase()] = false; });

  const pad = { f: 'w', b: 's', l: 'a', r: 'd' };
  document.querySelectorAll('#pad button').forEach((b) => {
    const k = pad[b.dataset.k];
    const on = (e) => { e.preventDefault(); keys[k] = true; };
    const off = (e) => { e.preventDefault(); keys[k] = false; };
    b.addEventListener('touchstart', on, { passive: false });
    b.addEventListener('touchend', off);
    b.addEventListener('mousedown', on);
    b.addEventListener('mouseup', off);
    b.addEventListener('mouseleave', off);
  });

  /* ---------- Camera Switching & Zoom ---------- */
  function cycleCamera() {
    const mode = MarsCameras.cycleNext();
    $('bCam').textContent = mode.name;
    $('camName').textContent = mode.name;
    $('zoomWrap').style.display = mode.id === 'mastcam' ? 'inline-flex' : 'none';
    if (currentRig && currentRig.root) {
      currentRig.root.visible = mode.id !== 'mastcam';
    }
    toast(`Camera: ${mode.name} — ${mode.desc}`);
  }
  $('bCam').onclick = cycleCamera;

  function adjustZoom(delta) {
    const z = MarsCameras.adjustZoom(delta, camera);
    $('zoomVal').textContent = z.toFixed(1) + 'x';
  }
  $('bZoomIn').onclick = () => adjustZoom(0.5);
  $('bZoomOut').onclick = () => adjustZoom(-0.5);

  /* ---------- Robotic Arm Controls ---------- */
  $('bArmStow').onclick = () => {
    MarsMechanics.setArmMode('RETRACTING');
    toast('Retracting robotic arm to travel stowage');
  };
  $('bArmInspect').onclick = () => {
    MarsMechanics.setArmMode('DEPLOYING');
    toast('Deploying 5-DOF robotic arm for surface contact inspection');
  };
  $('bArmDrill').onclick = () => {
    MarsMechanics.setArmMode('DRILLING');
    toast('Coring drill active! Extracting rock sample…');
    MarsCampaign.checkAction('drill', currentPos, (milestone) => {
      toast(`Milestone Complete: ${milestone.completedText}`, 4500);
    }, (campaign) => {
      onAllCampaignMilestonesCompleted(campaign);
    });
  };

  /* ---------- Helicopter Controls ---------- */
  $('bHeliTakeoff').onclick = () => {
    if (MarsMechanics.state.flightState === 'LANDED') {
      MarsMechanics.state.flightState = 'SPOOLING';
      toast('Ingenuity: Spooling rotors to 2,400 RPM for takeoff');
    } else {
      MarsMechanics.state.flightState = 'LANDING';
      toast('Ingenuity: Commencing descent and landing sequence');
    }
  };
  $('bHeliUp').onclick = () => {
    MarsMechanics.state.targetAlt = Math.min(14, MarsMechanics.state.targetAlt + 2.0);
    toast(`Target altitude: ${MarsMechanics.state.targetAlt.toFixed(1)} m`);
  };
  $('bHeliDown').onclick = () => {
    MarsMechanics.state.targetAlt = Math.max(1.0, MarsMechanics.state.targetAlt - 1.5);
    toast(`Target altitude: ${MarsMechanics.state.targetAlt.toFixed(1)} m`);
  };

  /* ---------- Lighting & Audio Toggles ---------- */
  function toggleLights() {
    const on = MarsLighting.toggleHeadlights(currentRig);
    $('bLights').textContent = `💡 Lights: ${on ? 'On' : 'Off'}`;
    $('bLights').classList.toggle('on', on);
  }
  $('bLights').onclick = toggleLights;

  function toggleAudio() {
    const on = MarsAudio.toggle();
    $('bAudio').textContent = `🔊 Audio: ${on ? 'On' : 'Off'}`;
    $('bAudio').classList.toggle('on', on);
  }
  $('bAudio').onclick = toggleAudio;

  const toggleProv = () => {
    terrainMat.uniforms.showProv.value = 1 - terrainMat.uniforms.showProv.value;
    nearMat.uniforms.showProv.value = terrainMat.uniforms.showProv.value;
    $('bProv').classList.toggle('on');
  };
  $('bProv').onclick = toggleProv;

  $('bScores').onclick = () => {
    const s = $('scores');
    s.style.display = s.style.display === 'block' ? 'none' : 'block';
  };

  /* ---------- Rover Character Selection Modal ---------- */
  const roverGrid = $('roverGrid');
  roverGrid.innerHTML = Object.values(RoverModels.SPECS).map((s) => `
    <div class="rover-card ${s.id === currentCraftSpec.id ? 'selected' : ''}" data-id="${s.id}">
      <h3>${s.name}</h3>
      <div class="spec-badge">📍 ${s.site} · ⚖️ ${s.mass} · ⚡ ${s.power}</div>
      <div class="highlight">${s.highlight}</div>
    </div>
  `).join('');

  let selectedRoverId = currentCraftSpec.id;
  roverGrid.querySelectorAll('.rover-card').forEach((card) => {
    card.onclick = () => {
      roverGrid.querySelectorAll('.rover-card').forEach((c) => c.classList.remove('selected'));
      card.classList.add('selected');
      selectedRoverId = card.dataset.id;
      if (window.MarsAudio) window.MarsAudio.playBeep(980, 0.05);
    };
  });

  $('bSelectRover').onclick = () => { $('charSelectModal').style.display = 'flex'; };
  $('bCloseChar').onclick = () => { $('charSelectModal').style.display = 'none'; };
  $('bLaunchMission').onclick = () => {
    $('charSelectModal').style.display = 'none';
    const camp = MarsCampaign.selectRover(selectedRoverId);
    setVehicle(selectedRoverId, camp.startCoord);
  };

  /* ---------- Component Inspector Modal ---------- */
  const compList = $('compList');
  compList.innerHTML = Object.values(RoverModels.COMPONENT_INFO).map((c) => `
    <div class="comp-card">
      <b>${c.name}</b>
      <div class="role">${c.role} — <i>${c.movement}</i></div>
      <div>${c.description}</div>
    </div>
  `).join('');

  $('bInspectComp').onclick = () => { $('compInspectModal').style.display = 'flex'; };
  $('bCloseComp').onclick = () => { $('compInspectModal').style.display = 'none'; };

  /* ---------- Free-Roam Mars Completion Banner ---------- */
  function onAllCampaignMilestonesCompleted(campaign) {
    $('celebrationModal').style.display = 'flex';
    $('freeRoamBadge').style.display = 'inline';
    if (window.MarsAudio) window.MarsAudio.playBeep(1400, 0.4);
  }
  $('bStartFreeRoam').onclick = () => {
    $('celebrationModal').style.display = 'none';
    toast('Free-Roam Mars World Activated! Drive and fly anywhere across Jezero Crater.');
  };

  /* ---------- Simulation Step & Frame Loop ---------- */
  const clock = new THREE.Clock();
  let fpsAcc = 0, fpsN = 0, fps = 0, provTimer = 0;

  function step(dt) {
    updateNear(currentPos.x, currentPos.z);

    // Survey boundary: active during campaign, removed in Free-Roam
    const surveyLimit = MarsCampaign.isFreeRoamUnlocked() ? null : (HALF - 12);

    const onBoundaryHit = (customMsg) => {
      toast(customMsg || 'Survey boundary: Complete campaign milestones to unlock unrestricted Free-Roam Mars!');
    };

    // Update vehicle kinematics & mechanics
    if (currentCraftSpec.isHelicopter) {
      MarsMechanics.updateHelicopter(dt, keys, currentRig, heightAt, currentPos, surveyLimit, onBoundaryHit);
    } else {
      MarsMechanics.updateRover(dt, keys, currentRig, heightAt, currentPos, surveyLimit, onBoundaryHit);
      MarsMechanics.updateRoboticArm(dt, currentRig);
    }

    // Update lighting & solar time
    MarsLighting.update(dt, currentPos, [terrainMat, nearMat], scene);

    // Update campaign milestones & waypoints
    const campStatus = MarsCampaign.update(dt, currentPos, heightAt, null, onAllCampaignMilestonesCompleted);
    if (campStatus && campStatus.milestone) {
      const m = campStatus.milestone;
      $('milestoneTitle').textContent = `MILESTONE ${campStatus.currentNumber}/${campStatus.totalMilestones}: SOL ${m.sol}`;
      $('milestoneObj').textContent = m.objective;
      $('waypointDist').textContent = `${campStatus.distance.toFixed(1)} m`;

      // Visual progress dots
      let dots = '';
      for (let i = 1; i <= campStatus.totalMilestones; i++) {
        dots += (i < campStatus.currentNumber ? '● ' : i === campStatus.currentNumber ? '◐ ' : '○ ');
      }
      $('progressDots').textContent = dots.trim();

      // If near target and action can be executed
      if (campStatus.isNearTarget) {
        $('bAction').style.display = 'inline-block';
        $('bAction').textContent = m.actionRequired === 'drill' ? 'DRILL SAMPLE' : 'CHECK-IN';
        $('bAction').onclick = () => {
          MarsCampaign.checkAction(m.actionRequired, currentPos, (milestone) => {
            toast(`Milestone Complete: ${milestone.completedText}`, 4500);
          }, onAllCampaignMilestonesCompleted);
        };
      } else {
        $('bAction').style.display = 'none';
      }
    } else {
      $('milestoneTitle').textContent = 'MARS FREE-ROAM WORLD';
      $('milestoneObj').textContent = 'Explore freely across Jezero Crater. Switch crafts, drill bedrock, and fly anywhere!';
      $('waypointDist').textContent = 'UNRESTRICTED';
      $('bAction').style.display = 'none';
      $('freeRoamBadge').style.display = 'inline';
    }

    // Multi-camera suite update
    MarsCameras.update(dt, camera, currentPos, MarsMechanics.state.yaw, MarsMechanics.state.pitch, MarsMechanics.state.roll, currentRig, gyro, deviceQuat);
    terrainMat.uniforms.camPos.value.copy(camera.position);

    // Update top telemetry bar
    $('solClock').textContent = MarsLighting.getFormattedTime();
    $('speedVal').textContent = `${Math.abs(MarsMechanics.state.v).toFixed(1)} m/s`;
    $('pitchRollVal').textContent = `${(MarsMechanics.state.pitch * 180 / Math.PI).toFixed(0)}° / ${(MarsMechanics.state.roll * 180 / Math.PI).toFixed(0)}°`;
  }

  function frame() {
    const dt = Math.min(0.05, clock.getDelta());
    step(dt);
    bakeStep();

    renderer.setRenderTarget(rt);
    renderer.render(scene, camera);
    renderer.setRenderTarget(null);
    renderer.render(postScene, postCam);

    fpsAcc += dt;
    fpsN++;
    if (fpsAcc > 1) {
      fps = fpsN / fpsAcc; fpsAcc = 0; fpsN = 0;
      const q = terrainMat.uniforms.quality; let ns = renderScale;
      if (fps < 27) {
        if (renderScale > 0.55) ns = Math.max(0.5, renderScale - 0.1);
      } else if (fps > 50) {
        if (renderScale < 0.95) ns = Math.min(1, renderScale + 0.05);
      }
      if (ns !== renderScale) { renderScale = ns; resize(); }
    }

    provTimer += dt;
    if (provTimer > 0.25) {
      provTimer = 0;
      const h = reticleHit();
      const p = h ? provenanceAt(h.point, mppAt(h.distance)) : '—';
      const cls = p.startsWith('measured (rover') ? 'm' : p.startsWith('simulated') ? 's' : 'n';
      $('provtxt').innerHTML = `under reticle:<br><span class="${cls}">${p}</span>`;
    }

    requestAnimationFrame(frame);
  }

  // Done loading
  $('loading').style.display = 'none';
  frame();
}

main().catch((e) => {
  $('loading').textContent = 'Failed to load data: ' + e.message;
  console.error(e);
});
})();
