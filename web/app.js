/* Jezero Ground Truth Explorer — no build step; three.js r128 from CDN.
 * Every rendered pixel has a provenance:
 *   measured (orbital)  : fused HiRISE imagery + HiRISE stereo DTM (always the base)
 *   measured (rover)    : real Perseverance Navcam ground texture, projected through the camera model
 *   simulated           : rover-texture detail for scales HiRISE cannot record (fuzzy surface-class mix)
 */
(() => {
'use strict';
const D = 'data/';
const $ = (id) => document.getElementById(id);
const toast = (msg, ms = 2600) => { const t = $('toast'); t.textContent = msg; t.style.display = 'block'; clearTimeout(toast._t); toast._t = setTimeout(() => t.style.display = 'none', ms); };
const loadImg = (src) => new Promise((ok, err) => { const i = new Image(); i.onload = () => ok(i); i.onerror = () => err(new Error('load ' + src)); i.src = src; });
const pixels = (img) => { const c = document.createElement('canvas'); c.width = img.width; c.height = img.height; const g = c.getContext('2d', { willReadFrequently: true }); g.drawImage(img, 0, 0); return g.getImageData(0, 0, img.width, img.height).data; };

/* ---------- tag ledger (stub; swap for Supabase/Firebase later) ---------- */
const Ledger = {
  key: 'jezero_tags_v1',
  list() { try { return JSON.parse(localStorage.getItem(this.key) || '[]'); } catch (e) { return []; } },
  add(tag) { const all = this.list(); all.push(tag); try { localStorage.setItem(this.key, JSON.stringify(all)); } catch (e) {} return all; },
};

async function main() {
  const man = await (await fetch(D + 'manifest.json')).json();
  let scores = null; try { scores = await (await fetch(D + 'scores.json')).json(); } catch (e) {}
  const [hImg, cImg, corImg] = await Promise.all([loadImg(D + man.height.file), loadImg(D + man.classes.file), loadImg(D + man.corridor.file)]);
  const SIZE = man.size_m, HALF = SIZE / 2;

  /* ---------- heights (12-bit packed: R high 8, G high nibble = low 4; same as the rover prototype) ---------- */
  const hp = pixels(hImg), HN = man.height.n, H = new Float32Array(HN * HN);
  for (let i = 0; i < HN * HN; i++) H[i] = man.height.emin + (((hp[i * 4] << 4) | (hp[i * 4 + 1] >> 4)) / 4095) * (man.height.emax - man.height.emin) - man.height.datum;
  const heightAt = (x, z) => {
    const fx = Math.min(HN - 1.001, Math.max(0, (x / SIZE + 0.5) * HN - 0.5)), fz = Math.min(HN - 1.001, Math.max(0, (z / SIZE + 0.5) * HN - 0.5));
    const x0 = fx | 0, z0 = fz | 0, tx = fx - x0, tz = fz - z0, a = H[z0 * HN + x0], b = H[z0 * HN + x0 + 1], c = H[(z0 + 1) * HN + x0], d = H[(z0 + 1) * HN + x0 + 1];
    return (a * (1 - tx) + b * tx) * (1 - tz) + (c * (1 - tx) + d * tx) * tz;
  };
  const corPx = pixels(corImg), corW = corImg.width;

  /* ---------- renderer + adaptive render scale with a sharpening upscale pass (FSR1-style, cheap) ---------- */
  const canvas = $('view');
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance' });
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
  const camera = new THREE.PerspectiveCamera(60, 1, 0.05, 3000);
  const resize = () => {
    const w = window.innerWidth, h = window.innerHeight;
    renderer.setSize(w, h, false); canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
    camera.aspect = w / h; camera.updateProjectionMatrix();
    const rw = Math.max(64, Math.round(w * renderScale)), rh = Math.max(64, Math.round(h * renderScale));
    rt.setSize(rw, rh); post.uniforms.texel.value.set(1 / rw, 1 / rh);
  };
  window.addEventListener('resize', resize); resize();

  /* ---------- terrain ---------- */
  const tl = new THREE.TextureLoader();
  const tex = (f, nearest) => { const t = tl.load(D + f); t.flipY = false; t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping; if (nearest) { t.minFilter = t.magFilter = THREE.NearestFilter; } else { t.minFilter = THREE.LinearMipmapLinearFilter; t.anisotropy = Math.min(+(new URLSearchParams(location.search).get('aniso') || 1), renderer.capabilities.getMaxAnisotropy()); } return t; };
  const baseTex = tex(man.base.file), clsTex = tex(man.classes.file), conTex = tex(man.contrast.file), corTex = tex(man.corridor.file, true);
  const atlasTex = tex(man.atlas.file); atlasTex.minFilter = THREE.LinearFilter; atlasTex.generateMipmaps = false;
  const MAXS = Math.max(1, man.corridor.stops.length);
  const stops = new Array(MAXS).fill(0).map(() => new THREE.Vector4(1e6, 1e6, 0, 0));
  man.corridor.stops.slice(0, MAXS).forEach((s, i) => stops[i].set(s.x, s.z, s.ratio, s.tile));
  // corridor lookup without shader loops: 1 m index map (which rover stop covers this spot) + per-stop parameter texture
  const NS = man.corridor.stops.length, IM = Math.ceil(SIZE), idxData = new Uint8Array(IM * IM * 4);
  for (let gz = 0; gz < IM; gz++) for (let gx = 0; gx < IM; gx++) {
    const x = gx + 0.5 - SIZE / 2, z = gz + 0.5 - SIZE / 2; let best = 0, bd = 1e9;
    man.corridor.stops.forEach((st, i) => { const dx = x - st.x, dz = z - st.z, d = dx * dx + dz * dz;
      if (Math.abs(dx) < man.corridor.half_m && Math.abs(dz) < man.corridor.half_m && d < bd) { bd = d; best = i + 1; } });
    idxData[(gz * IM + gx) * 4] = best;
  }
  const corIdx = new THREE.DataTexture(idxData, IM, IM, THREE.RGBAFormat); corIdx.minFilter = corIdx.magFilter = THREE.NearestFilter; corIdx.needsUpdate = true;
  const spData = new Uint8Array(Math.max(1, NS) * 2 * 4);
  man.corridor.stops.forEach((st, i) => {
    const ex = Math.round((st.x / SIZE + 0.5) * 65535), ez = Math.round((st.z / SIZE + 0.5) * 65535);
    spData.set([ex >> 8, ex & 255, ez >> 8, ez & 255], i * 4);
    spData.set([Math.round(Math.min(1, st.ratio / 4) * 255), st.tile, 0, 255], (NS + i) * 4);
  });
  const stopTex = new THREE.DataTexture(spData, Math.max(1, NS), 2, THREE.RGBAFormat); stopTex.minFilter = stopTex.magFilter = THREE.NearestFilter; stopTex.needsUpdate = true;
  const hc = man.corridor.half_m, xs = man.corridor.stops.map((s) => s.x), zs = man.corridor.stops.map((s) => s.z);
  const corBox = new THREE.Vector4(Math.min(...xs) - hc, Math.min(...zs) - hc, Math.max(...xs) + hc, Math.max(...zs) + hc);
  const URLP = new URLSearchParams(location.search);
  const seg = Math.min(+(URLP.get('seg') || 96), man.height.n - 1);       // far terrain: coarse; near patch below carries full DTM detail
  const geo = new THREE.PlaneGeometry(SIZE, SIZE, seg, seg); geo.rotateX(-Math.PI / 2);
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
      stops: { value: stops }, nStops: { value: new URLSearchParams(location.search).has('nocor') ? 0 : Math.min(MAXS, man.corridor.stops.length) }, stopHalf: { value: man.corridor.half_m },
      corBox: { value: corBox }, corIdx: { value: corIdx }, stopTex: { value: stopTex }, nStopsF: { value: Math.max(1, NS) },
      corCols: { value: man.corridor.cols }, corTile: { value: man.corridor.tile_px / corW },
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
        if (w.x > cutBox.x && w.x < cutBox.z && w.y > cutBox.y && w.y < cutBox.w) discard;   // far mesh yields to the near patch
        float g = texture2D(baseTex, uv).r;
        float mpp = max(fwidth(w.x), fwidth(w.y));          // metres per screen pixel
        float detail = 0.0, real = 0.0, fade = 0.0;
        #if QUALITY > 0
        fade = 1.0 - smoothstep(0.05, 0.22, mpp);           // fine detail only where it can be seen
        vec2 bt = (w - bakeOrigin) / bakeSize;              // detail is baked around the rover (one tap per pixel)
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
  const terrain = new THREE.Mesh(geo, terrainMat); scene.add(terrain);

  // ---- detail bake: the rover-texture detail field is rendered once per 8 m of driving into a 64 m texture
  //      (6.25 cm texels), in 8 strips over 8 frames, so per-frame cost is one texture read per pixel.
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
      float bomb(float cls, vec2 w){            // two half-offset layers, Hann-weighted, variance preserving
        #if QUALITY < 2
        return tileSample(cls, w);                      // tier 1: single layer (half the taps)
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
          if (cls.x > 0.03) detail += cls.x * ampRatio.x * bomb(0.0, w);   // skip near-zero classes (fewer texture taps)
          if (cls.y > 0.03) detail += cls.y * ampRatio.y * bomb(1.0, w);
          if (cls.z > 0.03) detail += cls.z * ampRatio.z * bomb(2.0, w);
          if (cls.w > 0.03) detail += cls.w * ampRatio.w * bomb(3.0, w);
          vec4 a2 = ampRatio * ampRatio;                  // variance-preserving fuzzy blend (matches pipeline/detail.py)
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
  const bakeScene = new THREE.Scene(); bakeScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), bakeMat));
  let bakeFront = 0, bakeStrip = -1, bakePending = null;
  const requestBake = (ox, oz) => { bakePending = [ox, oz]; bakeStrip = 0; };
  const bakeStep = () => {
    if (bakeStrip < 0 || terrainMat.defines.QUALITY === 0) return;
    const back = bakeRT[1 - bakeFront], h = BAKE_PX / STRIPS;
    bakeUniforms.bakeOrigin.value.set(bakePending[0], bakePending[1]);
    back.scissor.set(0, bakeStrip * h, BAKE_PX, h); back.scissorTest = true;
    renderer.setRenderTarget(back); renderer.render(bakeScene, postCam); renderer.setRenderTarget(null);
    if (++bakeStrip >= STRIPS) {                        // swap: the new field becomes visible all at once
      bakeFront = 1 - bakeFront; bakeStrip = -1;
      terrainMat.uniforms.bakeTex.value = bakeRT[bakeFront].texture;
      terrainMat.uniforms.bakeOrigin.value.set(bakePending[0], bakePending[1]);
    }
  };

  // near patch: full 1 m DTM detail around the rover, re-centred as it drives (cheap on phones)
  const NEAR = 96, NSEG = 96, nearUniforms = Object.assign({}, terrainMat.uniforms, { cutBox: { value: new THREE.Vector4(1e6, 1e6, -1e6, -1e6) } });
  const nearMat = new THREE.ShaderMaterial({ extensions: { derivatives: true }, defines: { QUALITY: 2 }, uniforms: nearUniforms, vertexShader: terrainMat.vertexShader, fragmentShader: terrainMat.fragmentShader });
  const ngeo = new THREE.PlaneGeometry(NEAR, NEAR, NSEG, NSEG); ngeo.rotateX(-Math.PI / 2);
  const nbase = ngeo.attributes.position.array.slice();
  const near = new THREE.Mesh(ngeo, nearMat); near.frustumCulled = false; scene.add(near);
  let nearC = [1e9, 1e9];
  const updateNear = (x, z) => {
    const cx = Math.round(x / 8) * 8, cz = Math.round(z / 8) * 8;
    if (cx === nearC[0] && cz === nearC[1]) return; nearC = [cx, cz];
    const a = ngeo.attributes.position;
    for (let i = 0; i < a.count; i++) { const px = nbase[i * 3] + cx, pz = nbase[i * 3 + 2] + cz; a.setXYZ(i, px, heightAt(px, pz), pz); }
    a.needsUpdate = true; ngeo.computeVertexNormals(); ngeo.computeBoundingSphere();
    const m = NEAR / 2 - 2; terrainMat.uniforms.cutBox.value.set(cx - m, cz - m, cx + m, cz + m);
    requestBake(cx - BAKE_M / 2, cz - BAKE_M / 2);
  };
  const sky = () => { const t = +$('tau').value; const c = new THREE.Color().setRGB(0.72 - 0.05 * t, 0.56 - 0.04 * t, 0.42 - 0.03 * t); scene.background = c; terrainMat.uniforms.tau.value = t; };
  $('tau').addEventListener('input', sky); sky();

  /* ---------- rover (6 wheels, rocker-style ground clamp) ---------- */
  const rover = new THREE.Group(); scene.add(rover);
  const mat = (c) => new THREE.MeshLambertMaterial({ color: c });
  scene.add(new THREE.HemisphereLight(0xffe6cc, 0x553322, 0.9)); const dl = new THREE.DirectionalLight(0xffffff, 0.7); dl.position.copy(sun).multiplyScalar(50); scene.add(dl);
  const body = new THREE.Mesh(new THREE.BoxGeometry(2.0, 0.6, 2.8), mat(0xd8d2c8)); body.position.y = 1.1; rover.add(body);
  const mast = new THREE.Mesh(new THREE.BoxGeometry(0.12, 1.1, 0.12), mat(0xbdb7ad)); mast.position.set(0.55, 1.95, -1.0); rover.add(mast);
  const head = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.25, 0.25), mat(0x8a8178)); head.position.set(0.55, 2.55, -1.05); rover.add(head);
  const WHEELS = [[-1.25, -1.1], [1.25, -1.1], [-1.3, 0.05], [1.3, 0.05], [-1.25, 1.15], [1.25, 1.15]];
  const wheels = WHEELS.map(([x, z]) => { const w = new THREE.Mesh(new THREE.CylinderGeometry(0.26, 0.26, 0.4, 14), mat(0x3a332e)); w.rotation.z = Math.PI / 2; w.position.set(x, 0.26, z); rover.add(w); return w; });
  const R = { x: man.start.x, z: man.start.z, yaw: man.start.yaw || 0, v: 0, cam: 'chase' };

  /* ---------- tag markers ---------- */
  const tagGroup = new THREE.Group(); scene.add(tagGroup);
  const drawTags = () => { tagGroup.clear(); for (const t of Ledger.list()) { const m = new THREE.Mesh(new THREE.ConeGeometry(0.25, 0.8, 8), mat(t.provenance.startsWith('measured (rover') ? 0x4fd27a : 0xe0b040)); m.position.set(t.x, t.y + 0.5, t.z); tagGroup.add(m); } };
  drawTags();

  /* ---------- provenance query (same rules as the shader) ---------- */
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

  /* ---------- input ---------- */
  const keys = {};
  addEventListener('keydown', (e) => { keys[e.key.toLowerCase()] = true; if (e.key === 'c') toggleCam(); if (e.key === 'p') toggleProv(); if (e.key === 't') tag(); });
  addEventListener('keyup', (e) => { keys[e.key.toLowerCase()] = false; });
  const pad = { f: 'w', b: 's', l: 'a', r: 'd' };
  document.querySelectorAll('#pad button').forEach((b) => {
    const k = pad[b.dataset.k]; const on = (e) => { e.preventDefault(); keys[k] = true; }, off = (e) => { e.preventDefault(); keys[k] = false; };
    b.addEventListener('touchstart', on, { passive: false }); b.addEventListener('touchend', off); b.addEventListener('mousedown', on); b.addEventListener('mouseup', off); b.addEventListener('mouseleave', off);
  });
  const toggleCam = () => { R.cam = R.cam === 'chase' ? 'cockpit' : 'chase'; $('bCam').textContent = R.cam === 'chase' ? 'Chase' : 'Cockpit'; rover.visible = R.cam === 'chase'; };
  const toggleProv = () => { terrainMat.uniforms.showProv.value = 1 - terrainMat.uniforms.showProv.value; $('bProv').classList.toggle('on'); };
  $('bCam').onclick = toggleCam; $('bProv').onclick = toggleProv; $('bTag').onclick = () => tag();
  $('bScores').onclick = () => { const s = $('scores'); s.style.display = s.style.display === 'block' ? 'none' : 'block'; };

  /* gyro look-around (iOS 13+ needs an explicit permission handshake) */
  let gyro = null;
  $('bGyro').onclick = async () => {
    if (gyro) { removeEventListener('deviceorientation', onOri); gyro = null; $('bGyro').classList.remove('on'); return; }
    try { if (typeof DeviceOrientationEvent !== 'undefined' && typeof DeviceOrientationEvent.requestPermission === 'function') { if ((await DeviceOrientationEvent.requestPermission()) !== 'granted') { toast('Gyro permission denied'); return; } } }
    catch (e) { toast('Gyro unavailable'); return; }
    gyro = { a: 0, b: 90, g: 0 }; addEventListener('deviceorientation', onOri); $('bGyro').classList.add('on'); if (R.cam !== 'cockpit') toggleCam(); toast('Gyro on: turn your phone to look around');
  };
  function onOri(e) { if (e.alpha == null) return; gyro = { a: e.alpha, b: e.beta, g: e.gamma }; }
  const zee = new THREE.Vector3(0, 0, 1), q1 = new THREE.Quaternion(-Math.sqrt(0.5), 0, 0, Math.sqrt(0.5)), eul = new THREE.Euler(), qd = new THREE.Quaternion(), q0 = new THREE.Quaternion();
  const deviceQuat = () => { const d = THREE.MathUtils.degToRad; eul.set(d(gyro.b), d(gyro.a), -d(gyro.g), 'YXZ'); qd.setFromEuler(eul); qd.multiply(q1); qd.multiply(q0.setFromAxisAngle(zee, -d(window.orientation || 0))); return qd; };

  /* ---------- tagging: raycast from the reticle ---------- */
  const ray = new THREE.Raycaster();
  const reticleHit = () => { ray.setFromCamera(new THREE.Vector2(0, 0), camera); const h = ray.intersectObjects([near, terrain]); return h.length ? h[0] : null; };
  const mppAt = (dist) => 2 * dist * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) / (window.innerHeight * renderScale);
  function tag() {
    const h = reticleHit(); if (!h) { toast('Aim the reticle at the ground'); return; }
    const prov = provenanceAt(h.point, mppAt(h.distance));
    if (prov.startsWith('simulated')) { toast('Blocked: that detail is simulated. Step back so the reticle sits on measured terrain, or tag on the rover corridor.'); return; }
    const label = $('tagLabel').value;
    const ll = toLatLon(h.point.x, h.point.z);
    Ledger.add({ id: Date.now().toString(36), x: h.point.x, y: h.point.y, z: h.point.z, lat: ll[0], lon: ll[1], label, provenance: prov, timestamp: new Date().toISOString(), votes: 0, validation: scores ? scores.summary_line : null });
    drawTags(); toast('Tagged "' + label + '" — ' + prov);
  }
  const toLatLon = (x, z) => { const Rm = 3394839.8133163, k = 180 / Math.PI; return [man.center.lat - (z / Rm) * k, man.center.lon + (x / (Rm * Math.cos(15 * Math.PI / 180))) * k]; };

  /* ---------- scores panel ---------- */
  if (scores) {
    const rows = Object.entries(scores.detail_layer).map(([k, v]) => `<tr><td>${k}</td><td>${v.spectral_match_pct.toFixed(0)}%</td><td>${v.texture_match_pct.toFixed(0)}%</td><td>${v.consistency_pct.toFixed(0)}%</td></tr>`).join('');
    $('scores').innerHTML = `<b>Validated against ${scores.n_held_out} held-out rover stops</b><br>Measured layer: ${scores.measured_layer.noise_reduction_vs_single_frame_pct.toFixed(0)}% less error than one raw HiRISE frame<br>
      <table><tr class="n"><td>method</td><td>spectral</td><td>texture</td><td>consistent</td></tr>${rows}</table><span class="n">100% = matches rover photos / measurement exactly</span>`;
  }

  /* ---------- simulation loop ---------- */
  const qp = new URLSearchParams(location.search).get('quality');
  const MAXQ = qp != null ? +qp : 2, MINQ = qp != null ? +qp : 0;
  const setTier = (t) => { terrainMat.uniforms.quality.value = t; for (const m of [terrainMat, nearMat, bakeMat]) { m.defines.QUALITY = t; m.needsUpdate = true; } if (t > 0 && nearC[0] < 1e8) requestBake(nearC[0] - BAKE_M / 2, nearC[1] - BAKE_M / 2); };
  setTier(MAXQ);
  const DBG = { nopost: URLP.has('nopost'), noray: URLP.has('noray'), noterrain: URLP.has('noterrain') };
  if (DBG.noterrain) { terrain.visible = false; near.visible = false; }
  const clock = new THREE.Clock(); let fpsAcc = 0, fpsN = 0, fps = 0, provTimer = 0;
  const camTarget = new THREE.Vector3(), camPosV = new THREE.Vector3();
  function step(dt) {
    updateNear(R.x, R.z);
    const acc = (keys['w'] || keys['arrowup'] ? 1 : 0) - (keys['s'] || keys['arrowdown'] ? 1 : 0);
    const turn = (keys['a'] || keys['arrowleft'] ? 1 : 0) - (keys['d'] || keys['arrowright'] ? 1 : 0);
    R.v += (acc * 3.0 - R.v * 1.8) * dt; R.yaw += turn * 0.9 * dt * (Math.abs(R.v) > 0.05 ? 1 : 0.6);
    const fx = -Math.sin(R.yaw), fz = -Math.cos(R.yaw);
    const nx = R.x + fx * R.v * dt, nz = R.z + fz * R.v * dt;
    const edge = HALF - 12;
    if (Math.abs(nx) > edge || Math.abs(nz) > edge) { R.v = 0; toast('Survey boundary: beyond here only low-resolution data exists.'); }
    else {
      const ahead = Math.sign(R.v) * 1.5, h0 = heightAt(R.x, R.z), h1 = heightAt(R.x + fx * ahead, R.z + fz * ahead);
      const slope = Math.atan(Math.abs(h1 - h0) / 1.5) * 180 / Math.PI;
      if (slope > 25 && Math.abs(R.v) > 0.05) { R.v = 0; toast(`Too steep: real DTM slope ${slope.toFixed(0)}° (limit 25°)`); }
      else { R.x = nx; R.z = nz; }
    }
    const c = Math.cos(R.yaw), s = Math.sin(R.yaw);
    const wh = WHEELS.map(([x, z]) => heightAt(R.x + x * c + z * s, R.z - x * s + z * c));
    const front = (wh[0] + wh[1]) / 2, back = (wh[4] + wh[5]) / 2, left = (wh[0] + wh[2] + wh[4]) / 3, right = (wh[1] + wh[3] + wh[5]) / 3;
    rover.position.set(R.x, wh.reduce((a, b) => a + b, 0) / 6, R.z);
    rover.rotation.set(Math.atan2(front - back, 2.25), R.yaw, Math.atan2(right - left, 2.55), 'YXZ');
    wheels.forEach((w, i) => { w.rotation.x += R.v * dt / 0.26; w.position.y = 0.26 + (wh[i] - rover.position.y) * 0.6; });
    if (R.cam === 'chase') {
      camTarget.set(R.x, rover.position.y + 1.2, R.z);
      camPosV.set(R.x - fx * 9, rover.position.y + 4.5, R.z - fz * 9);
      camera.position.lerp(camPosV, 1 - Math.exp(-dt * 4)); camera.lookAt(camTarget);
    } else {
      camera.position.set(R.x + 0.55 * c - 1.0 * s * 0, rover.position.y + 1.8 + 0.6, R.z);   // mast height 1.8 m above deck pivot
      const qh = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), R.yaw);
      if (gyro) camera.quaternion.copy(qh).multiply(deviceQuat());
      else camera.quaternion.copy(qh).multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -0.28));
    }
    terrainMat.uniforms.camPos.value.copy(camera.position);
  }
  function frame() {
    const dt = Math.min(0.05, clock.getDelta());
    step(dt);
    bakeStep();
    if (DBG.nopost) { renderer.setRenderTarget(null); renderer.render(scene, camera); }
    else { renderer.setRenderTarget(rt); renderer.render(scene, camera); renderer.setRenderTarget(null); renderer.render(postScene, postCam); }
    fpsAcc += dt; fpsN++;
    if (fpsAcc > 1) {
      fps = fpsN / fpsAcc; fpsAcc = 0; fpsN = 0;
      // adaptive quality: first trade resolution, then detail tier (each tier is a separately compiled shader)
      const q = terrainMat.uniforms.quality; let ns = renderScale;
      if (fps < 27) {
        if (renderScale > 0.55) ns = Math.max(0.5, renderScale - 0.1);
        else if (q.value > MINQ) { setTier(q.value - 1); ns = 0.75; }
      } else if (fps > 50) {
        if (renderScale < 0.95) ns = Math.min(1, renderScale + 0.05);
        else if (q.value < MAXQ) { setTier(q.value + 1); ns = 0.75; }
      }
      if (ns !== renderScale) { renderScale = ns; resize(); }
      window.__perf = { fps, renderScale, quality: q.value };
    }
    provTimer += dt;
    if (provTimer > 0.25 && !DBG.noray) {
      provTimer = 0;
      const h = reticleHit(); const ll = toLatLon(R.x, R.z);
      const p = h ? provenanceAt(h.point, mppAt(h.distance)) : '—';
      const cls = p.startsWith('measured (rover') ? 'm' : p.startsWith('simulated') ? 's' : 'n';
      $('provtxt').innerHTML = `under reticle:<br><span class="${cls}">${p}</span>`;
      $('hudtxt').innerHTML = `${ll[0].toFixed(5)}°N ${ll[1].toFixed(5)}°E · ${Math.abs(R.v).toFixed(1)} m/s<br>${fps.toFixed(0)} fps · render ${Math.round(renderScale * 100)}% · detail tier ${terrainMat.uniforms.quality.value} · ${R.cam}<br><span class="n">WASD/▲▼◀▶ drive · C camera · P provenance · T tag</span>`;
    }
    requestAnimationFrame(frame);
  }
  camera.position.set(R.x, heightAt(R.x, R.z) + 6, R.z + 10);
  $('loading').style.display = 'none';
  window.__ready = true;
  frame();
}
main().catch((e) => { $('loading').textContent = 'Failed to load data: ' + e.message; console.error(e); });
})();
