/* Mars Rover & Ingenuity Kinematics, NASA 3D Models & Suspension Engine
 * Integrates into Jezero Explorer LOD terrain quadtree streaming.
 */
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { MarsAudio } from "./audio.js";

const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const smoothstep = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };

export const ROVER_SPECS = {
  perseverance: {
    id: 'perseverance',
    name: 'Perseverance (Mars 2020)',
    site: 'Jezero Crater (Octavia E. Butler)',
    mass: '1,025 kg',
    modelFile: './assets/perseverance.glb',
    targetLength: 3.0,
    isHelicopter: false
  },
  ingenuity: {
    id: 'ingenuity',
    name: 'Ingenuity (Mars Helicopter)',
    site: 'Wright Brothers Field, Jezero',
    mass: '1.8 kg',
    modelFile: './assets/ingenuity.glb',
    targetLength: 1.3,
    isHelicopter: true
  },
  curiosity: {
    id: 'curiosity',
    name: 'Curiosity (MSL)',
    site: 'Gale Crater (Mount Sharp)',
    mass: '899 kg',
    modelFile: './assets/curiosity.glb',
    targetLength: 3.0,
    isHelicopter: false
  },
  opportunity: {
    id: 'opportunity',
    name: 'Opportunity (MER-B)',
    site: 'Meridiani Planum',
    mass: '185 kg',
    modelFile: './assets/mer_opportunity.glb',
    targetLength: 1.6,
    isHelicopter: false
  }
};

const CANON_WHEELS = ['FL', 'FR', 'ML', 'MR', 'RL', 'RR'];

function finalizeWheels(root, wheels) {
  root.updateMatrixWorld(true);
  const tmp = new THREE.Vector3();
  wheels.forEach((w) => {
    w.pivot.getWorldPosition(tmp);
    root.worldToLocal(tmp);
    w.restPos = tmp.clone();
    w.baseY = w.pivot.position.y;
    w.steer = 0;
    w.spin = 0;
    w.dy = 0;
  });
  return CANON_WHEELS.map((n) => wheels.find((w) => w.name === n)).filter(Boolean);
}

/**
 * Splits NASA's combined "Wheels_objs" mesh into 6 individually steerable,
 * spinning, and suspension-compliant wheel assemblies.
 */
function splitWheelsMesh(wheelsNode) {
  const meshes = [];
  wheelsNode.traverse((o) => { if (o.isMesh) meshes.push(o); });
  if (!meshes.length) return null;

  let seeds = [];
  [-1, 1].forEach((sx) => [-1.13, -0.055, 1.13].forEach((z) => {
    seeds.push({ x: sx * (Math.abs(z) > 0.5 ? 1.06 : 1.18), z });
  }));

  const tris = meshes.map((m) => {
    const pos = m.geometry.attributes.position, idx = m.geometry.index;
    const n = idx ? idx.count / 3 : pos.count / 3;
    const cx = new Float32Array(n), cz = new Float32Array(n);
    for (let t = 0; t < n; t++) {
      let sx = 0, sz = 0;
      for (let k = 0; k < 3; k++) {
        const vi = idx ? idx.getX(t * 3 + k) : t * 3 + k;
        sx += pos.getX(vi); sz += pos.getZ(vi);
      }
      cx[t] = sx / 3; cz[t] = sz / 3;
    }
    return { n, cx, cz, owner: new Uint8Array(n) };
  });

  for (let iter = 0; iter < 3; iter++) {
    const acc = seeds.map(() => ({ x: 0, z: 0, n: 0 }));
    tris.forEach((T) => {
      for (let t = 0; t < T.n; t++) {
        let best = 0, bd = 1e9;
        for (let s = 0; s < seeds.length; s++) {
          const dx = T.cx[t] - seeds[s].x, dz = T.cz[t] - seeds[s].z, d = dx * dx + dz * dz;
          if (d < bd) { bd = d; best = s; }
        }
        T.owner[t] = best;
        acc[best].x += T.cx[t]; acc[best].z += T.cz[t]; acc[best].n++;
      }
    });
    if (iter < 2) seeds = seeds.map((s, i) => acc[i].n ? { x: acc[i].x / acc[i].n, z: acc[i].z / acc[i].n } : s);
  }

  const bbs = seeds.map(() => new THREE.Box3());
  meshes.forEach((m, mi) => {
    const pos = m.geometry.attributes.position, idx = m.geometry.index, T = tris[mi];
    const v = new THREE.Vector3();
    for (let t = 0; t < T.n; t++) for (let k = 0; k < 3; k++) {
      const vi = idx ? idx.getX(t * 3 + k) : t * 3 + k;
      bbs[T.owner[t]].expandByPoint(v.fromBufferAttribute(pos, vi));
    }
  });
  const centers = bbs.map((b) => b.getCenter(new THREE.Vector3()));

  const out = seeds.map((s, i) => {
    const pivot = new THREE.Group();
    const hub = new THREE.Group();
    pivot.add(hub);
    pivot.position.copy(centers[i]);
    return { pivot, hub, center: centers[i], radius: (bbs[i].max.y - bbs[i].min.y) / 2, widthX: bbs[i].max.x - bbs[i].min.x };
  });

  meshes.forEach((m, mi) => {
    const g = m.geometry, T = tris[mi], idx = g.index;
    const names = Object.keys(g.attributes);
    for (let w = 0; w < seeds.length; w++) {
      const remap = new Map(), order = [], newIdx = [];
      for (let t = 0; t < T.n; t++) {
        if (T.owner[t] !== w) continue;
        for (let k = 0; k < 3; k++) {
          const vi = idx ? idx.getX(t * 3 + k) : t * 3 + k;
          let ni = remap.get(vi);
          if (ni === undefined) { ni = order.length; remap.set(vi, ni); order.push(vi); }
          newIdx.push(ni);
        }
      }
      if (!order.length) continue;
      const ng = new THREE.BufferGeometry();
      names.forEach((name) => {
        const src = g.attributes[name], sz = src.itemSize, arr = new src.array.constructor(order.length * sz);
        for (let i = 0; i < order.length; i++) for (let c = 0; c < sz; c++) arr[i * sz + c] = src.array[order[i] * sz + c];
        const att = new THREE.BufferAttribute(arr, sz, src.normalized);
        if (name === 'position') {
          for (let i = 0; i < order.length; i++) {
            arr[i * 3] -= centers[w].x; arr[i * 3 + 1] -= centers[w].y; arr[i * 3 + 2] -= centers[w].z;
          }
        }
        ng.setAttribute(name, att);
      });
      ng.setIndex(order.length > 65535 ? new THREE.Uint32BufferAttribute(newIdx, 1) : new THREE.Uint16BufferAttribute(newIdx, 1));
      ng.computeBoundingBox(); ng.computeBoundingSphere();
      const nm = new THREE.Mesh(ng, m.material);
      nm.castShadow = true; nm.receiveShadow = true;
      out[w].hub.add(nm);
    }
  });

  meshes.forEach((m) => { m.parent.remove(m); m.geometry.dispose(); });
  out.forEach((o) => wheelsNode.add(o.pivot));
  return out;
}

export class RoverKinematics {
  constructor(scene, terrainRef) {
    this.scene = scene;
    this.terrain = terrainRef;
    this.g = new THREE.Group();
    this.scene.add(this.g);

    this.currentSpec = ROVER_SPECS.perseverance;
    this.pos = new THREE.Vector3();
    this.heading = 0; // rad, 0 = north (-Z)
    this.speed = 0;
    this.odo = 0;
    this.keys = {};

    this.pitch = 0;
    this.roll = 0;
    this.steerAngle = 0;
    this.wheelSpin = 0;

    // Robotic arm state
    this.armMode = 'STOWED'; // 'STOWED', 'DEPLOYING', 'INSPECTING', 'DRILLING', 'RETRACTING'
    this.armProgress = 0;
    this.drillSpin = 0;

    // Helicopter flight dynamics
    this.flightState = 'LANDED';
    this.altitude = 0;
    this.rotorRPM = 0;
    this.rotorAngle = 0;
    this.heliPitch = 0;
    this.heliRoll = 0;
    this.targetAlt = 0;

    this.headlights = [];
    this.headlightsOn = false;

    this.activeRig = null;
    this.buildProceduralFallback();
  }

  buildProceduralFallback() {
    const m = c => new THREE.MeshStandardMaterial({ color: c, roughness: 0.6, metalness: 0.3 });
    const fallbackGroup = new THREE.Group();
    const body = new THREE.Mesh(new THREE.BoxGeometry(1.6, 0.65, 2.5), m(0xd6d0c4));
    body.position.y = 1.05;
    fallbackGroup.add(body);

    const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 1.2), m(0x8a847c));
    mast.position.set(0.55, 1.9, -0.95);
    fallbackGroup.add(mast);

    const head = new THREE.Mesh(new THREE.BoxGeometry(0.48, 0.24, 0.26), m(0x111122));
    head.position.set(0.55, 2.55, -0.95);
    fallbackGroup.add(head);

    this.g.add(fallbackGroup);
    this.placeholder = fallbackGroup;
  }

  loadCraft(craftId, onDone) {
    const spec = ROVER_SPECS[craftId] || ROVER_SPECS.perseverance;
    this.currentSpec = spec;
    this.speed = 0;
    this.steerAngle = 0;
    this.altitude = 0;
    this.flightState = 'LANDED';

    if (this.activeRig && this.activeRig.root) {
      this.g.remove(this.activeRig.root);
    }
    if (this.placeholder) {
      this.placeholder.visible = true;
    }

    new GLTFLoader().load(spec.modelFile, (gltf) => {
      if (this.placeholder) this.placeholder.visible = false;
      const root = new THREE.Group();
      root.name = 'Craft_' + spec.id;
      const model = gltf.scene;

      if (spec.isHelicopter) {
        model.rotation.z = Math.PI / 2;
      }

      model.traverse((o) => {
        if (o.isMesh) {
          o.castShadow = true;
          o.receiveShadow = true;
          if (o.material) {
            o.material.side = THREE.FrontSide;
            o.material.roughness = Math.max(0.3, o.material.roughness || 0.5);
            o.material.metalness = Math.min(0.85, o.material.metalness || 0.2);
          }
        }
      });

      const rig = { root, spec, glbScene: model, isHelicopter: spec.isHelicopter };

      if (!spec.isHelicopter && spec.id === 'perseverance') {
        // Native meters with wheel decomposition
        model.updateMatrixWorld(true);
        const wheelsNode = model.getObjectByName('Wheels_objs');
        const parts = wheelsNode ? splitWheelsMesh(wheelsNode) : null;

        if (parts) {
          model.updateMatrixWorld(true);
          const tmp = new THREE.Vector3();
          const modelPos = parts.map((p) => p.pivot.getWorldPosition(new THREE.Vector3()));
          const mx = modelPos.reduce((a, v) => a + v.x, 0) / 6;
          const mz = modelPos.reduce((a, v) => a + v.z, 0) / 6;
          const axleY = modelPos.reduce((a, v) => a + v.y, 0) / 6;
          const radius = parts.reduce((a, p) => a + p.radius, 0) / 6;

          // Flip 180 so front faces -Z (sim forward)
          model.rotation.y = Math.PI;
          model.position.set(mx, radius - axleY, mz);
          root.add(model);
          root.updateMatrixWorld(true);

          const wheels = parts.map((p) => {
            root.worldToLocal(p.pivot.getWorldPosition(tmp));
            const row = tmp.z < -0.5 ? 'F' : tmp.z > 0.5 ? 'R' : 'M';
            const side = tmp.x < 0 ? 'L' : 'R';
            return { name: row + side, pivot: p.pivot, hub: p.hub, steerable: row !== 'M' };
          });
          rig.wheels = finalizeWheels(root, wheels);
          rig.wheelRadius = radius;
          rig.spinSign = 1;
        } else {
          root.add(model);
        }
      } else {
        // Generic bounding-box scale
        model.updateMatrixWorld(true);
        const bbox = new THREE.Box3().setFromObject(model);
        const bSize = new THREE.Vector3();
        bbox.getSize(bSize);
        const maxDim = Math.max(bSize.x, bSize.z, bSize.y);
        const s = (spec.targetLength || 3.0) / Math.max(0.1, maxDim);
        model.scale.setScalar(s);
        model.rotation.y = Math.PI;

        const b2 = new THREE.Box3().setFromObject(model);
        const c = b2.getCenter(new THREE.Vector3());
        model.position.set(-c.x, -b2.min.y, -c.z);
        root.add(model);
      }

      // Hook animated nodes
      rig.rotor1 = model.getObjectByName('rotors_01');
      rig.rotor2 = model.getObjectByName('rotors_02');
      rig.armNode = model.getObjectByName('arm');
      rig.mastHead = model.getObjectByName('head');
      rig.drillBit = model.getObjectByName('corring_drill');

      // Headlights
      const hlL = new THREE.SpotLight(0xfffae0, 0, 45, Math.PI / 5, 0.4, 1.5);
      hlL.position.set(-0.55, 0.95, -1.5);
      hlL.target.position.set(-0.55, 0.2, -12);
      root.add(hlL); root.add(hlL.target);

      const hlR = new THREE.SpotLight(0xfffae0, 0, 45, Math.PI / 5, 0.4, 1.5);
      hlR.position.set(0.55, 0.95, -1.5);
      hlR.target.position.set(0.55, 0.2, -12);
      root.add(hlR); root.add(hlR.target);

      rig.headlights = [hlL, hlR];
      this.setHeadlights(this.headlightsOn, rig);

      this.activeRig = rig;
      this.g.add(root);
      if (onDone) onDone(spec);
    }, undefined, (err) => {
      console.warn('Failed loading GLB model:', err);
    });
  }

  place(e, n, headingDeg = 0) {
    this.pos.set(e, 0, -n);
    this.heading = THREE.MathUtils.degToRad(headingDeg);
    this.speed = 0;
    this.ground(true);
  }

  toggleHeadlights() {
    this.headlightsOn = !this.headlightsOn;
    this.setHeadlights(this.headlightsOn, this.activeRig);
    return this.headlightsOn;
  }

  setHeadlights(on, rig) {
    const r = rig || this.activeRig;
    if (r && r.headlights) {
      r.headlights.forEach((hl) => { hl.intensity = on ? 2.5 : 0.0; });
    }
  }

  ground(force) {
    const y = this.terrain.heightAt(this.pos.x, this.pos.z);
    if (y !== null) this.pos.y = y;
    else if (force) this.pos.y = 0;

    const f = new THREE.Vector3(Math.sin(this.heading), 0, -Math.cos(this.heading));
    const yf = this.terrain.heightAt(this.pos.x + f.x * 1.4, this.pos.z + f.z * 1.4);
    const yb = this.terrain.heightAt(this.pos.x - f.x * 1.4, this.pos.z - f.z * 1.4);
    const pitch = (yf !== null && yb !== null) ? Math.atan2(yf - yb, 2.8) : 0;

    this.g.position.copy(this.pos);
    this.g.rotation.set(0, 0, 0);
    this.g.rotateY(-this.heading);
    this.g.rotateX(pitch);
  }

  update(dt) {
    if (this.currentSpec.isHelicopter) {
      this.updateHelicopter(dt);
    } else {
      this.updateRover(dt);
    }
  }

  updateRover(dt) {
    const k = this.keys;
    const vmax = k.ShiftLeft || k.ShiftRight ? 16 : 4.5;
    const thr = (k.KeyW || k.ArrowUp ? 1 : 0) - (k.KeyS || k.ArrowDown ? 1 : 0);
    const turnInput = (k.KeyA || k.ArrowLeft ? 1 : 0) - (k.KeyD || k.ArrowRight ? 1 : 0);

    // Motor acceleration
    this.speed += (thr * vmax - this.speed) * Math.min(1, dt * (thr ? 2.2 : 3.5));
    if (Math.abs(this.speed) < 0.01 && thr === 0) this.speed = 0;

    // Steering actuator smoothing
    const MAX_STEER = 0.52;
    const targetSteer = turnInput * MAX_STEER;
    this.steerAngle += (targetSteer - this.steerAngle) * Math.min(1, dt * 6.0);

    // Heading change: Ackermann rolling twist + spot-turn near standstill
    const blend = smoothstep(0.05, 0.4, Math.abs(this.speed));
    const yawRate = blend * (this.speed * Math.tan(this.steerAngle) / 1.3) + (1 - blend) * this.steerAngle * 0.9;
    this.heading += yawRate * dt;

    // Displacement
    const step = this.speed * dt;
    this.pos.x += Math.sin(this.heading) * step;
    this.pos.z -= Math.cos(this.heading) * step;
    this.odo += Math.abs(step);

    // Rocker-Bogie 6-Wheel Terrain Probing
    const rig = this.activeRig;
    const wheelPts = [
      { x: -1.2, z: -1.1 }, { x: 1.2, z: -1.1 },
      { x: -1.3, z: 0.05 }, { x: 1.3, z: 0.05 },
      { x: -1.2, z: 1.15 }, { x: 1.2, z: 1.15 }
    ];

    const c = Math.cos(-this.heading), s = Math.sin(-this.heading);
    const wh = wheelPts.map((p) => {
      const wx = this.pos.x + p.x * c + p.z * s;
      const wz = this.pos.z - p.x * s + p.z * c;
      const h = this.terrain.heightAt(wx, wz);
      return h !== null ? h : this.pos.y;
    });

    const leftRockerPitch = Math.atan2(wh[0] - wh[4], 2.25);
    const rightRockerPitch = Math.atan2(wh[1] - wh[5], 2.25);
    const targetPitch = (leftRockerPitch + rightRockerPitch) * 0.5;

    const leftAvg = (wh[0] + wh[2] + wh[4]) / 3;
    const rightAvg = (wh[1] + wh[3] + wh[5]) / 3;
    const targetRoll = Math.atan2(rightAvg - leftAvg, 2.5);

    this.pitch += (targetPitch - this.pitch) * Math.min(1, dt * 5.0);
    this.roll += (targetRoll - this.roll) * Math.min(1, dt * 5.0);

    const groundY = this.terrain.heightAt(this.pos.x, this.pos.z);
    if (groundY !== null) this.pos.y = groundY;

    this.g.position.copy(this.pos);
    this.g.rotation.set(0, 0, 0);
    this.g.rotateY(-this.heading);
    this.g.rotateX(this.pitch);
    this.g.rotateZ(this.roll);

    // Per-wheel independent steering and spin
    if (rig && rig.wheels) {
      const radius = rig.wheelRadius || 0.28;
      rig.wheels.forEach((w) => {
        const a = -w.restPos.x;
        const l = -w.restPos.z;
        const vf = this.speed - yawRate * a;
        const vl = yawRate * l;
        const mag = Math.hypot(vf, vl);

        if (w.steerable && mag > 1e-3) {
          let ang = Math.atan2(vl, vf);
          if (ang > Math.PI / 2) ang -= Math.PI;
          else if (ang < -Math.PI / 2) ang += Math.PI;
          w.steer += clamp(ang - w.steer, -2.5 * dt, 2.5 * dt);
        }
        w.pivot.rotation.y = w.steer;

        const along = w.steerable ? vf * Math.cos(w.steer) + vl * Math.sin(w.steer) : vf;
        w.spin += (along * dt) / radius;
        w.hub.rotation.x = w.spin * (rig.spinSign || 1);
      });
    }

    // Update audio motor hum
    MarsAudio.updateRoverSpeed(this.speed);

    // Animate drill if drilling
    if (this.armMode === 'DRILLING' && rig && rig.drillBit) {
      this.drillSpin += dt * 35;
      rig.drillBit.rotation.y = this.drillSpin;
    }
  }

  updateHelicopter(dt) {
    const k = this.keys;
    const throttle = (k.KeyW || k.ArrowUp ? 1 : 0) - (k.KeyS || k.ArrowDown ? 1 : 0);
    const turn = (k.KeyA || k.ArrowLeft ? 1 : 0) - (k.KeyD || k.ArrowRight ? 1 : 0);
    const climb = (k.Space || k.KeyE ? 1 : 0) - (k.ShiftLeft || k.KeyQ ? 1 : 0);

    const groundY = this.terrain.heightAt(this.pos.x, this.pos.z) ?? this.pos.y;

    if (this.flightState === 'LANDED') {
      if (climb > 0 || throttle !== 0 || turn !== 0) {
        this.flightState = 'SPOOLING';
      }
    }

    if (this.flightState === 'SPOOLING') {
      this.rotorRPM = Math.min(1.0, this.rotorRPM + dt * 0.8);
      if (this.rotorRPM >= 0.85) {
        this.flightState = 'FLYING';
        this.targetAlt = 3.5;
      }
    } else if (this.flightState === 'FLYING') {
      this.rotorRPM = 1.0;
      this.targetAlt = Math.max(1.0, Math.min(18.0, this.targetAlt + climb * 4.0 * dt));
      this.altitude += (this.targetAlt - this.altitude) * Math.min(1, dt * 3.0);

      if (climb < 0 && this.altitude < 1.4) {
        this.flightState = 'LANDING';
      }

      // Horizontal flight & banking
      this.speed += (throttle * 6.5 - this.speed * 1.8) * dt;
      this.heading += turn * 1.6 * dt;

      const targetPitch = (this.speed / 6.5) * 0.35;
      const targetRoll = -turn * 0.32;
      this.heliPitch += (targetPitch - this.heliPitch) * Math.min(1, dt * 5.0);
      this.heliRoll += (targetRoll - this.heliRoll) * Math.min(1, dt * 5.0);

      const step = this.speed * dt;
      this.pos.x += Math.sin(this.heading) * step;
      this.pos.z -= Math.cos(this.heading) * step;
      this.odo += Math.abs(step);
    } else if (this.flightState === 'LANDING') {
      this.altitude = Math.max(0.0, this.altitude - dt * 2.0);
      if (this.altitude <= 0.05) {
        this.altitude = 0.0;
        this.rotorRPM = Math.max(0.0, this.rotorRPM - dt * 0.8);
        if (this.rotorRPM <= 0.05) this.flightState = 'LANDED';
      }
    }

    // Coaxial counter-rotating blades
    this.rotorAngle += this.rotorRPM * dt * 75;
    const rig = this.activeRig;
    if (rig) {
      if (rig.rotor1) rig.rotor1.rotation.y = this.rotorAngle;
      if (rig.rotor2) rig.rotor2.rotation.y = -this.rotorAngle;
    }

    this.pos.y = groundY + this.altitude;
    this.g.position.copy(this.pos);
    this.g.rotation.set(0, 0, 0);
    this.g.rotateY(-this.heading);
    this.g.rotateX(this.heliPitch);
    this.g.rotateZ(this.heliRoll);

    MarsAudio.updateHelicopter(this.rotorRPM, this.altitude > 0.3);
  }

  triggerDrill(onComplete) {
    this.armMode = 'DRILLING';
    MarsAudio.setDrilling(true);
    setTimeout(() => {
      this.armMode = 'STOWED';
      MarsAudio.setDrilling(false);
      MarsAudio.playBeep(1200, 0.3);
      if (onComplete) onComplete();
    }, 3500);
  }
}
