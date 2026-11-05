/* Mars Rover Kinematics, Rocker-Bogie Mechanics & Ingenuity Flight Dynamics */
window.MarsMechanics = (() => {
  'use strict';

  // --- Rover Physics State ---
  const state = {
    // Kinematics
    v: 0,              // forward velocity (m/s)
    yaw: 0,            // heading angle (rad)
    steerAngle: 0,     // current wheel steer angle (rad)
    wheelSpin: 0,      // total accumulated wheel rotation (rad)
    pitch: 0,
    roll: 0,

    // Arm state
    armMode: 'STOWED', // 'STOWED', 'DEPLOYING', 'INSPECTING', 'DRILLING', 'RETRACTING'
    armProgress: 0,    // 0.0 (stowed) to 1.0 (fully extended/drilling)
    drillSpin: 0,

    // Ingenuity flight state
    isHelicopter: false,
    flightState: 'LANDED', // 'LANDED', 'SPOOLING', 'TAKEOFF', 'HOVER', 'FLYING', 'LANDING'
    altitude: 0,       // meters above ground
    rotorRPM: 0,       // normalized 0.0 to 1.0
    rotorAngle: 0,
    heliPitch: 0,
    heliRoll: 0,
    targetAlt: 0
  };

  const WHEEL_RADIUS = 0.28;          // fallback only; rigs report their own rig.wheelRadius
  const MAX_STEER = 0.52;             // ~30 degrees commanded reference steer
  const MAX_SPEED = 2.4;              // m/s
  const MAX_HELI_ALT = 14.0;          // max historical Ingenuity altitude
  const STEER_SLEW = 2.2;             // rad/s steering-actuator slew limit per wheel
  const MAX_WHEEL_TRAVEL = 0.15;      // m, per-wheel vertical compliance clamp (single-mesh suspension can't flex further)
  const SPOT_TURN_RATE = 0.9;         // rad/s of yaw per rad of steer when (almost) stationary

  // Fallback layout for rigs that expose no wheels (Curiosity / MER GLBs): rover-root x,z
  const DEFAULT_WHEELS = [
    [-1.25, -1.1], [1.25, -1.1], [-1.35, 0.1], [1.35, 0.1], [-1.25, 1.15], [1.25, 1.15]
  ];

  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const smoothstep = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
  const _v = new THREE.Vector3();

  /** Wheel layout in rover-root space (forward = -Z, left = -X), cached on the rig. */
  function layoutOf(rig) {
    if (rig._layout) return rig._layout;
    const pts = rig.wheels
      ? rig.wheels.map((w) => ({ x: w.restPos.x, y: w.restPos.y, z: w.restPos.z }))
      : DEFAULT_WHEELS.map(([x, z]) => ({ x, y: WHEEL_RADIUS, z }));
    // front = most negative z, rear = most positive z, per side
    const side = (sign) => {
      const s = pts.map((p, i) => ({ p, i })).filter((o) => Math.sign(o.p.x) === sign);
      s.sort((a, b) => a.p.z - b.p.z);
      return { front: s[0].i, rear: s[s.length - 1].i, all: s.map((o) => o.i) };
    };
    const L = side(-1), R = side(1);
    const meanAbsX = pts.reduce((a, p) => a + Math.abs(p.x), 0) / pts.length;
    const Lf = Math.abs(pts[L.front].z - (pts.find((p, i) => L.all.includes(i) && i !== L.front && i !== L.rear) || { z: 0 }).z) || 1.2;
    rig._layout = {
      pts, L, R,
      wheelbase: pts[L.rear].z - pts[L.front].z,
      track: 2 * meanAbsX,
      Lf,
      radius: rig.wheelRadius || WHEEL_RADIUS
    };
    return rig._layout;
  }

  /**
   * Rover update: driving, true Ackermann/point-turn wheel kinematics, per-wheel spin (w = v/r),
   * rocker-bogie style terrain compliance (chassis pitch = mean of the two rocker pitches).
   * Rover-root convention: forward = -Z, left = -X, yaw>0 turns left.
   */
  function updateRover(dt, keys, rig, heightAt, currentPos, surveyLimit, onBoundaryHit) {
    if (!rig) return;
    const lay = layoutOf(rig);

    // 1. Throttle & Steering Inputs
    const throttle = (keys['w'] || keys['arrowup'] ? 1 : 0) - (keys['s'] || keys['arrowdown'] ? 1 : 0);
    const turnInput = (keys['a'] || keys['arrowleft'] ? 1 : 0) - (keys['d'] || keys['arrowright'] ? 1 : 0);

    // Motor acceleration & friction
    state.v += (throttle * 2.8 - state.v * 2.2) * dt;
    if (Math.abs(state.v) < 0.01 && throttle === 0) state.v = 0;

    // Steering command (reference/centre-line steer angle), actuator-smoothed
    const targetSteer = turnInput * MAX_STEER;
    state.steerAngle += (targetSteer - state.steerAngle) * Math.min(1, dt * 6.0);

    // Body twist. Rolling: bicycle-equivalent of 4-wheel steer  w = v * tan(delta) / Lf  (so the wheels' Ackermann
    // angles and the real heading change agree). Near standstill: spot-turn about the centre.
    const kappa = Math.tan(state.steerAngle) / lay.Lf;
    const blend = smoothstep(0.05, 0.4, Math.abs(state.v));
    const yawRate = blend * state.v * kappa + (1 - blend) * state.steerAngle * SPOT_TURN_RATE;
    state.yaw += yawRate * dt;

    // Compute forward displacement vector
    const fx = -Math.sin(state.yaw), fz = -Math.cos(state.yaw);
    const nx = currentPos.x + fx * state.v * dt;
    const nz = currentPos.z + fz * state.v * dt;

    // Survey boundary check (if boundary is active)
    if (surveyLimit && (Math.abs(nx) > surveyLimit || Math.abs(nz) > surveyLimit)) {
      state.v = 0;
      if (onBoundaryHit) onBoundaryHit();
    } else {
      // Slope check on real DTM terrain
      const ahead = Math.sign(state.v || 1) * 1.5;
      const h0 = heightAt(currentPos.x, currentPos.z);
      const h1 = heightAt(currentPos.x + fx * ahead, currentPos.z + fz * ahead);
      const slopeDeg = Math.atan(Math.abs(h1 - h0) / 1.5) * (180 / Math.PI);

      if (slopeDeg > 26 && Math.abs(state.v) > 0.05) {
        state.v = 0;
        if (onBoundaryHit) onBoundaryHit(`DTM Slope ${slopeDeg.toFixed(0)}° exceeds 26° safety limit.`);
      } else {
        currentPos.x = nx;
        currentPos.z = nz;
      }
    }

    // 2. Rocker-bogie terrain compliance: probe terrain under each of the 6 wheels
    const c = Math.cos(state.yaw), s = Math.sin(state.yaw);
    const wh = lay.pts.map((p) => heightAt(currentPos.x + p.x * c + p.z * s, currentPos.z - p.x * s + p.z * c));

    const { L, R, pts } = lay;
    const leftRockerPitch = Math.atan2(wh[L.front] - wh[L.rear], pts[L.rear].z - pts[L.front].z);
    const rightRockerPitch = Math.atan2(wh[R.front] - wh[R.rear], pts[R.rear].z - pts[R.front].z);
    // Differential bar averages left and right rocker pitches -> body tilts half as much as one rocker
    const targetPitch = (leftRockerPitch + rightRockerPitch) * 0.5;

    const avg = (ids) => ids.reduce((a, i) => a + wh[i], 0) / ids.length;
    const targetRoll = Math.atan2(avg(R.all) - avg(L.all), lay.track);

    state.pitch += (targetPitch - state.pitch) * Math.min(1, dt * 5.0);
    state.roll += (targetRoll - state.roll) * Math.min(1, dt * 5.0);

    const groundY = wh.reduce((a, b) => a + b, 0) / wh.length;
    currentPos.y = groundY;

    // Apply orientation to root
    rig.root.position.set(currentPos.x, groundY, currentPos.z);
    rig.root.rotation.set(state.pitch, state.yaw, state.roll, 'YXZ');

    // 3. Per-wheel steering, spin and ground contact
    if (rig.wheels) {
      rig.root.updateMatrixWorld(true);
      const cosTilt = Math.max(0.5, Math.cos(state.pitch) * Math.cos(state.roll));
      const maxSlew = STEER_SLEW * dt;
      let spinSum = 0;

      rig.wheels.forEach((w, i) => {
        const a = -w.restPos.x;        // lateral offset, + = left
        const l = -w.restPos.z;        // longitudinal offset, + = forward
        // Rigid-body twist -> velocity of this wheel's axle in (forward, left)
        const vf = state.v - yawRate * a;
        const vl = yawRate * l;
        const mag = Math.hypot(vf, vl);

        if (w.steerable && mag > 1e-3) {
          let ang = Math.atan2(vl, vf);                  // direction the wheel must point
          if (ang > Math.PI / 2) ang -= Math.PI;         // a wheel can roll either way: stay within +-90 deg
          else if (ang < -Math.PI / 2) ang += Math.PI;
          w.steer += clamp(ang - w.steer, -maxSlew, maxSlew);
        }
        w.pivot.rotation.y = w.steer;

        // Roll along the wheel's actual heading: w = v_along / r
        const along = w.steerable ? vf * Math.cos(w.steer) + vl * Math.sin(w.steer) : vf;
        w.spin += (along * dt) / lay.radius;
        spinSum += w.spin;
        w.hub.rotation.x = w.spin * (rig.spinSign || 1);

        // Vertical compliance: put the axle one radius above the terrain beneath this wheel
        _v.set(w.restPos.x, w.restPos.y, w.restPos.z).applyMatrix4(rig.root.matrixWorld);
        const dyLocal = clamp((wh[i] + lay.radius - _v.y) / cosTilt, -MAX_WHEEL_TRAVEL, MAX_WHEEL_TRAVEL);
        w.dy += (dyLocal - w.dy) * Math.min(1, dt * 12.0);
        w.pivot.position.y = w.baseY + w.dy;
      });
      state.wheelSpin = spinSum / rig.wheels.length;
      state.wheelSteer = rig.wheels.map((w) => w.steer);
    } else {
      state.wheelSpin += (state.v * dt) / WHEEL_RADIUS;
    }

    // Sound effect trigger
    if (window.MarsAudio) {
      window.MarsAudio.updateRoverSpeed(state.v);
    }
  }

  /**
   * Updates 5-DOF robotic arm extension, positioning, and sample drilling.
   */
  function updateRoboticArm(dt, rig) {
    if (!rig || !rig.armJoints) return;
    const { shoulderAz, shoulderEl, elbow, wrist, turret, drillBit } = rig.armJoints;

    if (state.armMode === 'DEPLOYING') {
      state.armProgress = Math.min(1.0, state.armProgress + dt * 0.6);
      if (state.armProgress >= 1.0) state.armMode = 'INSPECTING';
    } else if (state.armMode === 'RETRACTING') {
      state.armProgress = Math.max(0.0, state.armProgress - dt * 0.7);
      if (state.armProgress <= 0.0) state.armMode = 'STOWED';
    }

    const t = state.armProgress;

    // Interpolate joints from Stowed (t=0) to Deployed Ground Inspect (t=1)
    // Stowed: shoulderAz=0.8, shoulderEl=-1.45, elbow=2.45, wrist=-1.0
    // Deployed: shoulderAz=0.0, shoulderEl=0.25, elbow=0.85, wrist=-0.65
    shoulderAz.rotation.y = THREE.MathUtils.lerp(0.8, 0.0, t);
    shoulderEl.rotation.z = THREE.MathUtils.lerp(-1.45, 0.25, t);
    elbow.rotation.z = THREE.MathUtils.lerp(2.45, 0.85, t);
    wrist.rotation.z = THREE.MathUtils.lerp(-1.0, -0.65, t);

    // Drilling mode: high-speed drill spin & micro-vibration
    if (state.armMode === 'DRILLING') {
      state.drillSpin += dt * 35;
      if (drillBit) drillBit.rotation.y = state.drillSpin;
      // Slight mechanical vibration
      turret.position.x = (Math.random() - 0.5) * 0.008;
      turret.position.y = (Math.random() - 0.5) * 0.008;
      if (window.MarsAudio) window.MarsAudio.setDrilling(true);
    } else {
      if (window.MarsAudio) window.MarsAudio.setDrilling(false);
      turret.position.set(0, 0, 0);
    }
  }

  /**
   * Updates Ingenuity Mars Helicopter flight dynamics:
   * Spool-up, counter-rotating coaxial rotor blades, aerodynamic banking, and altitude control.
   */
  function updateHelicopter(dt, keys, rig, heightAt, currentPos, surveyLimit, onBoundaryHit) {
    if (!rig) return;

    // Controls
    const throttle = (keys['w'] || keys['arrowup'] ? 1 : 0) - (keys['s'] || keys['arrowdown'] ? 1 : 0);
    const turn = (keys['a'] || keys['arrowleft'] ? 1 : 0) - (keys['d'] || keys['arrowright'] ? 1 : 0);
    const climb = (keys[' '] || keys['e'] ? 1 : 0) - (keys['shift'] || keys['q'] ? 1 : 0);

    const groundY = heightAt(currentPos.x, currentPos.z);

    // Flight state transitions
    if (state.flightState === 'LANDED') {
      if (climb > 0 || throttle !== 0 || turn !== 0) {
        state.flightState = 'SPOOLING';
      }
    }

    if (state.flightState === 'SPOOLING') {
      state.rotorRPM = Math.min(1.0, state.rotorRPM + dt * 0.7);
      if (state.rotorRPM >= 0.85) {
        state.flightState = 'TAKEOFF';
        state.targetAlt = 3.0; // initial nominal survey altitude
      }
    } else if (state.flightState === 'TAKEOFF') {
      state.altitude = Math.min(state.targetAlt, state.altitude + dt * 2.2);
      if (state.altitude >= state.targetAlt - 0.1) {
        state.flightState = 'FLYING';
      }
    } else if (state.flightState === 'FLYING' || state.flightState === 'HOVER') {
      state.rotorRPM = 1.0;
      // Vertical climb / descent
      state.targetAlt = Math.max(1.0, Math.min(MAX_HELI_ALT, state.targetAlt + climb * 3.5 * dt));
      state.altitude += (state.targetAlt - state.altitude) * Math.min(1, dt * 3.0);

      // Check for landing command
      if (climb < 0 && state.altitude < 1.4) {
        state.flightState = 'LANDING';
      }
    } else if (state.flightState === 'LANDING') {
      state.altitude = Math.max(0.0, state.altitude - dt * 1.8);
      if (state.altitude <= 0.05) {
        state.altitude = 0.0;
        state.rotorRPM = Math.max(0.0, state.rotorRPM - dt * 0.8);
        if (state.rotorRPM <= 0.05) state.flightState = 'LANDED';
      }
    }

    // Horizontal flight & banking
    if (state.altitude > 0.4) {
      state.v += (throttle * 5.5 - state.v * 1.8) * dt;
      state.yaw += turn * 1.8 * dt;

      // Aerodynamic tilt (banking into motion)
      const targetPitch = (state.v / 5.5) * 0.35; // pitches forward
      const targetRoll = -turn * 0.3;              // rolls into turn
      state.heliPitch += (targetPitch - state.heliPitch) * Math.min(1, dt * 5.0);
      state.heliRoll += (targetRoll - state.heliRoll) * Math.min(1, dt * 5.0);

      const fx = -Math.sin(state.yaw), fz = -Math.cos(state.yaw);
      const nx = currentPos.x + fx * state.v * dt;
      const nz = currentPos.z + fz * state.v * dt;

      if (surveyLimit && (Math.abs(nx) > surveyLimit || Math.abs(nz) > surveyLimit)) {
        state.v = 0;
        if (onBoundaryHit) onBoundaryHit();
      } else {
        currentPos.x = nx;
        currentPos.z = nz;
      }
    } else {
      state.v = 0;
      state.heliPitch = 0;
      state.heliRoll = 0;
    }

    // 2. Counter-Rotating Dual Rotors Animation
    // Top rotor spins Clockwise, Bottom rotor spins Counter-Clockwise
    const spinSpeed = state.rotorRPM * dt * 75; // 2,400+ RPM visual blur
    state.rotorAngle += spinSpeed;

    if (rig.rotor1) rig.rotor1.rotation.y = state.rotorAngle;
    if (rig.rotor2) rig.rotor2.rotation.y = -state.rotorAngle;

    // Apply 3D transform to helicopter
    currentPos.y = groundY + state.altitude;
    rig.root.position.set(currentPos.x, currentPos.y, currentPos.z);
    rig.root.rotation.set(state.heliPitch, state.yaw, state.heliRoll, 'YXZ');

    // Sound effect
    if (window.MarsAudio) {
      window.MarsAudio.updateHelicopter(state.rotorRPM, state.altitude > 0.3);
    }
  }

  function setArmMode(mode) {
    state.armMode = mode;
  }

  function resetState() {
    state.v = 0;
    state.steerAngle = 0;
    state.armMode = 'STOWED';
    state.armProgress = 0;
    state.flightState = 'LANDED';
    state.altitude = 0;
    state.rotorRPM = 0;
  }

  return {
    state,
    updateRover,
    updateRoboticArm,
    updateHelicopter,
    setArmMode,
    resetState
  };
})();
