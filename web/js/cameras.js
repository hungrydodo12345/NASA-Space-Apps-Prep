/* Multi-Camera Scientific Imaging Suite for Mars Rovers & Ingenuity */
window.MarsCameras = (() => {
  'use strict';

  const MODES = [
    { id: 'chase', name: 'Chase Camera', desc: 'Third-person trailing survey view' },
    { id: 'mastcam', name: 'Mastcam-Z Stereo', desc: 'Remote Sensing Mast (2.0 m) with optical zoom' },
    { id: 'hazcam_front', name: 'Front Hazcam', desc: 'Low-chassis 120° wide-angle boulder hazard view' },
    { id: 'hazcam_rear', name: 'Rear Hazcam', desc: 'Reverse drive hazard detection' },
    { id: 'watson', name: 'WATSON Arm Micro-Cam', desc: 'Close-up microscopic petrology inspection on robotic arm' },
    { id: 'aerial_down', name: 'Downward RTE Cam', desc: 'Optical surface-drift tracking camera' }
  ];

  let currentModeIdx = 0;
  let zoomLevel = 1.0; // 1.0x to 5.0x zoom for Mastcam-Z
  const camPosTarget = new THREE.Vector3();
  const camLookTarget = new THREE.Vector3();

  function getCurrent() {
    return MODES[currentModeIdx];
  }

  function setMode(modeId) {
    const idx = MODES.findIndex((m) => m.id === modeId);
    if (idx !== -1) currentModeIdx = idx;
    return getCurrent();
  }

  function cycleNext() {
    currentModeIdx = (currentModeIdx + 1) % MODES.length;
    return getCurrent();
  }

  function setZoom(val, camera) {
    zoomLevel = Math.max(1.0, Math.min(5.0, val));
    if (camera) {
      camera.fov = 60 / zoomLevel;
      camera.updateProjectionMatrix();
    }
  }

  function adjustZoom(delta, camera) {
    setZoom(zoomLevel + delta, camera);
    return zoomLevel;
  }

  /**
   * Updates camera position and orientation based on active mode.
   * Completely immune to undefined or NaN values.
   */
  function update(dt, camera, roverPos, roverYaw, roverPitch, roverRoll, rig, gyro, deviceQuat) {
    if (!camera) return;

    // Sanitize position inputs
    const rx = (roverPos && typeof roverPos.x === 'number' && !isNaN(roverPos.x)) ? roverPos.x : 0;
    const ry = (roverPos && typeof roverPos.y === 'number' && !isNaN(roverPos.y)) ? roverPos.y : 0;
    const rz = (roverPos && typeof roverPos.z === 'number' && !isNaN(roverPos.z)) ? roverPos.z : 0;
    const yaw = (!isNaN(roverYaw)) ? roverYaw : 0;

    const mode = MODES[currentModeIdx].id;
    const isHeli = !!(rig && rig.isHelicopter);

    // Reset base FOV if not in Mastcam mode
    if (mode !== 'mastcam' && camera.fov !== 60) {
      camera.fov = 60;
      camera.updateProjectionMatrix();
    }

    const c = Math.cos(yaw), s = Math.sin(yaw);
    // Forward directional vector in world space: -sin(yaw), -cos(yaw)
    const fx = -s, fz = -c;

    if (mode === 'chase') {
      if (isHeli) {
        camLookTarget.set(rx, ry + 0.6, rz);
        camPosTarget.set(rx - fx * 6.5, ry + 3.4, rz - fz * 6.5);
      } else {
        camLookTarget.set(rx, ry + 1.2, rz);
        camPosTarget.set(rx - fx * 8.5, ry + 3.6, rz - fz * 8.5);
      }

      // Safe lerp: if current camera position is invalid or far away, snap immediately
      if (isNaN(camera.position.x) || isNaN(camera.position.y) || isNaN(camera.position.z) || camera.position.distanceTo(camPosTarget) > 100) {
        camera.position.copy(camPosTarget);
      } else {
        camera.position.lerp(camPosTarget, Math.min(1, dt * 5.0));
      }
      camera.lookAt(camLookTarget);

    } else if (mode === 'mastcam') {
      // 2.05 m above deck pivot on the Remote Sensing Mast
      const mx = rx + 0.55 * c - (-0.95) * s;
      const mz = rz - 0.55 * s + (-0.95) * c;
      const my = ry + 2.05;

      camera.position.set(mx, my, mz);

      const qh = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), yaw);
      if (gyro && deviceQuat) {
        const dq = deviceQuat();
        if (dq) camera.quaternion.copy(qh).multiply(dq);
      } else {
        camera.quaternion.copy(qh).multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -0.2));
      }

    } else if (mode === 'hazcam_front') {
      // Front bumper, 0.95m off ground, pointing forward-down toward ground hazards
      const bx = rx + fx * 1.35;
      const bz = rz + fz * 1.35;
      const by = ry + 0.95;

      camera.position.set(bx, by, bz);
      camLookTarget.set(bx + fx * 5.0, by - 0.5, bz + fz * 5.0);
      camera.lookAt(camLookTarget);

    } else if (mode === 'hazcam_rear') {
      // Rear hazard camera, looking backward
      const rxRear = rx - fx * 1.35;
      const rzRear = rz - fz * 1.35;
      const ryRear = ry + 1.05;

      camera.position.set(rxRear, ryRear, rzRear);
      camLookTarget.set(rxRear - fx * 5.0, ryRear - 0.4, rzRear - fz * 5.0);
      camera.lookAt(camLookTarget);

    } else if (mode === 'watson') {
      // Robotic arm / close-up sample target view
      const wx = rx - 0.6 * c + fx * 0.9;
      const wz = rz + 0.6 * s + fz * 0.9;
      const wy = ry + 0.85;

      camera.position.set(wx, wy, wz);
      camLookTarget.set(rx + fx * 1.6, ry + 0.1, rz + fz * 1.6);
      camera.lookAt(camLookTarget);

    } else if (mode === 'aerial_down') {
      // Looking straight down at ground beneath vehicle
      camera.position.set(rx, ry + (isHeli ? 0.4 : 1.8), rz);
      camera.rotation.set(-Math.PI / 2, 0, yaw, 'YXZ');
    }
  }

  return {
    MODES,
    getCurrent,
    setMode,
    cycleNext,
    setZoom,
    adjustZoom,
    getZoom: () => zoomLevel,
    update
  };
})();
