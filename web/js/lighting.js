/* Mars Diurnal Cycle, Solar Position & Time-of-Day Lighting */
window.MarsLighting = (() => {
  'use strict';

  let dirLight = null;
  let hemiLight = null;
  let sunDirVector = new THREE.Vector3();
  let roverRigRef = null;
  let headlightsOn = false;

  const state = {
    solHour: 11.5,       // 0.0 to 24.0 (Martian Local True Solar Time)
    autoAdvance: false,   // whether time automatically flows
    speedMultiplier: 300, // 1 real sec = 300 sim sec (full day in ~5 min)
    isNight: false,
    sunElevation: 0,
    sunAzimuth: 0
  };

  /**
   * Initializes scene lights and shadow maps.
   */
  function init(scene, initialSunDir) {
    // 1. Directional Sun Light with Shadows
    dirLight = new THREE.DirectionalLight(0xfffaed, 0.95);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.width = 1024;
    dirLight.shadow.mapSize.height = 1024;
    dirLight.shadow.camera.near = 0.5;
    dirLight.shadow.camera.far = 120;
    const d = 16;
    dirLight.shadow.camera.left = -d;
    dirLight.shadow.camera.right = d;
    dirLight.shadow.camera.top = d;
    dirLight.shadow.camera.bottom = -d;
    dirLight.shadow.bias = -0.001;
    scene.add(dirLight);

    // 2. Martian Ambient / Hemisphere Sky Light
    hemiLight = new THREE.HemisphereLight(0xffeedd, 0x442c1d, 0.65);
    scene.add(hemiLight);

    if (initialSunDir) {
      sunDirVector.copy(initialSunDir);
    }

    setSolTime(11.5, scene, null);
  }

  /**
   * Calculates sun vector, color temperatures, and ambient levels for a given Sol hour.
   * Latitude of Jezero landing site: ~18.44° N
   */
  function setSolTime(hour, scene, terrainMaterials) {
    state.solHour = Math.max(0, Math.min(24, hour));

    // Solar hour angle: 12.0 is noon (0 rad), 6.0 is sunrise (-PI/2), 18.0 is sunset (PI/2)
    const hourAngle = ((state.solHour - 12) / 12) * Math.PI;

    // Solar elevation angle: max ~72° at noon, negative at night
    const maxElevation = 1.25; // ~72 degrees
    const elevation = Math.sin(Math.PI / 2 - Math.abs(hourAngle)) * maxElevation;
    const azimuth = hourAngle + Math.PI; // East at morning, West at evening

    state.sunElevation = elevation * (180 / Math.PI);
    state.sunAzimuth = azimuth * (180 / Math.PI);
    state.isNight = elevation <= 0.05;

    // Calculate Cartesian unit sun direction
    if (elevation > 0) {
      sunDirVector.set(
        Math.sin(azimuth) * Math.cos(elevation),
        Math.sin(elevation),
        Math.cos(azimuth) * Math.cos(elevation)
      ).normalize();
    } else {
      // Below horizon
      sunDirVector.set(0, -0.2, 0).normalize();
    }

    // Color temperature & lighting intensity based on elevation
    let sunColor, skyColor, groundColor, sunIntensity, hemiIntensity;

    if (elevation > 0.4) {
      // Midday: bright warm tan sunlight, reddish-tan sky
      sunColor = new THREE.Color(0xfff6ea);
      skyColor = new THREE.Color(0xdcb088);
      groundColor = new THREE.Color(0x553522);
      sunIntensity = 1.05;
      hemiIntensity = 0.55;
    } else if (elevation > 0.08) {
      // Golden Hour (Early morning / Late afternoon): warm butterscotch & amber
      const t = (elevation - 0.08) / 0.32;
      sunColor = new THREE.Color().lerpColors(new THREE.Color(0xffaa55), new THREE.Color(0xfff6ea), t);
      skyColor = new THREE.Color().lerpColors(new THREE.Color(0xaa6644), new THREE.Color(0xdcb088), t);
      groundColor = new THREE.Color(0x381f12);
      sunIntensity = 0.4 + t * 0.65;
      hemiIntensity = 0.35 + t * 0.2;
    } else if (elevation > -0.05) {
      // Twilight / Dusk: deep violet-blue forward scattering (famous blue Martian sunset!)
      sunColor = new THREE.Color(0x5588cc);
      skyColor = new THREE.Color(0x403045);
      groundColor = new THREE.Color(0x181015);
      sunIntensity = 0.15;
      hemiIntensity = 0.2;
    } else {
      // Nighttime: faint starry luminescence and Phobos moon glow
      sunColor = new THREE.Color(0x111825);
      skyColor = new THREE.Color(0x0a0c14);
      groundColor = new THREE.Color(0x040508);
      sunIntensity = 0.0;
      hemiIntensity = 0.12;
    }

    if (dirLight) {
      dirLight.color.copy(sunColor);
      dirLight.intensity = sunIntensity;
    }

    if (hemiLight) {
      hemiLight.color.copy(skyColor);
      hemiLight.groundColor.copy(groundColor);
      hemiLight.intensity = hemiIntensity;
    }

    if (scene) {
      scene.background = skyColor;
    }

    // Update shaders if passed
    if (terrainMaterials) {
      terrainMaterials.forEach((m) => {
        if (m && m.uniforms && m.uniforms.sunDir) {
          m.uniforms.sunDir.value.copy(sunDirVector);
        }
      });
    }

    // Auto-enable headlights at night if user hasn't explicitly disabled
    if (state.isNight && !headlightsOn) {
      setHeadlights(true);
    }
  }

  /**
   * Centers the directional light around the rover to optimize shadow resolution.
   */
  function update(dt, roverPos, terrainMaterials, scene) {
    if (state.autoAdvance) {
      const hoursPerSec = (state.speedMultiplier / 3600);
      let newHour = state.solHour + dt * hoursPerSec;
      if (newHour >= 24) newHour -= 24;
      setSolTime(newHour, scene, terrainMaterials);
    }

    if (dirLight && roverPos) {
      const rx = (typeof roverPos.x === 'number' && !isNaN(roverPos.x)) ? roverPos.x : 0;
      const ry = (typeof roverPos.y === 'number' && !isNaN(roverPos.y)) ? roverPos.y : 0;
      const rz = (typeof roverPos.z === 'number' && !isNaN(roverPos.z)) ? roverPos.z : 0;
      dirLight.position.set(
        rx + sunDirVector.x * 35,
        ry + Math.max(12, sunDirVector.y * 35),
        rz + sunDirVector.z * 35
      );
      dirLight.target.position.set(rx, ry, rz);
      dirLight.target.updateMatrixWorld();
    }
  }

  function setHeadlights(on, rig) {
    headlightsOn = on;
    const r = rig || roverRigRef;
    if (r && r.headlights) {
      r.headlights.forEach((hl) => {
        hl.intensity = on ? 2.5 : 0.0;
      });
    }
  }

  function toggleHeadlights(rig) {
    setHeadlights(!headlightsOn, rig);
    return headlightsOn;
  }

  function registerRig(rig) {
    roverRigRef = rig;
    setHeadlights(headlightsOn, rig);
  }

  function getFormattedTime() {
    const totalMinutes = Math.round(state.solHour * 60);
    const h = Math.floor(totalMinutes / 60) % 24;
    const m = totalMinutes % 60;
    return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')} LST`;
  }

  return {
    state,
    init,
    setSolTime,
    update,
    setHeadlights,
    toggleHeadlights,
    registerRig,
    getSunDir: () => sunDirVector,
    getFormattedTime,
    isHeadlightsOn: () => headlightsOn
  };
})();
