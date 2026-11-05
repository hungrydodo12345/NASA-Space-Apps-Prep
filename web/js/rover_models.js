/* Mars Rover & Ingenuity 3D Models, Rigging & Component Breakdown */
window.RoverModels = (() => {
  'use strict';

  // Educational component encyclopedia for NASA Space Apps
  const COMPONENT_INFO = {
    rocker_bogie: {
      name: 'Rocker-Bogie Suspension',
      role: 'Rough-Terrain Compliance',
      movement: 'Passive differential rocker-bogie rocking with top balance bar',
      description: 'NASA patent mechanism without springs. Distributes vehicle weight evenly across all 6 wheels, allowing the rover to climb over obstacles larger than the wheel diameter (up to 40 cm) while reducing body tilt by 50%.'
    },
    wheels: {
      name: 'Aluminum Cleated Wheels & Steering Actuators',
      role: 'Traction & Independent Steering',
      movement: 'Rolling spin (v/r) + front and rear wheel Ackermann turning',
      description: 'Machined from single blocks of aluminum with curved titanium spokes for shock absorption. Four corner wheels have dedicated steering motors to enable driving, curved steering, and in-place 360° point turns.'
    },
    robotic_arm: {
      name: '5-DOF Robotic Science Arm & Turret',
      role: 'Contact Science & Sample Extraction',
      movement: 'Articulated 5-joint kinematic chain (azimuth, elevation, elbow, wrist, turret)',
      description: 'A 2.1 m long arm that positions a 45 kg turret against Martian bedrock. Houses the rotary-percussive coring drill, the PIXL X-ray spectrometer, and the WATSON microscopic imager.'
    },
    mast: {
      name: 'Remote Sensing Mast (RSM)',
      role: 'Panoramic & Spectral Reconnaissance',
      movement: '360° Pan and ±90° Tilt head pointing',
      description: 'Stands 2 m tall above the Martian surface to emulate human field of view. Houses Mastcam-Z dual stereo zoom cameras, Navcams for autonomous hazard avoidance, and SuperCam with pulsed laser rock ablation.'
    },
    power_source: {
      name: 'MMRTG (Plutonium Decaying Generator)',
      role: 'Electrical & Thermal Survival',
      movement: 'Static cooling fin thermal dissipation',
      description: 'Contains 4.8 kg of plutonium dioxide. Continuously generates ~110 Watts of electrical power and excess heat piped through internal fluid loops to keep sensitive computers warm in -90°C Martian nights.'
    },
    antenna: {
      name: 'Steerable High-Gain Antenna (HGA)',
      role: 'Earth & Deep Space Network Link',
      movement: 'Dual-axis gimbal pointing directly toward Earth',
      description: 'Transmits X-band scientific data directly to Earth over 250 million kilometers to the Deep Space Network (DSN) at Goldstone, Madrid, and Canberra.'
    },
    ingenuity_rotors: {
      name: 'Coaxial Counter-Rotating Blades',
      role: 'Atmospheric Aerodynamic Lift',
      movement: 'Top rotor CW, bottom rotor CCW at 2,400 - 2,800 RPM',
      description: 'Because Mars atmosphere has only 1% the density of Earth at sea level, twin 1.2 m counter-rotating carbon-fiber blades spin 10x faster than terrestrial helicopters to lift the 1.8 kg drone.'
    },
    ingenuity_solar: {
      name: 'Ingenuity Solar Panel & Avionics',
      role: 'Flight Power & Autonomous Guidance',
      movement: 'Down-facing optical RTE camera tracking surface drift at 30 Hz',
      description: 'Sits atop the rotor mast to recharge 6 Sony lithium-ion cells each sol. Features dual Qualcomm Snapdragon / FPGA guidance computing 500 navigational adjustments per second.'
    }
  };

  // Specifications for rover character selection
  const ROVER_SPECS = {
    perseverance: {
      id: 'perseverance',
      name: 'Perseverance (Mars 2020)',
      site: 'Jezero Crater (Octavia E. Butler)',
      landingSol: 'Feb 18, 2021',
      mass: '1,025 kg',
      power: 'MMRTG Nuclear',
      highlight: 'MOXIE oxygen generator, SuperCam laser, Coring Drill, sample caching for return to Earth.',
      modelFile: 'models/perseverance.glb',
      scale: 1.0,
      yOffset: 0.0,
      nativeMeters: true,   // GLB is authored in real metres (wheel dia 0.52 m): do NOT auto-rescale
      facesPlusZ: true,     // GLB front (arm, hazcams) is +Z; the sim drives toward -Z, so the model is flipped 180°
      companion: 'ingenuity'
    },
    curiosity: {
      id: 'curiosity',
      name: 'Curiosity (MSL)',
      site: 'Gale Crater (Mount Sharp)',
      landingSol: 'Aug 6, 2012',
      mass: '899 kg',
      power: 'MMRTG Nuclear',
      highlight: 'CheMin X-ray diffraction, SAM chemistry lab, 30+ km traversed studying ancient lakes.',
      modelFile: 'models/curiosity.glb',
      scale: 1.0,
      yOffset: 0.0,
      companion: null
    },
    opportunity: {
      id: 'opportunity',
      name: 'Opportunity (MER-B)',
      site: 'Meridiani Planum',
      landingSol: 'Jan 25, 2004',
      mass: '185 kg',
      power: 'Solar Arrays',
      highlight: 'Longest distance driven on another world (45.16 km); discovered hematite blueberries confirming liquid water.',
      modelFile: 'models/mer_opportunity.glb',
      scale: 1.05,
      yOffset: 0.0,
      companion: null
    },
    spirit: {
      id: 'spirit',
      name: 'Spirit (MER-A)',
      site: 'Gusev Crater (Columbia Hills)',
      landingSol: 'Jan 4, 2004',
      mass: '185 kg',
      power: 'Solar Arrays',
      highlight: 'Discovered pure silica deposits indicating ancient hydrothermal fumaroles/hot springs.',
      modelFile: 'models/mer_opportunity.glb',
      scale: 1.05,
      yOffset: 0.0,
      companion: null
    },
    ingenuity: {
      id: 'ingenuity',
      name: 'Ingenuity (Mars Helicopter)',
      site: 'Wright Brothers Field, Jezero',
      landingSol: 'Apr 3, 2021 (Deployment)',
      mass: '1.8 kg',
      power: 'Solar & Li-Ion Cells',
      highlight: 'First powered controlled flight on another planet. Flew 72 flights totaling over 17 km.',
      modelFile: 'models/ingenuity.glb',
      scale: 1.8,
      yOffset: 0.0,
      companion: null,
      isHelicopter: true
    }
  };

  /**
   * Builds an articulated high-fidelity procedural rover rig.
   * Isolates chassis, rocker-bogie, 6 wheels with steer pivots, mast, and robotic arm.
   */
  function createProceduralRover(spec = ROVER_SPECS.perseverance) {
    const root = new THREE.Group();
    root.name = 'Rover_' + spec.id;

    // Materials
    const matChassis = new THREE.MeshStandardMaterial({ color: 0xd6d0c4, roughness: 0.45, metalness: 0.35 });
    const matGoldFoil = new THREE.MeshStandardMaterial({ color: 0xd4af37, roughness: 0.25, metalness: 0.85 });
    const matTitanium = new THREE.MeshStandardMaterial({ color: 0x8a847c, roughness: 0.5, metalness: 0.6 });
    const matWheel = new THREE.MeshStandardMaterial({ color: 0x3d3834, roughness: 0.7, metalness: 0.4 });
    const matLens = new THREE.MeshStandardMaterial({ color: 0x111122, roughness: 0.1, metalness: 0.95 });
    const matRtg = new THREE.MeshStandardMaterial({ color: 0x444444, roughness: 0.6, metalness: 0.7 });

    // 1. Chassis Body
    const bodyGroup = new THREE.Group();
    bodyGroup.name = 'Chassis_Body';
    bodyGroup.userData.component = 'chassis';

    const mainBody = new THREE.Mesh(new THREE.BoxGeometry(1.6, 0.65, 2.5), matChassis);
    mainBody.position.y = 1.05;
    mainBody.castShadow = true;
    mainBody.receiveShadow = true;
    bodyGroup.add(mainBody);

    // Warm electronics box top deck with solar/foil details
    const topDeck = new THREE.Mesh(new THREE.BoxGeometry(1.5, 0.08, 2.3), spec.power.includes('Solar') ? matGoldFoil : matTitanium);
    topDeck.position.set(0, 1.41, 0);
    bodyGroup.add(topDeck);

    // RTG / Power unit on back
    const rtg = new THREE.Mesh(new THREE.CylinderGeometry(0.32, 0.35, 0.9, 12), matRtg);
    rtg.rotation.x = Math.PI / 2;
    rtg.position.set(0, 1.15, 1.45);
    rtg.userData.component = 'power_source';
    bodyGroup.add(rtg);

    // High gain antenna dish
    const hgaStem = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 0.5), matTitanium);
    hgaStem.position.set(-0.55, 1.65, 0.7);
    const hgaDish = new THREE.Mesh(new THREE.CylinderGeometry(0.28, 0.05, 0.06, 16), matTitanium);
    hgaDish.position.set(-0.55, 1.9, 0.7);
    hgaDish.rotation.z = 0.35;
    hgaDish.userData.component = 'antenna';
    bodyGroup.add(hgaStem);
    bodyGroup.add(hgaDish);

    // Front Hazcams
    const hazcamMat = new THREE.MeshBasicMaterial({ color: 0x1a2b3c });
    const fHazL = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.1, 0.12), hazcamMat);
    fHazL.position.set(-0.55, 0.95, -1.28);
    const fHazR = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.1, 0.12), hazcamMat);
    fHazR.position.set(0.55, 0.95, -1.28);
    bodyGroup.add(fHazL);
    bodyGroup.add(fHazR);

    // Headlights for night driving
    const headLightL = new THREE.SpotLight(0xfffae0, 0, 45, Math.PI / 5, 0.4, 1.5);
    headLightL.position.set(-0.5, 0.95, -1.3);
    headLightL.target.position.set(-0.5, 0.2, -10);
    bodyGroup.add(headLightL);
    bodyGroup.add(headLightL.target);

    const headLightR = new THREE.SpotLight(0xfffae0, 0, 45, Math.PI / 5, 0.4, 1.5);
    headLightR.position.set(0.5, 0.95, -1.3);
    headLightR.target.position.set(0.5, 0.2, -10);
    bodyGroup.add(headLightR);
    bodyGroup.add(headLightR.target);

    root.add(bodyGroup);

    // 2. Articulated Remote Sensing Mast (RSM)
    const mastBase = new THREE.Group();
    mastBase.name = 'Mast_Base';
    mastBase.position.set(0.55, 1.45, -0.95);
    mastBase.userData.component = 'mast';

    const mastShaft = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.07, 1.1, 10), matTitanium);
    mastShaft.position.y = 0.55;
    mastBase.add(mastShaft);

    const mastHead = new THREE.Group();
    mastHead.name = 'Mast_Head';
    mastHead.position.set(0, 1.12, 0);

    const headBox = new THREE.Mesh(new THREE.BoxGeometry(0.48, 0.24, 0.26), matChassis);
    mastHead.add(headBox);

    // Mastcam-Z Dual Zoom Lenses
    const lensL = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.07, 0.18, 14), matLens);
    lensL.rotation.x = Math.PI / 2;
    lensL.position.set(-0.16, 0.0, -0.15);
    const lensR = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.07, 0.18, 14), matLens);
    lensR.rotation.x = Math.PI / 2;
    lensR.position.set(0.16, 0.0, -0.15);
    mastHead.add(lensL);
    mastHead.add(lensR);

    // SuperCam mirror optic
    const superCam = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 0.12, 10), matTitanium);
    superCam.rotation.x = Math.PI / 2;
    superCam.position.set(0, 0.05, -0.14);
    mastHead.add(superCam);

    mastBase.add(mastHead);
    root.add(mastBase);

    // 3. 5-DOF Articulated Robotic Arm
    const armGroup = new THREE.Group();
    armGroup.name = 'Robotic_Arm';
    armGroup.position.set(-0.6, 1.05, -1.15);
    armGroup.userData.component = 'robotic_arm';

    // Joint 1: Shoulder Azimuth
    const shoulderAz = new THREE.Group();
    shoulderAz.name = 'Shoulder_Azimuth';
    const j1Mesh = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.1, 0.16, 12), matTitanium);
    shoulderAz.add(j1Mesh);

    // Joint 2: Shoulder Elevation
    const shoulderEl = new THREE.Group();
    shoulderEl.name = 'Shoulder_Elevation';
    shoulderEl.position.set(0, 0.12, 0);

    const upperArm = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 0.85, 10), matChassis);
    upperArm.position.set(0, 0.42, 0);
    shoulderEl.add(upperArm);

    // Joint 3: Elbow
    const elbow = new THREE.Group();
    elbow.name = 'Elbow_Joint';
    elbow.position.set(0, 0.85, 0);

    const forearm = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.055, 0.8, 10), matChassis);
    forearm.position.set(0, 0.4, 0);
    elbow.add(forearm);

    // Joint 4: Wrist & Joint 5: Turret
    const wrist = new THREE.Group();
    wrist.name = 'Wrist_Joint';
    wrist.position.set(0, 0.8, 0);

    const turret = new THREE.Group();
    turret.name = 'Arm_Turret';

    const turretHub = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.18, 0.22, 12), matTitanium);
    turret.add(turretHub);

    // Coring Drill tool
    const drillBit = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.015, 0.35, 10), matRtg);
    drillBit.rotation.x = Math.PI / 2;
    drillBit.position.set(0, 0, 0.22);
    turret.add(drillBit);

    // WATSON Micro-Camera
    const watsonCam = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.1, 0.14), matLens);
    watsonCam.position.set(0.12, 0.06, 0.12);
    turret.add(watsonCam);

    wrist.add(turret);
    elbow.add(wrist);
    shoulderEl.add(elbow);
    shoulderAz.add(shoulderEl);
    armGroup.add(shoulderAz);
    root.add(armGroup);

    // Default arm pose: stowed neatly across the front
    shoulderAz.rotation.y = 0.8;
    shoulderEl.rotation.z = -1.45;
    elbow.rotation.z = 2.45;
    wrist.rotation.z = -1.0;

    // 4. Rocker-Bogie Suspension & 6 Wheels
    // We create left and right rocker assemblies with independent wheel pivots
    const rockerGroup = new THREE.Group();
    rockerGroup.name = 'Rocker_Bogie_System';
    rockerGroup.userData.component = 'rocker_bogie';

    const WHEEL_NAMES = ['FL', 'FR', 'ML', 'MR', 'RL', 'RR'];
    const WHEEL_SPECS = [
      { name: 'FrontLeft', x: -1.25, z: -1.1, steerable: true },
      { name: 'FrontRight', x: 1.25, z: -1.1, steerable: true },
      { name: 'MidLeft', x: -1.35, z: 0.1, steerable: false },
      { name: 'MidRight', x: 1.35, z: 0.1, steerable: false },
      { name: 'RearLeft', x: -1.25, z: 1.15, steerable: true },
      { name: 'RearRight', x: 1.25, z: 1.15, steerable: true }
    ];

    const wheelMeshes = [];
    const steerPivots = [];
    const wheels = [];
    const PROC_WHEEL_R = 0.28;

    WHEEL_SPECS.forEach((ws, idx) => {
      // Steer pivot (rotates around Y axis); axle sits one wheel-radius above the rover ground plane
      const pivot = new THREE.Group();
      pivot.name = 'Pivot_' + ws.name;
      pivot.position.set(ws.x, PROC_WHEEL_R, ws.z);

      // Wheel mesh (rotates around X axis)
      const wheelHub = new THREE.Group();
      wheelHub.name = 'WheelHub_' + ws.name;

      const tireGeo = new THREE.CylinderGeometry(PROC_WHEEL_R, PROC_WHEEL_R, 0.38, 18);
      tireGeo.rotateZ(Math.PI / 2);
      const tire = new THREE.Mesh(tireGeo, matWheel);
      tire.castShadow = true;
      tire.receiveShadow = true;
      tire.userData.component = 'wheels';

      // Cleats / grousers pattern
      for (let c = 0; c < 8; c++) {
        const cleat = new THREE.Mesh(new THREE.BoxGeometry(0.4, 0.04, 0.06), matTitanium);
        const ang = (c / 8) * Math.PI * 2;
        cleat.position.set(0, Math.sin(ang) * PROC_WHEEL_R, Math.cos(ang) * PROC_WHEEL_R);
        cleat.rotation.x = ang;
        tire.add(cleat);
      }

      wheelHub.add(tire);
      pivot.add(wheelHub);
      rockerGroup.add(pivot);

      wheelMeshes.push(wheelHub);
      steerPivots.push({ pivot, steerable: ws.steerable, isFront: ws.z < 0 });
      wheels.push({ name: WHEEL_NAMES[idx], pivot, hub: wheelHub, steerable: ws.steerable });
    });

    root.add(rockerGroup);

    return {
      root,
      spec,
      bodyGroup,
      mastBase,
      mastHead,
      armGroup,
      armJoints: { shoulderAz, shoulderEl, elbow, wrist, turret, drillBit },
      wheelMeshes,
      steerPivots,
      wheels: finalizeWheels(root, wheels),
      wheelRadius: PROC_WHEEL_R,
      // Procedural hubs are NOT flipped, so a positive rotation.x moves the tread toward +Z (backwards in the sim).
      spinSign: -1,
      headlights: [headLightL, headLightR]
    };
  }

  /**
   * Builds an articulated Ingenuity Mars Helicopter rig.
   */
  function createProceduralIngenuity() {
    const root = new THREE.Group();
    root.name = 'Ingenuity_Helicopter';
    root.userData.isHelicopter = true;

    const matSolar = new THREE.MeshStandardMaterial({ color: 0x1f2e4d, roughness: 0.25, metalness: 0.8 });
    const matCarbon = new THREE.MeshStandardMaterial({ color: 0x222222, roughness: 0.7, metalness: 0.3 });
    const matGold = new THREE.MeshStandardMaterial({ color: 0xc89820, roughness: 0.3, metalness: 0.85 });
    const matLeg = new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.8, metalness: 0.2 });

    // 1. Fuselage Body (Cube holding batteries & avionics)
    const body = new THREE.Mesh(new THREE.BoxGeometry(0.36, 0.36, 0.36), matGold);
    body.position.y = 0.42;
    body.castShadow = true;
    body.userData.component = 'ingenuity_solar';
    root.add(body);

    // Downward Return-To-Earth (RTE) Camera
    const camLens = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 0.08, 12), new THREE.MeshBasicMaterial({ color: 0x050510 }));
    camLens.position.set(0, 0.22, 0);
    root.add(camLens);

    // 2. Landing Gear Legs (4 angled carbon-fiber legs)
    const legGeo = new THREE.CylinderGeometry(0.015, 0.015, 0.65, 8);
    const legAngles = [Math.PI / 4, 3 * Math.PI / 4, 5 * Math.PI / 4, 7 * Math.PI / 4];
    legAngles.forEach((ang) => {
      const leg = new THREE.Mesh(legGeo, matLeg);
      leg.position.set(Math.cos(ang) * 0.25, 0.22, Math.sin(ang) * 0.25);
      leg.rotation.y = -ang;
      leg.rotation.z = 0.55;
      root.add(leg);
    });

    // 3. Rotor Mast
    const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, 0.65, 10), matCarbon);
    mast.position.y = 0.75;
    root.add(mast);

    // 4. Coaxial Dual Counter-Rotating Rotors
    // Rotor 1 (Lower rotor)
    const rotor1 = new THREE.Group();
    rotor1.name = 'Rotor_Lower';
    rotor1.position.y = 0.82;
    rotor1.userData.component = 'ingenuity_rotors';

    const blade1A = new THREE.Mesh(new THREE.BoxGeometry(1.2, 0.015, 0.09), matCarbon);
    blade1A.castShadow = true;
    rotor1.add(blade1A);
    root.add(rotor1);

    // Rotor 2 (Upper rotor)
    const rotor2 = new THREE.Group();
    rotor2.name = 'Rotor_Upper';
    rotor2.position.y = 0.98;
    rotor2.userData.component = 'ingenuity_rotors';

    const blade2A = new THREE.Mesh(new THREE.BoxGeometry(1.2, 0.015, 0.09), matCarbon);
    blade2A.castShadow = true;
    rotor2.add(blade2A);
    root.add(rotor2);

    // 5. Top Solar Panel
    const solarPanel = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.015, 0.34), matSolar);
    solarPanel.position.y = 1.05;
    solarPanel.userData.component = 'ingenuity_solar';
    root.add(solarPanel);

    return {
      root,
      spec: ROVER_SPECS.ingenuity,
      rotor1,
      rotor2,
      body,
      isHelicopter: true
    };
  }

  /* ------------------------------------------------------------------ *
   *  Wheel rig helpers
   * ------------------------------------------------------------------ */
  const CANON_WHEELS = ['FL', 'FR', 'ML', 'MR', 'RL', 'RR'];

  /**
   * Records each wheel's rest axle position in rover-root space and orders the list FL,FR,ML,MR,RL,RR.
   * Rover-root convention (matches mechanics.js): forward = -Z, left = -X, +Y up, origin on the ground plane.
   */
  function finalizeWheels(root, wheels) {
    root.updateMatrixWorld(true);
    const tmp = new THREE.Vector3();
    wheels.forEach((w) => {
      w.pivot.getWorldPosition(tmp);
      root.worldToLocal(tmp);
      w.restPos = tmp.clone();
      w.baseY = w.pivot.position.y;   // pivot Y in its parent's space at rest (suspension travel is added to this)
      w.steer = 0;                    // current steer angle (rad)
      w.spin = 0;                     // accumulated roll angle (rad)
      w.dy = 0;                       // current suspension travel (m)
    });
    return CANON_WHEELS.map((n) => wheels.find((w) => w.name === n)).filter(Boolean);
  }

  /**
   * NASA's CAD export welded all six wheels into one 3-primitive mesh ("Wheels_objs").
   * Split every triangle of every primitive into the wheel it belongs to (nearest axle in X/Z),
   * re-centre each wheel's vertices on its own axle, and parent them to a steer-pivot > spin-hub pair.
   * Returns [{ pivot, hub, center(Vector3 in Wheels_objs space), radius, widthX }] (unordered).
   */
  function splitWheelsMesh(wheelsNode) {
    const meshes = [];
    wheelsNode.traverse((o) => { if (o.isMesh) meshes.push(o); });
    if (!meshes.length) return null;

    // Axle seeds in mesh space (measured from the GLB: corner wheels x=±1.06, middle wheels x=±1.18)
    let seeds = [];
    [-1, 1].forEach((sx) => [-1.13, -0.055, 1.13].forEach((z) => {
      seeds.push({ x: sx * (Math.abs(z) > 0.5 ? 1.06 : 1.18), z });
    }));

    // Triangle centroids -> nearest seed, with two Lloyd refinements so the split adapts to the real asset
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

    // Per-wheel bounding box (so the pivot lands exactly on the axle)
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
        nm.name = 'wheel_part';
        out[w].hub.add(nm);
      }
    });

    // Swap: drop the welded meshes, add the six pivots under the original node (keeps its transform)
    meshes.forEach((m) => { m.parent.remove(m); m.geometry.dispose(); });
    out.forEach((o) => wheelsNode.add(o.pivot));
    return out;
  }

  /** Two forward-facing headlights for GLB rovers (rover-root space, forward = -Z). */
  function addGLBHeadlights(root) {
    return [-0.55, 0.55].map((x) => {
      const hl = new THREE.SpotLight(0xfffae0, 0, 45, Math.PI / 5, 0.4, 1.5);
      hl.position.set(x, 0.95, -1.5);
      hl.target.position.set(x, 0.2, -12);
      root.add(hl); root.add(hl.target);
      return hl;
    });
  }

  /**
   * Loads the real NASA GLB model using Three.js GLTFLoader.
   * If GLTF fails or is slow, falls back seamlessly to procedural rig.
   */
  function loadGLBModel(spec, scene, onLoaded, onError) {
    if (!window.THREE || !window.THREE.GLTFLoader) {
      console.warn('GLTFLoader not found. Using procedural rig.');
      const proc = spec.isHelicopter ? createProceduralIngenuity() : createProceduralRover(spec);
      onLoaded(proc);
      return;
    }

    const loader = new THREE.GLTFLoader();
    const modelUrl = 'data/' + spec.modelFile;

    loader.load(
      modelUrl,
      (gltf) => {
        const root = new THREE.Group();
        root.name = 'NASA_' + spec.id;
        const model = gltf.scene;

        if (spec.isHelicopter) {
          model.rotation.z = Math.PI / 2;
        }

        model.traverse((child) => {
          if (child.isMesh) {
            child.castShadow = true;
            child.receiveShadow = true;
            if (child.material) {
              child.material.roughness = Math.max(0.3, child.material.roughness || 0.5);
              child.material.metalness = Math.min(0.85, child.material.metalness || 0.2);
            }
          }
        });

        const rig = { root, spec, glbScene: model, isHelicopter: !!spec.isHelicopter };

        if (spec.nativeMeters && !spec.isHelicopter) {
          /* ---- Real-metre rover path (Perseverance) ----
           * The GLB bbox includes the arm sticking out 2.4 m ahead, so bbox-based auto-scale/centering
           * (the generic path below) shrinks the rover ~22% and shifts it off its wheelbase.
           * Instead: keep native metres, split the wheels, then centre on the wheelbase and sit the wheels on y=0. */
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

            if (spec.facesPlusZ) {
              // flip 180° about Y so the NASA "front" points at -Z (sim forward); then centre on wheelbase
              model.rotation.y = Math.PI;
              model.position.set(mx, radius - axleY, mz);
            } else {
              model.position.set(-mx, radius - axleY, -mz);
            }
            root.add(model);
            root.updateMatrixWorld(true);

            const wheels = parts.map((p, i) => {
              // classify in rover-root space: forward = -Z, left = -X
              root.worldToLocal(p.pivot.getWorldPosition(tmp));
              const row = tmp.z < -0.5 ? 'F' : tmp.z > 0.5 ? 'R' : 'M';
              const side = tmp.x < 0 ? 'L' : 'R';
              return { name: row + side, pivot: p.pivot, hub: p.hub, steerable: row !== 'M' };
            });
            rig.wheels = finalizeWheels(root, wheels);
            rig.wheelRadius = radius;
            // GLB hubs sit under the 180°-flipped model: +rotation.x moves the tread toward -Z (forward)
            rig.spinSign = spec.facesPlusZ ? 1 : -1;
            rig.headlights = addGLBHeadlights(root);
          } else {
            console.warn('Wheels_objs not found; wheels will not animate.');
            root.add(model);
          }
        } else {
          // Generic path (Curiosity / MER / Ingenuity): auto-scale to a target size and ground the bbox
          model.updateMatrixWorld(true);
          const bbox = new THREE.Box3().setFromObject(model);
          const bSize = new THREE.Vector3();
          bbox.getSize(bSize);
          const maxDim = Math.max(bSize.x, bSize.y, bSize.z);

          const targetDim = spec.isHelicopter ? 1.3 : 3.2;
          const autoScale = (maxDim > 0) ? (targetDim / maxDim) : 1.0;
          model.scale.set(autoScale, autoScale, autoScale);

          model.updateMatrixWorld(true);
          const scaledBbox = new THREE.Box3().setFromObject(model);
          const scaledCenter = new THREE.Vector3();
          scaledBbox.getCenter(scaledCenter);
          model.position.x = -scaledCenter.x;
          model.position.z = -scaledCenter.z;
          model.position.y = -scaledBbox.min.y;
          root.add(model);
        }

        // Inspect and wire up animated nodes if available
        rig.rotor1 = model.getObjectByName('rotors_01');
        rig.rotor2 = model.getObjectByName('rotors_02');
        rig.armNode = model.getObjectByName('arm');
        rig.mastHead = model.getObjectByName('head');

        onLoaded(rig);
      },
      (progress) => {
        // progress callback
      },
      (err) => {
        console.warn('Failed loading GLB from ' + modelUrl + ', falling back to procedural rig.', err);
        const proc = spec.isHelicopter ? createProceduralIngenuity() : createProceduralRover(spec);
        if (onError) onError(err, proc);
        else onLoaded(proc);
      }
    );
  }

  return {
    SPECS: ROVER_SPECS,
    COMPONENT_INFO,
    createProceduralRover,
    createProceduralIngenuity,
    loadGLBModel
  };
})();
