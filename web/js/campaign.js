/* Gamified Rover Character Selection, Historic Sol Campaigns & Free-World Mars Unlock */
window.MarsCampaign = (() => {
  'use strict';

  const STORAGE_KEY = 'jezero_campaign_progress_v2';

  // Historic campaigns per vehicle with real NASA mission milestones
  const CAMPAIGNS = {
    perseverance: {
      roverId: 'perseverance',
      title: 'Perseverance — Jezero Crater Campaign',
      startCoord: { x: -6.13, z: -142.42, yaw: -2.28 },
      milestones: [
        {
          id: 'sol_13',
          sol: 13,
          title: 'Octavia E. Butler Landing Checkout',
          target: { x: -6.13, z: -145.42, r: 8.0 },
          objective: 'Drive rover 6 meters forward to complete initial wheel and instrument checkout.',
          actionRequired: 'drive',
          completedText: 'Sol 13: All 6 wheels functional. Navcam panoramas confirmed pristine terrain at Octavia E. Butler landing site.'
        },
        {
          id: 'sol_31',
          sol: 31,
          title: 'Flight Zone Alpha Reconnaissance',
          target: { x: 19.4, z: -178.18, r: 10.0 },
          objective: 'Navigate south-east to scout flat terrain for Ingenuity flight airfield.',
          actionRequired: 'drive',
          completedText: 'Sol 31: Suitable airfield identified at Wright Brothers Field. Terrain roughness under 3 cm.'
        },
        {
          id: 'sol_65',
          sol: 65,
          title: 'Citadelle Bedrock Sample Acquisition',
          target: { x: 25.5, z: -120.0, r: 12.0 },
          objective: 'Deploy 5-DOF Robotic Arm and perform coring drill operation on bedrock outcrop.',
          actionRequired: 'drill',
          completedText: 'Sol 65: Core sample sealed in titanium tube #3! Igneous basaltic rock confirmed by PIXL.'
        },
        {
          id: 'sol_104',
          sol: 104,
          title: 'Delta Front Traverse',
          target: { x: -77.2, z: 116.9, r: 14.0 },
          objective: 'Trek toward the ancient Jezero river delta, avoiding steep sand ripples.',
          actionRequired: 'drive',
          completedText: 'Sol 104: Over 1.5 km traversed! Sedimentary delta deposits reached.'
        }
      ]
    },

    curiosity: {
      roverId: 'curiosity',
      title: 'Curiosity — Gale Crater Ancient Lake Campaign',
      startCoord: { x: 0.0, z: -80.0, yaw: 0.0 },
      milestones: [
        {
          id: 'msl_yellowknife',
          sol: 56,
          title: 'Yellowknife Bay Mudstone',
          target: { x: 28.0, z: -55.0, r: 10.0 },
          objective: 'Drive to Yellowknife Bay lowlands to inspect clay-bearing mudstones.',
          actionRequired: 'drive',
          completedText: 'Sol 56: Mudstone confirmed ancient fresh-water lake that persisted for millions of years.'
        },
        {
          id: 'msl_drill',
          sol: 182,
          title: 'John Klein Core Drilling',
          target: { x: 45.0, z: -20.0, r: 10.0 },
          objective: 'Deploy robotic arm and drill grey subsurface tailings.',
          actionRequired: 'drill',
          completedText: 'Sol 182: Discovered carbon, hydrogen, oxygen, phosphorus, and sulfur—the building blocks of life!'
        },
        {
          id: 'msl_dunes',
          sol: 388,
          title: 'Bagnold Dunes Crossing',
          target: { x: 90.0, z: 65.0, r: 14.0 },
          objective: 'Navigate through active Martian dark basaltic sand dunes toward Mount Sharp.',
          actionRequired: 'drive',
          completedText: 'Sol 388: First in-situ investigation of extraterrestrial active dunes completed.'
        }
      ]
    },

    opportunity: {
      roverId: 'opportunity',
      title: 'Opportunity — Meridiani Planum Marathon Campaign',
      startCoord: { x: -40.0, z: -100.0, yaw: 1.57 },
      milestones: [
        {
          id: 'mer_eagle',
          sol: 1,
          title: 'Hole-in-One at Eagle Crater',
          target: { x: -35.0, z: -90.0, r: 8.0 },
          objective: 'Examine exposed bedrock layers in Eagle Crater wall.',
          actionRequired: 'drive',
          completedText: 'Sol 1: Discovered hematite spherules ("blueberries") proving past liquid water!'
        },
        {
          id: 'mer_endurance',
          sol: 134,
          title: 'Endurance Crater Deep Incline',
          target: { x: -10.0, z: -20.0, r: 12.0 },
          objective: 'Descend 18-degree crater slopes to analyze layered stratigraphy.',
          actionRequired: 'drive',
          completedText: 'Sol 134: Successfully climbed back out of Endurance Crater on Rocker-Bogie drive.'
        },
        {
          id: 'mer_marathon',
          sol: 3968,
          title: 'Olympic Marathon Distance (42.195 km)',
          target: { x: 50.0, z: 90.0, r: 14.0 },
          objective: 'Cross the finish line of the legendary 42.2 km Martian marathon traverse.',
          actionRequired: 'drive',
          completedText: 'Sol 3968: Record broken! First vehicle to drive a full marathon distance on another planet.'
        }
      ]
    },

    spirit: {
      roverId: 'spirit',
      title: 'Spirit — Gusev Crater & Columbia Hills Campaign',
      startCoord: { x: -80.0, z: 40.0, yaw: -0.8 },
      milestones: [
        {
          id: 'spirit_bonneville',
          sol: 66,
          title: 'Bonneville Crater Rim Survey',
          target: { x: -60.0, z: 15.0, r: 10.0 },
          objective: 'Trek to the rim of 200 m wide Bonneville crater and examine ejecta blocks.',
          actionRequired: 'drive',
          completedText: 'Sol 66: High-resolution pan-cam mosaic acquired from the crater rim.'
        },
        {
          id: 'spirit_husband_hill',
          sol: 582,
          title: 'Husband Hill Summit Ascent',
          target: { x: 0.0, z: -40.0, r: 12.0 },
          objective: 'Climb 106 meters up the steep summit of Husband Hill in the Columbia Hills.',
          actionRequired: 'drive',
          completedText: 'Sol 582: Summit reached! Spectacular 360-degree vista across Gusev Crater floor.'
        },
        {
          id: 'spirit_silica',
          sol: 1150,
          title: 'Home Plate Hydrothermal Silica',
          target: { x: 35.0, z: -10.0, r: 10.0 },
          objective: 'Inspect bright soil deposits churned up by the stalled right front wheel.',
          actionRequired: 'drill',
          completedText: 'Sol 1150: 90% pure silica discovered—evidence of ancient explosive volcanic fumaroles or hot springs!'
        }
      ]
    },

    ingenuity: {
      roverId: 'ingenuity',
      title: 'Ingenuity — Mars Aerial Scout Campaign',
      startCoord: { x: 19.4, z: -178.18, yaw: 0.0 },
      isHelicopter: true,
      milestones: [
        {
          id: 'flight_1',
          sol: 58,
          title: 'Flight 1: Historic First Powered Flight',
          target: { x: 19.4, z: -178.18, r: 8.0 },
          objective: 'Spool dual counter-rotating rotors, ascend to 3 meters altitude, hover for 30s, and land.',
          actionRequired: 'flight',
          completedText: 'Flight 1 Successful! First powered, controlled flight on another planet. Wright Brothers moment!'
        },
        {
          id: 'flight_26',
          sol: 414,
          title: 'Flight 26: Parachute & Backshell Survey',
          target: { x: -30.0, z: -110.0, r: 14.0 },
          objective: 'Fly 8 meters altitude over the Perseverance backshell and parachute debris field to acquire ground-truth photos.',
          actionRequired: 'flight',
          completedText: 'Flight 26: 3 cm/px imagery of the backshell acquired! Used for ground-truthing orbital super-resolution.'
        },
        {
          id: 'flight_72',
          sol: 1000,
          title: 'Long-Range Airfield Scout Traverse',
          target: { x: 80.0, z: -30.0, r: 15.0 },
          objective: 'Scout 500 meters ahead over sandy ridges to map safe driving paths for Perseverance.',
          actionRequired: 'flight',
          completedText: 'Airfield Scout Complete! 72 total flights logged, covering over 17 kilometers of Martian sky.'
        }
      ]
    }
  };

  let activeRoverId = 'perseverance';
  let currentMilestoneIdx = 0;
  let freeRoamUnlocked = false;
  let beaconMesh = null;

  function loadSavedProgress() {
    try {
      const data = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
      if (data.freeRoamUnlocked) freeRoamUnlocked = true;
      return data;
    } catch (e) {
      return {};
    }
  }

  function saveProgress() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({
        activeRoverId,
        freeRoamUnlocked
      }));
    } catch (e) {}
  }

  function init(scene) {
    loadSavedProgress();

    // Create a visual holographic pulsing waypoint beacon
    const beaconGroup = new THREE.Group();
    beaconGroup.name = 'Campaign_Beacon';

    const ringGeo = new THREE.RingGeometry(0.8, 1.4, 32);
    ringGeo.rotateX(-Math.PI / 2);
    const ringMat = new THREE.MeshBasicMaterial({ color: 0x5ec27a, side: THREE.DoubleSide, transparent: true, opacity: 0.85 });
    const ring = new THREE.Mesh(ringGeo, ringMat);

    const beamGeo = new THREE.CylinderGeometry(0.08, 0.4, 6.0, 16);
    beamGeo.translate(0, 3.0, 0);
    const beamMat = new THREE.MeshBasicMaterial({ color: 0x5ec27a, transparent: true, opacity: 0.45, side: THREE.DoubleSide });
    const beam = new THREE.Mesh(beamGeo, beamMat);

    beaconGroup.add(ring);
    beaconGroup.add(beam);
    beaconGroup.visible = false;
    scene.add(beaconGroup);
    beaconMesh = beaconGroup;
  }

  function selectRover(roverId) {
    if (CAMPAIGNS[roverId]) {
      activeRoverId = roverId;
      currentMilestoneIdx = 0;
      saveProgress();
    }
    return getActiveCampaign();
  }

  function getActiveCampaign() {
    return CAMPAIGNS[activeRoverId] || CAMPAIGNS.perseverance;
  }

  function getCurrentMilestone() {
    const c = getActiveCampaign();
    if (currentMilestoneIdx < c.milestones.length) {
      return c.milestones[currentMilestoneIdx];
    }
    return null;
  }

  /**
   * Evaluates distance to target waypoint and tests milestone completion.
   */
  function update(dt, roverPos, heightAt, onMilestoneComplete, onAllCompleted) {
    const m = getCurrentMilestone();

    if (!m) {
      if (beaconMesh) beaconMesh.visible = false;
      return null;
    }

    // Position beacon at milestone target
    if (beaconMesh) {
      const by = heightAt(m.target.x, m.target.z);
      beaconMesh.position.set(m.target.x, by + 0.1, m.target.z);
      beaconMesh.visible = true;

      // Pulse animation
      const scale = 1.0 + Math.sin(Date.now() * 0.005) * 0.15;
      beaconMesh.children[0].scale.set(scale, scale, scale);
    }

    const dx = roverPos.x - m.target.x;
    const dz = roverPos.z - m.target.z;
    const dist = Math.sqrt(dx * dx + dz * dz);

    const isNearTarget = dist <= m.target.r;

    return {
      milestone: m,
      distance: dist,
      isNearTarget,
      totalMilestones: getActiveCampaign().milestones.length,
      currentNumber: currentMilestoneIdx + 1
    };
  }

  /**
   * Action trigger (e.g. drilling action executed or flight reached waypoint).
   */
  function checkAction(actionType, roverPos, onComplete, onAllCompleted) {
    const m = getCurrentMilestone();
    if (!m) return false;

    const dx = roverPos.x - m.target.x;
    const dz = roverPos.z - m.target.z;
    const dist = Math.sqrt(dx * dx + dz * dz);

    if (dist <= m.target.r && (m.actionRequired === actionType || m.actionRequired === 'drive')) {
      currentMilestoneIdx++;

      if (window.MarsAudio) window.MarsAudio.playBeep(1200, 0.25);

      if (currentMilestoneIdx >= getActiveCampaign().milestones.length) {
        freeRoamUnlocked = true;
        saveProgress();
        if (onAllCompleted) onAllCompleted(getActiveCampaign());
      } else {
        if (onComplete) onComplete(m);
      }
      return true;
    }
    return false;
  }

  function unlockFreeRoam() {
    freeRoamUnlocked = true;
    saveProgress();
  }

  return {
    CAMPAIGNS,
    init,
    selectRover,
    getActiveCampaign,
    getCurrentMilestone,
    update,
    checkAction,
    unlockFreeRoam,
    isFreeRoamUnlocked: () => freeRoamUnlocked,
    getActiveRoverId: () => activeRoverId
  };
})();
