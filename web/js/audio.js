/* Mars Sound Synthesizer via Web Audio API (zero audio file dependencies) */
window.MarsAudio = (() => {
  let ctx = null;
  let motorOsc = null, motorGain = null;
  let windNode = null, windGain = null;
  let rotorOsc1 = null, rotorOsc2 = null, rotorGain = null;
  let drillNode = null, drillGain = null;
  let enabled = false;

  function init() {
    if (ctx) return;
    try {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      ctx = new AudioContext();

      // --- Wind Noise Generator ---
      const bufferSize = ctx.sampleRate * 2;
      const noiseBuffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
      const output = noiseBuffer.getChannelData(0);
      for (let i = 0; i < bufferSize; i++) {
        output[i] = Math.random() * 2 - 1;
      }
      const whiteNoise = ctx.createBufferSource();
      whiteNoise.buffer = noiseBuffer;
      whiteNoise.loop = true;

      const windFilter = ctx.createBiquadFilter();
      windFilter.type = 'lowpass';
      windFilter.frequency.value = 320;

      windGain = ctx.createGain();
      windGain.gain.value = 0.08;

      whiteNoise.connect(windFilter);
      windFilter.connect(windGain);
      windGain.connect(ctx.destination);
      whiteNoise.start(0);

      // --- Rover Electric Motor Oscillator ---
      motorOsc = ctx.createOscillator();
      motorOsc.type = 'sawtooth';
      motorOsc.frequency.value = 65;

      const motorFilter = ctx.createBiquadFilter();
      motorFilter.type = 'lowpass';
      motorFilter.frequency.value = 180;

      motorGain = ctx.createGain();
      motorGain.gain.value = 0.0;

      motorOsc.connect(motorFilter);
      motorFilter.connect(motorGain);
      motorGain.connect(ctx.destination);
      motorOsc.start(0);

      // --- Ingenuity Rotor Drone ---
      rotorOsc1 = ctx.createOscillator();
      rotorOsc1.type = 'sawtooth';
      rotorOsc1.frequency.value = 140;

      rotorOsc2 = ctx.createOscillator();
      rotorOsc2.type = 'square';
      rotorOsc2.frequency.value = 280;

      const rotorFilter = ctx.createBiquadFilter();
      rotorFilter.type = 'bandpass';
      rotorFilter.frequency.value = 350;

      rotorGain = ctx.createGain();
      rotorGain.gain.value = 0.0;

      rotorOsc1.connect(rotorFilter);
      rotorOsc2.connect(rotorFilter);
      rotorFilter.connect(rotorGain);
      rotorGain.connect(ctx.destination);
      rotorOsc1.start(0);
      rotorOsc2.start(0);

      // --- Drill Noise ---
      const drillBuffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
      const drillData = drillBuffer.getChannelData(0);
      for (let i = 0; i < bufferSize; i++) drillData[i] = (Math.random() * 2 - 1) * 0.7;
      const drillNoise = ctx.createBufferSource();
      drillNoise.buffer = drillBuffer;
      drillNoise.loop = true;

      const drillFilter = ctx.createBiquadFilter();
      drillFilter.type = 'bandpass';
      drillFilter.frequency.value = 850;

      drillGain = ctx.createGain();
      drillGain.gain.value = 0.0;

      drillNoise.connect(drillFilter);
      drillFilter.connect(drillGain);
      drillGain.connect(ctx.destination);
      drillNoise.start(0);

      enabled = true;
    } catch (e) {
      console.warn('Web Audio not available:', e);
    }
  }

  function playBeep(freq = 880, duration = 0.08) {
    if (!ctx) return;
    try {
      if (ctx.state === 'suspended') ctx.resume();
      const osc = ctx.createOscillator();
      const g = ctx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(freq, ctx.currentTime);
      g.gain.setValueAtTime(0.08, ctx.currentTime);
      g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration);
      osc.connect(g);
      g.connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + duration);
    } catch (e) {}
  }

  function updateRoverSpeed(speed) {
    if (!ctx || !enabled) return;
    const absSpeed = Math.abs(speed);
    if (absSpeed > 0.02) {
      motorGain.gain.setTargetAtTime(Math.min(0.12, 0.02 + absSpeed * 0.05), ctx.currentTime, 0.05);
      motorOsc.frequency.setTargetAtTime(55 + absSpeed * 45, ctx.currentTime, 0.05);
    } else {
      motorGain.gain.setTargetAtTime(0.0, ctx.currentTime, 0.1);
    }
  }

  function updateHelicopter(rpmNorm, isAirborne) {
    if (!ctx || !enabled) return;
    if (rpmNorm > 0.05) {
      rotorGain.gain.setTargetAtTime(Math.min(0.2, rpmNorm * (isAirborne ? 0.18 : 0.08)), ctx.currentTime, 0.05);
      rotorOsc1.frequency.setTargetAtTime(90 + rpmNorm * 130, ctx.currentTime, 0.05);
      rotorOsc2.frequency.setTargetAtTime(180 + rpmNorm * 260, ctx.currentTime, 0.05);
    } else {
      rotorGain.gain.setTargetAtTime(0.0, ctx.currentTime, 0.1);
    }
  }

  function setDrilling(active) {
    if (!ctx || !enabled) return;
    drillGain.gain.setTargetAtTime(active ? 0.16 : 0.0, ctx.currentTime, 0.08);
  }

  function toggle() {
    if (!ctx) {
      init();
      return true;
    }
    if (ctx.state === 'suspended') {
      ctx.resume();
      enabled = true;
      return true;
    } else if (ctx.state === 'running') {
      ctx.suspend();
      enabled = false;
      return false;
    }
    return enabled;
  }

  return {
    init,
    toggle,
    isEnabled: () => enabled && ctx && ctx.state === 'running',
    playBeep,
    updateRoverSpeed,
    updateHelicopter,
    setDrilling
  };
})();
