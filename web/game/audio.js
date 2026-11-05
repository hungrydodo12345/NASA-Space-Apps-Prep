/* Mars Ambient, Rover Motor & Flight Audio Synthesizer (Web Audio API)
 * Zero external audio files required. Real-time procedural synthesis.
 */
let ctx = null;
let masterGain = null;
let enabled = false;

// Motor sound nodes
let motorOsc = null;
let motorGain = null;

// Wind ambience nodes
let windSource = null;
let windFilter = null;
let windGain = null;

// Drill sound nodes
let drillSource = null;
let drillFilter = null;
let drillGain = null;

// Helicopter rotor drone
let heliOsc1 = null;
let heliOsc2 = null;
let heliGain = null;

export const MarsAudio = {
  init() {
    if (ctx) return;
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (!AudioContext) return;
    ctx = new AudioContext();

    masterGain = ctx.createGain();
    masterGain.gain.value = 0.5;
    masterGain.connect(ctx.destination);

    // 1. Rover Electric Motor Hum
    motorOsc = ctx.createOscillator();
    motorOsc.type = 'sawtooth';
    motorOsc.frequency.value = 45; // base idle hum (Hz)

    const motorFilter = ctx.createBiquadFilter();
    motorFilter.type = 'lowpass';
    motorFilter.frequency.value = 220;

    motorGain = ctx.createGain();
    motorGain.gain.value = 0.0;

    motorOsc.connect(motorFilter);
    motorFilter.connect(motorGain);
    motorGain.connect(masterGain);
    motorOsc.start();

    // 2. Martian Thin-Atmosphere Wind Ambience (Pinkish noise)
    const bufSize = ctx.sampleRate * 2;
    const noiseBuf = ctx.createBuffer(1, bufSize, ctx.sampleRate);
    const output = noiseBuf.getChannelData(0);
    let b0 = 0, b1 = 0, b2 = 0;
    for (let i = 0; i < bufSize; i++) {
      const white = Math.random() * 2 - 1;
      b0 = 0.99886 * b0 + white * 0.0555179;
      b1 = 0.99332 * b1 + white * 0.0750759;
      b2 = 0.96900 * b2 + white * 0.1538520;
      output[i] = (b0 + b1 + b2) * 0.08;
    }

    windSource = ctx.createBufferSource();
    windSource.buffer = noiseBuf;
    windSource.loop = true;

    windFilter = ctx.createBiquadFilter();
    windFilter.type = 'bandpass';
    windFilter.frequency.value = 320;
    windFilter.Q.value = 1.2;

    windGain = ctx.createGain();
    windGain.gain.value = 0.08;

    windSource.connect(windFilter);
    windFilter.connect(windGain);
    windGain.connect(masterGain);
    windSource.start();

    // 3. Rotary-Percussive Rock Drill Noise
    const drillBuf = ctx.createBuffer(1, bufSize, ctx.sampleRate);
    const dOut = drillBuf.getChannelData(0);
    for (let i = 0; i < bufSize; i++) {
      dOut[i] = (Math.random() * 2 - 1) * 0.35;
    }
    drillSource = ctx.createBufferSource();
    drillSource.buffer = drillBuf;
    drillSource.loop = true;

    drillFilter = ctx.createBiquadFilter();
    drillFilter.type = 'bandpass';
    drillFilter.frequency.value = 1100;
    drillFilter.Q.value = 3.5;

    drillGain = ctx.createGain();
    drillGain.gain.value = 0.0;

    drillSource.connect(drillFilter);
    drillFilter.connect(drillGain);
    drillGain.connect(masterGain);
    drillSource.start();

    // 4. Ingenuity Counter-Rotating Twin Rotors Drone
    heliOsc1 = ctx.createOscillator();
    heliOsc1.type = 'sawtooth';
    heliOsc1.frequency.value = 80;

    heliOsc2 = ctx.createOscillator();
    heliOsc2.type = 'sawtooth';
    heliOsc2.frequency.value = 82;

    const heliFilter = ctx.createBiquadFilter();
    heliFilter.type = 'lowpass';
    heliFilter.frequency.value = 380;

    heliGain = ctx.createGain();
    heliGain.gain.value = 0.0;

    heliOsc1.connect(heliFilter);
    heliOsc2.connect(heliFilter);
    heliFilter.connect(heliGain);
    heliGain.connect(masterGain);
    heliOsc1.start();
    heliOsc2.start();
  },

  toggle() {
    if (!ctx) this.init();
    if (!ctx) return false;
    if (ctx.state === 'suspended') ctx.resume();
    enabled = !enabled;
    if (masterGain) {
      masterGain.gain.setTargetAtTime(enabled ? 0.5 : 0.0, ctx.currentTime, 0.05);
    }
    return enabled;
  },

  isEnabled() {
    return enabled;
  },

  updateRoverSpeed(speedMps) {
    if (!ctx || !enabled || !motorGain) return;
    const absSpeed = Math.abs(speedMps);
    const targetFreq = 45 + Math.min(180, absSpeed * 45);
    const targetGain = absSpeed > 0.02 ? Math.min(0.22, 0.04 + absSpeed * 0.06) : 0.01;
    motorOsc.frequency.setTargetAtTime(targetFreq, ctx.currentTime, 0.08);
    motorGain.gain.setTargetAtTime(targetGain, ctx.currentTime, 0.08);
  },

  setDrilling(isDrilling) {
    if (!ctx || !enabled || !drillGain) return;
    drillGain.gain.setTargetAtTime(isDrilling ? 0.35 : 0.0, ctx.currentTime, 0.05);
  },

  updateHelicopter(rpmNorm, inFlight) {
    if (!ctx || !enabled || !heliGain) return;
    const f1 = 40 + rpmNorm * 180;
    const f2 = f1 * 1.025;
    heliOsc1.frequency.setTargetAtTime(f1, ctx.currentTime, 0.05);
    heliOsc2.frequency.setTargetAtTime(f2, ctx.currentTime, 0.05);
    const g = rpmNorm > 0.1 ? (inFlight ? 0.32 : 0.15 * rpmNorm) : 0.0;
    heliGain.gain.setTargetAtTime(g, ctx.currentTime, 0.08);
  },

  playBeep(freq = 880, duration = 0.1) {
    if (!ctx || !enabled) return;
    const osc = ctx.createOscillator();
    const g = ctx.createGain();
    osc.type = 'sine';
    osc.frequency.value = freq;
    g.gain.setValueAtTime(0.12, ctx.currentTime);
    g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration);
    osc.connect(g);
    g.connect(masterGain);
    osc.start();
    osc.stop(ctx.currentTime + duration);
  }
};
