// Headless phone-profile test: CPU throttled 4x, 390x844 @3x, software GL (SwiftShader) - pessimistic vs real phone GPUs.
// Usage: node scripts/perf_test.js <url> <screenshot_dir>
const { chromium } = require('playwright');
(async () => {
  const [url, shots] = process.argv.slice(2);
  const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 3, isMobile: true, hasTouch: true });
  const page = await ctx.newPage();
  const errors = []; page.on('pageerror', (e) => errors.push(String(e))); page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  const cdp = await ctx.newCDPSession(page); await cdp.send('Emulation.setCPUThrottlingRate', { rate: 4 });
  await page.goto(url); await page.waitForFunction('window.__ready === true', null, { timeout: 120000 });
  await page.waitForTimeout(3000);
  await page.screenshot({ path: `${shots}/chase.png` });
  const samples = [];
  await page.keyboard.down('w');
  for (let i = 0; i < 12; i++) { if (i === 5) await page.keyboard.down('a'); if (i === 8) await page.keyboard.up('a'); await page.waitForTimeout(1000); samples.push(await page.evaluate(() => window.__perf)); }
  await page.keyboard.up('w');
  await page.keyboard.press('p'); await page.waitForTimeout(1500); await page.screenshot({ path: `${shots}/provenance.png` });
  await page.keyboard.press('c'); await page.waitForTimeout(1500); await page.screenshot({ path: `${shots}/cockpit.png` });
  const prov = await page.evaluate(() => document.getElementById('provtxt').innerText);
  const s = samples.filter(Boolean);
  const fps = s.map((x) => x.fps), scale = s.map((x) => x.renderScale);
  console.log(JSON.stringify({ fps_mean: fps.reduce((a, b) => a + b, 0) / fps.length, fps_min: Math.min(...fps), render_scale_final: scale[scale.length - 1], samples: s.length, reticle: prov, errors }, null, 1));
  await browser.close();
})();
