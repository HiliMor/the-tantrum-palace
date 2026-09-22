import * as THREE from 'three/webgpu';
import {
  Fn, If, instanceIndex, instancedArray, uniform, float, vec2, vec3, vec4, color,
  hash, mix, smoothstep, length, sin, cos, pow, clamp, max, uv, varying,
  positionLocal, screenUV, pass, mx_noise_vec3, mx_noise_float, time,
} from 'three/tsl';
import { bloom } from 'three/addons/tsl/display/BloomNode.js';
import { rgbShift } from 'three/addons/tsl/display/RGBShiftNode.js';

// ---------------------------------------------------------------------------
// Painting layout (world units). Landmarks are measured on the original photo
// as normalized (u, v) with v going down from the top edge.
// ---------------------------------------------------------------------------
const IMG_ASPECT = 2268 / 4032;
const H = 10;
const W = H * IMG_ASPECT;
const BOTTOM = -H / 2;

const LANDMARKS = {
  eyeL: [0.449, 0.2],
  eyeR: [0.711, 0.195],
  mouth: [0.613, 0.29],
};

const uvToWorld = (u, v) => [(u - 0.5) * W, (0.5 - v) * H];
const smooth = (a, b, x) => {
  const t = Math.min(Math.max((x - a) / (b - a), 0), 1);
  return t * t * (3 - 2 * t);
};

// Fake depth: face bulges out as a dome, the dollhouse sits forward like a
// diorama, the orange backdrop gets pushed back.
function depthAt(u, v, r, g, b) {
  let z = 0;
  const fx = (u - 0.6) / 0.36;
  const fy = (v - 0.2) / 0.24;
  const fd = fx * fx + fy * fy;
  if (fd < 1) z += 0.9 * Math.sqrt(1 - fd);
  z += smooth(0.3, 0.95, v) * 1.3;
  const mx = Math.max(r, g, b);
  const sat = mx > 0 ? (mx - Math.min(r, g, b)) / mx : 0;
  if (r > 0.75 && sat > 0.55 && b < 0.35 && v < 0.55) z -= 0.7;
  z += (0.3 * r + 0.59 * g + 0.11 * b - 0.5) * 0.3;
  return z;
}

function faceMask(u, v) {
  const fx = (u - 0.6) / 0.36;
  const fy = (v - 0.2) / 0.26;
  return 1 - smooth(0.6, 1.0, Math.sqrt(fx * fx + fy * fy));
}

const loadImage = (src) => new Promise((resolve, reject) => {
  const img = new Image();
  img.onload = () => resolve(img);
  img.onerror = reject;
  img.src = src;
});

// ---------------------------------------------------------------------------
// Renderer / scene
// ---------------------------------------------------------------------------
const renderer = new THREE.WebGPURenderer({ antialias: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.prepend(renderer.domElement);
await renderer.init();

const isWebGPU = renderer.backend.isWebGPUBackend === true;
document.getElementById('badge').innerHTML = isWebGPU
  ? 'rendered with <b>WebGPU</b> compute'
  : 'WebGPU unavailable &middot; <b>WebGL2</b> fallback';

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(32, window.innerWidth / window.innerHeight, 0.1, 200);
let camDist = 20;

{
  const d = screenUV.sub(vec2(0.5, 0.42)).length();
  const glow = smoothstep(0.9, 0.0, d);
  scene.backgroundNode = mix(color(0x07010a), color(0x3a0620), glow.mul(glow))
    .add(mx_noise_float(vec3(screenUV.mul(3), time.mul(0.1))).mul(0.015));
}

scene.add(new THREE.HemisphereLight(0xfff0e0, 0x401030, 2.2));
const sun = new THREE.DirectionalLight(0xffffff, 2.5);
sun.position.set(3, 6, 8);
scene.add(sun);

// ---------------------------------------------------------------------------
// Shared uniforms
// ---------------------------------------------------------------------------
const uDt = uniform(0);
const uTime = uniform(0);
const uMouse = uniform(new THREE.Vector3(0, 0, 0));
const uMouseActive = uniform(0);
const uRadius = uniform(0.9);
const uTantrum = uniform(0);
const uBlast = uniform(0);
const uAssemble = uniform(0);
const uWater = uniform(BOTTOM - 0.4);
const uSpawn = uniform(0);
const uSeed = uniform(0);
const uDiorama = uniform(0); // 0 = flat painting, 1 = pop-up diorama
const uUnfold = uniform(1); // pop-up animation progress

const [mx, my] = uvToWorld(...LANDMARKS.mouth);
const uMouth = uniform(new THREE.Vector3(mx, my, 1));

// ---------------------------------------------------------------------------
// Painting → GPU particles
// ---------------------------------------------------------------------------
const [img, depthImg] = await Promise.all([
  loadImage(`${import.meta.env.BASE_URL}painting.jpg`),
  loadImage(`${import.meta.env.BASE_URL}depth.png`),
]);
const COLS = isWebGPU ? 360 : 200;
const ROWS = Math.round(COLS / IMG_ASPECT);
const SPACING = W / COLS;

function sampleImage(image) {
  const cvs = document.createElement('canvas');
  cvs.width = COLS;
  cvs.height = ROWS;
  const ctx = cvs.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(image, 0, 0, COLS, ROWS);
  return ctx.getImageData(0, 0, COLS, ROWS).data;
}
const pixels = sampleImage(img);
// Depth Anything V2 map of the painting (brighter = closer), precomputed offline.
const depthPx = sampleImage(depthImg);

const toLinear = new Float32Array(256);
for (let i = 0; i < 256; i++) {
  const c = i / 255;
  toLinear[i] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

// Diorama depth: the AI depth, partly snapped into terraces so the layers read
// like thick cardboard cut-outs.
const DIO_DEPTH = 5.2;
const dioZ = new Float32Array(COLS * ROWS);
const depth01 = new Float32Array(COLS * ROWS);
{
  let lo = 255, hi = 0;
  for (let i = 0; i < COLS * ROWS; i++) {
    lo = Math.min(lo, depthPx[i * 4]);
    hi = Math.max(hi, depthPx[i * 4]);
  }
  for (let i = 0; i < COLS * ROWS; i++) {
    const d = (depthPx[i * 4] - lo) / Math.max(1, hi - lo);
    const terraced = d + (Math.round(d * 7) / 7 - d) * 0.6;
    depth01[i] = d;
    dioZ[i] = (terraced - 0.35) * DIO_DEPTH;
  }
}
const dioZAt = (u, v) => dioZ[Math.min(ROWS - 1, (v * ROWS) | 0) * COLS + Math.min(COLS - 1, (u * COLS) | 0)];

// One particle per pixel, plus "side wall" particles wherever the diorama depth
// drops sharply, so cut-outs look solid instead of paper-thin from the side.
const homes = [];
const colors = [];
const dios = [];
for (let r = 0; r < ROWS; r++) {
  for (let c = 0; c < COLS; c++) {
    const i = r * COLS + c;
    const u = (c + 0.5) / COLS;
    const v = (r + 0.5) / ROWS;
    const R = pixels[i * 4], G = pixels[i * 4 + 1], B = pixels[i * 4 + 2];
    const [x, y] = uvToWorld(u, v);
    const z = depthAt(u, v, R / 255, G / 255, B / 255);
    const mask = faceMask(u, v);
    const lin = [toLinear[R], toLinear[G], toLinear[B]];
    homes.push(x, y, z, Math.random());
    colors.push(...lin, mask);
    dios.push(dioZ[i], 0, depth01[i], 0);

    let floor = dioZ[i];
    if (c > 0) floor = Math.min(floor, dioZ[i - 1]);
    if (c < COLS - 1) floor = Math.min(floor, dioZ[i + 1]);
    if (r > 0) floor = Math.min(floor, dioZ[i - COLS]);
    if (r < ROWS - 1) floor = Math.min(floor, dioZ[i + COLS]);
    const drop = dioZ[i] - floor;
    if (drop > SPACING * 2.5) {
      const slices = Math.min(40, Math.floor(drop / (SPACING * 1.2)));
      for (let k = 1; k <= slices; k++) {
        const shade = 0.42 + 0.18 * (1 - k / slices);
        homes.push(x, y, z - 0.002, Math.random());
        colors.push(lin[0] * shade, lin[1] * shade, lin[2] * shade, mask);
        dios.push(dioZ[i] - (drop * k) / (slices + 1), 1, depth01[i], 0);
      }
    }
  }
}
const COUNT = homes.length / 4;

const pHome = instancedArray(COUNT, 'vec4'); // xyz flat home, w random
const pPos = instancedArray(COUNT, 'vec4');
const pVel = instancedArray(COUNT, 'vec4');
const pCol = instancedArray(COUNT, 'vec4'); // rgb linear, w face mask
const pDio = instancedArray(COUNT, 'vec4'); // x diorama z, y side-wall flag, z depth 0..1

pHome.value.array.set(homes);
pCol.value.array.set(colors);
pDio.value.array.set(dios);
{
  // Start as an exploded nebula; the springs pull it together on load.
  const pos = pPos.value.array;
  for (let i = 0; i < COUNT; i++) {
    const th = Math.random() * Math.PI * 2;
    const ph = Math.acos(2 * Math.random() - 1);
    const rad = 14 + Math.random() * 30;
    pos.set([
      Math.sin(ph) * Math.cos(th) * rad,
      Math.sin(ph) * Math.sin(th) * rad,
      Math.cos(ph) * rad - 10,
      1,
    ], i * 4);
  }
}

const paintUpdate = Fn(() => {
  const pos = pPos.element(instanceIndex);
  const vel = pVel.element(instanceIndex);
  const home = pHome.element(instanceIndex);
  const mask = pCol.element(instanceIndex).w;
  const rnd = home.w;

  const p = pos.xyz.toVar();
  const v = vel.xyz.toVar();
  const dt = uDt;

  // Sobbing: the face heaves, and shakes its head in a tantrum.
  const sobSpeed = mix(5.0, 17.0, uTantrum);
  const sob = sin(uTime.mul(sobSpeed)).mul(mix(0.025, 0.07, uTantrum));
  const headShake = sin(uTime.mul(23.0)).mul(0.14).mul(uTantrum);
  // Diorama: swap in the AI depth, then unfold like a pop-up book, hinged at
  // the bottom edge. Far layers stand up first, the face pops up last.
  const dio = pDio.element(instanceIndex);
  const z = mix(home.z, dio.x, uDiorama);
  const up = smoothstep(0.0, 1.0, clamp(uUnfold.mul(1.8).sub(dio.z.mul(0.8)), 0.0, 1.0));
  const ang = float(1.0).sub(up).mul(Math.PI / 2).mul(uDiorama);
  const lift = home.y.sub(BOTTOM);
  const unfolded = vec3(home.x, lift.mul(cos(ang)).add(BOTTOM), z.sub(lift.mul(sin(ang))));
  const target = unfolded.add(vec3(headShake, sob, sob.mul(0.5)).mul(mask));

  // Spring home (weak while assembling or throwing a fit).
  const k = mix(20.0, 4.5, uTantrum).mul(mix(0.5, 1.5, rnd)).mul(uAssemble);
  v.addAssign(target.sub(p).mul(k).mul(dt));

  // Cursor melts paint: push away, then drip down.
  const d = p.xy.sub(uMouse.xy);
  const dist = length(d);
  const infl = smoothstep(uRadius, 0.0, dist).mul(uMouseActive);
  const push = d.div(dist.add(0.0001)).mul(10.0);
  v.addAssign(vec3(push.x, push.y.sub(34.0), mix(2.0, 9.0, rnd)).mul(infl).mul(dt));

  // Tantrum turbulence.
  const n = mx_noise_vec3(p.mul(0.55).add(vec3(0.0, 0.0, uTime.mul(0.5))));
  v.addAssign(n.mul(uTantrum).mul(9.0).mul(dt));

  // One-shot blast from the mouth.
  const fromMouth = p.sub(uMouth);
  const md = length(fromMouth);
  const falloff = smoothstep(11.0, 0.0, md).add(0.2);
  v.addAssign(fromMouth.div(md.add(0.15)).mul(uBlast).mul(mix(0.3, 1.7, rnd)).mul(falloff));

  // Underwater: everything gets wobbly and floaty.
  const under = smoothstep(uWater.add(0.05), uWater.sub(0.35), p.y);
  const bob = vec3(
    sin(p.y.mul(3.0).add(uTime.mul(2.0)).add(rnd.mul(6.28))).mul(1.6),
    cos(p.x.mul(2.5).add(uTime.mul(1.7))).mul(1.3),
    0.0,
  );
  v.addAssign(bob.mul(under).mul(dt));

  const damp = pow(mix(0.88, 0.94, uTantrum), dt.mul(60.0));
  v.mulAssign(damp);
  p.addAssign(v.mul(dt));

  pos.assign(vec4(p, 1.0));
  vel.assign(vec4(v, 0.0));
})().compute(COUNT);

const paintMat = new THREE.SpriteNodeMaterial();
{
  const pos = pPos.toAttribute();
  const speed = pVel.toAttribute().xyz.length();
  const base = pCol.toAttribute().xyz;
  const under = smoothstep(uWater.add(0.05), uWater.sub(0.35), pos.y);
  const glow = smoothstep(2.0, 9.0, speed);

  const wet = mix(base, base.mul(vec3(0.5, 0.78, 1.35)).add(vec3(0.0, 0.03, 0.09)), under.mul(0.75));
  const hot = wet.mul(1.6).add(vec3(0.5, 0.06, 0.2).mul(glow));
  paintMat.positionNode = pos.xyz;
  paintMat.colorNode = varying(mix(wet, hot, glow));
  const wall = pDio.toAttribute().y;
  paintMat.scaleNode = float(SPACING * 1.6).mul(mix(1.0, 0.5, glow)).mul(mix(1.0, uDiorama, wall));
}
const painting = new THREE.Sprite(paintMat);
painting.count = COUNT;
painting.frustumCulled = false;
scene.add(painting);

// ---------------------------------------------------------------------------
// Tears: two fountains that turn into lawn sprinklers during a tantrum
// ---------------------------------------------------------------------------
const TEARS = isWebGPU ? 60000 : 12000;
const tPos = instancedArray(TEARS, 'vec4'); // xyz, w life
const tVel = instancedArray(TEARS, 'vec4');

const eyeL = uvToWorld(...LANDMARKS.eyeL);
const eyeR = uvToWorld(...LANDMARKS.eyeR);
const uEyeL = uniform(new THREE.Vector3(eyeL[0], eyeL[1], 1.05));
const uEyeR = uniform(new THREE.Vector3(eyeR[0], eyeR[1], 0.95));
// Face landmarks move forward with the face in diorama mode.
const FLAT_Z = { eyeL: 1.05, eyeR: 0.95, mouth: 1 };
const DIO_Z = {
  eyeL: dioZAt(...LANDMARKS.eyeL) + 0.1,
  eyeR: dioZAt(...LANDMARKS.eyeR) + 0.1,
  mouth: dioZAt(...LANDMARKS.mouth) + 0.1,
};

const tearUpdate = Fn(() => {
  const pos = tPos.element(instanceIndex);
  const vel = tVel.element(instanceIndex);
  const id = instanceIndex.toFloat();
  const life = pos.w;

  If(life.lessThanEqual(0.0), () => {
    const r0 = hash(id.add(uSeed));
    If(r0.lessThan(uSpawn), () => {
      const r1 = hash(id.mul(1.37).add(uSeed).add(11.0));
      const r2 = hash(id.mul(2.11).add(uSeed).add(23.0));
      const r3 = hash(id.mul(0.73).add(uSeed).add(37.0));
      const side = id.mod(2.0);
      const dir = side.mul(2.0).sub(1.0);
      const eye = mix(uEyeL, uEyeR, side);

      const jitter = vec3(r1.sub(0.5).mul(0.12), r2.sub(0.5).mul(0.04), 0.0);
      pos.assign(vec4(eye.add(jitter), 1.0));

      const calm = vec3(dir.mul(0.04).mul(r1), r2.mul(-0.3), r3.mul(0.15).add(0.05));
      const fit = vec3(
        dir.mul(r1.mul(4.5).add(1.5)),
        r2.mul(4.0).add(2.0),
        r3.mul(2.5).add(0.5),
      );
      vel.assign(vec4(mix(calm, fit, uTantrum), 0.0));
    }).Else(() => {
      pos.assign(vec4(0.0, -100.0, 0.0, 0.0));
    });
  }).Else(() => {
    const v = vel.xyz.add(vec3(0.0, -5.5, 0.0).mul(uDt)).toVar();
    const p = pos.xyz.add(v.mul(uDt)).toVar();
    const l = life.sub(uDt.mul(0.3)).toVar();
    If(p.y.lessThan(uWater), () => { l.assign(0.0); });
    pos.assign(vec4(p, l));
    vel.assign(vec4(v, 0.0));
  });
})().compute(TEARS);

const tearMat = new THREE.SpriteNodeMaterial({
  transparent: true,
  depthWrite: false,
  blending: THREE.AdditiveBlending,
});
{
  const pos = tPos.toAttribute();
  const alive = smoothstep(0.0, 0.001, pos.w);
  const vel = tVel.toAttribute().xyz;
  tearMat.positionNode = pos.xyz;
  tearMat.scaleNode = vec2(0.035, clamp(vel.length().mul(0.02), 0.04, 0.14)).mul(alive);
  const d = uv().sub(0.5).length();
  const a = smoothstep(0.5, 0.0, d).mul(clamp(pos.w.mul(3.0), 0.0, 1.0)).mul(0.5);
  tearMat.colorNode = vec4(vec3(0.3, 0.7, 1.5), a);
}
const tears = new THREE.Sprite(tearMat);
tears.count = TEARS;
tears.frustumCulled = false;
scene.add(tears);

// ---------------------------------------------------------------------------
// The flood
// ---------------------------------------------------------------------------
const waveAt = (x, t, tantrum) =>
  Math.sin(x * 3 + t * 2.2) * 0.06 + Math.sin(x * 7.3 - t * 3.1) * 0.03 + Math.sin(x * 1.7 + t * 5) * 0.14 * tantrum;

function makeWater(z, tint, phase, opacity) {
  const geo = new THREE.PlaneGeometry(W * 8, 1, 480, 1);
  const mat = new THREE.MeshBasicNodeMaterial({ transparent: true, depthWrite: false, side: THREE.DoubleSide });
  const x = positionLocal.x;
  const t = uTime.add(phase);
  const top = uWater
    .add(sin(x.mul(3.0).add(t.mul(2.2))).mul(0.06))
    .add(sin(x.mul(7.3).sub(t.mul(3.1))).mul(0.03))
    .add(sin(x.mul(1.7).add(t.mul(5.0))).mul(0.14).mul(uTantrum));
  mat.positionNode = vec3(x, mix(float(BOTTOM - 3), top, uv().y), positionLocal.z);
  const surf = pow(uv().y, 40.0);
  const caustic = mx_noise_float(vec3(uv().mul(vec2(18.0, 3.0)), t.mul(0.6))).mul(0.5).add(0.5);
  const col = mix(tint, vec3(0.7, 0.95, 1.4), surf.add(caustic.mul(0.18)));
  mat.colorNode = vec4(col, mix(opacity, 0.95, surf));
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.z = z;
  mesh.frustumCulled = false;
  mesh.renderOrder = 2;
  scene.add(mesh);
  return mesh;
}
makeWater(-1.2, vec3(0.05, 0.12, 0.5), 1.3, 0.35);
makeWater(2.4, vec3(0.08, 0.3, 0.85), 0.0, 0.42);

// Rubber ducks. Obviously.
function makeDuck() {
  const g = new THREE.Group();
  const yellow = new THREE.MeshStandardNodeMaterial({ color: 0xffd21f, roughness: 0.35 });
  const orange = new THREE.MeshStandardNodeMaterial({ color: 0xff6a00, roughness: 0.4 });
  const black = new THREE.MeshStandardNodeMaterial({ color: 0x111111, roughness: 0.2 });
  const body = new THREE.Mesh(new THREE.SphereGeometry(0.24, 24, 16), yellow);
  body.scale.set(1.35, 0.85, 1);
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.15, 24, 16), yellow);
  head.position.set(0.2, 0.24, 0);
  const beak = new THREE.Mesh(new THREE.ConeGeometry(0.06, 0.14, 12), orange);
  beak.rotation.z = -Math.PI / 2;
  beak.position.set(0.37, 0.22, 0);
  const eye = new THREE.Mesh(new THREE.SphereGeometry(0.025, 8, 8), black);
  eye.position.set(0.27, 0.3, 0.11);
  const tail = new THREE.Mesh(new THREE.ConeGeometry(0.08, 0.18, 12), yellow);
  tail.rotation.z = Math.PI / 2.6;
  tail.position.set(-0.33, 0.1, 0);
  g.add(body, head, beak, eye, tail);
  return g;
}
const ducks = Array.from({ length: 5 }, (_, i) => {
  const d = makeDuck();
  d.userData = { x: (i / 4 - 0.5) * W * 0.9, speed: 0.15 + Math.random() * 0.25, flip: Math.random() < 0.5 ? -1 : 1, phase: Math.random() * 10 };
  d.scale.setScalar(0.9 + Math.random() * 0.4);
  d.position.z = 2.45;
  d.visible = false;
  scene.add(d);
  return d;
});

// ---------------------------------------------------------------------------
// Post: bloom + tantrum-driven chromatic split + vignette & grain
// ---------------------------------------------------------------------------
const uShift = uniform(0.0015);
const pipeline = new THREE.RenderPipeline(renderer);
{
  const scenePass = pass(scene, camera);
  const sceneColor = scenePass.getTextureNode('output');
  const bloomPass = bloom(sceneColor, 0.5, 0.4, 0.9);
  pipeline.userData = { bloomPass };
  const shifted = rgbShift(sceneColor.add(bloomPass), uShift, uTime.mul(3.0));
  const vig = smoothstep(1.15, 0.35, screenUV.sub(0.5).mul(vec2(1.2, 1.0)).length());
  // Grain is off in ?demo: it makes the recorded video huge.
  const grainAmt = new URLSearchParams(location.search).has('demo') ? 0 : 0.02;
  const grain = mx_noise_float(vec3(screenUV.mul(900.0), uTime.mul(40.0))).mul(grainAmt);
  pipeline.outputNode = vec4(shifted.rgb.mul(mix(0.45, 1.0, vig)).add(grain), 1.0);
}

// ---------------------------------------------------------------------------
// Sound: a formant-filtered sawtooth that goes "WAAAH"
// ---------------------------------------------------------------------------
const sound = {
  ctx: null, enabled: true, osc: null, gain: null,
  init() {
    if (this.ctx) return;
    // iOS mutes Web Audio when the ringer switch is on silent unless the page
    // declares itself as media playback.
    if (navigator.audioSession) navigator.audioSession.type = 'playback';
    const ac = new (window.AudioContext || window.webkitAudioContext)();
    this.ctx = ac;
    const osc = ac.createOscillator();
    osc.type = 'sawtooth';
    osc.frequency.value = 420;
    const vib = ac.createOscillator();
    vib.frequency.value = 7;
    const vibAmt = ac.createGain();
    vibAmt.gain.value = 14;
    vib.connect(vibAmt).connect(osc.frequency);
    const out = ac.createGain();
    out.gain.value = 0;
    // "aa" vowel formants
    [[850, 6, 1], [1250, 8, 0.6], [2800, 10, 0.25]].forEach(([f, q, g]) => {
      const bp = ac.createBiquadFilter();
      bp.type = 'bandpass';
      bp.frequency.value = f;
      bp.Q.value = q;
      const lvl = ac.createGain();
      lvl.gain.value = g;
      osc.connect(bp).connect(lvl).connect(out);
    });
    this.master = ac.createGain();
    this.master.connect(ac.destination);
    const comp = ac.createDynamicsCompressor();
    out.connect(comp).connect(this.master);
    osc.start();
    vib.start();
    this.osc = osc;
    this.gain = out;
  },
  // Mobile browsers only allow audio to start inside a gesture that counts as
  // user activation (on touch that is pointerup/touchend, not pointerdown), and
  // suspend the context again when the tab is backgrounded.
  unlock() {
    this.init();
    if (this.ctx.state !== 'running') this.ctx.resume().catch(() => {});
  },
  update(tantrum, t) {
    if (!this.ctx) return;
    const now = this.ctx.currentTime;
    // WAAH-aah-AAAH: a wailing contour that loops every ~1.6 s
    const cyc = (t % 1.6) / 1.6;
    const wail = cyc < 0.7 ? Math.sin((cyc / 0.7) * Math.PI) : Math.sin(((cyc - 0.7) / 0.3) * Math.PI) * 0.5;
    const f = 360 + wail * 260 + tantrum * 140;
    this.osc.frequency.setTargetAtTime(f, now, 0.04);
    // Between fits: a quiet, hiccuping whimper.
    const whimper = Math.pow(Math.max(0, Math.sin(t * 2.6)), 10) * 0.07 * (1 - tantrum) * (this.calm ? 0 : 1);
    const vol = this.enabled ? Math.max(0, tantrum - 0.05) * (0.25 + wail * 0.75) * 0.9 + whimper : 0;
    this.gain.gain.setTargetAtTime(vol, now, 0.05);
  },
  shh() {
    if (!this.ctx || !this.enabled) return;
    const ac = this.ctx;
    const len = ac.sampleRate * 1.4;
    const buf = ac.createBuffer(1, len, ac.sampleRate);
    const data = buf.getChannelData(0);
    for (let i = 0; i < len; i++) data[i] = (Math.random() * 2 - 1) * Math.sin((i / len) * Math.PI);
    const src = ac.createBufferSource();
    src.buffer = buf;
    const hp = ac.createBiquadFilter();
    hp.type = 'highpass';
    hp.frequency.value = 3000;
    const g = ac.createGain();
    g.gain.value = 0.18;
    src.connect(hp).connect(g).connect(this.master);
    src.start();
  },
  glug() {
    if (!this.ctx || !this.enabled) return;
    const ac = this.ctx;
    const t0 = ac.currentTime;
    for (let i = 0; i < 9; i++) {
      const o = ac.createOscillator();
      const g = ac.createGain();
      const t = t0 + i * 0.13 + Math.random() * 0.05;
      o.type = 'sine';
      o.frequency.setValueAtTime(180 + Math.random() * 260, t);
      o.frequency.exponentialRampToValueAtTime(700 + Math.random() * 500, t + 0.09);
      g.gain.setValueAtTime(0, t);
      g.gain.linearRampToValueAtTime(0.25, t + 0.01);
      g.gain.exponentialRampToValueAtTime(0.001, t + 0.12);
      o.connect(g).connect(this.master);
      o.start(t);
      o.stop(t + 0.15);
    }
  },
};

// Microphone: scream at your computer to feed the tantrum.
const mic = {
  analyser: null, data: null, level: 0, stream: null,
  async start() {
    sound.init();
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const src = sound.ctx.createMediaStreamSource(this.stream);
    this.analyser = sound.ctx.createAnalyser();
    this.analyser.fftSize = 1024;
    this.data = new Float32Array(this.analyser.fftSize);
    src.connect(this.analyser);
  },
  stop() {
    this.stream?.getTracks().forEach((t) => t.stop());
    this.analyser = null;
    this.stream = null;
    this.level = 0;
  },
  read() {
    if (!this.analyser) return 0;
    this.analyser.getFloatTimeDomainData(this.data);
    let sum = 0;
    for (let i = 0; i < this.data.length; i++) sum += this.data[i] * this.data[i];
    const rms = Math.sqrt(sum / this.data.length);
    this.level += (rms - this.level) * 0.3;
    return Math.min(Math.max((this.level - 0.03) * 9, 0), 1);
  },
};

// ---------------------------------------------------------------------------
// Interaction state
// ---------------------------------------------------------------------------
const state = {
  holding: false,
  tantrum: 0,
  calm: 0,
  water: BOTTOM - 0.4,
  liters: 0,
  drowned: 0,
  draining: 0,
  mouseActive: 0,
  pointer: new THREE.Vector2(),
  lastMove: -10,
  wahTimer: 0,
  micWasLoud: false,
  dio: 0,
  dioTarget: 0,
  unfold: 1,
};

const $ = (id) => document.getElementById(id);
const ui = {
  liters: $('liters'), flood: $('flood'), mood: $('mood'), drowned: $('drowned'),
  wahs: $('wahs'), banner: $('drown-banner'), tantrumBtn: $('btn-tantrum'),
};

const raycaster = new THREE.Raycaster();
const hitPlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), -0.6);
const hit = new THREE.Vector3();

function startTantrum() {
  sound.init();
  if (!state.holding) uBlast.value = 13;
  state.holding = true;
  state.calm = 0;
  ui.tantrumBtn.classList.add('on');
}
function stopTantrum() {
  state.holding = false;
  ui.tantrumBtn.classList.remove('on');
}
function pacify() {
  sound.init();
  sound.shh();
  state.calm = 4;
  state.draining = 0;
  stopTantrum();
}

const canvas = renderer.domElement;
window.addEventListener('pointermove', (e) => {
  state.pointer.set((e.clientX / window.innerWidth) * 2 - 1, -(e.clientY / window.innerHeight) * 2 + 1);
  state.lastMove = performance.now() / 1000;
});
canvas.addEventListener('pointerdown', (e) => {
  state.pointer.set((e.clientX / window.innerWidth) * 2 - 1, -(e.clientY / window.innerHeight) * 2 + 1);
  state.lastMove = performance.now() / 1000;
  startTantrum();
});
window.addEventListener('pointerup', stopTantrum);
for (const type of ['pointerdown', 'pointerup', 'touchend', 'click', 'keydown']) {
  window.addEventListener(type, () => sound.unlock(), { capture: true, passive: true });
}
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible' && sound.ctx) sound.ctx.resume().catch(() => {});
});
window.addEventListener('pointercancel', stopTantrum);
document.addEventListener('pointerleave', () => { state.lastMove = -10; });

ui.tantrumBtn.addEventListener('pointerdown', (e) => { e.preventDefault(); startTantrum(); });
$('btn-pacifier').addEventListener('click', pacify);
$('btn-sound').addEventListener('click', (e) => {
  sound.init();
  sound.enabled = !sound.enabled;
  e.currentTarget.textContent = sound.enabled ? '🔊' : '🔇';
});
// Diorama mode: fold everything flat, then pop it up in depth.
const dioBtn = $('btn-diorama');
function toggleDiorama(on = state.dioTarget === 0) {
  state.dioTarget = on ? 1 : 0;
  if (on) state.unfold = 0;
  dioBtn.classList.toggle('on', on);
  dioBtn.textContent = on ? '🖼 flatten' : '🎪 pop-up';
}
dioBtn.addEventListener('click', () => toggleDiorama());

$('btn-scream').addEventListener('click', async (e) => {
  const btn = e.currentTarget;
  if (mic.analyser) {
    mic.stop();
    btn.classList.remove('on');
    btn.textContent = '🎤 scream mode';
    return;
  }
  try {
    await mic.start();
    btn.classList.add('on');
    btn.textContent = '🎤 screaming…';
  } catch {
    btn.textContent = '🎤 mic denied 😢';
  }
});

window.addEventListener('keydown', (e) => {
  if (e.code === 'Space' && !e.repeat) { e.preventDefault(); startTantrum(); }
  if (e.code === 'KeyP') pacify();
  if (e.code === 'KeyD') toggleDiorama();
});
window.addEventListener('keyup', (e) => { if (e.code === 'Space') stopTantrum(); });

// ---------------------------------------------------------------------------
// "WAAAH" typography flying out of the mouth
// ---------------------------------------------------------------------------
const WAHS = ['WAAAH', 'WAAAAAAH', 'NOOO', 'MINE!', 'WHYYY', 'UNFAIR', 'I HATE BEDTIME', 'BWAAAH', 'NOT THE BLUE CUP', 'AAAAAA', 'HIC!', 'MOOOM'];
const mouthScreen = new THREE.Vector3();
function spawnWah(intensity) {
  mouthScreen.copy(uMouth.value).project(camera);
  const sx = (mouthScreen.x * 0.5 + 0.5) * window.innerWidth;
  const sy = (-mouthScreen.y * 0.5 + 0.5) * window.innerHeight;
  const el = document.createElement('div');
  el.className = 'wah';
  el.textContent = WAHS[(Math.random() * WAHS.length) | 0];
  const ang = Math.random() * Math.PI * 2;
  const dist = 180 + Math.random() * 380 * intensity;
  el.style.setProperty('--x0', `${sx}px`);
  el.style.setProperty('--y0', `${sy}px`);
  el.style.setProperty('--x1', `${sx + Math.cos(ang) * dist}px`);
  el.style.setProperty('--y1', `${sy + Math.sin(ang) * dist * 0.8}px`);
  el.style.setProperty('--r', `${(Math.random() - 0.5) * 50}deg`);
  el.style.fontSize = `${24 + Math.random() * 60 * intensity}px`;
  el.addEventListener('animationend', () => el.remove());
  ui.wahs.appendChild(el);
}

function moodText() {
  if (state.draining > 0) return 'glub glub';
  if (state.calm > 0) return 'soothed (temporarily)';
  if (state.tantrum > 0.75) return '☢ nuclear ☢';
  if (state.tantrum > 0.25) return 'escalating';
  if (state.water > BOTTOM + 3) return 'soggy & betrayed';
  if (state.water > BOTTOM) return 'damp. furious.';
  return 'mildly inconsolable';
}

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------
function resize() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  camera.aspect = w / h;
  const vFov = THREE.MathUtils.degToRad(camera.fov);
  const fitH = (H * 1.02) / 2 / Math.tan(vFov / 2);
  const fitW = (W * 1.12) / 2 / Math.tan(vFov / 2) / camera.aspect;
  camDist = Math.max(fitH, fitW) + 1.3;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}
window.addEventListener('resize', resize);
resize();

// ---------------------------------------------------------------------------
// ?demo: a scripted performance, used to record the showreel
// ---------------------------------------------------------------------------
const DEMO = new URLSearchParams(location.search).has('demo');
const DIORAMA_ON_LOAD = new URLSearchParams(location.search).has('diorama');
const DEMO_LENGTH = 26;
let floodBoost = 1;

// Shorter showreel for the pop-up diorama (?demo&diorama).
const DIORAMA_DEMO_LENGTH = 17;
function runDioramaDemo(t, now) {
  const look = (x, y) => { state.pointer.set(x, y); state.lastMove = -10; };
  if (t < 6.5) look(0, 0.05);
  else if (t < 12) look(Math.sin((t - 6.5) * 0.9) * 0.95, 0.15 + Math.sin(t * 0.6) * 0.2);
  else if (t < 14.5) { startTantrum(); look(0.55, 0.1); }
  else if (t < DIORAMA_DEMO_LENGTH) { stopTantrum(); look(0.3, 0); }
  else window.__demoDone = true;
}

function runDemo(t, now) {
  const hover = (x, y) => { state.pointer.set(x, y); state.lastMove = now; };
  const melt = (k) => hover(Math.sin(k * 1.3) * 0.2, Math.sin(k * 0.9) * 0.45 - 0.15);
  const hold = (on) => (on ? startTantrum() : stopTantrum());

  if (t < 3.2) state.pointer.set(0.35 * Math.sin(t * 0.4), 0.1);
  else if (t < 7.5) melt(t - 3.2);
  else if (t < 10) { hold(true); hover(Math.sin(t * 2) * 0.15, 0.2); }
  else if (t < 12.5) { hold(false); state.lastMove = -10; }
  else if (t < 20) { floodBoost = 2.4; hold(state.drowned === 0); hover(Math.sin(t * 1.7) * 0.12, 0.1); }
  else if (t < 21.5) { floodBoost = 1; hold(false); }
  else if (t < DEMO_LENGTH) {
    if (!demoPacified) { pacify(); demoPacified = true; }
    melt(t);
  } else window.__demoDone = true;
}
let demoPacified = false;

if (DEMO) {
  sound.init();
  window.__sound = sound;
  window.__demoLength = DEMO_LENGTH;
}

// ---------------------------------------------------------------------------
// Loop
// ---------------------------------------------------------------------------
const timer = new THREE.Timer();
const smoothPtr = new THREE.Vector2();
let elapsed = 0;
let frame = 0;

$('loader').classList.add('gone');

renderer.setAnimationLoop(() => {
  timer.update();
  const dt = Math.min(timer.getDelta(), 1 / 30);
  elapsed += dt;
  frame++;
  const now = performance.now() / 1000;
  if (DEMO) (DIORAMA_ON_LOAD ? runDioramaDemo : runDemo)(elapsed, now);

  // Tantrum level: button / space / pointer hold, or a real scream.
  const micLevel = mic.read();
  if (micLevel > 0.55 && !state.micWasLoud) uBlast.value = 9 + micLevel * 8;
  state.micWasLoud = micLevel > 0.55;
  const target = state.calm > 0 ? 0 : Math.max(state.holding ? 1 : 0, micLevel);
  state.tantrum += (target - state.tantrum) * (1 - Math.exp(-dt * (target > state.tantrum ? 5 : 1.6)));
  state.calm = Math.max(0, state.calm - dt);
  document.body.classList.toggle('tantrum', state.tantrum > 0.6);

  // Flood.
  if (state.draining > 0) {
    state.draining -= dt;
    state.water -= dt * 3.2;
  } else if (state.calm > 0) {
    state.water -= dt * 1.8;
  } else {
    state.water += dt * (0.03 + state.tantrum * 0.6) * floodBoost;
  }
  state.water = Math.max(state.water, BOTTOM - 0.4);
  const cryRate = state.calm > 0 ? 0.01 : 0.08 + state.tantrum * 2.8;
  state.liters += cryRate * dt;

  if (state.water > uMouth.value.y - 0.3 && state.draining <= 0) {
    state.drowned++;
    state.draining = 3.2;
    ui.banner.classList.add('show');
    sound.glug();
    setTimeout(() => ui.banner.classList.remove('show'), 2600);
    uBlast.value = 30;
  }

  // Cursor → world point on the painting.
  const hovering = now - state.lastMove < 1.2;
  state.mouseActive += ((hovering ? 1 : 0) - state.mouseActive) * (1 - Math.exp(-dt * 6));
  raycaster.setFromCamera(state.pointer, camera);
  if (raycaster.ray.intersectPlane(hitPlane, hit)) uMouse.value.copy(hit);

  // Diorama transition.
  if (DIORAMA_ON_LOAD && elapsed > 2.6 && !state.dioAuto) { state.dioAuto = true; toggleDiorama(true); }
  state.dio += (state.dioTarget - state.dio) * (1 - Math.exp(-dt * (state.dioTarget ? 12 : 3)));
  if (state.dioTarget) state.unfold = Math.min(1, state.unfold + dt * 0.42);
  for (const k of ['eyeL', 'eyeR']) {
    const u = k === 'eyeL' ? uEyeL : uEyeR;
    u.value.z = THREE.MathUtils.lerp(FLAT_Z[k], DIO_Z[k], state.dio * state.unfold);
  }
  uMouth.value.z = THREE.MathUtils.lerp(FLAT_Z.mouth, DIO_Z.mouth, state.dio * state.unfold);
  hitPlane.constant = -THREE.MathUtils.lerp(0.6, 1.4, state.dio);

  // Camera: orbit with the pointer (much wider in diorama mode) + tantrum shake.
  smoothPtr.lerp(state.pointer, 1 - Math.exp(-dt * 3));
  const shake = state.tantrum * 0.09;
  const yaw = smoothPtr.x * THREE.MathUtils.lerp(0.08, 0.75, state.dio) + Math.sin(elapsed * 0.25) * 0.12 * state.dio;
  const pitch = smoothPtr.y * THREE.MathUtils.lerp(0.05, 0.3, state.dio) + Math.sin(elapsed * 0.4) * 0.008 + 0.1 * state.dio;
  const focusZ = 0.8 * state.dio;
  camera.position.set(
    Math.sin(yaw) * Math.cos(pitch) * camDist + (Math.random() - 0.5) * shake,
    Math.sin(pitch) * camDist + (Math.random() - 0.5) * shake,
    focusZ + Math.cos(yaw) * Math.cos(pitch) * camDist,
  );
  camera.lookAt((Math.random() - 0.5) * shake * 0.5, 0, focusZ);

  // Uniforms.
  uDt.value = dt;
  uTime.value = elapsed;
  uTantrum.value = state.tantrum;
  uWater.value = state.water;
  uMouseActive.value = state.mouseActive;
  uAssemble.value = Math.min(1, 0.08 + elapsed * elapsed * 0.35);
  uDiorama.value = state.dio;
  uUnfold.value = state.unfold;
  uSpawn.value = state.calm > 0 ? 0.00015 : 0.0009 + state.tantrum * 0.006;
  uSeed.value = (frame * 7919) % 1000003;
  uShift.value = 0.001 + state.tantrum * 0.005;
  pipeline.userData.bloomPass.strength.value = 0.5 + state.tantrum * 0.9;

  // Ducks bob along the surface.
  for (const d of ducks) {
    const u = d.userData;
    u.x += u.speed * u.flip * dt * (1 + state.tantrum * 4);
    if (Math.abs(u.x) > W * 0.62) u.flip *= -1;
    const y = state.water + waveAt(u.x, elapsed, state.tantrum);
    const slope = (waveAt(u.x + 0.05, elapsed, state.tantrum) - waveAt(u.x - 0.05, elapsed, state.tantrum)) / 0.1;
    d.position.set(u.x, y - 0.02 + Math.sin(elapsed * 3 + u.phase) * 0.02, 2.45);
    d.rotation.z = Math.atan(slope) * 0.8 + Math.sin(elapsed * 2 + u.phase) * 0.08;
    d.rotation.y = u.flip > 0 ? 0 : Math.PI;
    d.visible = state.water > BOTTOM + 0.15;
  }

  // WAAAHs.
  state.wahTimer -= dt;
  if (state.tantrum > 0.35 && state.wahTimer <= 0) {
    spawnWah(state.tantrum);
    state.wahTimer = 0.28 - state.tantrum * 0.18;
  }
  sound.calm = state.calm > 0;
  sound.update(state.tantrum, elapsed);

  // HUD.
  if (frame % 4 === 0) {
    ui.liters.textContent = state.liters.toFixed(2);
    const pct = ((state.water - (BOTTOM - 0.4)) / (uMouth.value.y - 0.3 - (BOTTOM - 0.4))) * 100;
    ui.flood.style.width = `${Math.min(Math.max(pct, 0), 100)}%`;
    ui.mood.textContent = moodText();
    ui.drowned.textContent = state.drowned;
  }

  renderer.compute(paintUpdate);
  renderer.compute(tearUpdate);
  uBlast.value = 0;
  pipeline.render();
});
