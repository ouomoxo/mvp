import * as THREE from 'three';
import { OrbitControls } from '../vendor/OrbitControls.js';

/* ------------------------------------------------------------------ constants and state */
const N = 512;            // elevation grid
const G = 256;            // analysis grid
const SPAN = 16;          // scene units = km
const EXAG = 1.6;         // vertical exaggeration
const LIFT = 0.045;       // how far overlays float above the ground (scene units)
const $ = (s) => document.querySelector(s);
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

const state = {
  tool: 'look', edge: 'N', stealth: 0.7,
  objective: { x: 128, y: 128 }, posts: [], starts: [],
  layers: { sat: true, contour: true, vis: true, corr: true, iso: false, grid: false, hover: true, places: true },
  result: null, focus: -1,
  play: { active: false, playing: false, t: 0, T: 1 },
};

let meta, H, lcRaw, relief, renderer, scene, camera, controls, terrain, anaTex, worker;
let lineGroup, lines = [], markers = [], placeLabels = [], heads = [], events = [], shownEvents = -1;
const marksEl = $('#marks');

/* ------------------------------------------------------------------ terrain helpers */
function heightAt(u, v) {                 // metres, bilinear
  const x = clamp(u * N - 0.5, 0, N - 1.001), y = clamp(v * N - 0.5, 0, N - 1.001);
  const x0 = x | 0, y0 = y | 0, fx = x - x0, fy = y - y0, i = y0 * N + x0;
  return (H[i] * (1 - fx) + H[i + 1] * fx) * (1 - fy) + (H[i + N] * (1 - fx) + H[i + N + 1] * fx) * fy;
}
const yOf = (m) => (m - meta.hmin) / 1000 * EXAG;
const world = (u, v, lift = 0) => new THREE.Vector3((u - 0.5) * SPAN, yOf(heightAt(u, v)) + lift, (v - 0.5) * SPAN);
const cellUV = (x, y) => [(x + 0.5) / G, (y + 0.5) / G];
const cellWorld = (x, y, lift = LIFT) => world(...cellUV(x, y), lift);
function groundY(x, z) {
  const u = x / SPAN + 0.5, v = z / SPAN + 0.5;
  return (u < 0 || v < 0 || u > 1 || v > 1) ? -Infinity : yOf(heightAt(u, v));
}
const lonLat = (u, v) => [meta.west + u * (meta.east - meta.west), meta.north - v * (meta.north - meta.south)];

/* WGS84 -> MGRS (10 m precision) */
function mgrs(lat, lon) {
  const a = 6378137, f = 1 / 298.257223563, k0 = 0.9996, e2 = f * (2 - f), ep2 = e2 / (1 - e2);
  const zone = Math.floor((lon + 180) / 6) + 1, lon0 = ((zone - 1) * 6 - 180 + 3) * Math.PI / 180;
  const p = lat * Math.PI / 180, l = lon * Math.PI / 180;
  const Nn = a / Math.sqrt(1 - e2 * Math.sin(p) ** 2), T = Math.tan(p) ** 2, C = ep2 * Math.cos(p) ** 2, A = Math.cos(p) * (l - lon0);
  const M = a * ((1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * p - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * Math.sin(2 * p)
    + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * Math.sin(4 * p) - (35 * e2 ** 3 / 3072) * Math.sin(6 * p));
  const E = k0 * Nn * (A + (1 - T + C) * A ** 3 / 6 + (5 - 18 * T + T * T + 72 * C - 58 * ep2) * A ** 5 / 120) + 500000;
  const Nrt = k0 * (M + Nn * Math.tan(p) * (A * A / 2 + (5 - T + 9 * C + 4 * C * C) * A ** 4 / 24 + (61 - 58 * T + T * T + 600 * C - 330 * ep2) * A ** 6 / 720));
  const band = 'CDEFGHJKLMNPQRSTUVWXX'[Math.floor((lat + 80) / 8)];
  const col = ['ABCDEFGH', 'JKLMNPQR', 'STUVWXYZ'][(zone - 1) % 3][Math.floor(E / 100000) - 1];
  const row = (zone % 2 ? 'ABCDEFGHJKLMNPQRSTUV' : 'FGHJKLMNPQRSTUVABCDE')[Math.floor(Nrt / 100000) % 20];
  const d4 = (v) => String(Math.floor((v % 100000) / 10)).padStart(4, '0');
  return `${zone}${band} ${col}${row} ${d4(E)} ${d4(Nrt)}`;
}
window.__mgrs = mgrs;

const fmtH = (h) => { const m = Math.round(h * 60); return `${Math.floor(m / 60)}시간 ${String(m % 60).padStart(2, '0')}분`; };
const clock = (h) => { const m = Math.round(h * 60); return `H+${String(Math.floor(m / 60)).padStart(2, '0')}:${String(m % 60).padStart(2, '0')}`; };

/* ------------------------------------------------------------------ shaders */
const TERRAIN_VS = /* glsl */`
uniform sampler2D uDem; uniform float uRelief; uniform vec2 uObs; uniform float uObsOn; uniform float uEye;
attribute vec2 aUv;
varying vec2 vUv; varying float vH; varying float vPrev; varying float vRange;
void main() {
  vUv = aUv; vRange = 0.0;
  float h = texture2D(uDem, aUv).r;
  vH = h;
  float vis = 0.0;
  if (uObsOn > 0.5) {                                   // live line-of-sight preview from the cursor
    float h0 = texture2D(uDem, uObs).r + uEye;
    float excess = -1.0;
    for (int i = 1; i < 40; i++) {
      float t = float(i) / 40.0;
      excess = max(excess, texture2D(uDem, mix(uObs, aUv, t)).r - mix(h0, h + uEye * 0.5, t));
    }
    vRange = 1.0 - smoothstep(0.35, 0.375, distance(uObs, aUv));
    vis = (1.0 - smoothstep(0.0, 0.004, excess)) * vRange;
  }
  vPrev = vis;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position.x, h * uRelief, position.z, 1.0);
}`;

const TERRAIN_FS = /* glsl */`
uniform sampler2D uDem; uniform sampler2D uSat; uniform sampler2D uAna;
uniform float uHmin; uniform float uHspan; uniform float uTmax; uniform float uCell;
uniform float uLSat; uniform float uLContour; uniform float uLVis; uniform float uLCorr; uniform float uLIso; uniform float uLGrid;
uniform vec3 uBg;
varying vec2 vUv; varying float vH; varying float vPrev; varying float vRange;
float lineAt(float v, float w) { return 1.0 - smoothstep(0.0, w, abs(fract(v - 0.5) - 0.5)); }
void main() {
  float px = 1.0 / 512.0;
  float sx = (texture2D(uDem, vUv + vec2(px, 0.0)).r - texture2D(uDem, vUv - vec2(px, 0.0)).r) * uHspan / (2.0 * uCell);
  float sz = (texture2D(uDem, vUv + vec2(0.0, px)).r - texture2D(uDem, vUv - vec2(0.0, px)).r) * uHspan / (2.0 * uCell);
  vec3 n = normalize(vec3(-sx * 1.8, 1.0, -sz * 1.8));
  float shade = clamp(dot(n, normalize(vec3(-0.55, 0.62, -0.56))), 0.0, 1.0);     // light from the north-west

  vec3 sat = texture2D(uSat, vUv).rgb;
  vec3 photo = mix(vec3(dot(sat, vec3(0.3, 0.59, 0.11))), sat, 0.82) * 1.5;
  vec3 topo = mix(vec3(0.17, 0.20, 0.21), vec3(0.46, 0.48, 0.45), vH);
  vec3 col = mix(topo, photo, uLSat) * (0.42 + 0.9 * shade);

  float elev = uHmin + vH * uHspan;
  float c1 = elev / 20.0, c5 = elev / 100.0, w1 = fwidth(c1);
  float minor = lineAt(c1, 1.1 * w1) * (1.0 - smoothstep(0.10, 0.30, w1));
  float major = lineAt(c5, 1.6 * fwidth(c5));
  col = mix(col, vec3(1.0), uLContour * (0.07 * minor + 0.20 * major * (1.0 - smoothstep(0.25, 0.7, fwidth(c5)))));

  vec4 a = texture2D(uAna, vUv);
  float hatch = step(0.5, fract((vUv.x + vUv.y) * 230.0));
  col = mix(col, vec3(0.34, 0.66, 1.0), uLVis * a.r * (0.13 + 0.07 * hatch));
  col = mix(col, vec3(1.0, 0.26, 0.20), uLCorr * a.g * a.g * 0.34);
  float th = a.b * uTmax * 2.0;
  col = mix(col, vec3(1.0, 0.75, 0.33), uLIso * lineAt(th, 1.5 * fwidth(th)) * 0.6 * step(a.b, 0.995));
  col *= 1.0 - 0.38 * vRange * (1.0 - vPrev);
  col = mix(col, vec3(0.86, 0.97, 1.0), vPrev * 0.26);

  vec2 gk = vUv * 16.0;
  col = mix(col, vec3(1.0), uLGrid * max(lineAt(gk.x, 1.2 * fwidth(gk.x)), lineAt(gk.y, 1.2 * fwidth(gk.y))) * 0.14);

  float e = min(min(vUv.x, 1.0 - vUv.x), min(vUv.y, 1.0 - vUv.y));
  gl_FragColor = vec4(mix(uBg, col, smoothstep(0.0, 0.018, e)), 1.0);
}`;

const LINE_VS = /* glsl */`
uniform vec2 uRes; uniform float uWidth;
attribute vec3 aPrev; attribute vec3 aNext; attribute float aSide; attribute float aT;
varying float vT; varying float vSide;
void main() {
  mat4 m = projectionMatrix * modelViewMatrix;
  vec4 c = m * vec4(position, 1.0); vec4 p = m * vec4(aPrev, 1.0); vec4 n = m * vec4(aNext, 1.0);
  vec2 d = n.xy / n.w * uRes - p.xy / p.w * uRes;
  float l = length(d); d = l > 1e-4 ? d / l : vec2(1.0, 0.0);
  c.xy += vec2(-d.y, d.x) * aSide * uWidth / uRes * c.w;
  gl_Position = c; vT = aT; vSide = aSide;
}`;
const LINE_FS = /* glsl */`
uniform vec3 uColor; uniform float uAlpha; uniform float uProgress; uniform float uTime; uniform float uDash; uniform float uLen;
varying float vT; varying float vSide;
void main() {
  if (vT > uProgress) discard;
  float a = uAlpha * (1.0 - smoothstep(0.55, 1.0, abs(vSide)));
  if (uDash > 0.0) { float f = fract(vT * uLen * uDash - uTime); a *= mix(0.3, 1.0, smoothstep(0.0, 0.08, f) * (1.0 - smoothstep(0.5, 0.58, f))); }
  gl_FragColor = vec4(uColor, a);
}`;

/* ------------------------------------------------------------------ scene */
const uRes = { value: new THREE.Vector2(1, 1) }, uTime = { value: 0 };

function buildTerrain(demTex, satTex) {
  const M = 256, verts = (M + 1) * (M + 1);
  const pos = new Float32Array(verts * 3), uv = new Float32Array(verts * 2), idx = new Uint32Array(M * M * 6);
  for (let j = 0, k = 0; j <= M; j++) for (let i = 0; i <= M; i++, k++) {
    const u = i / M, v = j / M;
    pos[3 * k] = (u - 0.5) * SPAN; pos[3 * k + 2] = (v - 0.5) * SPAN; uv[2 * k] = u; uv[2 * k + 1] = v;
  }
  for (let j = 0, t = 0; j < M; j++) for (let i = 0; i < M; i++) {
    const a = j * (M + 1) + i, b = a + 1, c = a + M + 1, d = c + 1;
    idx[t++] = a; idx[t++] = c; idx[t++] = b; idx[t++] = b; idx[t++] = c; idx[t++] = d;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('aUv', new THREE.BufferAttribute(uv, 2));
  geo.setIndex(new THREE.BufferAttribute(idx, 1));
  anaTex = new THREE.DataTexture(new Uint8Array(G * G * 4), G, G, THREE.RGBAFormat);
  anaTex.magFilter = anaTex.minFilter = THREE.LinearFilter; anaTex.needsUpdate = true;
  const mat = new THREE.ShaderMaterial({
    vertexShader: TERRAIN_VS, fragmentShader: TERRAIN_FS,
    uniforms: {
      uDem: { value: demTex }, uSat: { value: satTex }, uAna: { value: anaTex },
      uRelief: { value: relief }, uHmin: { value: meta.hmin }, uHspan: { value: meta.hmax - meta.hmin }, uCell: { value: meta.cell_m },
      uObs: { value: new THREE.Vector2(0.5, 0.5) }, uObsOn: { value: 0 }, uEye: { value: 2 / (meta.hmax - meta.hmin) },
      uTmax: { value: 6 }, uBg: { value: new THREE.Color(0x0a0d10) },
      uLSat: { value: 1 }, uLContour: { value: 1 }, uLVis: { value: 1 }, uLCorr: { value: 1 }, uLIso: { value: 0 }, uLGrid: { value: 0 },
    },
  });
  terrain = new THREE.Mesh(geo, mat);
  terrain.frustumCulled = false;
  scene.add(terrain);
}

function makeLine(pts, o) {
  const n = pts.length, pos = new Float32Array(n * 6), prev = new Float32Array(n * 6), next = new Float32Array(n * 6);
  const side = new Float32Array(n * 2), tt = new Float32Array(n * 2), idx = new Uint32Array((n - 1) * 6), cum = [0];
  for (let i = 1; i < n; i++) cum.push(cum[i - 1] + pts[i].distanceTo(pts[i - 1]));
  for (let i = 0; i < n; i++) {
    const p = pts[i], a = pts[Math.max(0, i - 1)], b = pts[Math.min(n - 1, i + 1)];
    for (let s = 0; s < 2; s++) {
      const k = (2 * i + s) * 3;
      pos[k] = p.x; pos[k + 1] = p.y; pos[k + 2] = p.z; prev[k] = a.x; prev[k + 1] = a.y; prev[k + 2] = a.z; next[k] = b.x; next[k + 1] = b.y; next[k + 2] = b.z;
      side[2 * i + s] = s ? 1 : -1; tt[2 * i + s] = cum[i] / cum[n - 1];
    }
  }
  for (let i = 0, t = 0; i < n - 1; i++) { const a = 2 * i; idx[t++] = a; idx[t++] = a + 1; idx[t++] = a + 2; idx[t++] = a + 1; idx[t++] = a + 3; idx[t++] = a + 2; }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('aPrev', new THREE.BufferAttribute(prev, 3));
  geo.setAttribute('aNext', new THREE.BufferAttribute(next, 3));
  geo.setAttribute('aSide', new THREE.BufferAttribute(side, 1));
  geo.setAttribute('aT', new THREE.BufferAttribute(tt, 1));
  geo.setIndex(new THREE.BufferAttribute(idx, 1));
  const mat = new THREE.ShaderMaterial({
    vertexShader: LINE_VS, fragmentShader: LINE_FS, transparent: true, depthTest: o.depthTest, depthWrite: false, side: THREE.DoubleSide,
    uniforms: { uRes, uTime, uWidth: { value: o.width }, uColor: { value: new THREE.Color(o.color) }, uAlpha: { value: o.alpha },
      uProgress: { value: 1 }, uDash: { value: o.dash || 0 }, uLen: { value: o.len || 1 } },
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.frustumCulled = false; mesh.renderOrder = o.order || 1;
  mesh.userData = { baseWidth: o.width, baseAlpha: o.alpha };
  return mesh;
}

function smoothPath(xy) {                               // moving average to remove the grid staircase
  const n = xy.length / 2, out = [];
  for (let i = 0; i < n; i++) {
    let sx = 0, sy = 0, c = 0;
    const r = Math.min(2, i, n - 1 - i);
    for (let k = i - r; k <= i + r; k++) { sx += xy[2 * k]; sy += xy[2 * k + 1]; c++; }
    out.push(cellWorld(sx / c, sy / c));
  }
  return out;
}

/* ------------------------------------------------------------------ markers (DOM, projected each frame) */
function addMarker(cls, html, pos, anchor = 'bottom') {
  const el = document.createElement('div');
  el.className = cls; el.innerHTML = html;
  marksEl.appendChild(el);
  const m = { el, pos, anchor, hidden: false };
  markers.push(m);
  return m;
}
function clearMarkers() { markers.forEach(m => m.el.remove()); markers = []; heads = []; }

const _v = new THREE.Vector3();
function occluded(pos) {
  const c = camera.position;
  for (let i = 1; i < 22; i++) {
    const t = 0.04 + 0.92 * i / 22;
    if (groundY(c.x + (pos.x - c.x) * t, c.z + (pos.z - c.z) * t) > c.y + (pos.y - c.y) * t + 0.03) return true;
  }
  return false;
}
function placeAll(list, w, h, checkOcclusion) {
  for (const m of list) {
    _v.copy(m.pos).project(camera);
    const off = _v.z > 1 || Math.abs(_v.x) > 1.15 || Math.abs(_v.y) > 1.15;
    if (off || m.off) { m.el.style.display = 'none'; continue; }
    m.el.style.display = '';
    const x = (_v.x * 0.5 + 0.5) * w, y = (-_v.y * 0.5 + 0.5) * h;
    m.el.style.transform = `translate(${x.toFixed(1)}px,${y.toFixed(1)}px) translate(-50%,${m.anchor === 'bottom' ? '-100%' : '-50%'})`;
    if (checkOcclusion) m.el.classList.toggle('hid', occluded(m.pos));
  }
}

/* ------------------------------------------------------------------ results -> scene and panels */
function applyResult(r) {
  state.result = r;
  anaTex.image.data.set(r.tex); anaTex.needsUpdate = true;
  terrain.material.uniforms.uTmax.value = r.tmax;

  // route names A, B, C from west to east (or north to south)
  const horiz = state.edge === 'N' || state.edge === 'S';
  const order = r.routes.map((_, i) => i).sort((a, b) => horiz ? r.routes[a].start.x - r.routes[b].start.x : r.routes[a].start.y - r.routes[b].start.y);
  order.forEach((ri, k) => { r.routes[ri].name = 'ABCDEF'[k]; });

  // lines
  lines.forEach(l => { l.geometry.dispose(); l.material.dispose(); lineGroup.remove(l); });
  lines = [];
  const dpr = renderer.getPixelRatio();
  r.routes.forEach((rt, ri) => {
    rt.pts = smoothPath(rt.xy);
    const w = (rt.rank === 1 ? 5 : 3.6) * dpr;
    const ghost = makeLine(smoothPath(rt.ghost), { width: 1.6 * dpr, color: 0xffffff, alpha: 0.42, depthTest: false, dash: 4, len: rt.km, order: 2 });
    const xray = makeLine(rt.pts, { width: w, color: 0xff5a48, alpha: 0.26, depthTest: false, order: 3 });
    const main = makeLine(rt.pts, { width: w, color: 0xff5a48, alpha: 1, depthTest: true, dash: 1.6, len: rt.km, order: 4 });
    [ghost, xray, main].forEach(l => { l.userData.route = ri; lineGroup.add(l); lines.push(l); });
    ghost.userData.ghost = true;
    rt.lines = [xray, main];
  });

  // markers
  clearMarkers();
  const ob = r.objective;
  addMarker('mk mk-obj', `<div class="tag">목표 <small>${state.objectiveName || '대곡리'}</small></div><div class="stem"></div><div class="dot"></div>`, cellWorld(ob.x, ob.y));
  r.routes.forEach((rt) => {
    addMarker('mk mk-enemy', `<div class="tag">적 ${rt.name}</div><div class="stem"></div><div class="dot"></div>`, cellWorld(rt.start.x, rt.start.y));
    const hd = addMarker('mk mk-head', '', rt.pts[0].clone(), 'center'); hd.off = true; hd.route = rt; heads.push(hd);
  });
  r.posts.forEach((p, i) => {
    const m = addMarker('mk mk-post', `<div class="tag">OP${i + 1}<em>E${p.threatRank}</em></div><div class="stem"></div><div class="dot"></div>`, cellWorld(p.x, p.y));
    m.el.title = r.stats.autoPosts ? '자동 배치된 관측소' : '클릭하면 이 관측소를 뺍니다';
    m.el.addEventListener('click', () => { if (!r.stats.autoPosts) { state.posts.splice(i, 1); requestAnalysis(); } });
  });
  r.trps.forEach((t, i) => { t.marker = addMarker('mk mk-trp', `<div class="tag">T${i + 1}</div><div class="stem"></div><div class="dot"></div>`, cellWorld(t.x, t.y)); });

  renderPanels(r, order);
  buildEvents(r);
  state.play.T = Math.max(...r.routes.map(q => q.hours));
  $('#clockEnd').textContent = ` / ${clock(state.play.T)}`;
  setFocus(-1);
  updatePlayback(true);
  if (needFrame) { needFrame = false; frameRoutes(); }
}

function bearingName(dx, dy) {
  return ['동쪽', '남동쪽', '남쪽', '남서쪽', '서쪽', '북서쪽', '북쪽', '북동쪽'][((Math.round(Math.atan2(dy, dx) / (Math.PI / 4)) % 8) + 8) % 8];
}
function renderBrief(r) {
  const best = r.routes.find(q => q.rank === 1), parts = [];
  parts.push(`가능성이 가장 높은 침투로는 <b>${best.name}</b>입니다(${Math.round(best.prob * 100)}%). 목표의 ${bearingName(best.start.x - r.objective.x, best.start.y - r.objective.y)}에서 들어와 ${best.km.toFixed(1)} km를 ${fmtH(best.hours)}에 이동하며, 구간의 ${Math.round(best.forest * 100)}%가 산림입니다.`);
  if (r.trps.length) parts.push(`화력은 <b>T1</b>(목표 ${r.trps[0].distObjKm.toFixed(1)} km 전방)을 우선 계획합니다.${r.trps[0].seenBy.length ? '' : ' 다만 이 지점은 현재 어느 관측소에서도 보이지 않습니다.'}`);
  const top = r.posts.findIndex(p => p.threatRank === 1);
  if (top >= 0) parts.push(`적이 가장 먼저 제압하려 할 곳은 <b>OP${top + 1}</b>(표고 ${Math.round(r.posts[top].elev)} m)입니다.`);
  $('#brief').innerHTML = parts.join(' ');
}

function renderPanels(r, order) {
  renderBrief(r);
  $('#stat').textContent = `계산 ${r.stats.ms} ms · 관측 면적 ${r.stats.visKm2.toFixed(1)} km²`;
  const byRank = r.routes.map((_, i) => i).sort((a, b) => r.routes[a].rank - r.routes[b].rank);
  $('#routes').innerHTML = byRank.map((ri) => {
    const q = r.routes[ri];
    return `<li data-i="${ri}"><div class="rt-h"><i>${q.name}</i><b>침투로 ${q.name}</b><em>${Math.round(q.prob * 100)}%</em></div>
      <div class="rt-m">${q.km.toFixed(1)} km · ${fmtH(q.hours)} · 관측 노출 ${Math.round(q.exposed * 100)}% · 산림 ${Math.round(q.forest * 100)}%</div>
      <div class="meter"><i style="width:${Math.round(q.prob * 100)}%"></i></div></li>`;
  }).join('');
  $('#routes').querySelectorAll('li').forEach(li => {
    const i = +li.dataset.i;
    li.addEventListener('mouseenter', () => setFocus(i));
    li.addEventListener('mouseleave', () => setFocus(-1));
    li.addEventListener('click', () => { const q = r.routes[i]; flyTo(q.pts[Math.floor(q.pts.length * 0.6)], 9); });
  });

  const names = (arr) => arr.map(i => r.routes[i].name).join('·');
  $('#trps').innerHTML = r.trps.length ? r.trps.map((t, i) => {
    const s1 = t.routes.length > 1 ? `침투로 ${names(t.routes)}가 합쳐지는 구간입니다.` : `침투로 ${names(t.routes)}의 회랑이 가장 좁아지는 구간입니다.`;
    const s3 = t.seenBy.length ? `${t.seenBy.map(k => 'OP' + (k + 1)).join('·')}에서 관측됩니다.` : '어느 관측소에서도 보이지 않습니다. 관측소를 옮기거나 추가해야 관측 사격이 가능합니다.';
    return `<li data-i="${i}"><span class="code">T${i + 1}</span><div><b>표고 ${Math.round(t.elev)} m · 목표까지 ${t.distObjKm.toFixed(1)} km</b><p>${s1} 회랑 폭 약 ${Math.round(t.widthM / 10) * 10} m. ${s3}</p></div></li>`;
  }).join('') : '<li class="empty">조건에 맞는 지점이 없습니다.</li>';
  $('#trps').querySelectorAll('li[data-i]').forEach(li => li.addEventListener('click', () => { const t = r.trps[+li.dataset.i]; flyTo(cellWorld(t.x, t.y), 6); }));

  const ranked = r.posts.map((_, i) => i).sort((a, b) => r.posts[a].threatRank - r.posts[b].threatRank);
  $('#tgts').innerHTML = ranked.length ? ranked.map((pi, k) => {
    const p = r.posts[pi];
    const tail = k === 0 ? ' 적 입장에서 가장 먼저 무력화할 가치가 큰 지점입니다.' : '';
    return `<li data-i="${pi}"><span class="code">E${p.threatRank}</span><div><b>OP${pi + 1} · 표고 ${Math.round(p.elev)} m</b><p>관측소가 없을 때의 침투로 ${r.routes.length}개 중 ${p.seenRoutes.length}개, 총 ${p.seenKm.toFixed(1)} km를 감시합니다. 가시 면적 ${p.areaKm2.toFixed(1)} km².${tail}</p></div></li>`;
  }).join('') : '<li class="empty">관측소가 없습니다. 관측소 도구로 배치하거나 자동 배치를 누르십시오.</li>';
  $('#tgts').querySelectorAll('li[data-i]').forEach(li => li.addEventListener('click', () => { const p = r.posts[+li.dataset.i]; flyTo(cellWorld(p.x, p.y), 6); }));
}

function setFocus(i) {
  state.focus = i;
  lines.forEach(l => {
    const dim = i >= 0 && l.userData.route !== i;
    l.material.uniforms.uAlpha.value = l.userData.baseAlpha * (dim ? 0.18 : 1);
  });
  $('#routes').querySelectorAll('li').forEach(li => li.classList.toggle('on', +li.dataset.i === i));
}

/* ------------------------------------------------------------------ playback */
function buildEvents(r) {
  events = [];
  r.routes.forEach((rt, ri) => {
    const m = rt.cumH.length;
    let wasSeen = 0;
    for (let j = 0; j < m; j++) {
      if (rt.seen[j] && !wasSeen) {
        const ops = []; for (let k = 0; k < 8; k++) if (rt.seen[j] & (1 << k)) ops.push('OP' + (k + 1));
        events.push({ t: rt.cumH[j], text: `침투로 ${rt.name}: ${ops.join('·')} 관측 구간에 진입` });
      }
      wasSeen = rt.seen[j];
    }
    r.trps.forEach((t, ti) => {
      if (!t.routes.includes(ri)) return;
      let best = -1, bd = 1e9;
      for (let j = 0; j < m; j++) { const d = Math.hypot(rt.xy[2 * j] - t.x, rt.xy[2 * j + 1] - t.y); if (d < bd) { bd = d; best = j; } }
      if (best >= 0) events.push({ t: rt.cumH[best], text: `침투로 ${rt.name}: T${ti + 1} 통과`, trp: ti });
    });
    events.push({ t: rt.hours, text: `침투로 ${rt.name}: 목표 도달` });
  });
  events.sort((a, b) => a.t - b.t);
  shownEvents = -1;
}

function updatePlayback(force) {
  const p = state.play, r = state.result;
  if (!r) return;
  $('#clock').textContent = clock(p.active ? p.t : 0);
  $('#scrub').value = p.active ? Math.round(p.t / p.T * 1000) : 0;
  $('#play').textContent = p.playing ? '❚❚' : '▶';
  $('#logWrap').hidden = !p.active;
  r.routes.forEach((rt, ri) => {
    const hd = heads[ri];
    if (!p.active) { rt.lines.forEach(l => { l.material.uniforms.uProgress.value = 1; }); if (hd) hd.off = true; return; }
    const m = rt.cumH.length;
    let j = 0; while (j < m - 2 && rt.cumH[j + 1] < p.t) j++;
    const span = rt.cumH[j + 1] - rt.cumH[j], f = p.t >= rt.hours ? 1 : clamp(span > 0 ? (p.t - rt.cumH[j]) / span : 0, 0, 1);
    const km = rt.cumKm[j] + (rt.cumKm[j + 1] - rt.cumKm[j]) * f;
    rt.lines.forEach(l => { l.material.uniforms.uProgress.value = km / rt.km; });
    if (hd) { hd.off = false; hd.pos.lerpVectors(rt.pts[j], rt.pts[j + 1], f); hd.el.classList.toggle('seen', !!rt.seen[f > 0.5 ? j + 1 : j] && p.t < rt.hours); }
  });
  if (p.active) {
    let n = 0; while (n < events.length && events[n].t <= p.t) n++;
    if (n !== shownEvents || force) {
      if (n > shownEvents && shownEvents >= 0) for (let k = shownEvents; k < n; k++) {
        const e = events[k];
        if (e.trp != null && r.trps[e.trp] && r.trps[e.trp].marker) { const el = r.trps[e.trp].marker.el; el.classList.remove('fire'); void el.offsetWidth; el.classList.add('fire'); }
      }
      shownEvents = n;
      $('#log').innerHTML = events.slice(0, n).reverse().map(e => `<li><time>${clock(e.t)}</time><span>${e.text}</span></li>`).join('') || '<li><time>H+00:00</time><span>적 출발</span></li>';
    }
  }
}

/* ------------------------------------------------------------------ camera helpers */
let fly = null, needFrame = true;
function frameRoutes() {
  const r = state.result, box = new THREE.Box3();
  r.routes.forEach(q => q.pts.forEach(pt => box.expandByPoint(pt)));
  box.expandByPoint(cellWorld(r.objective.x, r.objective.y, 0));
  const c = box.getCenter(new THREE.Vector3()), size = box.getSize(new THREE.Vector3());
  c.y = groundY(c.x, c.z);
  const W = window.innerWidth, open = (id, px) => (W > 900 && !$(id).classList.contains('closed')) ? px : 0;
  const frac = clamp((W - open('#panelL', 332) - open('#panelR', 358)) / W, 0.3, 1);
  const tanY = Math.tan(camera.fov * Math.PI / 360), tanX = tanY * camera.aspect * frac;
  const dist = clamp(Math.max(size.x / 2 / tanX, size.z * 0.75 / 2 / tanY) * 1.15 + 2, 8, 34);
  const dir = new THREE.Vector3(0.07, 0.62, 0.78).normalize();
  fly = { t: 0, fromT: controls.target.clone(), toT: c, fromP: camera.position.clone(), toP: c.clone().add(dir.multiplyScalar(dist)), dur: 1.5 };
}
function flyTo(target, dist) {
  const dir = camera.position.clone().sub(controls.target).normalize();
  fly = { t: 0, fromT: controls.target.clone(), toT: target.clone(), fromP: camera.position.clone(), toP: target.clone().add(dir.multiplyScalar(dist)) };
}

const ray = new THREE.Raycaster();
function pick(cx, cy) {
  const rect = renderer.domElement.getBoundingClientRect();
  ray.setFromCamera(new THREE.Vector2((cx - rect.left) / rect.width * 2 - 1, -((cy - rect.top) / rect.height) * 2 + 1), camera);
  const o = ray.ray.origin, d = ray.ray.direction;
  let t0 = 0;
  for (let t = 0.05; t < 80; t += 0.06) {
    const x = o.x + d.x * t, y = o.y + d.y * t, z = o.z + d.z * t;
    if (y <= groundY(x, z)) {
      let a = t0, b = t;
      for (let k = 0; k < 10; k++) { const m = (a + b) / 2; if (o.y + d.y * m <= groundY(o.x + d.x * m, o.z + d.z * m)) b = m; else a = m; }
      const u = (o.x + d.x * b) / SPAN + 0.5, v = (o.z + d.z * b) / SPAN + 0.5;
      return (u >= 0 && v >= 0 && u <= 1 && v <= 1) ? { u, v } : null;
    }
    if (y < -0.5) return null;
    t0 = t;
  }
  return null;
}

/* ------------------------------------------------------------------ analysis requests */
let pending = false, queued = false, reqId = 0, firstResult = true;
function requestAnalysis() {
  if (pending) { queued = true; return; }
  pending = true;
  $('#stat').textContent = '계산 중…';
  worker.postMessage({ type: 'analyze', id: ++reqId, query: { objective: state.objective, edge: state.edge, stealth: state.stealth, posts: state.posts, starts: state.starts, autoPosts: true } });
}
let toastTimer = 0;
function toast(msg) { const t = $('#toast'); t.textContent = msg; t.hidden = false; clearTimeout(toastTimer); toastTimer = setTimeout(() => { t.hidden = true; }, 2600); }

/* ------------------------------------------------------------------ UI wiring */
function wireUI() {
  const seg = (root, cb) => root.querySelectorAll('button').forEach(b => b.addEventListener('click', () => {
    root.querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b)); cb(b.dataset.v);
  }));
  seg($('#edge'), (v) => { state.edge = v; state.starts = []; needFrame = true; requestAnalysis(); });
  seg($('#tools'), (v) => {
    state.tool = v; document.body.dataset.tool = v;
    if (v === 'post') toast('지도를 클릭하면 관측소를 놓습니다. 최대 4곳, 관측소를 클릭하면 뺍니다.');
    if (v === 'enemy') toast('지도를 클릭하면 적 출발점을 지정합니다. 최대 3곳.');
    if (v === 'objective') toast('지도를 클릭하면 목표를 옮깁니다.');
  });
  let deb = 0;
  $('#stealth').addEventListener('input', (e) => {
    state.stealth = e.target.value / 100; $('#stealthOut').textContent = `${e.target.value}%`;
    clearTimeout(deb); deb = setTimeout(requestAnalysis, 90);
  });
  $('#resetPosts').addEventListener('click', () => { state.posts = []; requestAnalysis(); toast('관측소를 자동으로 배치했습니다.'); });
  $('#resetStarts').addEventListener('click', () => { state.starts = []; needFrame = true; requestAnalysis(); toast('진입점을 자동으로 골랐습니다.'); });
  $('#layers').querySelectorAll('input').forEach(c => c.addEventListener('change', () => { state.layers[c.dataset.v] = c.checked; applyLayers(); }));
  document.querySelectorAll('.fold').forEach(b => b.addEventListener('click', () => {
    const p = $('#' + b.dataset.fold); p.classList.toggle('closed');
    if (b.dataset.fold === 'panelL') document.body.classList.toggle('l-closed', p.classList.contains('closed'));
  }));

  const p = state.play;
  $('#play').addEventListener('click', () => {
    if (!p.active || p.t >= p.T) { p.active = true; p.t = 0; p.playing = true; shownEvents = 0; }
    else p.playing = !p.playing;
    updatePlayback(true);
  });
  $('#stop').addEventListener('click', () => { p.active = false; p.playing = false; p.t = 0; updatePlayback(true); });
  $('#scrub').addEventListener('input', (e) => { p.active = true; p.playing = false; p.t = e.target.value / 1000 * p.T; shownEvents = -1; updatePlayback(true); });

  // small screens: panels start closed and open from two buttons
  const tabs = document.createElement('div'); tabs.className = 'mobile-tabs';
  tabs.innerHTML = '<button data-p="panelL">설정</button><button data-p="panelR">결과</button>';
  document.body.appendChild(tabs);
  tabs.querySelectorAll('button').forEach(b => b.addEventListener('click', () => {
    const el = $('#' + b.dataset.p), open = el.classList.contains('closed');
    ['#panelL', '#panelR'].forEach(s => $(s).classList.add('closed'));
    if (open) el.classList.remove('closed');
  }));
  if (matchMedia('(max-width: 900px)').matches) { $('#panelL').classList.add('closed'); $('#panelR').classList.add('closed'); }

  // pointer: hover preview, coordinates, tool clicks
  const cv = renderer.domElement;
  let down = null, hoverQueued = false, last = null;
  cv.addEventListener('pointerdown', (e) => { down = { x: e.clientX, y: e.clientY, t: performance.now() }; });
  cv.addEventListener('pointermove', (e) => {
    last = e;
    if (hoverQueued) return;
    hoverQueued = true;
    requestAnimationFrame(() => {
      hoverQueued = false;
      const h = pick(last.clientX, last.clientY), U = terrain.material.uniforms;
      if (!h) { U.uObsOn.value = 0; return; }
      U.uObs.value.set(h.u, h.v); U.uObsOn.value = state.layers.hover ? 1 : 0;
      const [lon, lat] = lonLat(h.u, h.v);
      $('#hudMgrs').textContent = mgrs(lat, lon);
      $('#hudLL').textContent = `${lat.toFixed(5)}N ${lon.toFixed(5)}E`;
      $('#hudEl').textContent = `표고 ${Math.round(heightAt(h.u, h.v))} m`;
    });
  });
  cv.addEventListener('pointerleave', () => { terrain.material.uniforms.uObsOn.value = 0; });
  cv.addEventListener('pointerup', (e) => {
    if (!down || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 5 || performance.now() - down.t > 500 || state.tool === 'look') return;
    const h = pick(e.clientX, e.clientY); if (!h) return;
    const c = { x: clamp(Math.floor(h.u * G), 4, G - 5), y: clamp(Math.floor(h.v * G), 4, G - 5) };
    if (state.tool === 'post') { state.posts.push(c); if (state.posts.length > 4) state.posts.shift(); }
    else if (state.tool === 'enemy') { state.starts.push(c); if (state.starts.length > 3) state.starts.shift(); }
    else if (state.tool === 'objective') { state.objective = c; state.objectiveName = '지정 위치'; controls.target.copy(cellWorld(c.x, c.y, 0)); }
    requestAnalysis();
  });
}

function applyLayers() {
  const U = terrain.material.uniforms, L = state.layers;
  U.uLSat.value = L.sat ? 1 : 0; U.uLContour.value = L.contour ? 1 : 0; U.uLVis.value = L.vis ? 1 : 0;
  U.uLCorr.value = L.corr ? 1 : 0; U.uLIso.value = L.iso ? 1 : 0; U.uLGrid.value = L.grid ? 1 : 0;
  if (!L.hover) U.uObsOn.value = 0;
  placeLabels.forEach(m => { m.off = !L.places; });
}

/* ------------------------------------------------------------------ boot */
const EMB = window.__EMBED || null;
const b64buf = (t) => { const b = atob(t), a = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) a[i] = b.charCodeAt(i); return a.buffer; };
const boot = (p, msg) => { $('#bootBar').style.width = `${p}%`; if (msg) $('#bootMsg').textContent = msg; };

async function main() {
  const canvas = $('#gl');
  try { renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: 'high-performance' }); }
  catch (e) { boot(0, '이 브라우저에서는 WebGL을 사용할 수 없어 3D 지도를 표시하지 못합니다.'); return; }
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x000000, 0);
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(38, 1, 0.05, 200);

  boot(10);
  const get = (u, kind) => fetch(u).then(r => { if (!r.ok) throw new Error(u); return r[kind](); });
  const [m, demBuf, lcBuf, places, satTex] = EMB
    ? [EMB.area, b64buf(EMB.dem), b64buf(EMB.lc), EMB.places, await new THREE.TextureLoader().loadAsync(EMB.sat)]
    : await Promise.all([
      get('./assets/data/area.json', 'json'), get('./assets/data/dem.u16', 'arrayBuffer'), get('./assets/data/landcover.u8', 'arrayBuffer'),
      get('./assets/data/places.json', 'json'), new THREE.TextureLoader().loadAsync('./assets/data/sat.jpg'),
    ]);
  meta = m; boot(55, '지형을 구성하는 중입니다');
  relief = (meta.hmax - meta.hmin) / 1000 * EXAG;

  const raw = new Uint16Array(demBuf), half = new Uint16Array(N * N);
  H = new Float32Array(N * N); lcRaw = new Uint8Array(lcBuf);
  for (let i = 0; i < N * N; i++) { const t = raw[i] / 65535; H[i] = meta.hmin + t * (meta.hmax - meta.hmin); half[i] = THREE.DataUtils.toHalfFloat(t); }
  const demTex = new THREE.DataTexture(half, N, N, THREE.RedFormat, THREE.HalfFloatType);
  demTex.magFilter = demTex.minFilter = THREE.LinearFilter; demTex.wrapS = demTex.wrapT = THREE.ClampToEdgeWrapping; demTex.needsUpdate = true;
  satTex.flipY = false; satTex.colorSpace = THREE.NoColorSpace; satTex.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
  satTex.wrapS = satTex.wrapT = THREE.ClampToEdgeWrapping; satTex.needsUpdate = true;

  buildTerrain(demTex, satTex);
  lineGroup = new THREE.Group(); scene.add(lineGroup);

  // objective = the village the scenario is named after
  const cu = (meta.center.lon - meta.west) / (meta.east - meta.west), cv2 = (meta.north - meta.center.lat) / (meta.north - meta.south);
  state.objective = { x: clamp(Math.floor(cu * G), 4, G - 5), y: clamp(Math.floor(cv2 * G), 4, G - 5) };
  state.objectiveName = meta.center.name;

  controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true; controls.dampingFactor = 0.08; controls.maxPolarAngle = 1.42; controls.minDistance = 2.2; controls.maxDistance = 36;
  controls.target.copy(cellWorld(state.objective.x, state.objective.y, 0));
  camera.position.copy(controls.target).add(new THREE.Vector3(1.5, 19, 21));

  places.forEach(p => {
    const u = (p.lon - meta.west) / (meta.east - meta.west), v = (meta.north - p.lat) / (meta.north - meta.south);
    if (u < 0.02 || v < 0.02 || u > 0.98 || v > 0.98 || p.name === meta.center.name) return;
    const peak = p.kind === 'peak', el = document.createElement('div');
    el.className = 'pl' + (peak ? ' peak' : ''); el.textContent = peak ? `▲ ${p.name} ${Math.round(heightAt(u, v))}` : p.name;
    marksEl.appendChild(el);
    placeLabels.push({ el, pos: world(u, v, 0.12), anchor: 'center' });
  });

  $('#src').innerHTML = `표고 ${meta.dem_source}<br>위성영상 ${meta.sat ? meta.sat.source + ' · ' + meta.sat.date : '-'}<br>토지피복 ${meta.landcover_source}<br>지명 © OpenStreetMap contributors<br>구역 16 km × 16 km · ${meta.south.toFixed(3)}–${meta.north.toFixed(3)}N, ${meta.west.toFixed(3)}–${meta.east.toFixed(3)}E`;

  wireUI(); applyLayers(); resize();
  window.addEventListener('resize', resize);

  boot(75, '분석 엔진을 준비하는 중입니다');
  const Hg = new Float32Array(G * G), LCg = new Uint8Array(G * G);
  for (let y = 0; y < G; y++) for (let x = 0; x < G; x++) {
    const i = (2 * y) * N + 2 * x;
    Hg[y * G + x] = (H[i] + H[i + 1] + H[i + N] + H[i + N + 1]) / 4; LCg[y * G + x] = lcRaw[i];
  }
  worker = new Worker(EMB ? URL.createObjectURL(new Blob([EMB.worker], { type: 'text/javascript' })) : './assets/js/analysis.worker.js');
  worker.onmessage = (e) => {
    if (e.data.type === 'ready') { boot(90, '침투로를 계산하는 중입니다'); requestAnalysis(); return; }
    pending = false;
    applyResult(e.data);
    if (firstResult) {
      firstResult = false; boot(100); $('#boot').classList.add('done');
    }
    if (queued) { queued = false; requestAnalysis(); }
  };
  worker.postMessage({ type: 'init', G, cell: meta.span_m / G, H: Hg, LC: LCg });

  let prev = performance.now();
  renderer.setAnimationLoop((now) => {
    const dt = Math.min(0.1, (now - prev) / 1000); prev = now;
    uTime.value += dt * 0.9;
    const p = state.play;
    if (p.playing) { p.t += dt * p.T / 28; if (p.t >= p.T) { p.t = p.T; p.playing = false; } updatePlayback(); }
    if (fly) {
      fly.t = Math.min(1, fly.t + dt / (fly.dur || 0.9));
      const k = 1 - Math.pow(1 - fly.t, 3);
      controls.target.lerpVectors(fly.fromT, fly.toT, k); camera.position.lerpVectors(fly.fromP, fly.toP, k);
      if (fly.t >= 1) fly = null;
    }
    controls.update();
    renderer.render(scene, camera);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    placeAll(markers, w, h, true); placeAll(placeLabels, w, h, true);
    const az = Math.atan2(camera.position.x - controls.target.x, camera.position.z - controls.target.z);
    $('#compass').style.transform = `rotate(${az}rad)`;
  });
  window.__app = { state, flyTo, requestAnalysis, pick, get result() { return state.result; } };
}

function resize() {
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h; camera.updateProjectionMatrix();
  renderer.getDrawingBufferSize(uRes.value);
}

main().catch((e) => { console.error(e); boot(0, '자료를 불러오지 못했습니다. 페이지를 새로 고쳐 주십시오.'); });
