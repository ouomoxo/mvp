/* Terrain analysis engine (runs in a Web Worker).
 * Everything here is computed from public elevation and land-cover grids only.
 * Grid coordinates: x east, y south, origin at the north-west corner. */
'use strict';

let G = 0, CELL = 0, H = null, LC = null, EB = null;   // EB[a*8+k]: hours for a -> neighbour k

// Land-cover lookups (ESA WorldCover classes): movement multiplier, concealment 0..1
const MOVE = new Float32Array(256).fill(1.1);
const CONC = new Float32Array(256).fill(0.3);
[[10, 1.35, 1.0], [20, 1.2, 0.6], [30, 1.0, 0.25], [40, 1.1, 0.2], [50, 1.0, 0.0], [60, 1.0, 0.0], [80, 8.0, 0.0], [90, 3.0, 0.2]]
  .forEach(([k, m, c]) => { MOVE[k] = m; CONC[k] = c; });

const NB = [[1, 0, 1], [-1, 0, 1], [0, 1, 1], [0, -1, 1], [1, 1, Math.SQRT2], [1, -1, Math.SQRT2], [-1, 1, Math.SQRT2], [-1, -1, Math.SQRT2]];
const OPP = [1, 0, 3, 2, 7, 6, 5, 4];

/* Walking time in hours for one step a -> b (Tobler's hiking function, off-road land-cover factor). */
function stepHours(a, b, dCells) {
  const d = dCells * CELL;
  const s = (H[b] - H[a]) / d;
  const v = 6 * Math.exp(-3.5 * Math.abs(s + 0.05));            // km/h
  return (d / 1000) / v * 0.5 * (MOVE[LC[a]] + MOVE[LC[b]]);
}

function buildEdges() {
  EB = new Float32Array(G * G * 8);
  for (let y = 0; y < G; y++) for (let x = 0; x < G; x++) for (let k = 0; k < 8; k++) {
    const nx = x + NB[k][0], ny = y + NB[k][1];
    if (nx >= 0 && ny >= 0 && nx < G && ny < G) EB[(y * G + x) * 8 + k] = stepHours(y * G + x, ny * G + nx, NB[k][2]);
  }
}

/* Dijkstra over the 8-connected grid.
 * reverse=false: cost of travelling seeds -> cell.  reverse=true: cost of travelling cell -> seeds.
 * P is the per-cell penalty multiplier (concealment and exposure). */
function dijkstra(seeds, P, reverse) {
  const n = G * G;
  const dist = new Float32Array(n).fill(Infinity);
  const link = new Int32Array(n).fill(-1);
  const hn = new Int32Array(n * 9), hc = new Float32Array(n * 9);
  let hs = 0;
  const push = (node, c) => {
    let i = hs++;
    while (i > 0) { const p = (i - 1) >> 1; if (hc[p] <= c) break; hn[i] = hn[p]; hc[i] = hc[p]; i = p; }
    hn[i] = node; hc[i] = c;
  };
  for (const s of seeds) { dist[s] = 0; push(s, 0); }
  while (hs > 0) {
    const node = hn[0], c = hc[0];
    hs--;
    if (hs > 0) {
      const ln = hn[hs], lc = hc[hs];
      let i = 0;
      for (;;) {
        let l = 2 * i + 1; if (l >= hs) break;
        const r = l + 1; if (r < hs && hc[r] < hc[l]) l = r;
        if (hc[l] >= lc) break;
        hn[i] = hn[l]; hc[i] = hc[l]; i = l;
      }
      hn[i] = ln; hc[i] = lc;
    }
    if (c > dist[node]) continue;
    const x = node % G, y = (node / G) | 0;
    for (let k = 0; k < 8; k++) {
      const nx = x + NB[k][0], ny = y + NB[k][1];
      if (nx < 0 || ny < 0 || nx >= G || ny >= G) continue;
      const m = ny * G + nx;
      const base = reverse ? EB[m * 8 + OPP[k]] : EB[node * 8 + k];
      const nc = c + base * 0.5 * (P[node] + P[m]);
      if (nc < dist[m]) { dist[m] = nc; link[m] = node; push(m, nc); }
    }
  }
  return { dist, link };
}

/* Line-of-sight viewshed from one cell: a ray to every cell on the bounding square,
 * keeping the running maximum slope along each ray. eye/target heights in metres, R in cells. */
function viewshed(ox, oy, R, eye, tgt) {
  const out = new Uint8Array(G * G);
  const h0 = H[oy * G + ox] + eye;
  out[oy * G + ox] = 1;
  const ray = (tx, ty) => {
    const dx = tx - ox, dy = ty - oy, n = Math.max(Math.abs(dx), Math.abs(dy));
    const sx = dx / n, sy = dy / n, sl = Math.hypot(sx, sy);
    let maxS = -Infinity;
    for (let k = 1; k <= n; k++) {
      const x = (ox + sx * k + 0.5) | 0, y = (oy + sy * k + 0.5) | 0;
      if (x < 0 || y < 0 || x >= G || y >= G) break;
      const d = k * sl;
      if (d > R) break;
      const i = y * G + x, hh = H[i] - h0;
      if ((hh + tgt) / d >= maxS) out[i] = 1;
      const sT = hh / d; if (sT > maxS) maxS = sT;
    }
  };
  for (let t = -R; t <= R; t++) { ray(ox + t, oy - R); ray(ox + t, oy + R); ray(ox - R, oy + t); ray(ox + R, oy + t); }
  return out;
}

function penalty(stealth, vis) {
  const n = G * G, P = new Float32Array(n);
  const wC = 2.0 * stealth, wO = 6.0 * stealth;
  for (let i = 0; i < n; i++) P[i] = 1 + wC * (1 - CONC[LC[i]]) + (vis ? wO * (vis[i] ? 1 : 0) : 0);
  return P;
}

function edgeCells(edge) {
  const out = [], m = 3;
  for (let k = m; k < G - m; k++) {
    if (edge === 'N') out.push(m * G + k);
    else if (edge === 'S') out.push((G - 1 - m) * G + k);
    else if (edge === 'W') out.push(k * G + m);
    else out.push(k * G + (G - 1 - m));
  }
  return out;
}

/* Best entry points on an edge: lowest cost-to-objective, at least `sep` cells apart. */
function pickEntries(edge, toGo, count, sep) {
  const cells = edgeCells(edge).filter(c => isFinite(toGo[c])).sort((a, b) => toGo[a] - toGo[b]);
  const picked = [];
  for (const c of cells) {
    const x = c % G, y = (c / G) | 0;
    if (picked.every(p => Math.hypot(p % G - x, ((p / G) | 0) - y) >= sep)) picked.push(c);
    if (picked.length === count) break;
  }
  return picked;
}

function tracePath(start, link, goal) {
  const path = [start];
  let c = start, guard = G * G;
  while (c !== goal && link[c] >= 0 && guard-- > 0) { c = link[c]; path.push(c); }
  return path;
}

function corridorOf(fwd, toGo, start, eps, out) {
  const best = toGo[start], n = G * G;
  for (let i = 0; i < n; i++) {
    const ex = fwd[i] + toGo[i] - best;
    if (ex < eps * best) { const v = 1 - ex / (eps * best); if (v > out[i]) out[i] = v; }
  }
}

/* Greedy choice of observation posts covering the unobserved-approach corridor. */
function autoPosts(obj, corridor, count) {
  const ox = obj % G, oy = (obj / G) | 0, R = Math.round(3500 / CELL), cands = [];
  for (let y = Math.max(4, oy - R); y <= Math.min(G - 5, oy + R); y++) {
    for (let x = Math.max(4, ox - R); x <= Math.min(G - 5, ox + R); x++) {
      if (Math.hypot(x - ox, y - oy) > R) continue;
      const i = y * G + x, h = H[i];
      let isMax = true, sum = 0, cnt = 0;
      for (let dy = -4; dy <= 4 && isMax; dy++) for (let dx = -4; dx <= 4; dx++) {
        const hh = H[(y + dy) * G + x + dx];
        if (Math.abs(dx) <= 2 && Math.abs(dy) <= 2 && hh > h) { isMax = false; break; }
        sum += hh; cnt++;
      }
      if (isMax && LC[i] !== 80) cands.push({ i, x, y, tpi: h - sum / cnt });
    }
  }
  cands.sort((a, b) => b.tpi - a.tpi);
  const pool = cands.slice(0, 36), VR = Math.round(6000 / CELL);
  pool.forEach(c => { c.vs = viewshed(c.x, c.y, VR, 2, 1); });
  const covered = new Uint8Array(G * G), picked = [];
  let total = 0; for (let i = 0; i < corridor.length; i++) total += corridor[i];
  for (let k = 0; k < count; k++) {
    let best = null, bestGain = 0;
    for (const c of pool) {
      if (picked.some(p => Math.hypot(p.x - c.x, p.y - c.y) < 10)) continue;
      let gain = 0;
      for (let i = 0; i < corridor.length; i++) if (c.vs[i] && !covered[i]) gain += corridor[i];
      if (gain > bestGain) { bestGain = gain; best = c; }
    }
    if (!best || bestGain < 0.02 * total) break;
    picked.push(best);
    for (let i = 0; i < corridor.length; i++) if (best.vs[i]) covered[i] = 1;
  }
  return picked.map(p => p.i);
}

function analyze(q) {
  const t0 = performance.now(), n = G * G;
  const obj = q.objective.y * G + q.objective.x, stealth = q.stealth;
  const VR = Math.round(6000 / CELL);

  // 1) approach with concealment only (before any observation post is considered)
  const P0 = penalty(stealth, null);
  const go0 = dijkstra([obj], P0, true);
  const starts = (q.starts && q.starts.length) ? q.starts.map(s => s.y * G + s.x) : pickEntries(q.edge, go0.dist, 3, Math.round(2500 / CELL));
  const ghost = starts.map(s => tracePath(s, go0.link, obj));
  const corridor0 = new Float32Array(n);
  ghost.forEach(g => g.forEach(c => {
    const x = c % G, y = (c / G) | 0;
    for (let dy = -4; dy <= 4; dy++) for (let dx = -4; dx <= 4; dx++) {
      const xx = x + dx, yy = y + dy, d = Math.hypot(dx, dy);
      if (xx < 0 || yy < 0 || xx >= G || yy >= G || d > 4) continue;
      const v = 1 - d / 5; if (v > corridor0[yy * G + xx]) corridor0[yy * G + xx] = v;
    }
  }));

  // 2) observation posts: given by the user, or chosen to cover that corridor
  const postCells = (q.posts && q.posts.length) ? q.posts.map(p => p.y * G + p.x) : (q.autoPosts ? autoPosts(obj, corridor0, 3) : []);
  const postVs = postCells.map(c => viewshed(c % G, (c / G) | 0, VR, 2, 1));
  const vis = new Uint8Array(n);                                    // bit i = seen by post i
  postVs.forEach((v, k) => { for (let i = 0; i < n; i++) if (v[i]) vis[i] |= (1 << k); });

  // 3) approach again, now avoiding observed ground
  const P1 = penalty(stealth, vis);
  const go1 = dijkstra([obj], P1, true);
  const corridor1 = new Float32Array(n);
  const routes = starts.map((s, k) => {
    corridorOf(dijkstra([s], P1, false).dist, go1.dist, s, 0.06, corridor1);
    const path = tracePath(s, go1.link, obj), m = path.length;
    const xy = new Float32Array(m * 2), cumKm = new Float32Array(m), cumH = new Float32Array(m), seen = new Uint8Array(m);
    let km = 0, hrs = 0, expKm = 0, forestKm = 0;
    for (let j = 0; j < m; j++) {
      const c = path[j]; xy[2 * j] = c % G; xy[2 * j + 1] = (c / G) | 0; seen[j] = vis[c];
      if (j > 0) {
        const p = path[j - 1], dc = Math.hypot(c % G - p % G, ((c / G) | 0) - ((p / G) | 0)), d = dc * CELL / 1000;
        km += d; hrs += stepHours(p, c, dc);
        if (vis[c]) expKm += d;
        if (LC[c] === 10) forestKm += d;
      }
      cumKm[j] = km; cumH[j] = hrs;
    }
    const g = ghost[k], gxy = new Float32Array(g.length * 2);
    g.forEach((c, j) => { gxy[2 * j] = c % G; gxy[2 * j + 1] = (c / G) | 0; });
    return { start: { x: s % G, y: (s / G) | 0 }, cells: path, xy, cumKm, cumH, seen, ghost: gxy,
             km, hours: hrs, exposed: km ? expKm / km : 0, forest: km ? forestKm / km : 0, cost: go1.dist[s] };
  });
  const cmin = Math.min(...routes.map(r => r.cost));
  let z = 0; routes.forEach(r => { r.w = Math.exp(-(r.cost - cmin) / (0.15 * cmin)); z += r.w; });
  routes.forEach(r => { r.prob = r.w / z; delete r.w; });
  const order = routes.map((_, i) => i).sort((a, b) => routes[a].cost - routes[b].cost);
  order.forEach((ri, rank) => { routes[ri].rank = rank + 1; });

  // 4) fire-plan candidates: the narrowest point of each approach within 0.7-5 km of the objective
  const near = new Uint8Array(n);                                   // how many routes pass within 2 cells
  routes.forEach(r => {
    const mark = new Uint8Array(n);
    r.cells.forEach(c => { const x = c % G, y = (c / G) | 0;
      for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) { const xx = x + dx, yy = y + dy; if (xx >= 0 && yy >= 0 && xx < G && yy < G) mark[yy * G + xx] = 1; } });
    for (let i = 0; i < n; i++) near[i] += mark[i];
  });
  let trps = [];
  routes.forEach((r, ri) => {
    const m = r.cells.length, total = r.cumKm[m - 1];
    let best = null;
    for (let j = 0; j < m; j++) {
      const dObj = total - r.cumKm[j];
      if (dObj < 0.7 || dObj > 5) continue;
      const c = r.cells[j], x = c % G, y = (c / G) | 0;
      let width = 0;
      for (let dy = -5; dy <= 5; dy++) for (let dx = -5; dx <= 5; dx++) {
        const xx = x + dx, yy = y + dy;
        if (xx >= 0 && yy >= 0 && xx < G && yy < G && dx * dx + dy * dy <= 25 && corridor1[yy * G + xx] > 0.3) width++;
      }
      const score = (1 - Math.min(1, width / 60)) + (vis[c] ? 0.8 : 0) + 0.6 * (near[c] - 1) - 0.05 * Math.abs(dObj - 2);
      if (!best || score > best.score) best = { x, y, c, score, width, dObj, routes: [ri] };
    }
    if (best) trps.push(best);
  });
  trps.sort((a, b) => b.score - a.score);
  const merged = [];
  for (const t of trps) {
    const m = merged.find(o => Math.hypot(o.x - t.x, o.y - t.y) <= 6);
    if (m) m.routes.push(...t.routes); else merged.push(t);
  }
  merged.sort((a, b) => b.routes.length - a.routes.length || Math.min(...a.routes.map(i => routes[i].rank)) - Math.min(...b.routes.map(i => routes[i].rank)));
  trps = merged.map(t => ({ x: t.x, y: t.y, elev: H[t.c], routes: t.routes.sort(), distObjKm: t.dObj,
    widthM: Math.sqrt(t.width) * CELL, seenBy: postCells.map((_, k) => k).filter(k => vis[t.c] & (1 << k)) }));

  // 5) what the attacker would want to suppress first: the posts that watch the unconstrained approaches
  let corTotal = 0; for (let i = 0; i < n; i++) corTotal += corridor0[i];
  const posts = postCells.map((c, k) => {
    let cov = 0; for (let i = 0; i < n; i++) if (postVs[k][i]) cov += corridor0[i];
    let seenKm = 0; const seenRoutes = [];
    ghost.forEach((g, ri) => {
      let km = 0;
      for (let j = 1; j < g.length; j++) if (postVs[k][g[j]]) km += Math.hypot(g[j] % G - g[j - 1] % G, ((g[j] / G) | 0) - ((g[j - 1] / G) | 0)) * CELL / 1000;
      if (km > 0.2) seenRoutes.push(ri);
      seenKm += km;
    });
    let area = 0; for (let i = 0; i < n; i++) area += postVs[k][i];
    return { x: c % G, y: (c / G) | 0, elev: H[c], cover: corTotal ? cov / corTotal : 0, seenKm, seenRoutes, areaKm2: area * CELL * CELL / 1e6 };
  });
  const threat = posts.map((_, i) => i).sort((a, b) => posts[b].seenKm - posts[a].seenKm || posts[b].cover - posts[a].cover);
  threat.forEach((pi, rank) => { posts[pi].threatRank = rank + 1; });

  // 6) raster layers for the renderer: R observed, G corridor, B time-to-objective, A corridor before posts
  const tex = new Uint8Array(n * 4), TMAX = 6;
  let visCells = 0;
  for (let i = 0; i < n; i++) {
    tex[4 * i] = vis[i] ? 255 : 0;
    tex[4 * i + 1] = Math.round(255 * corridor1[i]);
    tex[4 * i + 2] = Math.round(255 * Math.min(1, go1.dist[i] / TMAX));
    tex[4 * i + 3] = Math.round(255 * corridor0[i]);
    if (vis[i]) visCells++;
  }
  return { tex, tmax: TMAX, routes, trps, posts, objective: q.objective,
    stats: { ms: Math.round(performance.now() - t0), visKm2: visCells * CELL * CELL / 1e6, autoPosts: !(q.posts && q.posts.length) } };
}

self.onmessage = (e) => {
  const d = e.data;
  if (d.type === 'init') { G = d.G; CELL = d.cell; H = d.H; LC = d.LC; buildEdges(); self.postMessage({ type: 'ready' }); }
  else if (d.type === 'analyze') {
    const r = analyze(d.query);
    r.type = 'result'; r.id = d.id;
    self.postMessage(r, [r.tex.buffer]);
  }
};
