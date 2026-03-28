/**
 * ExCuSe  —  Exclusion-Based Pupil Detection  (PupilExt port)
 * =============================================================
 * v2 — ambient RGB / webcam tuning (brightThr bug fixed, double blur, etc.)
 * v2.1 — failStage diagnostic + maximally loose gates
 * v2.2 — silent localStorage diagnostic logger (_silentLog)
 *
 * ExCuSe is architecturally the best fit for ambient RGB because it works
 * from a dark-blob region rather than needing clean connected edges.  Gates
 * have been widened aggressively here; noisy measurements should be filtered
 * downstream with a Savitzky-Golay or median filter.
 *
 * Every return includes failStage: null on success, or a string:
 *   "no_zone" | "no_cand_pixels" | "no_eroded_pixels" | "no_components" |
 *   "no_component_near_centre" | "hull_lt6" | "aspect_or_size" | null
 *
 * Diagnostic log: written to localStorage key 'pl_algo_log' only during trials
 * (when window._plTrialActive === true). Download via the Algo Log button in app.
 */
(function(){
'use strict';
const U = window.PupilAlgoUtils;

window.PupilAlgos['excuse'] = {
  id:    'excuse',
  label: 'ExCuSe',

  run(inp) {
    const { imageData, irisRadPx: irisRad, cx, cy,
            upperLidY, lowerLidY, mmPerPx } = inp;
    const { width: w, height: h } = imageData;

    const mmToPx = irisRad / 5.85;
    const outerR = 4.0 * mmToPx;
    const innerR = 0.8 * mmToPx;

    const { gray, zone } = U.extractGray(
      imageData, cx, cy, outerR, 0, upperLidY, lowerLidY, 4
    );
    if (!zone.some(Boolean)) return _fail('excuse', inp, 'no_zone');

    // Double blur
    const blurred  = new Float32Array(w * h);
    const blurred2 = new Float32Array(w * h);
    U.gaussianBlur5(gray,    blurred,  w, h, zone);
    U.gaussianBlur5(blurred, blurred2, w, h, zone);

    // Thresholds — loosened for ambient RGB
    const darkThr   = U.percentile(blurred2, zone, 35);
    const brightThr = U.percentile(blurred2, zone, 70);

    // Candidate mask: dark AND not a catchlight
    const cand = new Uint8Array(w * h);
    let candCount = 0;
    for (let i = 0; i < w * h; i++) {
      if (!zone[i]) continue;
      if (blurred2[i] <= darkThr && blurred2[i] < brightThr) { cand[i] = 1; candCount++; }
    }
    if (candCount === 0) return _fail('excuse', inp, 'no_cand_pixels', {
      darkThr: darkThr.toFixed(1), brightThr: brightThr.toFixed(1)
    });

    // Gradient exclusion
    const { mag } = U.sobel(blurred2, w, h, zone);
    let maxG = 0;
    for (let i = 0; i < w * h; i++) maxG = Math.max(maxG, mag[i]);
    const edgeThr = maxG * 0.25;
    for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
      if (!cand[y * w + x]) continue;
      if (Math.hypot(x - cx, y - cy) > outerR * 0.80 && mag[y * w + x] > edgeThr)
        cand[y * w + x] = 0;
    }

    // Morphological close
    const closed = new Uint8Array(w * h);
    for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
      if (!cand[y * w + x]) continue;
      for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
        const ny = y + dy, nx = x + dx;
        if (ny >= 0 && ny < h && nx >= 0 && nx < w) closed[ny * w + nx] = 1;
      }
    }
    const eroded = new Uint8Array(w * h);
    let erodedCount = 0;
    for (let y = 2; y < h - 2; y++) for (let x = 2; x < w - 2; x++) {
      if (!closed[y * w + x]) continue;
      let all = true;
      for (let dy = -2; dy <= 2 && all; dy++) for (let dx = -2; dx <= 2; dx++)
        if (!closed[(y + dy) * w + (x + dx)]) { all = false; break; }
      if (all) { eroded[y * w + x] = 1; erodedCount++; }
    }
    if (erodedCount === 0) return _fail('excuse', inp, 'no_eroded_pixels', { candCount });

    // Connected components
    const label = new Int16Array(w * h);
    let nlbls = 0;
    const compSz = [], compSx = [], compSy = [], compPts = [];
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      if (!eroded[y * w + x] || label[y * w + x]) continue;
      nlbls++;
      const lbl = nlbls, stk = [y * w + x]; label[y * w + x] = lbl;
      let cnt = 0, sx = 0, sy = 0; const pts = [];
      while (stk.length) {
        const i2 = stk.pop(), px = i2 % w, py = (i2 - px) / w;
        cnt++; sx += px; sy += py; pts.push([px, py]);
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          const ny = py + dy, nx = px + dx;
          if (ny < 0 || ny >= h || nx < 0 || nx >= w) continue;
          const ni = ny * w + nx;
          if (eroded[ni] && !label[ni]) { label[ni] = lbl; stk.push(ni); }
        }
      }
      compSz.push(cnt); compSx.push(sx); compSy.push(sy); compPts.push(pts);
    }
    if (!nlbls) return _fail('excuse', inp, 'no_components', { erodedCount });

    // Score — distance gate widened to outerR*0.75
    let best = null, bestScore = -1;
    const compDetails = [];
    for (let l = 0; l < nlbls; l++) {
      const mx = compSx[l] / compSz[l], my = compSy[l] / compSz[l];
      const dist  = Math.hypot(mx - cx, my - cy);
      const areaR = Math.sqrt(compSz[l] / Math.PI);
      compDetails.push({ dist: dist.toFixed(1), areaR: areaR.toFixed(1), sz: compSz[l] });
      if (dist > outerR * 0.75) continue;
      if (areaR < innerR * 0.4 || areaR > outerR * 1.1) continue;
      const score = compSz[l] / (dist + 1);
      if (score > bestScore) { bestScore = score; best = l; }
    }
    if (best === null) return _fail('excuse', inp, 'no_component_near_centre',
      { nlbls, outerR: outerR.toFixed(1), innerR: innerR.toFixed(1), compDetails });

    const mx = compSx[best] / compSz[best], my = compSy[best] / compSz[best];
    const SECTORS = 36;
    const buckets = Array.from({ length: SECTORS }, () => null);
    for (const [px, py] of compPts[best]) {
      const a  = Math.atan2(py - my, px - mx);
      const si = ((Math.round(a / 2 / Math.PI * SECTORS) % SECTORS) + SECTORS) % SECTORS;
      const d  = Math.hypot(px - mx, py - my);
      if (!buckets[si] || d > buckets[si].d) buckets[si] = { x: px, y: py, d };
    }
    const hull = buckets.filter(Boolean).map(b => [b.x, b.y]);
    if (hull.length < 6) return _fail('excuse', inp, 'hull_lt6', {
      hullLen: hull.length, compSz: compSz[best]
    });

    const el = U.fitEllipse(hull);
    if (!el) {
      // Circular fallback — always return something
      const r    = Math.sqrt(compSz[best] / Math.PI);
      const conf = Math.max(0, 0.35 - Math.hypot(mx - cx, my - cy) / (outerR + 1));
      const dbg  = buildDebug(w, h, zone, blurred2, eroded, best, label, mx, my, r, null);
      const out  = { pupilRadPx: r, pupilMm: r * 2 * mmPerPx, confidence: conf,
                     debugPixels: dbg, failStage: null };
      _silentLog('excuse', inp, out);
      return out;
    }

    const { a, b: bEl, cx: ex, cy: ey } = el;
    const pupilR = (a + bEl) / 2;
    const aspect = bEl / (a + 1e-6);
    if (aspect < 0.15 || pupilR < innerR * 0.5 || pupilR > outerR * 1.1)
      return _fail('excuse', inp, 'aspect_or_size', {
        aspect: aspect.toFixed(2), pupilR: pupilR.toFixed(1),
        innerR: innerR.toFixed(1), outerR: outerR.toFixed(1)
      });

    const dist2 = Math.hypot(ex - cx, ey - cy);
    const conf  = Math.min(1, aspect) * Math.max(0, 1 - dist2 / (outerR * 0.75));

    const dbg = buildDebug(w, h, zone, blurred2, eroded, best, label, ex, ey, pupilR, hull);
    const out = { pupilRadPx: pupilR, pupilMm: pupilR * 2 * mmPerPx, confidence: conf,
                  debugPixels: dbg, failStage: null };
    _silentLog('excuse', inp, out);
    return out;
  }
};

function buildDebug(w, h, zone, blurred2, eroded, bestLbl, label, cx, cy, r, hull) {
  const dbg = U.makeDebugRGBA(w, h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x, idx = i * 4;
    if (!zone[i]) { dbg[idx]=10; dbg[idx+1]=10; dbg[idx+2]=16; dbg[idx+3]=255; continue; }
    if (eroded[i]) {
      const isB = label[i] === bestLbl + 1;
      dbg[idx]=isB?60:90; dbg[idx+1]=isB?255:60; dbg[idx+2]=isB?120:60; dbg[idx+3]=255;
    } else {
      const v = Math.round(blurred2[i] * 0.4);
      dbg[idx]=v; dbg[idx+1]=v; dbg[idx+2]=v; dbg[idx+3]=255;
    }
  }
  if (hull) hull.forEach(([px, py]) => {
    const idx = (py * w + px) * 4;
    dbg[idx]=255; dbg[idx+1]=200; dbg[idx+2]=0; dbg[idx+3]=255;
  });
  U.drawCircle(dbg, w, h, cx, cy, r, [0, 255, 220, 220]);
  return dbg;
}

// ── Diagnostic helpers ────────────────────────────────────────────────────────
// In-memory ring buffer flushed to localStorage at most every 2s.
// Avoids JSON.parse+stringify on every frame which would kill 30fps.
// window._plLogBuf is shared across all algo IIFEs via the window object.

function _fail(algoId, inp, stage, detail) {
  if (detail === undefined) detail = {};
  const out = { pupilRadPx: null, pupilMm: null, confidence: 0,
                debugPixels: null, failStage: stage, failDetail: detail };
  _silentLog(algoId, inp, out);
  return out;
}

function _silentLog(algoId, inp, result) {
  try {
    if (!window._plTrialActive) return;
    if (!window._plLogBuf) window._plLogBuf = [];
    window._plLogBuf.push({
      t:          Date.now(),
      algo:       algoId,
      side:       inp._side || '?',
      failStage:  result.failStage || null,
      failDetail: result.failDetail || null,
      conf:       result.confidence != null ? result.confidence : null,
      pupilMm:    result.pupilMm   != null ? result.pupilMm   : null,
      irisRad:    inp.irisRadPx != null ? +inp.irisRadPx.toFixed(1) : null,
    });
    // Flush at most every 2000ms — batches all the per-frame writes
    const now = Date.now();
    if (!window._plLogLastFlush || now - window._plLogLastFlush > 2000) {
      window._plLogLastFlush = now;
      try {
        const stored = localStorage.getItem('pl_algo_log');
        const existing = stored ? JSON.parse(stored) : [];
        const merged = existing.concat(window._plLogBuf);
        if (merged.length > 2000) merged.splice(0, merged.length - 2000);
        localStorage.setItem('pl_algo_log', JSON.stringify(merged));
        window._plLogBuf = [];
      } catch(e) { window._plLogBuf = []; /* quota exceeded — drop buffer */ }
    }
  } catch(e) { /* never throw from logger */ }
}

})();
