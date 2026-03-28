/**
 * ElSe  —  Ellipse Selection  (PupilExt port)
 * ============================================
 * v2 — ambient RGB / webcam tuning (see else.js changelog)
 * v2.1 — added failStage diagnostic to return object
 * v2.2 — silent localStorage diagnostic logger (_silentLog)
 *
 * Every return now includes failStage: null on success, or a string
 * describing exactly where the algo gave up, e.g.:
 *   "no_zone" | "eq_empty" | "no_contours_after_canny" |
 *   "no_contour_ge6" | "all_filtered_size" | "all_filtered_dist" |
 *   "all_filtered_aspect" | "fitEllipse_all_null" | null
 *
 * Diagnostic log: written to localStorage key 'pl_algo_log' only during trials
 * (when window._plTrialActive === true). Download via the Algo Log button in app.
 */
(function(){
'use strict';
const U = window.PupilAlgoUtils;

window.PupilAlgos['else'] = {
  id:    'else',
  label: 'ElSe',

  run(inp) {
    const { imageData, irisRadPx: irisRad, cx, cy,
            upperLidY, lowerLidY, mmPerPx } = inp;
    const { width: w, height: h } = imageData;

    const mmToPx = irisRad / 5.85;
    const outerR = 4.0 * mmToPx;
    const innerR = 1.0 * mmToPx;

    const { gray, zone } = U.extractGray(
      imageData, cx, cy, outerR, 0, upperLidY, lowerLidY, 4
    );
    if (!zone.some(Boolean)) return _fail('else', inp, 'no_zone');

    // Histogram equalisation
    const hist = new Float32Array(256);
    let zCnt = 0;
    for (let i = 0; i < gray.length; i++)
      if (zone[i]) { hist[Math.floor(gray[i])]++; zCnt++; }
    if (zCnt < 20) return _fail('else', inp, 'eq_empty', { zCnt });
    let cumul = 0;
    const lut = new Float32Array(256);
    for (let i = 0; i < 256; i++) { cumul += hist[i]; lut[i] = (cumul / zCnt) * 255; }
    const eq = new Float32Array(w * h);
    for (let i = 0; i < w * h; i++) if (zone[i]) eq[i] = lut[Math.floor(gray[i])];

    // Specular flood-fill
    const zv = [];
    for (let i = 0; i < w * h; i++) if (zone[i]) zv.push(eq[i]);
    zv.sort((a, b) => a - b);
    const zmean = zv.reduce((s, v) => s + v, 0) / zv.length;
    const p90 = zv[Math.floor(zv.length * 0.90)];
    const p80 = zv[Math.floor(zv.length * 0.80)];
    const masked   = new Float32Array(eq);
    const specFill = new Uint8Array(w * h);
    const sfStk = [];
    for (let i = 0; i < w * h; i++) if (zone[i] && masked[i] >= p90) sfStk.push(i);
    while (sfStk.length) {
      const idx = sfStk.pop();
      if (specFill[idx]) continue;
      specFill[idx] = 1; masked[idx] = zmean;
      const fx = idx % w, fy = (idx - fx) / w;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const ny = fy + dy, nx = fx + dx;
        if (ny < 0 || ny >= h || nx < 0 || nx >= w) continue;
        const ni = ny * w + nx;
        if (zone[ni] && !specFill[ni] && masked[ni] > p80) sfStk.push(ni);
      }
    }

    // Double blur → edges
    const blurred  = new Float32Array(w * h);
    const blurred2 = new Float32Array(w * h);
    U.gaussianBlur5(masked,  blurred,  w, h, zone);
    U.gaussianBlur5(blurred, blurred2, w, h, zone);
    const { mag, ang } = U.sobel(blurred2, w, h, zone);
    const nmsMap = U.nms(mag, ang, w, h, zone);
    let   edges  = U.canny(nmsMap, mag, w, h);
    edges          = U.thin(edges, w, h);

    const edgePxCount = edges.reduce((s, v) => s + v, 0);

    // Connected components
    const label    = new Int16Array(w * h);
    const contours = [];
    let nlbls = 0;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      if (!edges[y * w + x] || label[y * w + x]) continue;
      nlbls++;
      const lbl = nlbls, pts = [], stk = [y * w + x];
      label[y * w + x] = lbl;
      while (stk.length) {
        const i2 = stk.pop(), px = i2 % w, py = (i2 - px) / w;
        pts.push([px, py]);
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          const ny = py + dy, nx = px + dx;
          if (ny < 0 || ny >= h || nx < 0 || nx >= w) continue;
          const ni = ny * w + nx;
          if (edges[ni] && !label[ni]) { label[ni] = lbl; stk.push(ni); }
        }
      }
      contours.push(pts);
    }

    if (contours.length === 0) return _fail('else', inp, 'no_contours_after_canny', { edgePxCount });
    const ge6 = contours.filter(p => p.length >= 6);
    if (ge6.length === 0) return _fail('else', inp, 'no_contour_ge6', {
      contourCount: contours.length,
      maxContourLen: Math.max(...contours.map(p=>p.length))
    });

    // Photometric hint
    const ir2 = (outerR * 0.5) ** 2;
    let iS = 0, iC = 0, oS = 0, oC = 0;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      if (!zone[y * w + x]) continue;
      const dx = x - cx, dy = y - cy, r2 = dx * dx + dy * dy;
      if (r2 < ir2)               { iS += blurred2[y * w + x]; iC++; }
      else if (r2 <= outerR ** 2) { oS += blurred2[y * w + x]; oC++; }
    }
    const innerDarker = (iC && oC) && iS / iC < oS / oC - 1.5;

    // Fit + score
    let best = null, bestScore = -1;
    let nFitAttempts = 0, nFitNull = 0, nFailSize = 0, nFailDist = 0, nFailAspect = 0;
    for (const pts of contours) {
      if (pts.length < 6) continue;
      nFitAttempts++;
      const el = U.fitEllipse(pts);
      if (!el) { nFitNull++; continue; }
      const { cx: ex, cy: ey, a, b } = el;
      if (a < innerR || a > outerR) { nFailSize++; continue; }
      const aspect  = b / (a + 1e-6);
      if (aspect < 0.2) { nFailAspect++; continue; }
      const dist    = Math.hypot(ex - cx, ey - cy);
      const maxDist = innerDarker ? irisRad * 0.55 : irisRad * 0.50;
      if (dist > maxDist) { nFailDist++; continue; }
      const score = aspect * (1 - dist / (maxDist + 1)) * Math.log(pts.length + 1);
      if (score > bestScore) { bestScore = score; best = { el, pts }; }
    }

    if (!best) {
      const detail = { nFitAttempts, nFitNull, nFailSize, nFailDist, nFailAspect,
                       innerDarker, irisRad: irisRad.toFixed(1),
                       innerR: innerR.toFixed(1), outerR: outerR.toFixed(1) };
      if (nFitNull === nFitAttempts) return _fail('else', inp, 'fitEllipse_all_null', detail);
      if (nFailSize > 0 && nFailDist === 0 && nFailAspect === 0) return _fail('else', inp, 'all_filtered_size', detail);
      if (nFailDist > 0 && nFailSize === 0) return _fail('else', inp, 'all_filtered_dist', detail);
      return _fail('else', inp, 'all_filtered_mixed', detail);
    }

    const { a, b, cx: ex, cy: ey } = best.el;
    const pupilR = (a + b) / 2;
    const aspect = b / (a + 1e-6);
    const dist   = Math.hypot(ex - cx, ey - cy);
    const conf   = Math.min(1, aspect) * Math.max(0, 1 - dist / (irisRad * 0.6));

    const dbg = U.makeDebugRGBA(w, h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x, idx = i * 4;
      if (!zone[i])    { dbg[idx]=10; dbg[idx+1]=10;  dbg[idx+2]=16;  dbg[idx+3]=255; continue; }
      if (specFill[i]) { dbg[idx]=0;  dbg[idx+1]=180; dbg[idx+2]=200; dbg[idx+3]=255; continue; }
      if (edges[i])    { dbg[idx]=255;dbg[idx+1]=180; dbg[idx+2]=30;  dbg[idx+3]=255; continue; }
      const v = Math.round(blurred2[i] * 0.4);
      dbg[idx]=v; dbg[idx+1]=v; dbg[idx+2]=v; dbg[idx+3]=255;
    }
    for (const [px, py] of best.pts) {
      const idx = (py * w + px) * 4;
      dbg[idx]=60; dbg[idx+1]=255; dbg[idx+2]=120; dbg[idx+3]=255;
    }
    U.drawCircle(dbg, w, h, ex, ey, pupilR, [0, 255, 220, 220]);

    const out = { pupilRadPx: pupilR, pupilMm: pupilR * 2 * mmPerPx, confidence: conf,
                  debugPixels: dbg, failStage: null };
    _silentLog('else', inp, out);
    return out;
  }
};

// ── Diagnostic helpers ────────────────────────────────────────────────────────

function _fail(algoId, inp, stage, detail = {}) {
  const out = { pupilRadPx: null, pupilMm: null, confidence: 0,
                debugPixels: null, failStage: stage, failDetail: detail };
  _silentLog(algoId, inp, out);
  return out;
}

function _silentLog(algoId, inp, result) {
  try {
    if (!window._plTrialActive) return;
    const row = {
      t:          Date.now(),
      algo:       algoId,
      side:       inp._side || '?',
      failStage:  result.failStage || null,
      failDetail: result.failDetail || null,
      conf:       result.confidence ?? null,
      pupilMm:    result.pupilMm   ?? null,
      irisRad:    inp.irisRadPx != null ? +inp.irisRadPx.toFixed(1) : null,
    };
    const log = JSON.parse(localStorage.getItem('pl_algo_log') || '[]');
    log.push(row);
    if (log.length > 2000) log.splice(0, log.length - 2000);
    localStorage.setItem('pl_algo_log', JSON.stringify(log));
  } catch(e) { /* never throw */ }
}

})();
