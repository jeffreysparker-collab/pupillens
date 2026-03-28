/**
 * PuRe  —  Pure Edge-Based Pupil Detection  (PupilExt port)
 * ===========================================================
 * v2 — ambient RGB / webcam tuning (curvature fix, spec fill, double blur etc.)
 * v2.1 — failStage diagnostic
 * v2.2 — silent localStorage diagnostic logger (_silentLog)
 *
 * Every return includes failStage: null on success, or a string:
 *   "no_zone" | "no_edges" | "no_segments_ge_minseg" | "no_arc_segs" |
 *   "no_merged_ge6" | "no_valid_ellipse" | null
 *
 * Diagnostic log: written to localStorage key 'pl_algo_log' only during trials
 * (when window._plTrialActive === true). Download via the Algo Log button in app.
 */
(function(){
'use strict';
const U = window.PupilAlgoUtils;

const MIN_SEG_LEN = 5;
const MERGE_GAP   = 8;

window.PupilAlgos['pure'] = {
  id:    'pure',
  label: 'PuRe',

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
    if (!zone.some(Boolean)) return _fail('pure', inp, 'no_zone');

    // p1-p99 stretch
    const vals = [];
    for (let i = 0; i < gray.length; i++) if (zone[i]) vals.push(gray[i]);
    if (vals.length < 20) return _fail('pure', inp, 'no_zone');
    vals.sort((a, b) => a - b);
    const lo = vals[Math.floor(vals.length * 0.01)] || 0;
    const hi = vals[Math.floor(vals.length * 0.99)] || 255;
    const span = hi - lo || 1;
    const norm = new Float32Array(w * h);
    for (let i = 0; i < w * h; i++)
      if (zone[i]) norm[i] = Math.min(255, (gray[i] - lo) / span * 255);

    // Specular flood-fill
    const nv = [];
    for (let i = 0; i < w * h; i++) if (zone[i]) nv.push(norm[i]);
    nv.sort((a, b) => a - b);
    const zmean = nv.reduce((s, v) => s + v, 0) / nv.length;
    const p90   = nv[Math.floor(nv.length * 0.90)];
    const p80   = nv[Math.floor(nv.length * 0.80)];
    const masked   = new Float32Array(norm);
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
    if (edgePxCount === 0) return _fail('pure', inp, 'no_edges', { irisRad: irisRad.toFixed(1) });

    // Chain tracing — admit ≥ 3
    const visited  = new Uint8Array(w * h);
    const segments = [];
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      if (!edges[y * w + x] || visited[y * w + x]) continue;
      const chain = [];
      let cx2 = x, cy2 = y;
      while (true) {
        const idx = cy2 * w + cx2;
        if (visited[idx]) break;
        visited[idx] = 1;
        chain.push([cx2, cy2]);
        let nx2 = -1, ny2 = -1;
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          if (!dy && !dx) continue;
          const ny = cy2 + dy, nx = cx2 + dx;
          if (ny < 0 || ny >= h || nx < 0 || nx >= w) continue;
          if (edges[ny * w + nx] && !visited[ny * w + nx]) { ny2 = ny; nx2 = nx; break; }
        }
        if (nx2 < 0) break;
        cx2 = nx2; cy2 = ny2;
      }
      if (chain.length >= 3) segments.push(chain);
    }

    const segsGeMinseg = segments.filter(s => s.length >= MIN_SEG_LEN).length;
    if (segsGeMinseg === 0)
      return _fail('pure', inp, 'no_segments_ge_minseg', {
        totalSegs: segments.length, edgePxCount,
        maxSegLen: segments.length ? Math.max(...segments.map(s=>s.length)) : 0
      });

    // Curvature filter — circumradius formula
    function segCurvatureR(pts) {
      if (pts.length < 3) return Infinity;
      let sumR = 0, cnt = 0;
      for (let i = 1; i < pts.length - 1; i++) {
        const dx1 = pts[i][0]-pts[i-1][0], dy1 = pts[i][1]-pts[i-1][1];
        const dx2 = pts[i+1][0]-pts[i][0],  dy2 = pts[i+1][1]-pts[i][1];
        const l1 = Math.hypot(dx1, dy1), l2 = Math.hypot(dx2, dy2);
        const l3 = Math.hypot(pts[i+1][0]-pts[i-1][0], pts[i+1][1]-pts[i-1][1]);
        const area = Math.abs(dx1 * dy2 - dy1 * dx2) / 2;
        if (area < 1e-6 || l1 < 0.5 || l2 < 0.5) continue;
        sumR += (l1 * l2 * l3) / (4 * area);
        cnt++;
      }
      return cnt ? sumR / cnt : Infinity;
    }

    const arcSegs = segments.filter(seg => {
      if (seg.length < MIN_SEG_LEN) return false;
      const R = segCurvatureR(seg);
      return R >= innerR * 0.4 && R <= outerR * 2.5;
    });

    if (arcSegs.length === 0) {
      const Rs = segments.filter(s=>s.length>=MIN_SEG_LEN).map(s=>segCurvatureR(s).toFixed(1));
      return _fail('pure', inp, 'no_arc_segs', {
        segsGeMinseg, sampleRs: Rs.slice(0,10),
        innerR: innerR.toFixed(1), outerR: outerR.toFixed(1)
      });
    }

    // Segment merging
    function segMeanAngle(seg) {
      let sx = 0, sy = 0;
      for (let i = 1; i < seg.length; i++) {
        sx += seg[i][0]-seg[i-1][0]; sy += seg[i][1]-seg[i-1][1];
      }
      return Math.atan2(sy, sx);
    }
    const merged = [...arcSegs];
    let didMerge = true;
    while (didMerge) {
      didMerge = false;
      outer:
      for (let i = 0; i < merged.length; i++) {
        for (let j = i + 1; j < merged.length; j++) {
          const si = merged[i], sj = merged[j];
          const d1 = Math.hypot(si[si.length-1][0]-sj[0][0], si[si.length-1][1]-sj[0][1]);
          const d2 = Math.hypot(sj[sj.length-1][0]-si[0][0], sj[sj.length-1][1]-si[0][1]);
          if (d1 > MERGE_GAP && d2 > MERGE_GAP) continue;
          const ai = segMeanAngle(si), aj = segMeanAngle(sj);
          let da = Math.abs(ai - aj) % Math.PI;
          if (da > Math.PI / 2) da = Math.PI - da;
          if (da > 40 * Math.PI / 180) continue;
          const candidate = d1 <= d2 ? [...si, ...sj] : [...sj, ...si];
          const mcx = candidate.reduce((s,p)=>s+p[0],0)/candidate.length;
          const mcy = candidate.reduce((s,p)=>s+p[1],0)/candidate.length;
          if (Math.hypot(mcx-cx, mcy-cy) > outerR * 0.75) continue;
          merged[i] = candidate; merged.splice(j, 1);
          didMerge = true; break outer;
        }
      }
    }

    const mergedGe6 = merged.filter(s => s.length >= 6);
    if (mergedGe6.length === 0)
      return _fail('pure', inp, 'no_merged_ge6', {
        arcSegCount: arcSegs.length, mergedCount: merged.length,
        maxMergedLen: merged.length ? Math.max(...merged.map(s=>s.length)) : 0
      });

    // Candidate generation
    let bestEl = null, bestScore = -1, bestPts = [];
    let nFitNull = 0, nFailSize = 0, nFailDist = 0, nFailAspect = 0;
    for (const seg of merged) {
      if (seg.length < 6) continue;
      const pts = seg.length > 80
        ? seg.filter((_, i) => i % Math.ceil(seg.length / 80) === 0) : seg;
      const el = U.fitEllipse(pts);
      if (!el) { nFitNull++; continue; }
      const { a, b, cx: ex, cy: ey } = el;
      if (a < innerR || a > outerR) { nFailSize++; continue; }
      const dist = Math.hypot(ex - cx, ey - cy);
      if (dist > outerR * 0.70) { nFailDist++; continue; }
      const aspect = b / (a + 1e-6);
      if (aspect < 0.15) { nFailAspect++; continue; }
      const circ    = Math.PI * (3*(a+b) - Math.sqrt((3*a+b)*(a+3*b)));
      const arcComp = Math.min(1, (pts.length * 1.5) / (circ + 1));
      const score   = aspect * arcComp * (1 - dist/(outerR*0.70+1)) * Math.log(pts.length+1);
      if (score > bestScore) { bestScore = score; bestEl = el; bestPts = seg; }
    }

    if (!bestEl) return _fail('pure', inp, 'no_valid_ellipse',
      { nFitNull, nFailSize, nFailDist, nFailAspect,
        mergedGe6: mergedGe6.length, innerR: innerR.toFixed(1), outerR: outerR.toFixed(1) });

    const { a, b, cx: ex, cy: ey } = bestEl;
    const pupilR = (a + b) / 2;
    const aspect = b / (a + 1e-6);
    const dist   = Math.hypot(ex - cx, ey - cy);
    const conf   = Math.min(1, aspect) * Math.max(0, 1 - dist / (outerR * 0.70));

    const dbg = U.makeDebugRGBA(w, h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x, idx = i * 4;
      if (!zone[i])    { dbg[idx]=10; dbg[idx+1]=10;  dbg[idx+2]=16;  dbg[idx+3]=255; continue; }
      if (specFill[i]) { dbg[idx]=0;  dbg[idx+1]=180; dbg[idx+2]=200; dbg[idx+3]=255; continue; }
      if (edges[i])    { dbg[idx]=255;dbg[idx+1]=180; dbg[idx+2]=30;  dbg[idx+3]=255; continue; }
      const v = Math.round(blurred2[i] * 0.4);
      dbg[idx]=v; dbg[idx+1]=v; dbg[idx+2]=v; dbg[idx+3]=255;
    }
    for (const [px, py] of bestPts) {
      const idx = (py * w + px) * 4;
      dbg[idx]=60; dbg[idx+1]=255; dbg[idx+2]=120; dbg[idx+3]=255;
    }
    U.drawCircle(dbg, w, h, ex, ey, pupilR, [0, 255, 220, 220]);

    const out = { pupilRadPx: pupilR, pupilMm: pupilR * 2 * mmPerPx, confidence: conf,
                  debugPixels: dbg, failStage: null };
    _silentLog('pure', inp, out);
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
