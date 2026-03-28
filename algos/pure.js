/**
 * PuRe  —  Pure Edge-Based Pupil Detection  (PupilExt port)
 * ===========================================================
 * Reference: Santini et al., "PuRe: Robust Pupil Detection for
 *            Real-World Gaze Estimation" (ETRA 2018)
 *
 * Key innovations over ElSe:
 *  1. Curved edge filtering: retain only edge segments whose local
 *     curvature is consistent with a circular arc of pupil-plausible radius
 *  2. Edge segment merging: groups nearby co-curved segments
 *  3. Candidates ranked by combined arc completeness + circularity + size
 *
 * Especially robust on partial occlusions (heavy eyelids).
 *
 * ─── Changelog ────────────────────────────────────────────────────────────────
 * v2 — ambient RGB / webcam tuning
 *
 *  1. Specular flood-fill added (ported from starburst).
 *     Small bright blobs produce short high-curvature edge segments that
 *     pass the arc filter and score well.  Patching catchlights before edge
 *     detection eliminates this entire class of false positives.
 *
 *  2. Double Gaussian blur before Sobel.
 *     RGB sensor noise fragments the pupil-arc chain into many short dead-end
 *     segments.  Most are dropped by MIN_SEG_LEN before the curvature filter
 *     even sees them.  A second blur pass suppresses the fragmentation.
 *
 *  3. Chain tracer admission threshold: MIN_SEG_LEN/2 (=4) → 3.
 *     Even with double blur, short arc fragments occur.  Lowering the raw
 *     admission floor means the curvature filter is the quality gate, not
 *     raw length.
 *
 *  4. MIN_SEG_LEN: 8 → 5.
 *     Consistent with fix 3 — shorter valid arcs can now reach the curvature
 *     filter and ellipse fit.
 *
 *  5. BUG: curvature formula underestimates κ for pixel-step chains.
 *     Original: κ = |cross| / (l1² · l2).  For 1px steps this gives
 *     κ ≈ |cross|, which is not normalised for arc length and produces
 *     wrong radius estimates that cause valid pupil arcs to fail the
 *     [innerR*0.5, outerR*2] range check.
 *     Fix: three-point circumradius R = (l1·l2·l3) / (4·area), which is
 *     geometrically exact at any step size.  R accumulated directly avoids
 *     the reciprocal instability of the original 1/(mean κ) approach.
 *
 *  6. Segment merge centroid guard added.
 *     Lid edges have tangent directions nearly parallel to pupil-arc tangents
 *     (both roughly horizontal).  The angle check alone let lid segments
 *     merge with pupil arcs.  After the angle check we now verify the merged
 *     segment centroid stays within outerR*0.65 of the iris centre.
 *
 *  7. Candidate dist gate: outerR*0.6 → outerR*0.65.
 *     Webcam gaze variation shifts the fitted ellipse centre further from the
 *     MediaPipe iris estimate than 0.6 allows.
 *
 *  8. Aspect ratio floor: 0.25 → 0.20.
 *     Oblique angles produce squashed ellipses that 0.25 rejects needlessly.
 * ──────────────────────────────────────────────────────────────────────────────
 */
(function(){
'use strict';
const U = window.PupilAlgoUtils;

const MIN_SEG_LEN = 5;   // was 8
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
    if (!zone.some(Boolean)) return fail();

    // Normalise contrast (p1–p99 stretch)
    const vals = [];
    for (let i = 0; i < gray.length; i++) if (zone[i]) vals.push(gray[i]);
    if (vals.length < 20) return fail();
    vals.sort((a, b) => a - b);
    const lo   = vals[Math.floor(vals.length * 0.01)] || 0;
    const hi   = vals[Math.floor(vals.length * 0.99)] || 255;
    const span = hi - lo || 1;
    const norm = new Float32Array(w * h);
    for (let i = 0; i < w * h; i++)
      if (zone[i]) norm[i] = Math.min(255, (gray[i] - lo) / span * 255);

    // ── v2: Specular flood-fill ───────────────────────────────────────────────
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

    // ── v2: Double Gaussian → Sobel → NMS → Canny → thin ─────────────────────
    const blurred  = new Float32Array(w * h);
    const blurred2 = new Float32Array(w * h);
    U.gaussianBlur5(masked,  blurred,  w, h, zone);
    U.gaussianBlur5(blurred, blurred2, w, h, zone);
    const { mag, ang } = U.sobel(blurred2, w, h, zone);
    const nmsMap = U.nms(mag, ang, w, h, zone);
    let   edges  = U.canny(nmsMap, mag, w, h);
    edges          = U.thin(edges, w, h);

    // Chain tracing — admit ≥ 3 pts; curvature filter is the quality gate
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
      if (chain.length >= 3) segments.push(chain); // was MIN_SEG_LEN/2 = 4
    }

    // ── v2: Curvature filter using three-point circumradius ───────────────────
    // Original κ = |cross|/(l1²·l2) is not normalised for step length and
    // gives wrong R estimates for pixel-step chains.  Circumradius is exact.
    function segCurvatureR(pts) {
      if (pts.length < 3) return Infinity;
      let sumR = 0, cnt = 0;
      for (let i = 1; i < pts.length - 1; i++) {
        const dx1 = pts[i][0] - pts[i-1][0], dy1 = pts[i][1] - pts[i-1][1];
        const dx2 = pts[i+1][0] - pts[i][0],  dy2 = pts[i+1][1] - pts[i][1];
        const l1   = Math.hypot(dx1, dy1);
        const l2   = Math.hypot(dx2, dy2);
        const l3   = Math.hypot(pts[i+1][0] - pts[i-1][0], pts[i+1][1] - pts[i-1][1]);
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
      return R >= innerR * 0.5 && R <= outerR * 2.0;
    });

    // Segment merging
    function segMeanAngle(seg) {
      let sx = 0, sy = 0;
      for (let i = 1; i < seg.length; i++) {
        sx += seg[i][0] - seg[i-1][0]; sy += seg[i][1] - seg[i-1][1];
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
          const endI   = si[si.length - 1], startJ = sj[0];
          const endJ   = sj[sj.length - 1], startI = si[0];
          const d1 = Math.hypot(endI[0]-startJ[0], endI[1]-startJ[1]);
          const d2 = Math.hypot(endJ[0]-startI[0], endJ[1]-startI[1]);
          if (d1 > MERGE_GAP && d2 > MERGE_GAP) continue;
          const ai = segMeanAngle(si), aj = segMeanAngle(sj);
          let da = Math.abs(ai - aj) % Math.PI;
          if (da > Math.PI / 2) da = Math.PI - da;
          if (da > 40 * Math.PI / 180) continue;
          // ── v2: centroid proximity guard ─────────────────────────────────────
          const candidate = d1 <= d2 ? [...si, ...sj] : [...sj, ...si];
          const mcx = candidate.reduce((s, p) => s + p[0], 0) / candidate.length;
          const mcy = candidate.reduce((s, p) => s + p[1], 0) / candidate.length;
          if (Math.hypot(mcx - cx, mcy - cy) > outerR * 0.65) continue;
          merged[i] = candidate; merged.splice(j, 1);
          didMerge = true; break outer;
        }
      }
    }

    // Candidate generation
    let bestEl = null, bestScore = -1, bestPts = [];
    for (const seg of merged) {
      if (seg.length < 6) continue;
      const pts = seg.length > 80
        ? seg.filter((_, i) => i % Math.ceil(seg.length / 80) === 0)
        : seg;
      const el = U.fitEllipse(pts);
      if (!el) continue;
      const { a, b, cx: ex, cy: ey } = el;
      if (a < innerR || a > outerR) continue;
      const dist = Math.hypot(ex - cx, ey - cy);
      if (dist > outerR * 0.65) continue;                            // was 0.6
      const aspect = b / (a + 1e-6);
      if (aspect < 0.20) continue;                                   // was 0.25
      const circ    = Math.PI * (3*(a+b) - Math.sqrt((3*a+b)*(a+3*b)));
      const arcComp = Math.min(1, (pts.length * 1.5) / (circ + 1));
      const score   = aspect * arcComp * (1 - dist / (outerR * 0.65 + 1)) * Math.log(pts.length + 1);
      if (score > bestScore) { bestScore = score; bestEl = el; bestPts = seg; }
    }

    if (!bestEl) return fail();

    const { a, b, cx: ex, cy: ey } = bestEl;
    const pupilR = (a + b) / 2;
    const aspect = b / (a + 1e-6);
    const dist   = Math.hypot(ex - cx, ey - cy);
    const conf   = Math.min(1, aspect) * Math.max(0, 1 - dist / (outerR * 0.65));

    // Debug
    const dbg = U.makeDebugRGBA(w, h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x, idx = i * 4;
      if (!zone[i])      { dbg[idx]=10; dbg[idx+1]=10;  dbg[idx+2]=16;  dbg[idx+3]=255; continue; }
      if (specFill[i])   { dbg[idx]=0;  dbg[idx+1]=180; dbg[idx+2]=200; dbg[idx+3]=255; continue; }
      if (edges[i])      { dbg[idx]=255;dbg[idx+1]=180; dbg[idx+2]=30;  dbg[idx+3]=255; continue; }
      const v = Math.round(blurred2[i] * 0.4);
      dbg[idx]=v; dbg[idx+1]=v; dbg[idx+2]=v; dbg[idx+3]=255;
    }
    for (const [px, py] of bestPts) {
      const idx = (py * w + px) * 4;
      dbg[idx]=60; dbg[idx+1]=255; dbg[idx+2]=120; dbg[idx+3]=255;
    }
    U.drawCircle(dbg, w, h, ex, ey, pupilR, [0, 255, 220, 220]);

    return { pupilRadPx: pupilR, pupilMm: pupilR * 2 * mmPerPx, confidence: conf, debugPixels: dbg };
  }
};

function fail() { return { pupilRadPx: null, pupilMm: null, confidence: 0, debugPixels: null }; }

})();
