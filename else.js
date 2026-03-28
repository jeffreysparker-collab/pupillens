/**
 * ElSe  —  Ellipse Selection  (PupilExt port)
 * ============================================
 * Reference: Fuhl et al., "ElSe: Ellipse Selection for Robust Pupil Detection
 *            in Real-World Environments" (ETRA 2016)
 *
 * Key steps:
 *  1. Greyscale crop → histogram equalisation
 *  2. Specular flood-fill (patch corneal reflections before edge detection)
 *  3. Double Gaussian blur → Sobel → NMS → Canny → thin
 *  4. Connected-component contour extraction
 *  5. Fit ellipse to each contour ≥ 6 pts
 *  6. Rank candidates by:
 *     (a) ellipse aspect ratio (circularity)
 *     (b) centroid proximity to iris centre
 *     (c) inner-darker photometric score
 *     (d) size within [innerR, outerR]
 *  7. Best candidate → pupil estimate
 *
 * ─── Changelog ────────────────────────────────────────────────────────────────
 *
 * v2 — ambient RGB / webcam + iPhone front-camera tuning
 *   Problem: algo never fired on ambient-light RGB cameras (webcam, iPhone
 *   front cam).  Original was calibrated for IR-illuminated eye trackers.
 *
 *   1. Added specular flood-fill (ported from starburst.js) before edge
 *      detection.  Corneal catchlights on RGB cameras land right in the pupil
 *      zone and produce false edges that corrupt the contour filter.  We seed
 *      a flood-fill from p90 pixels and replace them with the zone mean,
 *      matching what starburst does successfully.
 *
 *   2. Double Gaussian blur before Sobel.  A single gaussianBlur5 pass leaves
 *      enough sensor noise on RGB to fragment the pupil-boundary edge into
 *      many small disconnected components.  A second pass suppresses this
 *      without meaningfully spreading the pupil edge location.
 *
 *   3. innerDarker threshold: 5 → 1.5.  The original 5-unit gap is calibrated
 *      for IR where the pupil is very dark relative to the iris.  In ambient
 *      RGB the contrast is much lower; 5 almost never triggers, forcing
 *      maxDist to its tight fallback of irisRad*0.35 and rejecting valid fits.
 *      1.5 fires appropriately in normal indoor lighting.
 *
 *   4. maxDist fallback: irisRad*0.35 → irisRad*0.50.  Even when innerDarker
 *      is false, 0.35 is too tight for webcam/phone front cameras where gaze
 *      variation and slight head angle cause the MediaPipe iris-centre estimate
 *      to drift from the true pupil centroid.  0.50 matches the permissive
 *      branch (0.55) more closely while still rejecting edge contours from
 *      lids and lashes.
 *
 *   5. Aspect ratio floor: 0.3 → 0.2.  Oblique camera angles and non-frontal
 *      gaze produce slightly squashed ellipses.  0.3 was discarding these;
 *      0.2 accepts them while still rejecting obviously degenerate fits.
 *
 *   6. Photometric test now uses blurred2 (post-double-blur) rather than the
 *      single-pass blurred, so inner/outer brightness comparison is on the
 *      same smoothed signal used for edge detection.
 * ──────────────────────────────────────────────────────────────────────────────
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

    const mmToPx  = irisRad / 5.85;
    const outerR  = 4.0 * mmToPx;   // search radius (mm-derived, unchanged)
    const innerR  = 1.0 * mmToPx;

    const { gray, zone } = U.extractGray(
      imageData, cx, cy, outerR, 0, upperLidY, lowerLidY, 4
    );

    // Histogram equalisation within zone
    const hist = new Float32Array(256);
    let zCnt = 0;
    for (let i=0; i<gray.length; i++) if (zone[i]) { hist[Math.floor(gray[i])]++; zCnt++; }
    let cumul = 0;
    const lut = new Float32Array(256);
    for (let i=0; i<256; i++) { cumul += hist[i]; lut[i] = (cumul/zCnt)*255; }
    const eq = new Float32Array(gray.length);
    for (let i=0; i<gray.length; i++) if (zone[i]) eq[i] = lut[Math.floor(gray[i])];

    // ── v2: Specular flood-fill ───────────────────────────────────────────────
    // Patch bright corneal reflections with the zone mean before edge
    // detection so they don't seed false contours in the pupil region.
    const zv = [];
    for (let i = 0; i < eq.length; i++) if (zone[i]) zv.push(eq[i]);
    zv.sort((a, b) => a - b);
    const zmean = zv.length ? zv.reduce((s, v) => s + v, 0) / zv.length : 128;
    const p90   = zv[Math.floor(zv.length * 0.90)] ?? 230;
    const p80   = zv[Math.floor(zv.length * 0.80)] ?? 200;
    const masked    = new Float32Array(eq);
    const specFill  = new Uint8Array(w * h);
    const sfStk     = [];
    for (let i = 0; i < w * h; i++) if (zone[i] && masked[i] >= p90) sfStk.push(i);
    while (sfStk.length) {
      const idx = sfStk.pop();
      if (specFill[idx]) continue;
      specFill[idx] = 1;
      masked[idx]   = zmean;
      const fx = idx % w, fy = (idx - fx) / w;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const ny = fy + dy, nx = fx + dx;
        if (ny < 0 || ny >= h || nx < 0 || nx >= w) continue;
        const ni = ny * w + nx;
        if (zone[ni] && !specFill[ni] && masked[ni] > p80) sfStk.push(ni);
      }
    }
    // ─────────────────────────────────────────────────────────────────────────

    // ── v2: Double Gaussian blur → Sobel → NMS → Canny → thin ────────────────
    // Second blur pass suppresses RGB sensor noise that fragments pupil-edge
    // contours into many small disconnected components.
    const blurred  = new Float32Array(w * h);
    const blurred2 = new Float32Array(w * h);
    U.gaussianBlur5(masked,   blurred,  w, h, zone);
    U.gaussianBlur5(blurred,  blurred2, w, h, zone);
    const { mag, ang } = U.sobel(blurred2, w, h, zone);
    const nmsMap = U.nms(mag, ang, w, h, zone);
    let   edges  = U.canny(nmsMap, mag, w, h);
    edges         = U.thin(edges, w, h);
    // ─────────────────────────────────────────────────────────────────────────

    // Extract contour point sets via connected components
    const label    = new Int16Array(w * h);
    const contours = [];   // each = [[x,y],...]
    let nlbls = 0;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      if (!edges[y*w+x] || label[y*w+x]) continue;
      nlbls++;
      const lbl = nlbls, pts = [], stk = [y*w+x];
      label[y*w+x] = lbl;
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

    // ── v2: Photometric hint uses blurred2 (same signal as edge detection) ────
    // Threshold lowered 5 → 1.5: ambient RGB has far less IR-style pupil
    // contrast; the original threshold almost never triggered.
    const ir2 = (outerR * 0.5) ** 2;
    let iS = 0, iC = 0, oS = 0, oC = 0;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      if (!zone[y*w+x]) continue;
      const dx = x - cx, dy = y - cy, r2 = dx*dx + dy*dy;
      if (r2 < ir2)               { iS += blurred2[y*w+x]; iC++; }
      else if (r2 <= outerR ** 2) { oS += blurred2[y*w+x]; oC++; }
    }
    const innerDarker = (iC && oC) && iS/iC < oS/oC - 1.5;  // was 5
    // ─────────────────────────────────────────────────────────────────────────

    // Fit + score each contour
    let best = null, bestScore = -1;
    for (const pts of contours) {
      if (pts.length < 6) continue;
      const el = U.fitEllipse(pts);
      if (!el) continue;
      const { cx: ex, cy: ey, a, b } = el;
      if (a < innerR || a > outerR) continue;
      // ── v2: aspect floor 0.3 → 0.2 (oblique angles produce squashed fits)
      const aspect = b / (a + 1e-6);
      if (aspect < 0.2) continue;
      // ── v2: maxDist fallback 0.35 → 0.50 (webcam gaze drift + head angle)
      const dist    = Math.hypot(ex - cx, ey - cy);
      const maxDist = innerDarker ? irisRad * 0.55 : irisRad * 0.50;
      if (dist > maxDist) continue;
      const score = aspect * (1 - dist / (maxDist + 1)) * Math.log(pts.length + 1);
      if (score > bestScore) { bestScore = score; best = { el, pts }; }
    }

    if (!best) return fail();

    const { a, b, cx: ex, cy: ey } = best.el;
    const pupilR = (a + b) / 2;
    const aspect = b / (a + 1e-6);
    const dist   = Math.hypot(ex - cx, ey - cy);
    const conf   = Math.min(1, aspect) * Math.max(0, 1 - dist / (irisRad * 0.6));

    // Debug
    const dbg = U.makeDebugRGBA(w, h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y*w+x, idx = i*4;
      if (!zone[i])       { dbg[idx]=10; dbg[idx+1]=10;  dbg[idx+2]=16;  dbg[idx+3]=255; continue; }
      if (specFill[i])    { dbg[idx]=0;  dbg[idx+1]=180; dbg[idx+2]=200; dbg[idx+3]=255; continue; }
      if (edges[i])       { dbg[idx]=255;dbg[idx+1]=180; dbg[idx+2]=30;  dbg[idx+3]=255; continue; }
      const v = Math.round(blurred2[i] * 0.4);
      dbg[idx]=v; dbg[idx+1]=v; dbg[idx+2]=v; dbg[idx+3]=255;
    }
    for (const [px, py] of best.pts) {
      const idx = (py*w+px)*4;
      dbg[idx]=60; dbg[idx+1]=255; dbg[idx+2]=120; dbg[idx+3]=255;
    }
    U.drawCircle(dbg, w, h, ex, ey, pupilR, [0, 255, 220, 220]);

    return { pupilRadPx: pupilR, pupilMm: pupilR*2*mmPerPx, confidence: conf, debugPixels: dbg };
  }
};

function fail() { return { pupilRadPx: null, pupilMm: null, confidence: 0, debugPixels: null }; }

})();
