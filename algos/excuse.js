/**
 * ExCuSe  —  Exclusion-Based Pupil Detection  (PupilExt port)
 * =============================================================
 * Reference: Fuhl et al., "ExCuSe: Robust Pupil Detection in Real-World
 *            Scenarios" (DAGM GCPR 2015)
 *
 * Core idea: rather than finding pupil edges directly, EXCLUDE all pixels
 * that are demonstrably NOT the pupil:
 *  1. Exclude bright pixels (corneal reflection / iris)
 *  2. Exclude edge-strong regions outside a plausible pupil zone
 *  3. What remains = candidate dark region
 *  4. Find the largest connected dark blob near the centre
 *  5. Fit ellipse to its convex hull points
 *
 * Works well in high-glare conditions where Canny-based methods fail.
 *
 * ─── Changelog ────────────────────────────────────────────────────────────────
 *
 * v2 — ambient RGB / webcam + iPhone front-camera tuning
 *
 *   Problem: algo rarely fired on webcam / iPhone front camera.  Five issues
 *   identified, one of which was a latent bug:
 *
 *   1. BUG — `brightThr` computed but never used.
 *      The candidate mask only tested `blurred[i] <= darkThr` so corneal
 *      reflections were never excluded from the dark-blob search.  On RGB
 *      cameras catchlights sit right in the pupil zone; after dilation they
 *      merge with the pupil blob and pull its centroid off-centre, causing the
 *      dist > outerR*0.6 guard to reject it.  Fix: exclude pixels above
 *      brightThr from the candidate mask before the gradient-exclusion step.
 *
 *   2. darkThr percentile: 40 → 30.
 *      In ambient RGB the pupil is not as distinctly dark as under IR — the
 *      histogram is wider and flatter.  p40 grabs too many iris/limbal pixels
 *      alongside the pupil, creating large blobs that dwarf the pupil and win
 *      the size-based scoring.  p30 isolates the genuinely dark core better.
 *
 *   3. brightThr percentile: 75 → 70 (now actually applied, see #1).
 *      Lower threshold ensures the full catchlight region is masked before
 *      dilation can bridge it into the dark blob.
 *
 *   4. Gradient exclusion radius: outerR*0.65 → outerR*0.75.
 *      At 0.65 the exclusion circle clips the pupil boundary on larger pupils
 *      or when the MediaPipe iris-centre estimate drifts slightly.  0.75 gives
 *      the dark blob enough room without letting too many lid-edge pixels in.
 *
 *   5. Blob centroid distance gate: outerR*0.6 → outerR*0.65.
 *      Same drift issue as ElSe — webcam gaze variation and head angle shift
 *      the pupil centroid further from the MediaPipe iris estimate than the
 *      original IR-calibrated threshold allowed.
 *
 *   6. Double Gaussian blur (same fix as ElSe v2).
 *      Single pass leaves RGB sensor noise intact; this fragments the dark
 *      blob's boundary and can cause the morphological close to fail to bridge
 *      the pupil into one connected component.  Second pass costs little and
 *      meaningfully improves blob coherence.
 *
 *   7. Aspect ratio floor (final ellipse check): 0.25 → 0.20.
 *      Consistent with ElSe v2 — oblique angles produce squashed fits that
 *      0.25 rejects unnecessarily.
 * ──────────────────────────────────────────────────────────────────────────────
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
    if (!zone.some(Boolean)) return fail();

    // ── v2: Double Gaussian blur ──────────────────────────────────────────────
    // Second pass suppresses RGB sensor noise that fragments the dark blob's
    // boundary and prevents the morphological close from bridging it cleanly.
    const blurred  = new Float32Array(w*h);
    const blurred2 = new Float32Array(w*h);
    U.gaussianBlur5(gray,    blurred,  w, h, zone);
    U.gaussianBlur5(blurred, blurred2, w, h, zone);
    // ─────────────────────────────────────────────────────────────────────────

    // ── v2: percentiles lowered; brightThr now actually applied (was a bug) ──
    const darkThr   = U.percentile(blurred2, zone, 30);   // was 40
    const brightThr = U.percentile(blurred2, zone, 70);   // was 75, now used

    // Build candidate mask: dark AND not a catchlight
    // Previously only tested darkThr; brightThr was computed but never applied,
    // meaning corneal reflections merged into the dark blob after dilation.
    const cand = new Uint8Array(w*h);
    for (let i = 0; i < w*h; i++) {
      if (!zone[i]) continue;
      if (blurred2[i] <= darkThr && blurred2[i] < brightThr) cand[i] = 1;
    }
    // ─────────────────────────────────────────────────────────────────────────

    // Additional exclusion: high-gradient regions outside the pupil zone
    // ── v2: radius loosened outerR*0.65 → outerR*0.75 to avoid clipping the
    //        pupil boundary when MediaPipe iris centre drifts slightly.
    const { mag } = U.sobel(blurred2, w, h, zone);
    let maxG = 0;
    for (let i = 0; i < w*h; i++) maxG = Math.max(maxG, mag[i]);
    const edgeThr = maxG * 0.25;
    for (let y = 1; y < h-1; y++) for (let x = 1; x < w-1; x++) {
      if (!cand[y*w+x]) continue;
      const dx = x-cx, dy = y-cy, r = Math.hypot(dx, dy);
      if (r > outerR*0.75 && mag[y*w+x] > edgeThr) cand[y*w+x] = 0;  // was 0.65
    }

    // Morphological close (dilate 5×5 then erode 5×5) to fill holes
    const closed = new Uint8Array(w*h);
    for (let y = 1; y < h-1; y++) for (let x = 1; x < w-1; x++) {
      if (!cand[y*w+x]) continue;
      for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
        const ny = y+dy, nx = x+dx;
        if (ny >= 0 && ny < h && nx >= 0 && nx < w) closed[ny*w+nx] = 1;
      }
    }
    const eroded = new Uint8Array(w*h);
    for (let y = 2; y < h-2; y++) for (let x = 2; x < w-2; x++) {
      if (!closed[y*w+x]) continue;
      let all = true;
      for (let dy = -2; dy <= 2 && all; dy++) for (let dx = -2; dx <= 2; dx++)
        if (!closed[(y+dy)*w+(x+dx)]) { all = false; break; }
      if (all) eroded[y*w+x] = 1;
    }

    // Connected components on eroded mask
    const label = new Int16Array(w*h);
    let nlbls = 0;
    const compSz=[], compSx=[], compSy=[], compPts=[];
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      if (!eroded[y*w+x] || label[y*w+x]) continue;
      nlbls++;
      const lbl = nlbls, stk = [y*w+x]; label[y*w+x] = lbl;
      let cnt=0, sx=0, sy=0; const pts=[];
      while (stk.length) {
        const i2 = stk.pop(), px = i2%w, py = (i2-px)/w;
        cnt++; sx+=px; sy+=py; pts.push([px,py]);
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          const ny=py+dy, nx=px+dx;
          if (ny<0||ny>=h||nx<0||nx>=w) continue;
          const ni = ny*w+nx;
          if (eroded[ni] && !label[ni]) { label[ni]=lbl; stk.push(ni); }
        }
      }
      compSz.push(cnt); compSx.push(sx); compSy.push(sy); compPts.push(pts);
    }

    if (!nlbls) return fail();

    // Score components: favour large blobs near centre
    // ── v2: distance gate outerR*0.6 → outerR*0.65 (webcam gaze drift)
    let best = null, bestScore = -1;
    for (let l = 0; l < nlbls; l++) {
      const mx = compSx[l]/compSz[l], my = compSy[l]/compSz[l];
      const dist = Math.hypot(mx-cx, my-cy);
      if (dist > outerR*0.65) continue;                   // was 0.6
      const areaR = Math.sqrt(compSz[l]/Math.PI);
      if (areaR < innerR*0.5 || areaR > outerR) continue;
      const score = compSz[l] / (dist+1);
      if (score > bestScore) { bestScore=score; best=l; }
    }
    if (best === null) return fail();

    // Convex-hull approximation: furthest point per angular sector
    const mx = compSx[best]/compSz[best], my = compSy[best]/compSz[best];
    const SECTORS = 36;
    const buckets = Array.from({length:SECTORS}, ()=>null);
    for (const [px,py] of compPts[best]) {
      const a  = Math.atan2(py-my, px-mx);
      const si = ((Math.round(a/2/Math.PI*SECTORS)%SECTORS)+SECTORS)%SECTORS;
      const d  = Math.hypot(px-mx, py-my);
      if (!buckets[si] || d > buckets[si].d) buckets[si] = {x:px, y:py, d};
    }
    const hull = buckets.filter(Boolean).map(b => [b.x, b.y]);
    if (hull.length < 6) return fail();

    const el = U.fitEllipse(hull);
    if (!el) {
      // Fallback: circular estimate from area
      const r    = Math.sqrt(compSz[best]/Math.PI);
      const conf = Math.max(0, 0.4 - Math.hypot(mx-cx, my-cy)/(outerR+1));
      const dbg  = buildDebug(w,h,zone,blurred2,eroded,best,label,mx,my,r,null);
      return { pupilRadPx:r, pupilMm:r*2*mmPerPx, confidence:conf, debugPixels:dbg };
    }

    const { a, b: bEl, cx: ex, cy: ey } = el;
    const pupilR = (a + bEl) / 2;
    const aspect = bEl / (a + 1e-6);
    // ── v2: aspect floor 0.25 → 0.20 (oblique angles produce squashed fits)
    if (aspect < 0.20 || pupilR < innerR || pupilR > outerR) return fail();
    const dist2 = Math.hypot(ex-cx, ey-cy);
    const conf  = Math.min(1, aspect) * Math.max(0, 1 - dist2/(outerR*0.65));

    const dbg = buildDebug(w,h,zone,blurred2,eroded,best,label,ex,ey,pupilR,hull);
    return { pupilRadPx:pupilR, pupilMm:pupilR*2*mmPerPx, confidence:conf, debugPixels:dbg };
  }
};

function buildDebug(w,h,zone,blurred2,eroded,bestLbl,label,cx,cy,r,hull) {
  const dbg = U.makeDebugRGBA(w,h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y*w+x, idx = i*4;
    if (!zone[i]) { dbg[idx]=10; dbg[idx+1]=10; dbg[idx+2]=16; dbg[idx+3]=255; continue; }
    if (eroded[i]) {
      const isB = label[i] === bestLbl+1;
      dbg[idx]=isB?60:90; dbg[idx+1]=isB?255:60; dbg[idx+2]=isB?120:60; dbg[idx+3]=255;
    } else {
      const v = Math.round(blurred2[i]*0.4);
      dbg[idx]=v; dbg[idx+1]=v; dbg[idx+2]=v; dbg[idx+3]=255;
    }
  }
  if (hull) hull.forEach(([px,py]) => {
    const idx = (py*w+px)*4;
    dbg[idx]=255; dbg[idx+1]=200; dbg[idx+2]=0; dbg[idx+3]=255;
  });
  U.drawCircle(dbg, w, h, cx, cy, r, [0,255,220,220]);
  return dbg;
}

function fail() { return { pupilRadPx:null, pupilMm:null, confidence:0, debugPixels:null }; }

})();
