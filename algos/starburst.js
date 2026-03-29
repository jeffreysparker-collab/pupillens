/**
 * Starburst + Median Radius  (PupilLens original)
 * =====================================================
 * v7   — original
 * v7.1 — replaced gaussianBlur5 with inline 3×3 Gaussian
 *
 * At 6–10px pupil radius a 5×5 kernel (σ≈1px) spans nearly the full pupil
 * boundary.  The resulting gradient peak is wide and low, making NMS more
 * likely to suppress it.  A 3×3 kernel (σ≈0.85px) removes 1px noise spikes
 * while keeping the edge peak sharp and detectable.
 *
 * Everything else unchanged — specular fill, connected-component filter,
 * 24-ray starburst, median + MAD quality gate.
 */
(function(){
'use strict';
const U = window.PupilAlgoUtils;

const RAYS        = 48;
const RADIAL_TOL  = 65 * Math.PI / 180;

// Inline 3×3 Gaussian  [1 2 1 / 2 4 2 / 1 2 1] × (1/16)
function blur3(src, dst, w, h, zone) {
  const K = [1,2,1, 2,4,2, 1,2,1];
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    if (!zone[i]) { dst[i] = 0; continue; }
    let s = 0, wt = 0;
    for (let ky = -1; ky <= 1; ky++) for (let kx = -1; kx <= 1; kx++) {
      const ny = y + ky, nx = x + kx;
      if (ny < 0 || ny >= h || nx < 0 || nx >= w) continue;
      const ni = ny * w + nx;
      const k  = K[(ky + 1) * 3 + (kx + 1)];
      s  += src[ni] * k;
      wt += k;
    }
    dst[i] = s / wt;
  }
}

window.PupilAlgos['starburst'] = {
  id:    'starburst',
  label: 'Starburst / Median',

  run(inp) {
    const { imageData, irisRadPx: irisRad, cx, cy,
            upperLidY, lowerLidY, mmPerPx } = inp;
    const { width: w, height: h } = imageData;

    const outerR   = irisRad * 0.95;
    const innerR   = Math.max(2, irisRad * 0.15);
    const mmToPx   = irisRad / 5.85;
    const outerRmm = 4.0 * mmToPx;

    const { gray, zone } = U.extractGray(
      imageData, cx, cy, outerRmm, 0, upperLidY, lowerLidY, 4
    );

    // Specular flood-fill
    const zv = [];
    for (let i = 0; i < gray.length; i++) if (zone[i]) zv.push(gray[i]);
    if (zv.length < 20) return fail();
    zv.sort((a, b) => a - b);
    const zmean = zv.reduce((s, v) => s + v, 0) / zv.length;
    const p90   = zv[Math.floor(zv.length * 0.90)];
    const p80   = zv[Math.floor(zv.length * 0.80)];
    const masked   = new Float32Array(gray);
    const specFill = new Uint8Array(w * h);
    const stk = [];
    for (let i = 0; i < w * h; i++) if (zone[i] && masked[i] >= p90) stk.push(i);
    while (stk.length) {
      const idx = stk.pop();
      if (specFill[idx]) continue;
      specFill[idx] = 1; masked[idx] = zmean;
      const fx = idx % w, fy = (idx - fx) / w;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const ny = fy + dy, nx = fx + dx;
        if (ny < 0 || ny >= h || nx < 0 || nx >= w) continue;
        const ni = ny * w + nx;
        if (zone[ni] && !specFill[ni] && masked[ni] > p80) stk.push(ni);
      }
    }

    // v7.1: single 3×3 blur — preserves gradient peak at small pupil scale
    const blurred = new Float32Array(w * h);
    blur3(masked, blurred, w, h, zone);

    const { mag, ang } = U.sobel(blurred, w, h, zone);
    const nmsMap = U.nms(mag, ang, w, h, zone);
    let edges    = U.canny(nmsMap, mag, w, h);
    edges          = U.thin(edges, w, h);

    // Connected components → filter by size + proximity
    const label = new Int16Array(w * h);
    const csz = [], scx = [], scy = [];
    let nlbls = 0;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      if (!edges[y * w + x] || label[y * w + x]) continue;
      nlbls++;
      const lbl = nlbls, stk2 = [y * w + x]; label[y * w + x] = lbl;
      let cnt = 0, sx = 0, sy = 0;
      while (stk2.length) {
        const i2 = stk2.pop(), px = i2 % w, py = (i2 - px) / w;
        cnt++; sx += px; sy += py;
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          const ny = py + dy, nx = px + dx;
          if (ny < 0 || ny >= h || nx < 0 || nx >= w) continue;
          const ni = ny * w + nx;
          if (edges[ni] && !label[ni]) { label[ni] = lbl; stk2.push(ni); }
        }
      }
      csz.push(cnt); scx.push(sx); scy.push(sy);
    }

    const innerR2 = (outerRmm * 0.5) ** 2;
    let iS = 0, iC = 0, oS = 0, oC = 0;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      if (!zone[y * w + x]) continue;
      const dx = x - cx, dy = y - cy, r2 = dx * dx + dy * dy;
      if (r2 < innerR2)                  { iS += blurred[y * w + x]; iC++; }
      else if (r2 <= outerRmm ** 2)      { oS += blurred[y * w + x]; oC++; }
    }
    const innerDarker = (iC && oC) ? iS / iC < oS / oC - 1.5 : false;

    const goodLbl = new Uint8Array(nlbls + 1);
    for (let l = 1; l <= nlbls; l++) {
      if (csz[l-1] < 5) continue;
      const mx2 = scx[l-1] / csz[l-1], my2 = scy[l-1] / csz[l-1];
      const d = Math.hypot(mx2 - cx, my2 - cy);
      if (d > (innerDarker ? irisRad * 0.65 : irisRad * 0.55)) continue;
      goodLbl[l] = 1;
    }
    const goodEdge = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) if (label[i] && goodLbl[label[i]]) goodEdge[i] = 1;

    // Starburst rays — scan each ray inward, record first radially-aligned edge hit.
    // Do NOT break on non-radial hits; keep scanning so noise/eyelid edges don't
    // stop the ray before it reaches the actual pupil boundary.
    const hitX = [], hitY = [];
    for (let r = 0; r < RAYS; r++) {
      const a = r * 2 * Math.PI / RAYS;
      const cos = Math.cos(a), sin = Math.sin(a);
      for (let t = innerR; t <= outerR; t += 0.7) {
        const px = Math.round(cx + cos * t), py = Math.round(cy + sin * t);
        if (px < 0 || px >= w || py < 0 || py >= h) break;
        if (goodEdge[py * w + px]) {
          const ga = ang[py * w + px];
          let diff = Math.abs(ga - a) % Math.PI;
          if (diff > Math.PI / 2) diff = Math.PI - diff;
          if (diff < RADIAL_TOL) { hitX.push(px); hitY.push(py); break; }
          // non-radial edge (eyelid/noise) — keep scanning, don't break
        }
      }
    }
    if (hitX.length < 8) return fail('insufficient_hits', { hitCount: hitX.length });

    // Median radius — always return a result if we have hits.
    // MAD informs confidence only; hard gating belongs in the downstream
    // Kalman / SG filter which can weight by confidence.
    const radii   = hitX.map((x, i) => Math.hypot(x - cx, hitY[i] - cy));
    const r       = U.median(radii);
    if (r == null) return fail('median_null');
    if (r < innerR || r > outerR)
      return fail('radius_out_of_range', { r: +r.toFixed(1),
                                           innerR: +innerR.toFixed(1),
                                           outerR: +outerR.toFixed(1) });

    const mad     = U.mad(radii, r);
    const quality = r > 0 ? mad / r : 1;
    // Confidence: full at quality=0, zero at quality=QUALITY_CONF_FLOOR (0.35)
    // Frames with high MAD get low weight rather than being dropped entirely.
    const QUALITY_CONF_FLOOR = 0.35;
    const conf = Math.max(0, 1 - quality / QUALITY_CONF_FLOOR)
               * Math.min(1, hitX.length / RAYS);

    const dbg = U.makeDebugRGBA(w, h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x, idx = i * 4;
      if (!zone[i])        { dbg[idx]=10; dbg[idx+1]=10;  dbg[idx+2]=16;  dbg[idx+3]=255; }
      else if (specFill[i]){ dbg[idx]=0;  dbg[idx+1]=180; dbg[idx+2]=200; dbg[idx+3]=255; }
      else if (goodEdge[i]){ dbg[idx]=60; dbg[idx+1]=255; dbg[idx+2]=120; dbg[idx+3]=255; }
      else if (edges[i])   { dbg[idx]=200;dbg[idx+1]=40;  dbg[idx+2]=40;  dbg[idx+3]=255; }
      else { const v = Math.round(blurred[i] * 0.4); dbg[idx]=v; dbg[idx+1]=v; dbg[idx+2]=v; dbg[idx+3]=255; }
    }
    hitX.forEach((hx, i) => {
      const hy = hitY[i]; if (hy < 0 || hy >= h || hx < 0 || hx >= w) return;
      const idx = (hy * w + hx) * 4;
      dbg[idx]=255; dbg[idx+1]=230; dbg[idx+2]=0; dbg[idx+3]=255;
    });
    U.drawCircle(dbg, w, h, cx, cy, r, [0, 255, 220, 220]);

    return { pupilRadPx: r, pupilMm: r * 2 * mmPerPx, confidence: conf,
             debugPixels: dbg, failStage: null,
             quality: +quality.toFixed(3), hitCount: hitX.length };
  }
};

function fail(stage, detail) {
  return { pupilRadPx: null, pupilMm: null, confidence: 0,
           debugPixels: null, failStage: stage || 'unknown', failDetail: detail || {} };
}

})();
