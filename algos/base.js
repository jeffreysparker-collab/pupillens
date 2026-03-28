/**
 * PupilLens Algorithm Plugin Interface
 * =====================================
 * Every algo receives an AlgoInput and must return an AlgoResult.
 *
 * INPUT  (AlgoInput)
 * ------------------
 * {
 *   imageData   : ImageData  — grayscale-capable RGBA crop, square, centred on iris
 *   irisRadPx   : number     — iris radius in pixels (from MediaPipe boundary landmarks)
 *   cx          : number     — iris centre X in crop-local coordinates (= irisRadPx)
 *   cy          : number     — iris centre Y in crop-local coordinates (= irisRadPx)
 *   upperLidY   : number     — upper eyelid Y in crop-local coords (or -1 if unknown)
 *   lowerLidY   : number     — lower eyelid Y in crop-local coords (or size+1 if unknown)
 *   mmPerPx     : number     — mm per pixel derived from HVID iris size calibration
 * }
 *
 * OUTPUT  (AlgoResult)
 * --------------------
 * {
 *   pupilRadPx  : number | null   — pupil radius in pixels
 *   pupilMm     : number | null   — pupil diameter in mm (= pupilRadPx * 2 * mmPerPx)
 *   confidence  : number          — [0,1]  0=failed/uncertain, 1=excellent
 *   debugPixels : Uint8ClampedArray | null  — optional RGBA debug overlay (same size as imageData)
 * }
 *
 * REGISTRATION
 * ------------
 * Each file sets:  window.PupilAlgos['AlgoName'] = { id, label, run }
 *   id     : string  (slug, used as key)
 *   label  : string  (display name)
 *   run    : function(AlgoInput) → AlgoResult
 */

// ── Shared utilities all algos can import via top-level window.PupilAlgoUtils ──
(function(){
'use strict';

const U = {};

// Extract RGBA → greyscale Float32Array, zone mask, within an annular region.
// zone[i]=1 if pixel is inside the annular ring AND not lid-excluded.
U.extractGray = function(imageData, cx, cy, outerR, innerR, upperLidY, lowerLidY, lidBuf=4) {
  const { data, width: w, height: h } = imageData;
  const gray = new Float32Array(w * h);
  const zone = new Uint8Array(w * h);
  const outerR2 = outerR * outerR;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const dx = x - cx, dy = y - cy;
      if (dx*dx + dy*dy > outerR2) continue;
      if (y < upperLidY + lidBuf || y > lowerLidY - lidBuf) continue;
      const i = (y * w + x) * 4;
      gray[y*w+x] = data[i]*0.2126 + data[i+1]*0.7152 + data[i+2]*0.0722;
      zone[y*w+x] = 1;
    }
  }
  return { gray, zone };
};

// Percentile of a Float32Array where mask[i]=1
U.percentile = function(arr, mask, p) {
  const vals = [];
  for (let i = 0; i < arr.length; i++) if (mask[i]) vals.push(arr[i]);
  if (!vals.length) return 0;
  vals.sort((a, b) => a - b);
  return vals[Math.floor(vals.length * p / 100)];
};

// 5×5 separable Gaussian blur [1,4,6,4,1]/16
U.gaussianBlur5 = function(src, dst, w, h, mask) {
  const K = [1,4,6,4,1];
  const tmp = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    if (!mask[y*w+x]) continue;
    let s = 0, n = 0;
    for (let k = -2; k <= 2; k++) {
      const nx = x+k;
      if (nx < 0 || nx >= w || !mask[y*w+nx]) continue;
      s += src[y*w+nx]*K[k+2]; n += K[k+2];
    }
    tmp[y*w+x] = n ? s/n : src[y*w+x];
  }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    if (!mask[y*w+x]) { dst[y*w+x] = 0; continue; }
    let s = 0, n = 0;
    for (let k = -2; k <= 2; k++) {
      const ny = y+k;
      if (ny < 0 || ny >= h || !mask[ny*w+x]) continue;
      s += tmp[ny*w+x]*K[k+2]; n += K[k+2];
    }
    dst[y*w+x] = n ? s/n : tmp[y*w+x];
  }
};

// Sobel gradient magnitude + angle
U.sobel = function(src, w, h, zone) {
  const mag = new Float32Array(w * h);
  const ang = new Float32Array(w * h);
  for (let y = 1; y < h-1; y++) for (let x = 1; x < w-1; x++) {
    if (!zone[y*w+x]) continue;
    const p = (dy, dx) => zone[(y+dy)*w+(x+dx)] ? src[(y+dy)*w+(x+dx)] : src[y*w+x];
    const gx = -p(-1,-1)-2*p(0,-1)-p(1,-1)+p(-1,1)+2*p(0,1)+p(1,1);
    const gy = -p(-1,-1)-2*p(-1,0)-p(-1,1)+p(1,-1)+2*p(1,0)+p(1,1);
    mag[y*w+x] = Math.sqrt(gx*gx + gy*gy);
    ang[y*w+x] = Math.atan2(gy, gx);
  }
  return { mag, ang };
};

// Non-maximum suppression (1-pixel thin edges)
U.nms = function(mag, ang, w, h, zone) {
  const out = new Uint8Array(w * h);
  for (let y = 1; y < h-1; y++) for (let x = 1; x < w-1; x++) {
    if (!zone[y*w+x] || !mag[y*w+x]) continue;
    const deg = ((ang[y*w+x]*180/Math.PI) % 180 + 180) % 180;
    let nx1, ny1, nx2, ny2;
    if      (deg < 22.5  || deg >= 157.5) { nx1=1;ny1=0;  nx2=-1;ny2=0;  }
    else if (deg < 67.5)                   { nx1=1;ny1=1;  nx2=-1;ny2=-1; }
    else if (deg < 112.5)                  { nx1=0;ny1=1;  nx2=0; ny2=-1; }
    else                                   { nx1=-1;ny1=1; nx2=1; ny2=-1; }
    const m  = mag[y*w+x];
    const m1 = zone[(y+ny1)*w+(x+nx1)] ? mag[(y+ny1)*w+(x+nx1)] : 0;
    const m2 = zone[(y+ny2)*w+(x+nx2)] ? mag[(y+ny2)*w+(x+nx2)] : 0;
    if (m >= m1 && m >= m2) out[y*w+x] = 1;
  }
  return out;
};

// Canny hysteresis (strong + connected weak pixels)
U.canny = function(nmsMap, mag, w, h, hiRatio=0.30, loRatio=0.10) {
  let maxG = 0;
  for (let i = 0; i < w*h; i++) if (nmsMap[i]) maxG = Math.max(maxG, mag[i]);
  if (maxG < 1) return new Uint8Array(w * h);
  const hi = maxG * hiRatio, lo = maxG * loRatio;
  const strong = new Uint8Array(w * h);
  const weak   = new Uint8Array(w * h);
  for (let i = 0; i < w*h; i++) {
    if (!nmsMap[i]) continue;
    if (mag[i] >= hi) strong[i] = 1;
    else if (mag[i] >= lo) weak[i] = 1;
  }
  for (let y = 1; y < h-1; y++) for (let x = 1; x < w-1; x++) {
    if (!weak[y*w+x]) continue;
    let ok = false;
    for (let dy = -1; dy <= 1 && !ok; dy++)
      for (let dx = -1; dx <= 1; dx++)
        if (strong[(y+dy)*w+(x+dx)]) ok = true;
    if (ok) strong[y*w+x] = 1;
  }
  return strong;
};

// Morphological thinning — remove junction blobs and L-corners
U.thin = function(edges, w, h) {
  const out = new Uint8Array(edges);
  for (let y = 2; y < h-2; y++) for (let x = 2; x < w-2; x++) {
    if (!out[y*w+x]) continue;
    let n = 0;
    for (let dy = -1; dy <= 1; dy++)
      for (let dx = -1; dx <= 1; dx++)
        if (out[(y+dy)*w+(x+dx)]) n++;
    if (n > 4) { out[y*w+x] = 0; continue; }
    const N=out[(y-1)*w+x], S=out[(y+1)*w+x];
    const E=out[y*w+(x+1)], W=out[y*w+(x-1)];
    if ((E&&S)||(E&&N)||(W&&S)||(W&&N)) out[y*w+x] = 0;
  }
  return out;
};

// Median of array
U.median = function(arr) {
  if (!arr.length) return null;
  const s = arr.slice().sort((a,b) => a-b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m-1]+s[m])/2;
};

// MAD (Median Absolute Deviation)
U.mad = function(arr, med) {
  const diffs = arr.map(v => Math.abs(v - med));
  return U.median(diffs) ?? 0;
};

// Fit ellipse to point set using Fitzgibbon/Halir-Flusser algebraic method
// Returns {cx, cy, a, b, angle} or null.  a >= b (semi-axes).
U.fitEllipse = function(pts) {
  const n = pts.length;
  if (n < 6) return null;
  // Design matrix D (n×6): [x², xy, y², x, y, 1]
  const D = [];
  for (const [x, y] of pts) D.push([x*x, x*y, y*y, x, y, 1]);
  // S = D^T D
  const S = Array.from({length:6}, (_,i) => Array.from({length:6}, (_,j) => {
    let s = 0; for (const row of D) s += row[i]*row[j]; return s;
  }));
  // Constraint matrix C for ellipse (a*c - b²/4 = 1)
  const C = Array.from({length:6}, () => new Array(6).fill(0));
  C[0][2] = C[2][0] = 2; C[1][1] = -1;
  // Solve generalised eigen problem S·v = λ·C·v via simple power iteration won't work here.
  // Use the Halir-Flusser decomposition: split into 3×3 blocks.
  // S = [[S1, S2], [S2^T, S3]], solve S1^{-1} S2 S3^{-1} S2^T a1 = λ a1
  const inv3 = (m) => {
    const [a,b,c,d,e,f,g,h,k] = m.flat();
    const det = a*(e*k-f*h) - b*(d*k-f*g) + c*(d*h-e*g);
    if (Math.abs(det) < 1e-10) return null;
    const inv = [[e*k-f*h, c*h-b*k, b*f-c*e],
                 [f*g-d*k, a*k-c*g, c*d-a*f],
                 [d*h-e*g, b*g-a*h, a*e-b*d]];
    return inv.map(row => row.map(v => v/det));
  };
  const mul3 = (A, B) => A.map(row => [0,1,2].map(j => row.reduce((s,v,k)=>s+v*B[k][j],0)));
  const S1 = [0,1,2].map(i => [0,1,2].map(j => S[i][j]));
  const S2 = [0,1,2].map(i => [3,4,5].map(j => S[i][j]));
  const S3 = [3,4,5].map(i => [3,4,5].map(j => S[i][j]));
  const C1 = [0,1,2].map(i => [0,1,2].map(j => C[i][j]));
  const invS3 = inv3(S3);
  if (!invS3) return null;
  const S2T = [0,1,2].map(i => [0,1,2].map(j => S2[j][i]));
  const invS1 = inv3(S1);
  if (!invS1) return null;
  const M = mul3(inv3(C1) ?? S1, mul3(S1, mul3(invS3.map(r=>r.map(v=>-v)), mul3(S2T, invS1))));
  // Power iteration for dominant eigenvector (sufficient for single ellipse)
  let v = [1, 0, 0];
  for (let iter = 0; iter < 100; iter++) {
    const nv = M.map(row => row.reduce((s, x, k) => s + x*v[k], 0));
    const norm = Math.sqrt(nv.reduce((s,x)=>s+x*x,0));
    if (norm < 1e-12) return null;
    v = nv.map(x => x/norm);
  }
  // a2 = -S3^{-1} S2^T a1
  const a1 = v;
  const a2 = invS3.map(row => -row.reduce((s,x,k)=>s+x*S2T[k].reduce((ss,xx,kk)=>ss+xx*a1[kk],0),0));
  const co = [...a1, ...a2]; // [A,B,C,D,E,F]
  const [A,B,C2,D2,E,F] = co;
  // Convert general conic to geometric params
  const denom = B*B - 4*A*C2;
  if (Math.abs(denom) < 1e-10) return null;
  const cx = (2*C2*D2 - B*E) / denom;
  const cy = (2*A*E - B*D2) / denom;
  const num = 2*(A*E*E + C2*D2*D2 - B*D2*E + denom*F);
  const s1  = Math.sqrt((A-C2)**2 + B*B);
  if (num === 0) return null;
  const a = -Math.sqrt(num*(A+C2+s1)) / denom;
  const b2 = -Math.sqrt(num*(A+C2-s1)) / denom;
  if (isNaN(a) || isNaN(b2) || a <= 0 || b2 <= 0) return null;
  const angle = 0.5 * Math.atan2(B, A-C2);
  return { cx, cy, a: Math.max(a,b2), b: Math.min(a,b2), angle };
};

// Build empty RGBA debug overlay (same dimensions as imageData)
U.makeDebugRGBA = function(w, h) {
  return new Uint8ClampedArray(w * h * 4);
};

// Draw circle outline on RGBA debug buffer
U.drawCircle = function(rgba, w, h, cx, cy, r, color) {
  const steps = Math.max(64, Math.round(2 * Math.PI * r));
  for (let i = 0; i < steps; i++) {
    const a = i / steps * 2 * Math.PI;
    const px = Math.round(cx + Math.cos(a) * r);
    const py = Math.round(cy + Math.sin(a) * r);
    if (px < 0 || px >= w || py < 0 || py >= h) continue;
    const idx = (py * w + px) * 4;
    rgba[idx]   = color[0];
    rgba[idx+1] = color[1];
    rgba[idx+2] = color[2];
    rgba[idx+3] = color[3] ?? 255;
  }
};

window.PupilAlgoUtils = U;
window.PupilAlgos    = window.PupilAlgos || {};

})();
