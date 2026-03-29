/**
 * PupilLens Algorithm Plugin Interface
 * =====================================
 * Shared utilities for all algo plugins.
 * window.PupilAlgoUtils  — utility functions
 * window.PupilAlgos      — algo registry {id → {id, label, run}}
 *
 * v2 — fitEllipse rewritten: power-iteration eigensolver replaced with
 *      direct 3×3 real symmetric eigensolver (Cardano / closed-form).
 *      Power iteration diverges on small (6–30 pt), noisy contours typical
 *      of 6–10px pupils, returning semi-axes in the thousands.  The direct
 *      method is O(1), always terminates, and is stable for the 3×3 case.
 */
(function(){
'use strict';

const U = {};

// ── extractGray ────────────────────────────────────────────────────────────
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

// ── percentile ─────────────────────────────────────────────────────────────
U.percentile = function(arr, mask, p) {
  const vals = [];
  for (let i = 0; i < arr.length; i++) if (mask[i]) vals.push(arr[i]);
  if (!vals.length) return 0;
  vals.sort((a, b) => a - b);
  return vals[Math.floor(vals.length * p / 100)];
};

// ── 5×5 separable Gaussian ─────────────────────────────────────────────────
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

// ── Sobel ──────────────────────────────────────────────────────────────────
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

// ── NMS ────────────────────────────────────────────────────────────────────
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

// ── Canny hysteresis ───────────────────────────────────────────────────────
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

// ── Morphological thinning ─────────────────────────────────────────────────
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

// ── Median ────────────────────────────────────────────────────────────────
U.median = function(arr) {
  if (!arr.length) return null;
  const s = arr.slice().sort((a,b) => a-b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m-1]+s[m])/2;
};

// ── MAD ───────────────────────────────────────────────────────────────────
U.mad = function(arr, med) {
  const diffs = arr.map(v => Math.abs(v - med));
  return U.median(diffs) ?? 0;
};

// ── fitEllipse v2 ─────────────────────────────────────────────────────────
// Halir-Flusser algebraic ellipse fit with direct closed-form 3×3 eigensolver.
//
// Replaces the v1 power-iteration approach which diverged on small (6–30 pt)
// noisy contours, producing semi-axes in the thousands.
//
// The reduced 3×3 eigenproblem  M·v = λ·v  (after Halir-Flusser block
// decomposition) is solved analytically via the Cardano cubic formula.
// This is O(1), always terminates, and is numerically stable for 3×3.
//
// Returns {cx, cy, a, b, angle} with a >= b (semi-axes), or null on failure.
U.fitEllipse = function(pts) {
  const n = pts.length;
  if (n < 6) return null;

  // Build 6×6 scatter matrix S = D^T D
  // D row: [x², xy, y², x, y, 1]
  const S = new Float64Array(36);   // 6×6 row-major
  for (const [x, y] of pts) {
    const row = [x*x, x*y, y*y, x, y, 1];
    for (let i = 0; i < 6; i++)
      for (let j = 0; j < 6; j++)
        S[i*6+j] += row[i] * row[j];
  }

  // Extract 3×3 blocks: S = [[S1,S2],[S2T,S3]]
  const s = (r,c) => S[r*6+c];
  const S1 = [[s(0,0),s(0,1),s(0,2)],[s(1,0),s(1,1),s(1,2)],[s(2,0),s(2,1),s(2,2)]];
  const S2 = [[s(0,3),s(0,4),s(0,5)],[s(1,3),s(1,4),s(1,5)],[s(2,3),s(2,4),s(2,5)]];
  const S3 = [[s(3,3),s(3,4),s(3,5)],[s(4,3),s(4,4),s(4,5)],[s(5,3),s(5,4),s(5,5)]];

  // 3×3 matrix inversion
  const inv3 = (m) => {
    const [[a,b,c],[d,e,f],[g,h,k]] = m;
    const det = a*(e*k-f*h) - b*(d*k-f*g) + c*(d*h-e*g);
    if (Math.abs(det) < 1e-12) return null;
    return [
      [(e*k-f*h)/det, (c*h-b*k)/det, (b*f-c*e)/det],
      [(f*g-d*k)/det, (a*k-c*g)/det, (c*d-a*f)/det],
      [(d*h-e*g)/det, (b*g-a*h)/det, (a*e-b*d)/det],
    ];
  };

  // 3×3 matrix multiply
  const mul3 = (A, B) =>
    A.map(row => [0,1,2].map(j => row.reduce((s,v,k) => s + v*B[k][j], 0)));

  // 3×3 transpose
  const tr3 = (A) => [0,1,2].map(i => [0,1,2].map(j => A[j][i]));

  const iS3 = inv3(S3);
  if (!iS3) return null;

  // M = C1^{-1} · (S1 - S2 · S3^{-1} · S2^T)
  // C1^{-1} for ellipse constraint [0,0,0.5 / 0,-1,0 / 0.5,0,0]:
  //   C1 = [[0,0,2],[0,-1,0],[2,0,0]]  →  C1^{-1} = [[0,0,0.5],[0,-1,0],[0.5,0,0]]
  const C1inv = [[0,0,0.5],[0,-1,0],[0.5,0,0]];
  const S2T  = tr3(S2);
  const T    = mul3(S2, mul3(iS3, S2T));
  // inner = S1 - T
  const inner = S1.map((row, i) => row.map((v, j) => v - T[i][j]));
  const M    = mul3(C1inv, inner);

  // ── Direct 3×3 real-symmetric eigendecomposition ─────────────────────────
  // M is generally NOT symmetric, but we need the eigenvector corresponding
  // to the positive eigenvalue with positive ellipse constraint.
  // Use the characteristic polynomial det(M - λI) = 0 → cubic in λ.
  // Solve with Cardano, then back-substitute for eigenvectors.
  const [[m00,m01,m02],[m10,m11,m12],[m20,m21,m22]] = M;

  // Characteristic polynomial coefficients: -λ³ + (tr)λ² - (...)λ + det = 0
  // Rearranged as λ³ + pλ² + qλ + r = 0  (monic)
  const tr  = m00+m11+m22;
  const p   = -(tr);
  const q   = (m00*m11+m11*m22+m00*m22) - (m01*m10+m12*m21+m02*m20);
  const r2  = -(m00*(m11*m22-m12*m21) - m01*(m10*m22-m12*m20) + m02*(m10*m21-m11*m20));

  // Depressed cubic t³ + pt2·t + qt2 via substitution λ = t - p/3
  const p3 = p/3;
  const pt2 = q - p*p/3;
  const qt2 = r2 - p*q/3 + 2*p*p*p/27;
  const disc = (qt2/2)*(qt2/2) + (pt2/3)*(pt2/3)*(pt2/3);

  let roots;
  if (disc >= 0) {
    // One real root (or repeated)
    const sqD = Math.sqrt(disc);
    const u = Math.cbrt(-qt2/2 + sqD);
    const v = Math.cbrt(-qt2/2 - sqD);
    roots = [u + v - p3];
  } else {
    // Three real roots (casus irreducibilis — use trig method)
    const rho   = Math.sqrt(-(pt2/3)*(pt2/3)*(pt2/3));
    const theta = Math.acos(Math.max(-1, Math.min(1, -qt2/(2*rho))));
    const m     = 2 * Math.cbrt(rho);
    roots = [
      m * Math.cos(theta/3)          - p3,
      m * Math.cos((theta + 2*Math.PI)/3) - p3,
      m * Math.cos((theta + 4*Math.PI)/3) - p3,
    ];
  }

  // For each eigenvalue, find eigenvector and check ellipse positivity constraint
  // a·c - b²/4 > 0  (i.e. 4·a0·a2 - a1² > 0 for conic [a0,a1,a2,...])
  const eigenvector = (lam) => {
    // Null space of (M - λI) via two cross products of rows
    const A = [[m00-lam, m01, m02],[m10, m11-lam, m12],[m20, m21, m22-lam]];
    // Cross-product pairs to find null-space vector
    const cross = (u, v) => [
      u[1]*v[2]-u[2]*v[1],
      u[2]*v[0]-u[0]*v[2],
      u[0]*v[1]-u[1]*v[0],
    ];
    const candidates = [
      cross(A[0], A[1]),
      cross(A[0], A[2]),
      cross(A[1], A[2]),
    ];
    let best = null, bestLen = 0;
    for (const c of candidates) {
      const len = Math.sqrt(c[0]*c[0]+c[1]*c[1]+c[2]*c[2]);
      if (len > bestLen) { bestLen = len; best = c; }
    }
    if (!best || bestLen < 1e-10) return null;
    return best.map(v => v / bestLen);
  };

  let a1 = null;
  for (const lam of roots) {
    const v = eigenvector(lam);
    if (!v) continue;
    // Ellipse positivity: 4*v[0]*v[2] - v[1]^2 > 0
    if (4*v[0]*v[2] - v[1]*v[1] > 0) { a1 = v; break; }
  }
  if (!a1) {
    // Fallback: pick root whose eigenvector maximises the constraint
    let bestConstraint = -Infinity, bestV = null;
    for (const lam of roots) {
      const v = eigenvector(lam);
      if (!v) continue;
      const c = 4*v[0]*v[2] - v[1]*v[1];
      if (c > bestConstraint) { bestConstraint = c; bestV = v; }
    }
    if (!bestV) return null;
    a1 = bestV;
  }

  // a2 = -S3^{-1} S2^T a1
  const S2Ta1 = [0,1,2].map(i => S2T[i].reduce((s,v,k) => s+v*a1[k], 0));
  const a2    = iS3.map(row => -row.reduce((s,v,k) => s+v*S2Ta1[k], 0));
  const coef  = [...a1, ...a2];   // [A, B, C, D, E, F]
  const [A, B, C2, D2, E, F] = coef;

  // Convert general conic to geometric parameters
  const denom = B*B - 4*A*C2;
  if (Math.abs(denom) < 1e-10) return null;
  const cx_e = (2*C2*D2 - B*E)  / denom;
  const cy_e = (2*A*E   - B*D2) / denom;
  const num  = 2*(A*E*E + C2*D2*D2 - B*D2*E + denom*F);
  const s1_e = Math.sqrt(Math.max(0, (A-C2)*(A-C2) + B*B));
  if (num === 0) return null;
  const rawA = -Math.sqrt(Math.max(0, num*(A+C2+s1_e))) / denom;
  const rawB = -Math.sqrt(Math.max(0, num*(A+C2-s1_e))) / denom;
  if (isNaN(rawA) || isNaN(rawB) || rawA <= 0 || rawB <= 0) return null;

  const angle = 0.5 * Math.atan2(B, A-C2);
  return {
    cx: cx_e, cy: cy_e,
    a: Math.max(rawA, rawB),
    b: Math.min(rawA, rawB),
    angle,
  };
};

// ── makeDebugRGBA ──────────────────────────────────────────────────────────
U.makeDebugRGBA = function(w, h) {
  return new Uint8ClampedArray(w * h * 4);
};

// ── drawCircle ─────────────────────────────────────────────────────────────
U.drawCircle = function(rgba, w, h, cx, cy, r, color) {
  const steps = Math.max(64, Math.round(2 * Math.PI * r));
  for (let i = 0; i < steps; i++) {
    const a  = i / steps * 2 * Math.PI;
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
