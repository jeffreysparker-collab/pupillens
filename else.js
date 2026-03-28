/**
 * ElSe  —  Ellipse Selection  (PupilExt port)
 * ============================================
 * Reference: Fuhl et al., "ElSe: Ellipse Selection for Robust Pupil Detection
 *            in Real-World Environments" (ETRA 2016)
 *
 * Key steps:
 *  1. Greyscale crop → histogram equalisation
 *  2. Canny edges (Sobel→NMS→hysteresis)
 *  3. Morphological thinning
 *  4. Connected-component contour extraction
 *  5. Fit ellipse to each contour ≥ 6 pts
 *  6. Rank candidates by:
 *     (a) ellipse aspect ratio (circularity)
 *     (b) centroid proximity to iris centre
 *     (c) inner-darker photometric score
 *     (d) size within [innerR, outerR]
 *  7. Best candidate → pupil estimate
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
    const outerR  = 4.0 * mmToPx;
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

    // Gaussian → Sobel → NMS → Canny → thin
    const blurred = new Float32Array(w*h);
    U.gaussianBlur5(eq, blurred, w, h, zone);
    const { mag, ang } = U.sobel(blurred, w, h, zone);
    const nmsMap = U.nms(mag, ang, w, h, zone);
    let   edges  = U.canny(nmsMap, mag, w, h);
    edges         = U.thin(edges, w, h);

    // Extract contour point sets via connected components
    const label = new Int16Array(w*h);
    const contours = [];  // each = [[x,y],...]
    let nlbls = 0;
    for (let y=0;y<h;y++) for (let x=0;x<w;x++){
      if (!edges[y*w+x]||label[y*w+x]) continue;
      nlbls++;
      const lbl=nlbls, pts=[], stk=[y*w+x]; label[y*w+x]=lbl;
      while (stk.length){
        const i2=stk.pop(), px=i2%w, py=(i2-px)/w;
        pts.push([px,py]);
        for (let dy=-1;dy<=1;dy++) for (let dx=-1;dx<=1;dx++){
          const ny=py+dy, nx=px+dx;
          if (ny<0||ny>=h||nx<0||nx>=w) continue;
          const ni=ny*w+nx;
          if (edges[ni]&&!label[ni]){label[ni]=lbl;stk.push(ni);}
        }
      }
      contours.push(pts);
    }

    // Photometric hint
    const ir2 = (outerR*0.5)**2;
    let iS=0,iC=0,oS=0,oC=0;
    for (let y=0;y<h;y++) for (let x=0;x<w;x++){
      if (!zone[y*w+x]) continue;
      const dx=x-cx,dy=y-cy,r2=dx*dx+dy*dy;
      if (r2<ir2){iS+=blurred[y*w+x];iC++;}
      else if (r2<=(outerR**2)){oS+=blurred[y*w+x];oC++;}
    }
    const innerDarker = (iC&&oC) && iS/iC < oS/oC - 5;

    // Fit + score each contour
    let best = null, bestScore = -1;
    for (const pts of contours) {
      if (pts.length < 6) continue;
      const el = U.fitEllipse(pts);
      if (!el) continue;
      const { cx: ex, cy: ey, a, b } = el;
      // Semi-major axis ≈ pupil radius; check size range
      if (a < innerR || a > outerR) continue;
      // Circularity
      const aspect = b / (a + 1e-6);
      if (aspect < 0.3) continue;
      // Distance from iris centre
      const dist = Math.hypot(ex-cx, ey-cy);
      const maxDist = innerDarker ? irisRad*0.55 : irisRad*0.35;
      if (dist > maxDist) continue;
      // Score: high aspect, small distance, larger (more complete) contour
      const score = aspect * (1 - dist/(maxDist+1)) * Math.log(pts.length+1);
      if (score > bestScore) { bestScore = score; best = { el, pts }; }
    }

    if (!best) return fail();

    const { a, b, cx: ex, cy: ey } = best.el;
    const pupilR = (a + b) / 2;   // mean semi-axis ≈ radius for near-circles
    const aspect = b / (a + 1e-6);
    const dist   = Math.hypot(ex-cx, ey-cy);
    // Confidence: penalise low aspect ratio and off-centre fits
    const conf   = Math.min(1, aspect) * Math.max(0, 1 - dist/(irisRad*0.6));

    // Debug
    const dbg = U.makeDebugRGBA(w, h);
    for (let y=0;y<h;y++) for (let x=0;x<w;x++){
      const i=y*w+x,idx=i*4;
      if (!zone[i])  { dbg[idx]=10;dbg[idx+1]=10;dbg[idx+2]=16;dbg[idx+3]=255; continue; }
      if (edges[i])  { dbg[idx]=255;dbg[idx+1]=180;dbg[idx+2]=30;dbg[idx+3]=255; continue; }
      const v=Math.round(blurred[i]*0.4);dbg[idx]=v;dbg[idx+1]=v;dbg[idx+2]=v;dbg[idx+3]=255;
    }
    // Mark best contour in green
    for (const [px,py] of best.pts) {
      const idx=(py*w+px)*4; dbg[idx]=60;dbg[idx+1]=255;dbg[idx+2]=120;dbg[idx+3]=255;
    }
    U.drawCircle(dbg, w, h, ex, ey, pupilR, [0,255,220,220]);

    return { pupilRadPx: pupilR, pupilMm: pupilR*2*mmPerPx, confidence: conf, debugPixels: dbg };
  }
};

function fail() { return { pupilRadPx: null, pupilMm: null, confidence: 0, debugPixels: null }; }

})();
