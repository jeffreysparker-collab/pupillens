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
 *
 * v2 — ambient RGB / webcam + iPhone front-camera tuning
 *
 *   Five issues identified, two of which are logic bugs:
 *
 *   1. BUG — chain tracer discards too many short chains before curvature filter.
 *      MIN_SEG_LEN/2 (= 4) was the admission threshold for the raw tracer output.
 *      On RGB webcam the thinned pupil arc is interrupted every few pixels by
 *      sensor noise, so the greedy walk produces many 3–5 px fragments that get
 *      dropped here before the curvature filter can assess them.  Fix: admit
 *      chains ≥ 3 pts into segments[], and lower MIN_SEG_LEN 8 → 5 so the
 *      curvature filter (not raw length) is the primary quality gate.
 *
 *   2. BUG — discrete curvature formula underestimates κ for pixel-step chains.
 *      Original: κ = |cross| / (l1² * l2).  This is not normalised for step
 *      length; with l1≈l2≈1 px it gives κ ≈ |cross| which for a nearly-straight
 *      segment is close to zero, making all near-straight (large-radius) arcs
 *      look correct, but for a tight curve (small R) it also underestimates,
 *      causing valid pupil arcs to be rejected.  Replaced with the three-point
 *      circumradius formula R = (l1 * l2 * l3) / (4 * area), which is
 *      geometrically exact and accumulates R directly rather than κ.  This
 *      avoids reciprocal instability near zero curvature and gives correct
 *      radius estimates regardless of step length.
 *
 *   3. Segment merge contamination: centroid proximity guard added.
 *      The angle check alone allowed lid-edge segments (which have tangents
 *      nearly parallel to pupil-arc tangents) to merge with genuine pupil arcs,
 *      corrupting the ellipse fit.  After the angle check we now verify that the
 *      centroid of the merged segment stays within outerR*0.65 of the iris
 *      centre before accepting the merge.
 *
 *   4. dist gate: outerR*0.6 → outerR*0.65.
 *      Same issue as ElSe/ExCuSe — webcam gaze variation shifts the fitted
 *      ellipse centre further from the MediaPipe iris estimate than 0.6 allows.
 *      0.65 is consistent with the confidence denominator already in use.
 *
 *   5. Specular flood-fill + double Gaussian blur (same as ElSe v2 / ExCuSe v2).
 *      Catchlight edges pass the curvature filter (small bright blobs have high
 *      curvature that can fall inside [innerR*0.5, outerR*2]).  Single blur
 *      leaves fragmented RGB noise edges that fragment chains.  Both fixes
 *      are identical to the other algos.
 * ──────────────────────────────────────────────────────────────────────────────
 */
(function(){
'use strict';
const U = window.PupilAlgoUtils;

const MIN_SEG_LEN  = 5;    // was 8; lowered so curvature filter is the quality gate
const MERGE_GAP    = 8;    // max pixel gap to merge co-curved segments

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

    // Normalise contrast (p1–p99 stretch to [0,255])
    const vals=[];
    for (let i=0;i<gray.length;i++) if (zone[i]) vals.push(gray[i]);
    if (vals.length < 20) return fail();
    vals.sort((a,b)=>a-b);
    const lo = vals[Math.floor(vals.length*0.01)] || 0;
    const hi = vals[Math.floor(vals.length*0.99)] || 255;
    const span = hi - lo || 1;
    const norm = new Float32Array(w*h);
    for (let i=0;i<w*h;i++)
      if (zone[i]) norm[i] = Math.min(255, (gray[i]-lo)/span*255);

    // ── v2: Specular flood-fill ───────────────────────────────────────────────
    // Catchlights produce short high-curvature edge segments that pass the arc
    // filter.  Patch them before edge detection (same approach as ElSe v2).
    const nvals=[];
    for (let i=0;i<w*h;i++) if (zone[i]) nvals.push(norm[i]);
    nvals.sort((a,b)=>a-b);
    const zmean  = nvals.length ? nvals.reduce((s,v)=>s+v,0)/nvals.length : 128;
    const p90    = nvals[Math.floor(nvals.length*0.90)] ?? 230;
    const p80    = nvals[Math.floor(nvals.length*0.80)] ?? 200;
    const masked = new Float32Array(norm);
    const specFill = new Uint8Array(w*h);
    const sfStk = [];
    for (let i=0;i<w*h;i++) if (zone[i] && masked[i]>=p90) sfStk.push(i);
    while (sfStk.length){
      const idx=sfStk.pop();
      if (specFill[idx]) continue;
      specFill[idx]=1; masked[idx]=zmean;
      const fx=idx%w, fy=(idx-fx)/w;
      for (let dy=-1;dy<=1;dy++) for (let dx=-1;dx<=1;dx++){
        const ny=fy+dy, nx=fx+dx;
        if (ny<0||ny>=h||nx<0||nx>=w) continue;
        const ni=ny*w+nx;
        if (zone[ni]&&!specFill[ni]&&masked[ni]>p80) sfStk.push(ni);
      }
    }
    // ─────────────────────────────────────────────────────────────────────────

    // ── v2: Double Gaussian blur → Sobel → NMS → Canny → thin ────────────────
    // Second pass reduces RGB sensor noise that fragments pupil-arc chains.
    const blurred  = new Float32Array(w*h);
    const blurred2 = new Float32Array(w*h);
    U.gaussianBlur5(masked,   blurred,  w, h, zone);
    U.gaussianBlur5(blurred,  blurred2, w, h, zone);
    const {mag, ang} = U.sobel(blurred2, w, h, zone);
    const nmsMap = U.nms(mag, ang, w, h, zone);
    let   edges  = U.canny(nmsMap, mag, w, h);
    edges          = U.thin(edges, w, h);
    // ─────────────────────────────────────────────────────────────────────────

    // Extract edge segments via 8-connected chain tracing
    // ── v2: admit chains ≥ 3 (was MIN_SEG_LEN/2 = 4); curvature filter is
    //        the quality gate, not raw length.
    const visited = new Uint8Array(w*h);
    const segments = [];
    for (let y=0;y<h;y++) for (let x=0;x<w;x++){
      if (!edges[y*w+x]||visited[y*w+x]) continue;
      const chain=[];
      let cx2=x, cy2=y;
      while (true){
        const idx=cy2*w+cx2;
        if (visited[idx]) break;
        visited[idx]=1;
        chain.push([cx2,cy2]);
        let nx2=-1, ny2=-1;
        for (let dy=-1;dy<=1;dy++) for (let dx=-1;dx<=1;dx++){
          if (!dy&&!dx) continue;
          const ny=cy2+dy, nx=cx2+dx;
          if (ny<0||ny>=h||nx<0||nx>=w) continue;
          if (edges[ny*w+nx]&&!visited[ny*w+nx]){ny2=ny;nx2=nx;break;}
        }
        if (nx2<0) break;
        cx2=nx2; cy2=ny2;
      }
      if (chain.length >= 3) segments.push(chain);   // was MIN_SEG_LEN/2
    }

    // ── Curved-arc filtering ─────────────────────────────────────────────────
    // ── v2: replaced biased κ formula with three-point circumradius.
    //    Original: κ = |cross|/(l1²·l2) — underestimates for pixel-step chains.
    //    New: R = (l1·l2·l3)/(4·area) — geometrically exact, accumulates R
    //    directly to avoid reciprocal instability.
    function segCurvatureR(pts) {
      if (pts.length < 3) return Infinity;
      let sumR = 0, cnt = 0;
      for (let i=1; i<pts.length-1; i++){
        const dx1=pts[i][0]-pts[i-1][0], dy1=pts[i][1]-pts[i-1][1];
        const dx2=pts[i+1][0]-pts[i][0], dy2=pts[i+1][1]-pts[i][1];
        const l1 = Math.hypot(dx1,dy1);
        const l2 = Math.hypot(dx2,dy2);
        const l3 = Math.hypot(pts[i+1][0]-pts[i-1][0], pts[i+1][1]-pts[i-1][1]);
        const area = Math.abs(dx1*dy2 - dy1*dx2) / 2;
        if (area < 1e-6 || l1 < 0.5 || l2 < 0.5) continue;
        sumR += (l1 * l2 * l3) / (4 * area);
        cnt++;
      }
      if (!cnt) return Infinity;
      return sumR / cnt;
    }

    const arcSegs = segments.filter(seg => {
      if (seg.length < MIN_SEG_LEN) return false;
      const R = segCurvatureR(seg);
      return R >= innerR*0.5 && R <= outerR*2.0;
    });

    // ── Segment merging ───────────────────────────────────────────────────────
    function segMeanAngle(seg){
      let sx=0, sy=0;
      for (let i=1;i<seg.length;i++){
        sx+=seg[i][0]-seg[i-1][0]; sy+=seg[i][1]-seg[i-1][1];
      }
      return Math.atan2(sy,sx);
    }
    const merged=[...arcSegs];
    let didMerge=true;
    while (didMerge){
      didMerge=false;
      outer:
      for (let i=0;i<merged.length;i++){
        for (let j=i+1;j<merged.length;j++){
          const si=merged[i], sj=merged[j];
          const endI=si[si.length-1], startJ=sj[0];
          const endJ=sj[sj.length-1], startI=si[0];
          const d1=Math.hypot(endI[0]-startJ[0],endI[1]-startJ[1]);
          const d2=Math.hypot(endJ[0]-startI[0],endJ[1]-startI[1]);
          if (d1>MERGE_GAP&&d2>MERGE_GAP) continue;
          // Angle check
          const ai=segMeanAngle(si), aj=segMeanAngle(sj);
          let da=Math.abs(ai-aj)%Math.PI;
          if (da>Math.PI/2) da=Math.PI-da;
          if (da>40*Math.PI/180) continue;
          // ── v2: centroid proximity guard — prevent lid segments merging with
          //        pupil arcs just because they have similar tangent directions.
          const candidate = d1<=d2 ? [...si,...sj] : [...sj,...si];
          const mcx = candidate.reduce((s,p)=>s+p[0],0)/candidate.length;
          const mcy = candidate.reduce((s,p)=>s+p[1],0)/candidate.length;
          if (Math.hypot(mcx-cx, mcy-cy) > outerR*0.65) continue;
          merged[i]=candidate; merged.splice(j,1);
          didMerge=true; break outer;
        }
      }
    }

    // ── Candidate generation ──────────────────────────────────────────────────
    let bestEl=null, bestScore=-1, bestPts=[];
    for (const seg of merged){
      if (seg.length<6) continue;
      const pts=seg.length>80
        ? seg.filter((_,i)=>i%Math.ceil(seg.length/80)===0)
        : seg;
      const el=U.fitEllipse(pts);
      if (!el) continue;
      const {a,b,cx:ex,cy:ey}=el;
      if (a<innerR||a>outerR) continue;
      const dist=Math.hypot(ex-cx,ey-cy);
      if (dist>outerR*0.65) continue;           // was 0.6
      const aspect=b/(a+1e-6);
      if (aspect<0.20) continue;                // was 0.25
      const circ=Math.PI*(3*(a+b)-Math.sqrt((3*a+b)*(a+3*b)));
      const arcComp=Math.min(1,(pts.length*1.5)/(circ+1));
      const score=aspect*arcComp*(1-dist/(outerR*0.65+1))*Math.log(pts.length+1);
      if (score>bestScore){bestScore=score;bestEl=el;bestPts=seg;}
    }

    if (!bestEl) return fail();

    const {a,b,cx:ex,cy:ey}=bestEl;
    const pupilR=(a+b)/2;
    const aspect=b/(a+1e-6);
    const dist=Math.hypot(ex-cx,ey-cy);
    const conf=Math.min(1,aspect)*Math.max(0,1-dist/(outerR*0.65));

    // Debug
    const dbg=U.makeDebugRGBA(w,h);
    for (let y=0;y<h;y++) for (let x=0;x<w;x++){
      const i=y*w+x, idx=i*4;
      if (!zone[i])       { dbg[idx]=10; dbg[idx+1]=10;  dbg[idx+2]=16;  dbg[idx+3]=255; }
      else if (specFill[i]){ dbg[idx]=0;  dbg[idx+1]=180; dbg[idx+2]=200; dbg[idx+3]=255; }
      else if (edges[i])  { dbg[idx]=255; dbg[idx+1]=180; dbg[idx+2]=30;  dbg[idx+3]=255; }
      else { const v=Math.round(blurred2[i]*0.4); dbg[idx]=v;dbg[idx+1]=v;dbg[idx+2]=v;dbg[idx+3]=255; }
    }
    for (const [px,py] of bestPts){
      const idx=(py*w+px)*4; dbg[idx]=60;dbg[idx+1]=255;dbg[idx+2]=120;dbg[idx+3]=255;
    }
    U.drawCircle(dbg,w,h,ex,ey,pupilR,[0,255,220,220]);

    return{pupilRadPx:pupilR, pupilMm:pupilR*2*mmPerPx, confidence:conf, debugPixels:dbg};
  }
};

function fail(){return{pupilRadPx:null,pupilMm:null,confidence:0,debugPixels:null};}

})();
