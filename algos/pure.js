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
 */
(function(){
'use strict';
const U = window.PupilAlgoUtils;

const MIN_SEG_LEN  = 8;    // minimum points in a valid edge segment
const CURVE_WINDOW = 5;    // half-window for local curvature estimate
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

    // Normalise contrast (stretch to [0,255])
    const vals=[];
    for (let i=0;i<gray.length;i++) if (zone[i]) vals.push(gray[i]);
    vals.sort((a,b)=>a-b);
    const lo=vals[Math.floor(vals.length*0.01)]||0;
    const hi=vals[Math.floor(vals.length*0.99)]||255;
    const span=hi-lo||1;
    const norm=new Float32Array(w*h);
    for (let i=0;i<w*h;i++) if (zone[i]) norm[i]=Math.min(255,(gray[i]-lo)/span*255);

    // Gaussian → Sobel → NMS → Canny → thin
    const blurred=new Float32Array(w*h);
    U.gaussianBlur5(norm, blurred, w, h, zone);
    const {mag, ang}=U.sobel(blurred, w, h, zone);
    const nmsMap=U.nms(mag, ang, w, h, zone);
    let   edges =U.canny(nmsMap, mag, w, h);
    edges         =U.thin(edges, w, h);

    // Extract edge segments via 8-connected chain tracing
    const visited=new Uint8Array(w*h);
    const segments=[];
    for (let y=0;y<h;y++) for (let x=0;x<w;x++){
      if (!edges[y*w+x]||visited[y*w+x]) continue;
      // Trace chain from this seed
      const chain=[];
      let cx2=x, cy2=y;
      while (true){
        const idx=cy2*w+cx2;
        if (visited[idx]) break;
        visited[idx]=1;
        chain.push([cx2,cy2]);
        // Find next unvisited edge neighbour
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
      if (chain.length >= MIN_SEG_LEN/2) segments.push(chain);
    }

    // ── Curved-arc filtering ────────────────────────────────────────────────
    // For each segment, compute local curvature. Keep segments where the
    // mean curvature radius is consistent with pupil size [innerR, outerR].
    // Curvature κ at point i = |cross(T_i, T_i+1)| / |T_i|^3 (discrete)
    // Radius of curvature R = 1/κ
    function segCurvatureR(pts) {
      if (pts.length < 3) return Infinity;
      let sumK = 0, cnt = 0;
      for (let i = 1; i < pts.length-1; i++) {
        const dx1=pts[i][0]-pts[i-1][0], dy1=pts[i][1]-pts[i-1][1];
        const dx2=pts[i+1][0]-pts[i][0], dy2=pts[i+1][1]-pts[i][1];
        const cross=dx1*dy2-dy1*dx2;
        const l1=Math.hypot(dx1,dy1), l2=Math.hypot(dx2,dy2);
        if (l1<0.5||l2<0.5) continue;
        const kappa=Math.abs(cross)/(l1*l1*l2+1e-6);
        sumK+=kappa; cnt++;
      }
      if (!cnt) return Infinity;
      return 1/(sumK/cnt+1e-6);
    }

    const arcSegs=segments.filter(seg => {
      if (seg.length < MIN_SEG_LEN) return false;
      const R=segCurvatureR(seg);
      return R >= innerR*0.5 && R <= outerR*2.0;
    });

    // ── Segment merging ──────────────────────────────────────────────────────
    // Two segments merge if endpoints are within MERGE_GAP px AND their
    // mean gradient angles are within 40°.
    function segMeanAngle(seg){
      let sx=0,sy=0; const n=seg.length;
      for (let i=1;i<n;i++){
        const dx=seg[i][0]-seg[i-1][0], dy=seg[i][1]-seg[i-1][1];
        sx+=dx; sy+=dy;
      }
      return Math.atan2(sy,sx);
    }
    const merged=[...arcSegs];
    let didMerge=true;
    while (didMerge){
      didMerge=false;
      for (let i=0;i<merged.length;i++){
        for (let j=i+1;j<merged.length;j++){
          const si=merged[i], sj=merged[j];
          // Check endpoint gaps
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
          // Merge
          const merged2=d1<=d2?[...si,...sj]:[...sj,...si];
          merged[i]=merged2; merged.splice(j,1);
          didMerge=true; break;
        }
        if (didMerge) break;
      }
    }

    // ── Candidate generation ─────────────────────────────────────────────────
    // Fit an ellipse to every group of ≥6 points and score it.
    let bestEl=null, bestScore=-1, bestPts=[];
    for (const seg of merged){
      if (seg.length<6) continue;
      // Subsample if long (keep speed)
      const pts=seg.length>80 ? seg.filter((_,i)=>i%Math.ceil(seg.length/80)===0) : seg;
      const el=U.fitEllipse(pts);
      if (!el) continue;
      const {a,b,cx:ex,cy:ey}=el;
      if (a<innerR||a>outerR) continue;
      const dist=Math.hypot(ex-cx,ey-cy);
      if (dist>outerR*0.6) continue;
      const aspect=b/(a+1e-6);
      if (aspect<0.25) continue;
      // Arc completeness: fraction of the ellipse circumference covered by these pts
      const circ=Math.PI*(3*(a+b)-Math.sqrt((3*a+b)*(a+3*b)));
      const arcComp=Math.min(1, (pts.length*1.5) / (circ+1));
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
      if (!zone[i]){dbg[idx]=10;dbg[idx+1]=10;dbg[idx+2]=16;dbg[idx+3]=255;}
      else if (edges[i]){dbg[idx]=255;dbg[idx+1]=180;dbg[idx+2]=30;dbg[idx+3]=255;}
      else {const v=Math.round(blurred[i]*0.4);dbg[idx]=v;dbg[idx+1]=v;dbg[idx+2]=v;dbg[idx+3]=255;}
    }
    // Best segment in bright green
    for (const [px,py] of bestPts){
      const idx=(py*w+px)*4; dbg[idx]=60;dbg[idx+1]=255;dbg[idx+2]=120;dbg[idx+3]=255;
    }
    U.drawCircle(dbg,w,h,ex,ey,pupilR,[0,255,220,220]);

    return{pupilRadPx:pupilR, pupilMm:pupilR*2*mmPerPx, confidence:conf, debugPixels:dbg};
  }
};

function fail(){return{pupilRadPx:null,pupilMm:null,confidence:0,debugPixels:null};}

})();
