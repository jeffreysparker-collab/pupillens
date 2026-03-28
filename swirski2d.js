/**
 * Swirski2D  —  Dark-Pupil Ellipse Detection  (PupilExt port)
 * =============================================================
 * Reference: Swirski et al., "Robust Real-Time Pupil Detection in Highly
 *            Off-Axis Images" (ETRA 2012)
 *
 * Core idea:
 *  1. CLAHE-enhanced crop
 *  2. Threshold the darkest N% of pixels within the iris zone
 *  3. Remove specular reflections (bright blobs) from the dark mask
 *  4. Find largest dark connected component near centre
 *  5. Inertia-based ellipse fit (covariance of component pixels)
 *  6. Refine: run Canny on the same ROI, fit algebraic ellipse to edges
 *     inside the covariance estimate
 *
 * Best suited for high-contrast dark-pupil visible-light images.
 */
(function(){
'use strict';
const U = window.PupilAlgoUtils;

const DARK_PCT = 25;   // threshold: darkest N% of iris zone pixels

window.PupilAlgos['swirski2d'] = {
  id:    'swirski2d',
  label: 'Swirski2D',

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

    // CLAHE-like local contrast enhancement (global version for speed)
    const allVals=[];
    for (let i=0;i<gray.length;i++) if (zone[i]) allVals.push(gray[i]);
    if (allVals.length<20) return fail();
    allVals.sort((a,b)=>a-b);
    const pLo=allVals[Math.floor(allVals.length*0.02)]||0;
    const pHi=allVals[Math.floor(allVals.length*0.98)]||255;
    const span=pHi-pLo||1;
    const enhanced=new Float32Array(w*h);
    for (let i=0;i<w*h;i++)
      if (zone[i]) enhanced[i]=Math.min(255,Math.max(0,(gray[i]-pLo)/span*255));

    // Dark threshold
    const darkThr = U.percentile(enhanced, zone, DARK_PCT);

    // Specular removal: flood-fill bright blobs above p92
    const p92 = U.percentile(enhanced, zone, 92);
    const specMask = new Uint8Array(w*h);
    const specStk = [];
    for (let i=0;i<w*h;i++) if (zone[i]&&enhanced[i]>=p92&&!specMask[i]) specStk.push(i);
    while (specStk.length){
      const idx=specStk.pop(); if (specMask[idx]) continue;
      specMask[idx]=1;
      const fx=idx%w, fy=(idx-fx)/w;
      for (let dy=-1;dy<=1;dy++) for (let dx=-1;dx<=1;dx++){
        const ny=fy+dy, nx=fx+dx;
        if (ny<0||ny>=h||nx<0||nx>=w) continue;
        const ni=ny*w+nx;
        if (zone[ni]&&!specMask[ni]&&enhanced[ni]>p92*0.85) specStk.push(ni);
      }
    }

    // Dark mask (exclude specular)
    const darkMask = new Uint8Array(w*h);
    for (let i=0;i<w*h;i++)
      if (zone[i] && !specMask[i] && enhanced[i] <= darkThr) darkMask[i]=1;

    // Morphological open (erode then dilate) to remove noise specks
    const kern=2;
    const eroded=new Uint8Array(w*h);
    for (let y=kern;y<h-kern;y++) for (let x=kern;x<w-kern;x++){
      if (!darkMask[y*w+x]) continue;
      let all=true;
      for (let dy=-kern;dy<=kern&&all;dy++) for (let dx=-kern;dx<=kern;dx++)
        if (!darkMask[(y+dy)*w+(x+dx)]){all=false;break;}
      if (all) eroded[y*w+x]=1;
    }
    const opened=new Uint8Array(w*h);
    for (let y=0;y<h;y++) for (let x=0;x<w;x++){
      if (!eroded[y*w+x]) continue;
      for (let dy=-kern;dy<=kern;dy++) for (let dx=-kern;dx<=kern;dx++){
        const ny=y+dy, nx=x+dx;
        if (ny>=0&&ny<h&&nx>=0&&nx<w) opened[ny*w+nx]=1;
      }
    }

    // Connected components
    const label=new Int16Array(w*h); let nlbls=0;
    const csz=[],scx=[],scy=[],sxx=[],syy=[],sxy=[];
    for (let y=0;y<h;y++) for (let x=0;x<w;x++){
      if (!opened[y*w+x]||label[y*w+x]) continue;
      nlbls++;
      const lbl=nlbls, stk=[y*w+x]; label[y*w+x]=lbl;
      let cnt=0,sx=0,sy=0,xx=0,yy=0,xy=0;
      while (stk.length){
        const i2=stk.pop(), px=i2%w, py=(i2-px)/w;
        cnt++;sx+=px;sy+=py;xx+=px*px;yy+=py*py;xy+=px*py;
        for (let dy=-1;dy<=1;dy++) for (let dx=-1;dx<=1;dx++){
          const ny=py+dy, nx=px+dx;
          if (ny<0||ny>=h||nx<0||nx>=w) continue;
          const ni=ny*w+nx;
          if (opened[ni]&&!label[ni]){label[ni]=lbl;stk.push(ni);}
        }
      }
      csz.push(cnt);scx.push(sx);scy.push(sy);sxx.push(xx);syy.push(yy);sxy.push(xy);
    }

    // Score components
    let best=null, bestScore=-1;
    for (let l=0;l<nlbls;l++){
      const mx=scx[l]/csz[l], my=scy[l]/csz[l];
      const dist=Math.hypot(mx-cx,my-cy);
      if (dist>outerR*0.6) continue;
      const areaR=Math.sqrt(csz[l]/Math.PI);
      if (areaR<innerR*0.5||areaR>outerR) continue;
      const score=csz[l]/(dist+1);
      if (score>bestScore){bestScore=score;best=l;}
    }
    if (best===null) return fail();

    // Covariance ellipse from second moments
    const n=csz[best];
    const mx=scx[best]/n, my=scy[best]/n;
    const vx=sxx[best]/n - mx*mx;
    const vy=syy[best]/n - my*my;
    const vxy=sxy[best]/n - mx*my;
    const tr=vx+vy, det=vx*vy-vxy*vxy;
    if (tr<0||det<0) return fail();
    const disc=Math.sqrt(Math.max(0,(tr/2)**2-det));
    const lam1=tr/2+disc, lam2=tr/2-disc;
    if (lam1<0||lam2<0) return fail();
    const aEl=2*Math.sqrt(lam1), bEl=2*Math.sqrt(lam2);
    if (aEl<innerR||aEl>outerR) return fail();

    // Refine with Canny edge fit within covariance estimate×1.3
    const refR = aEl * 1.3;
    const blurred=new Float32Array(w*h);
    U.gaussianBlur5(enhanced, blurred, w, h, zone);
    const {mag, ang}=U.sobel(blurred,w,h,zone);
    const nmsMap=U.nms(mag,ang,w,h,zone);
    const edges=U.canny(nmsMap,mag,w,h);

    // Collect edge points inside refinement radius
    const ePts=[];
    for (let y=0;y<h;y++) for (let x=0;x<w;x++){
      if (!edges[y*w+x]) continue;
      if (Math.hypot(x-mx,y-my)<=refR) ePts.push([x,y]);
    }
    let pupilR, finalCx, finalCy, conf;
    if (ePts.length>=6){
      const el2=U.fitEllipse(ePts);
      if (el2&&el2.a>=innerR&&el2.a<=outerR){
        pupilR=(el2.a+el2.b)/2;
        finalCx=el2.cx; finalCy=el2.cy;
        const aspect=el2.b/(el2.a+1e-6);
        const d=Math.hypot(finalCx-cx,finalCy-cy);
        conf=Math.min(1,aspect)*Math.max(0,1-d/(outerR*0.65));
      }
    }
    if (!pupilR) {
      pupilR=(aEl+bEl)/2;
      finalCx=mx; finalCy=my;
      const aspect=bEl/(aEl+1e-6);
      const d=Math.hypot(mx-cx,my-cy);
      conf=Math.min(1,aspect)*0.7*Math.max(0,1-d/(outerR*0.65));
    }

    // Debug
    const dbg=U.makeDebugRGBA(w,h);
    for (let y=0;y<h;y++) for (let x=0;x<w;x++){
      const i=y*w+x, idx=i*4;
      if (!zone[i]){dbg[idx]=10;dbg[idx+1]=10;dbg[idx+2]=16;dbg[idx+3]=255;}
      else if (specMask[i]){dbg[idx]=0;dbg[idx+1]=180;dbg[idx+2]=200;dbg[idx+3]=255;}
      else if (opened[i]&&label[i]===best+1){dbg[idx]=60;dbg[idx+1]=255;dbg[idx+2]=120;dbg[idx+3]=255;}
      else if (edges[i]){dbg[idx]=255;dbg[idx+1]=180;dbg[idx+2]=30;dbg[idx+3]=255;}
      else {const v=Math.round(blurred[i]*0.4);dbg[idx]=v;dbg[idx+1]=v;dbg[idx+2]=v;dbg[idx+3]=255;}
    }
    U.drawCircle(dbg,w,h,finalCx,finalCy,pupilR,[0,255,220,220]);

    return {pupilRadPx:pupilR, pupilMm:pupilR*2*mmPerPx, confidence:conf, debugPixels:dbg};
  }
};

function fail(){return{pupilRadPx:null,pupilMm:null,confidence:0,debugPixels:null};}

})();
