/**
 * IrisClipRecorder
 * ================
 * Records a side-by-side composited video of both eye crops (iris-clipped)
 * using an OffscreenCanvas drawn each frame, piped into MediaRecorder.
 *
 * Usage:
 *   const rec = new IrisClipRecorder({ cellSize: 120, fps: 30 });
 *   rec.start();                         // begins recording
 *   rec.writeFrame(video, mL, mR);       // call each detection frame
 *   const blob = await rec.stop();       // returns WebM/MP4 blob
 *   rec.download(blob, 'iris_clip.webm');
 *
 * writeFrame inputs:
 *   video  : HTMLVideoElement
 *   mL     : { cx, cy, irisRadPx, pupilRadPx, pupilMm, confidence }  (left eye)
 *   mR     : same structure for right eye
 *   debugL : Uint8ClampedArray RGBA overlay from current algo (optional)
 *   debugR : same for right eye (optional)
 */

class IrisClipRecorder {
  constructor({ cellSize=160, fps=30, showDebug=false }={}) {
    this.cellSize  = cellSize;
    this.fps       = fps;
    this.showDebug = showDebug;
    this._stream   = null;
    this._recorder = null;
    this._chunks   = [];
    this._canvas   = null;
    this._ctx      = null;
    this._running  = false;
    this._startMs  = 0;
    this._frameN   = 0;
  }

  /** Start recording. Returns true if MediaRecorder is available. */
  start() {
    if (this._running) return false;

    const W = this.cellSize * 2;   // L eye | R eye
    const H = this.cellSize + 24;  // eye crop + label strip

    this._canvas = document.createElement('canvas');
    this._canvas.width  = W;
    this._canvas.height = H;
    this._ctx    = this._canvas.getContext('2d', { willReadFrequently: false });

    // Fill once so first frame isn't blank
    this._ctx.fillStyle = '#050508';
    this._ctx.fillRect(0, 0, W, H);

    try {
      this._stream = this._canvas.captureStream(this.fps);
    } catch(e) {
      console.error('IrisClipRecorder: captureStream failed', e);
      return false;
    }

    // Prefer VP9 → VP8 → default
    const mimeTypes = [
      'video/webm;codecs=vp9',
      'video/webm;codecs=vp8',
      'video/webm',
      'video/mp4',
    ];
    const mime = mimeTypes.find(m => MediaRecorder.isTypeSupported(m)) || '';

    this._recorder = new MediaRecorder(this._stream, mime ? { mimeType: mime } : {});
    this._chunks   = [];
    this._recorder.ondataavailable = e => { if (e.data.size) this._chunks.push(e.data); };

    this._recorder.start(200);   // 200ms time-slice chunks
    this._running  = true;
    this._startMs  = Date.now();
    this._frameN   = 0;
    return true;
  }

  /** Returns true if currently recording */
  get recording() { return this._running; }

  /**
   * Draw one frame to the canvas.  Call this inside your existing detection loop.
   *
   * @param {HTMLVideoElement} video
   * @param {{cx,cy,irisRadPx,pupilRadPx,pupilMm,confidence}} mL   left eye measurement
   * @param {{cx,cy,irisRadPx,pupilRadPx,pupilMm,confidence}} mR   right eye measurement
   * @param {Uint8ClampedArray|null} debugL   RGBA debug pixels for left eye (from algo)
   * @param {Uint8ClampedArray|null} debugR   RGBA debug pixels for right eye
   */
  writeFrame(video, mL, mR, debugL=null, debugR=null) {
    if (!this._running || !this._ctx) return;
    const ctx      = this._ctx;
    const size     = this.cellSize;
    const W        = size * 2;
    const H        = size + 24;
    const elapsed  = ((Date.now() - this._startMs) / 1000).toFixed(1);

    ctx.fillStyle = '#050508';
    ctx.fillRect(0, 0, W, H);

    const eyes = [
      { m: mL, dbg: debugL, col: 0, label: 'L', color: '#7b8fff' },
      { m: mR, dbg: debugR, col: 1, label: 'R', color: '#ff6b9d' },
    ];

    for (const { m, dbg, col, label, color } of eyes) {
      if (!m || !video.videoWidth) continue;
      const { cx, cy, irisRadPx } = m;
      const half = irisRadPx;
      const srcX = Math.round(cx - half);
      const srcY = Math.round(cy - half);
      const srcS = Math.round(half * 2);
      const dstX = col * size;

      if (srcX >= 0 && srcY >= 0 && srcX+srcS <= video.videoWidth && srcY+srcS <= video.videoHeight) {
        // Draw raw eye crop
        ctx.drawImage(video, srcX, srcY, srcS, srcS, dstX, 0, size, size);

        // Overlay debug pixels if available and showDebug is on
        if (this.showDebug && dbg && dbg.length === srcS * srcS * 4) {
          const tmpCanvas  = document.createElement('canvas');
          tmpCanvas.width  = srcS;
          tmpCanvas.height = srcS;
          const tmpCtx     = tmpCanvas.getContext('2d');
          const id = tmpCtx.createImageData(srcS, srcS);
          id.data.set(dbg);
          tmpCtx.putImageData(id, 0, 0);
          ctx.globalAlpha = 0.5;
          ctx.drawImage(tmpCanvas, 0, 0, srcS, srcS, dstX, 0, size, size);
          ctx.globalAlpha = 1.0;
        }
      } else {
        ctx.fillStyle = '#0a0a1a';
        ctx.fillRect(dstX, 0, size, size);
      }

      // Iris ring
      ctx.beginPath();
      ctx.arc(dstX + size/2, size/2, size/2 - 1, 0, Math.PI*2);
      ctx.strokeStyle = color + '88';
      ctx.lineWidth = 1;
      ctx.stroke();

      // Pupil circle
      if (m.pupilRadPx != null && m.pupilRadPx > 0) {
        const scale = size / (srcS || 1);
        ctx.beginPath();
        ctx.arc(dstX + size/2, size/2, m.pupilRadPx * scale, 0, Math.PI*2);
        ctx.strokeStyle = 'rgba(0,255,220,0.85)';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      // Eye label
      ctx.fillStyle = color;
      ctx.font      = 'bold 10px Space Mono,monospace';
      ctx.fillText(label, dstX + 3, 12);

      // Confidence dot
      if (m.confidence != null) {
        const cConf = m.confidence >= 0.7 ? '#4af0c8' : m.confidence >= 0.4 ? '#ffb347' : '#ff6b9d';
        ctx.beginPath();
        ctx.arc(dstX + size - 8, 8, 4, 0, Math.PI*2);
        ctx.fillStyle = cConf;
        ctx.fill();
      }
    }

    // Label strip at bottom
    const lMm = mL?.pupilMm?.toFixed(2) ?? '—';
    const rMm = mR?.pupilMm?.toFixed(2) ?? '—';
    const lC  = mL?.confidence?.toFixed(2) ?? '—';
    const rC  = mR?.confidence?.toFixed(2) ?? '—';
    ctx.fillStyle = '#0d0d18';
    ctx.fillRect(0, size, W, 24);
    ctx.fillStyle = 'rgba(74,240,200,0.6)';
    ctx.font      = '9px Space Mono,monospace';
    ctx.fillText(`${elapsed}s  L:${lMm}mm c${lC}  R:${rMm}mm c${rC}  f${this._frameN}`, 4, size + 14);

    this._frameN++;
  }

  /**
   * Stop recording and return a Promise<Blob> of the video.
   */
  stop() {
    if (!this._running) return Promise.resolve(null);
    this._running = false;
    return new Promise(resolve => {
      this._recorder.onstop = () => {
        const mime = this._recorder.mimeType || 'video/webm';
        const blob = new Blob(this._chunks, { type: mime });
        this._chunks = [];
        resolve(blob);
      };
      this._recorder.stop();
      this._stream.getTracks().forEach(t => t.stop());
    });
  }

  /** Trigger browser download of a recorded blob */
  download(blob, filename='iris_clip.webm') {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const a   = document.createElement('a');
    a.href     = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 30000);
  }

  /** Duration of current/last recording in seconds */
  get durationSec() { return (Date.now() - this._startMs) / 1000; }
}

window.IrisClipRecorder = IrisClipRecorder;
