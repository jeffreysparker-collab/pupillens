# PupilLens v6

Browser-based iris tracking and pupil light reflex (PLR) protocol — no install, no server, runs entirely client-side.

**Live demo:** `https://<your-username>.github.io/<repo-name>/`

---

## What it does

- Tracks pupil and iris size in real time using [MediaPipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh) via TensorFlow.js
- Measures **pupil diameter in mm** (calibrated against the ~11.7 mm iris population average)
- Runs a configurable **PLR trial**: dark adaptation → baseline → light flash → recovery
- Also supports a **colour sweep** trial (R → G → B → W)
- Exports per-frame data as **CSV** with full metadata (camera exposure, flash parameters, segment timestamps)
- Works offline after first load (service worker caches all assets)

- Flood-fill specular mask (p90 seed → p80 grow) — this is attacking the glare problem at source, replacing reflection blobs with mean intensity before edge detection
- Sobel + NMS + Canny hysteresis — proper edge pipeline, not just raw gradients
- Morphological thinning — junction blob removal + L-corner removal (borrowed from ElSe/PuRe)
- Connected components + photometric filter — inner darker than outer check (ElSe-style), centroid proximity filter
- Radial gradient filter on starburst hits — 50° tolerance, this IS essentially the Swirski2D image-aware support function
- Median pre-filter on radii — 28% deviation gate before RANSAC
- RANSAC with anatomical fast-reject — radius bounds + center tolerance
- 75% consistency check on ALL hits — not just RANSAC inliers, this is strong
- PuRe outline contrast confidence — photometric validation post-fit
- Taubin fallback — median filter only, no 75% gate, lower bar
---

## Files

| File | Purpose |
|---|---|
| `index.html` | Single-file app — all HTML, CSS, and JS |
| `sw.js` | Service worker — offline/PWA support |
| `manifest.json` | Web app manifest — installable PWA |
| `icon.svg` | Vector icon |
| `icon-192.png` | PNG icon (192 × 192) |
| `icon-512.png` | PNG icon (512 × 512) |

---

## Deploying to GitHub Pages

1. Push this folder's contents to a GitHub repository (root or `docs/` branch).
2. In **Settings → Pages**, set source to `main` branch, root `/` (or `/docs` if you used that).
3. GitHub will publish the site at `https://<username>.github.io/<repo>/`.

> **HTTPS is required.** `getUserMedia` (camera) only works on secure origins. GitHub Pages always serves over HTTPS, so you're good.

---

## CDN dependencies (all pinned versions)

All loaded from [jsDelivr](https://www.jsdelivr.com/) — no npm build step needed.

| Package | Version |
|---|---|
| `@mediapipe/face_mesh` | 0.4.1633559619 |
| `@tensorflow/tfjs-core` | 4.17.0 |
| `@tensorflow/tfjs-backend-webgl` | 4.17.0 |
| `@tensorflow-models/face-detection` | 1.0.2 |
| `@tensorflow-models/face-landmarks-detection` | 1.0.5 |

---

## Usage tips

- Use a **dim room** for best pupil contrast
- Sit **35–50 cm** from the screen
- Look **straight at the camera**
- The **half-screen flash** option (right half only) reduces camera auto-exposure washout — recommended for quantitative trials
- Exported CSV includes a commented header with all trial parameters and camera settings

---

## CSV output columns

| Column | Description |
|---|---|
| `t_ms` | Wall-clock timestamp (Unix ms) |
| `elapsed_ms` | Time since trial start |
| `segment` | `dark_adapt`, `baseline`, `flash`, `recovery` |
| `left/right_pupil_px_smooth` | EMA-smoothed pupil radius in video pixels |
| `left/right_pupil_px_raw` | Per-frame raw pupil radius |
| `left/right_pupil_mm` | Pupil diameter in mm |
| `left/right_iris_radius_px` | Iris outer radius (ruler for mm calibration) |
| `left/right_iris_mm` | Iris diameter in mm (sanity check, should ≈ 11.7) |
| `tilt_ratio` | Eye-level tilt (1.0 = perfectly level) |
| `quality` | `good` / `fair` / `poor` |

---

## Notes on camera permissions

GitHub Pages (and any HTTPS origin) will prompt the user for camera access on first use. The app never sends video data anywhere — all processing is local in the browser.

---

## License

MIT — do whatever you like, attribution appreciated.
