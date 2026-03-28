
// PupilLens v8 — Service Worker
// Cache key: bump CACHE_VER whenever index.html or assets change.
// Strategy:
//   • App shell (index, manifest, icons) → cache-first, update in background
//   • CDN scripts (MediaPipe, TF.js)     → cache-first (pinned versions)
//   • Everything else                    → network-first with cache fallback

const CACHE_VER = 'pupillens-v8-2026032x';

const SHELL = [
  './',
  './index.html',
  './manifest.json',
  './icon.svg',
  './icon-192.png',
  './icon-512.png',
];

const CDN_SCRIPTS = [
  'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1633559619/face_mesh.js',
  'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-core@4.17.0/dist/tf-core.min.js',
  'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgl@4.17.0/dist/tf-backend-webgl.min.js',
  'https://cdn.jsdelivr.net/npm/@tensorflow-models/face-detection@1.0.2/dist/face-detection.min.js',
  'https://cdn.jsdelivr.net/npm/@tensorflow-models/face-landmarks-detection@1.0.5/dist/face-landmarks-detection.min.js',
];

// ── Install: pre-cache app shell ─────────────────────────────────────────
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_VER)
      .then(cache => cache.addAll(SHELL))
      .then(() => self.skipWaiting())
  );
});

// ── Activate: delete old caches, claim clients, notify for reload ─────────
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys()
      .then(keys => Promise.all(
        keys.filter(k => k !== CACHE_VER).map(k => caches.delete(k))
      ))
      .then(() => self.clients.claim())
      .then(() => self.clients.matchAll({type:'window'}))
      .then(clients => clients.forEach(c => c.postMessage({type:'SW_UPDATED'})))
  );
});

// ── Fetch ─────────────────────────────────────────────────────────────────
self.addEventListener('fetch', event => {
  const {request} = event;
  const url = request.url;

  if(request.method !== 'GET') return;

  // App shell or pinned CDN → cache-first
  if(
    SHELL.some(s => url.endsWith(s.replace('./',''))) ||
    CDN_SCRIPTS.some(s => url === s) ||
    url.includes('cdn.jsdelivr.net/npm/@mediapipe/')
  ){
    event.respondWith(
      caches.match(request).then(cached => {
        if(cached) return cached;
        return fetch(request).then(response => {
          if(response.ok){
            const clone = response.clone();
            caches.open(CACHE_VER).then(cache => cache.put(request, clone));
          }
          return response;
        });
      })
    );
    return;
  }

  // Everything else → network-first, cache fallback
  event.respondWith(
    fetch(request)
      .then(response => {
        if(response.ok){
          const clone = response.clone();
          caches.open(CACHE_VER).then(cache => cache.put(request, clone));
        }
        return response;
      })
      .catch(() => caches.match(request))
  );
});
