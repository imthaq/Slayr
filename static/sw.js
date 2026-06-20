const CACHE_NAME = 'slayr-cache-v1';
const urlsToCache = [
  '/',
  '/static/css/style.css',
  '/static/css/components.css',
  '/static/assets/new_logo.png',
  '/static/assets/sila_stylist.png'
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(urlsToCache))
  );
});

self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request)
      .then(response => response || fetch(event.request))
  );
});
