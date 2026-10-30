'use strict';
const RELEASE = 'collection-0ae10daccd9b';
const COLLECTION_URL = 'assets/collection.7d3f41125fdc.json';
const EXPECTED_COUNT = 10;
const $ = id => document.getElementById(id);
const GROUPS = {
  ALL: {title: 'All examples', description: 'Choose a moment. Video and event labels play together.'},
  REJECT: {title: 'Everyday movements', description: 'Common activities that do not call for a response.'},
  INCOMPLETE: {title: 'Signing in progress', description: 'Follow an utterance as the signer continues.'},
  WAIT: {title: 'A pause within a turn', description: 'Brief holds and pauses before signing continues.'},
  COMPLETE: {title: 'The end of an utterance', description: 'Watch the transition as a signed turn comes to an end.'},
  BARGE: {title: 'Finish a turn. Begin another.', description: 'Follow a complete signed turn, then new signing that interrupts the response.'}
};
const DESCRIPTIONS = {
  REJECT: 'No response is needed for this movement.',
  INCOMPLETE: 'The signed utterance is still in progress.',
  WAIT: 'A brief pause within the current utterance.',
  COMPLETE: 'The model detects the end of the utterance.',
  BARGE: 'The model detects new signing during a response.'
};
const EVENTS = Object.keys(DESCRIPTIONS);
const isEventRow = row => row.kind === 'prediction' || row.kind === 'reference_label';
let samples = [], current = null, category = 'ALL', renderKey = '', generation = 0;
const prepared = new Map(), downloads = new Map(), retryAfter = new Map();
const CACHE_NAME = 'signduplex-video-v1';
let mediaCache = null, cacheTimer = null, restoring = true, preferredOrigin = '';
const player = document.createElement('video');
player.controls = true; player.playsInline = true; player.preload = 'auto'; player.muted = true;
player.hidden = true; $('video-stage').append(player);
const clock = value => { const n = Math.floor(Math.max(0, Number(value) || 0)); return `${String(Math.floor(n / 60)).padStart(2, '0')}:${String(n % 60).padStart(2, '0')}`; };
const absolute = path => new URL(path, location.href).href;
function message(text) { $('player-status').textContent = text; }
function sourcesFor(sample) {
  const sources = [...new Set([sample.file, ...(sample.sources || [])].map(absolute))];
  return sources.sort((a, b) => Number(new URL(b).origin === preferredOrigin) - Number(new URL(a).origin === preferredOrigin));
}
function updateProgress() {
  const n = prepared.size;
  $('cache-label').textContent = n === samples.length ? `All ${n} examples saved for replay` : `${samples.length} examples · choose any to play`;
  $('cache-status').className = `cache-status ${n === samples.length ? 'ready' : ''}`;
  $('cache-status').dataset.cached = String(n);
  $('retry').hidden = !current?.failed; $('retry').textContent = 'Retry video';
  $('prepare-count').textContent = ''; $('prepare-progress').style.width = '0%';
  document.querySelectorAll('.sample-card').forEach(button => {
    const ready = prepared.has(button.dataset.id);
    button.dataset.status = ready ? 'ready' : 'available'; button.disabled = false;
    button.querySelector('.sample-cache-state').textContent = ready ? 'Saved for replay' : 'Play example';
  });
  controls();
}
async function requestBytes(path, controller = new AbortController()) {
  // A slow but progressing transfer must not hit a whole-file deadline.
  let timer;
  const arm = delay => { clearTimeout(timer); timer = setTimeout(() => controller.abort(), delay); };
  arm(8000);
  try {
    const response = await fetch(path, {cache: 'force-cache', signal: controller.signal});
    if (!response.ok) throw Error(`HTTP ${response.status}`);
    arm(20000);
    if (!response.body) return await response.arrayBuffer();
    const reader = response.body.getReader(), chunks = []; let length = 0;
    while (true) {
      const {done, value} = await reader.read(); if (done) break;
      chunks.push(value); length += value.length; arm(20000);
    }
    const bytes = new Uint8Array(length); let offset = 0;
    for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
    return bytes.buffer;
  } finally { clearTimeout(timer); }
}
function validCollection(data) {
  if (data?.schema !== 2 || data.release !== RELEASE || !Array.isArray(data.samples) || data.samples.length !== EXPECTED_COUNT) return false;
  const ids = new Set();
  for (const s of data.samples) {
    if (ids.has(s.id) || !s.id || !(DESCRIPTIONS[s.category] || s.category === 'EXAMPLE' && s.source_type === 'reference_labels') || typeof s.title !== 'string' || !s.poster.startsWith('data:image/') || !s.file.includes(s.sha256.slice(0, 12)) || !s.bytes || !Array.isArray(s.record?.records) || !s.record.records.some(isEventRow)) return false;
    if (s.record.records.some(r => !Number.isFinite(Number(r.time)) || isEventRow(r) && !DESCRIPTIONS[r.event])) return false;
    ids.add(s.id);
  }
  return true;
}
async function getCollection() {
  const inline = $('example-data');
  if (inline) { try { const data = JSON.parse(inline.textContent); if (validCollection(data)) return data; } catch {} }
  const data = JSON.parse(new TextDecoder().decode(await requestBytes(COLLECTION_URL)));
  if (!validCollection(data)) throw Error('Collection version mismatch');
  return data;
}
async function checkedBlob(sample, bytes) {
  if (bytes.byteLength !== sample.bytes) throw Error('Incomplete video');
  if (globalThis.crypto?.subtle) {
    const digest = await crypto.subtle.digest('SHA-256', bytes);
    const hash = Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, '0')).join('');
    if (hash !== sample.sha256) throw Error('Video version mismatch');
  }
  return new Blob([bytes], {type: 'video/mp4'});
}
function remember(sample, blob) {
  if (!prepared.has(sample.id)) prepared.set(sample.id, {source: URL.createObjectURL(blob)});
  updateProgress(); // Do not change the source of a video already playing.
}
async function restoreSavedSamples() {
  try { mediaCache = await caches.open(CACHE_NAME); } catch { mediaCache = null; }
  if (!mediaCache) return;
  await Promise.all(samples.map(async sample => {
    try {
      const saved = await mediaCache.match(absolute(sample.file));
      if (saved) remember(sample, await checkedBlob(sample, await saved.arrayBuffer()));
    } catch { try { await mediaCache.delete(absolute(sample.file)); } catch {} }
  }));
}
function cancelBackground() {
  clearTimeout(cacheTimer);
  for (const job of downloads.values()) { job.cancelled = true; job.controller.abort(); }
}
function scheduleCache(delay = 0) {
  clearTimeout(cacheTimer); cacheTimer = setTimeout(cacheNext, delay);
}
function bufferedAhead() {
  for (let i = 0; i < player.buffered.length; i++) {
    if (player.buffered.start(i) <= player.currentTime + .05 && player.buffered.end(i) >= player.currentTime) return player.buffered.end(i) - player.currentTime;
  }
  return 0;
}
async function cacheNext() {
  if (restoring || downloads.size || navigator.onLine === false || prepared.size === samples.length) return;
  // Playback owns the connection until it has enough headroom.
  if (current?.wantsPlay && !prepared.has(current.sample.id) && !player.ended && (player.readyState < 3 || bufferedAhead() < (player.duration || current.sample.duration) - player.currentTime - .1)) {
    scheduleCache(1000); return;
  }
  const pending = samples.filter(s => !prepared.has(s.id) && (retryAfter.get(s.id) || 0) < Date.now());
  const sample = pending.find(s => s.id === current?.sample.id) || pending.find(s => s.category === category) || pending[0];
  if (!sample) { scheduleCache(15000); return; }
  const job = {controller: new AbortController(), cancelled: false}; downloads.set(sample.id, job);
  let complete = false;
  try {
    for (const source of sourcesFor(sample)) {
      if (job.cancelled) break;
      job.controller = new AbortController();
      try {
        const blob = await checkedBlob(sample, await requestBytes(source, job.controller));
        if (job.cancelled) break;
        if (mediaCache) { try { await mediaCache.put(absolute(sample.file), new Response(blob, {headers: {'Content-Type': 'video/mp4'}})); } catch {} }
        remember(sample, blob); complete = true; retryAfter.delete(sample.id);
        break;
      } catch { if (job.cancelled) break; }
    }
  } finally {
    if (!complete && !job.cancelled) retryAfter.set(sample.id, Date.now() + 60000);
    downloads.delete(sample.id); scheduleCache(complete ? 80 : 250);
  }
}
function controls() {
  $('play').disabled = !current; $('restart').disabled = !current;
  $('play-label').textContent = current && !player.paused ? 'Pause' : player.ended ? 'Replay example' : 'Play example';
}
function playCurrent() {
  if (!current) return;
  if (current.failed) { selectSample(current.sample.id); return; }
  const selected = current;
  const sourceGeneration = selected.sourceGeneration;
  const attempt = selected.playAttempt = (selected.playAttempt || 0) + 1;
  selected.wantsPlay = true; selected.lastActivity = performance.now();
  if (player.ended) player.currentTime = 0;
  player.play().catch(error => {
    if (current !== selected || selected.sourceGeneration !== sourceGeneration || selected.playAttempt !== attempt) return;
    if (error.name === 'AbortError' && player.paused) { selected.wantsPlay = false; selected.switching = false; controls(); return; }
    if (error.name === 'NotAllowedError') { selected.wantsPlay = false; message('Press the video’s play button to begin.'); }
  });
  controls();
}
function openSource(source, resumeTime = 0) {
  if (!current) return;
  current.sourceGeneration = (current.sourceGeneration || 0) + 1;
  current.source = source; current.tried.add(source); current.resumeTime = resumeTime; current.switching = true;
  current.lastActivity = performance.now(); current.lastTime = resumeTime; current.lastBuffer = 0; current.failed = false;
  player.src = source; player.load(); player.playbackRate = Number($('playback-speed').value);
  if (current.wantsPlay) playCurrent();
}
function fallbackCurrent() {
  if (!current || current.failed) return;
  const cached = prepared.get(current.sample.id)?.source;
  const next = [cached, ...current.sources].find(source => source && !current.tried.has(source));
  if (next) { openSource(next, Math.max(player.currentTime || 0, current.resumeTime || 0)); return; }
  current.failed = true; current.wantsPlay = false; current.switching = false;
  message('The video connection was interrupted. Click this example to retry.'); updateProgress();
}
function selectSample(id, autoplay = true) {
  const sample = samples.find(s => s.id === id); if (!sample) return;
  cancelBackground(); player.pause(); renderKey = '';
  current = {sample, recording: sample.record, video: player, generation: ++generation, sources: sourcesFor(sample), tried: new Set(), wantsPlay: autoplay};
  player.poster = sample.poster; player.hidden = false; player.setAttribute('aria-label', sample.title);
  $('prepare-cover').hidden = true; $('video-name').textContent = sample.title; $('selected-category').textContent = sample.category;
  $('reference-line').hidden = !sample.reference; $('reference-text').textContent = sample.reference || '';
  $('event-source').textContent = sample.source_type === 'reference_labels' ? 'Reference event labels' : 'Recorded model predictions';
  $('selected-position').textContent = `${String(samples.findIndex(s => s.id === id) + 1).padStart(2, '0')} / ${samples.length}`;
  document.querySelectorAll('.sample-card').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.id === id)));
  openSource(prepared.get(id)?.source || current.sources[0]);
  render(0, true); controls(); message(''); updateProgress(); scheduleCache(1000);
  if (autoplay) { const rect = $('video-stage').getBoundingClientRect(); if (rect.bottom < 0 || rect.top > innerHeight) $('demo').scrollIntoView({behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth'}); }
}
function render(time, force = false) {
  if (!current) return;
  const {sample, recording, video} = current;
  time = Math.max(0, Math.min(Number(time) || 0, video.currentTime));
  // TXT labels may round the final boundary beyond EOF by less than one frame.
  // Preserve the supplied times; include that terminal label only at video end.
  const endTolerance = video.ended && sample.source_type === 'reference_labels' ? sample.terminal_label_tolerance || 0 : 0;
  const rows = recording.records.filter(r => Number(r.time) <= time + Math.max(1e-7, endTolerance));
  const predictions = rows.filter(isEventRow), last = predictions.at(-1);
  const state = rows.at(-1)?.state_after || recording.initial_state || sample.initial_state || 'RECEIVING';
  $('media-clock').textContent = `${clock(time)} / ${clock(video.duration || sample.duration)}`;
  $('prediction-panel').dataset.mediaTime = String(time);
  const key = `${sample.id}|${predictions.length}|${state}`; if (!force && key === renderKey) return; renderKey = key;
  $('prediction-panel').dataset.event = last?.event || ''; $('prediction-panel').dataset.predictionTime = last ? String(last.time) : '';
  $('system-state').textContent = state; $('system-state').classList.toggle('responding', state === 'RESPONDING');
  $('prediction-event').textContent = last?.event || '';
  const referenceDescriptions = {COMPLETE: 'The signed utterance ends.', BARGE: 'New signing during a response.'};
  $('prediction-description').textContent = last ? (sample.source_type === 'reference_labels' && referenceDescriptions[last.event]) || DESCRIPTIONS[last.event] || '' : '';
  $('event-index').textContent = last ? `${String(EVENTS.indexOf(last.event) + 1).padStart(2, '0')} / 05` : '';
  const changes = predictions.filter((row, i) => i === 0 || row.event !== predictions[i - 1].event);
  $('event-sequence').replaceChildren(...changes.map(row => { const badge = document.createElement('span'); badge.className = `sequence-event ${row.event.toLowerCase()}`; badge.textContent = row.event; badge.title = `${Number(Number(row.time).toFixed(3))} s`; return badge; }));
  if (!changes.length) { const empty = document.createElement('span'); empty.className = 'sequence-empty'; empty.textContent = 'Events appear as the video plays.'; $('event-sequence').append(empty); }
}
function setCategory(next) {
  category = next;
  document.querySelectorAll('.category-tab').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.category === next)));
  document.querySelectorAll('.sample-card').forEach(b => { b.hidden = next !== 'ALL' && b.dataset.category !== next; });
  $('category-title').textContent = GROUPS[next].title; $('category-description').textContent = GROUPS[next].description;
}
function buildLibrary() {
  $('collection-count').textContent = `${samples.length} examples · 5 events`;
  for (const name of ['ALL']) {
    const n = name === 'ALL' ? samples.length : samples.filter(s => s.category === name).length;
    const button = document.createElement('button'); button.className = 'category-tab'; button.dataset.category = name; button.setAttribute('aria-pressed', String(name === category));
    const label = document.createElement('span'); label.textContent = name === 'ALL' ? 'All examples' : name; const count = document.createElement('small'); count.textContent = n;
    button.append(label, count); button.addEventListener('click', () => setCategory(name)); $('category-tabs').append(button);
  }
  samples.forEach((sample, index) => {
    const button = document.createElement('button'); button.className = 'sample-card'; button.dataset.id = sample.id; button.dataset.category = sample.category; button.disabled = false; button.setAttribute('aria-pressed', 'false'); button.setAttribute('aria-label', `Play ${sample.category}: ${sample.title}`);
    const frame = document.createElement('div'); frame.className = 'sample-image'; const image = document.createElement('img'); image.src = sample.poster; image.alt = ''; image.decoding = 'async'; const icon = document.createElement('span'); icon.className = 'sample-play'; icon.textContent = '▶'; icon.setAttribute('aria-hidden', 'true'); const duration = document.createElement('span'); duration.className = 'sample-duration'; duration.textContent = clock(sample.duration); frame.append(image, icon, duration);
    const copy = document.createElement('div'); copy.className = 'sample-copy'; const tag = document.createElement('span'); tag.textContent = `${sample.display_category || sample.category} / ${String(sample.display_index || index + 1).padStart(2, '0')}`; const title = document.createElement('strong'); title.textContent = sample.title; copy.append(tag, title); const cacheState = document.createElement('small'); cacheState.className = 'sample-cache-state'; cacheState.textContent = 'Play example'; copy.append(cacheState); button.append(frame, copy); button.addEventListener('click', () => selectSample(sample.id)); $('sample-grid').append(button);
  });
  setCategory('ALL');
}
$('play').addEventListener('click', () => { if (!current) return; if (player.paused) playCurrent(); else { current.wantsPlay = false; player.pause(); } });
$('restart').addEventListener('click', () => { if (current) { if (current.failed) selectSample(current.sample.id); else { player.currentTime = 0; render(0, true); playCurrent(); } } });
$('playback-speed').addEventListener('change', () => { if (current) player.playbackRate = Number($('playback-speed').value); });
$('retry').addEventListener('click', () => { if (!samples.length) initialize(); else if (current) selectSample(current.sample.id); });
for (const name of ['timeupdate', 'seeking', 'seeked', 'pause', 'ended', 'ratechange', 'loadedmetadata']) {
  player.addEventListener(name, () => {
    if (!current) return;
    if (name === 'loadedmetadata') {
      player.playbackRate = Number($('playback-speed').value);
      if (current.resumeTime > 0) { player.currentTime = Math.min(current.resumeTime, player.duration); current.resumeTime = 0; }
    }
    if (name === 'pause' && player.paused && !player.error) current.wantsPlay = false;
    if (name === 'ended') { current.wantsPlay = false; scheduleCache(); }
    render(player.currentTime); controls();
  });
}
player.addEventListener('play', () => { if (current) { current.wantsPlay = true; current.lastActivity = performance.now(); } controls(); message(''); });
player.addEventListener('playing', () => {
  if (!current) return;
  current.failed = false; current.wantsPlay = true; current.switching = false;
  current.lastActivity = performance.now(); message(''); updateProgress();
  if (!current.source.startsWith('blob:')) preferredOrigin = new URL(current.source).origin;
  scheduleCache(1200);
});
player.addEventListener('loadeddata', () => { if (current) current.switching = false; });
player.addEventListener('progress', () => { if (current) current.lastActivity = performance.now(); });
player.addEventListener('error', () => { if (current && player.error) fallbackCurrent(); });
setInterval(() => {
  if (!current || current.failed || !current.wantsPlay || player.ended) return;
  const end = player.buffered.length ? player.buffered.end(player.buffered.length - 1) : 0;
  if (player.currentTime > (current.lastTime || 0) + .02 || end > (current.lastBuffer || 0) + .05) current.lastActivity = performance.now();
  current.lastTime = player.currentTime; current.lastBuffer = end;
  if (performance.now() - current.lastActivity > 4500) fallbackCurrent();
}, 500);
window.addEventListener('online', () => { retryAfter.clear(); if (current?.failed) selectSample(current.sample.id); scheduleCache(); });
window.addEventListener('offline', cancelBackground);
function followPlayback() { if (current && !player.paused) render(player.currentTime); requestAnimationFrame(followPlayback); }
requestAnimationFrame(followPlayback);
async function initialize() {
  try {
    const data = await getCollection(); samples = data.samples;
    document.documentElement.dataset.release = RELEASE;
    $('category-tabs').replaceChildren(); $('sample-grid').replaceChildren(); buildLibrary(); updateProgress();
    await restoreSavedSamples(); restoring = false;
    if (!current) selectSample(samples[0].id, false);
    scheduleCache();
  } catch {
    $('cache-status').className = 'cache-status failed'; $('cache-label').textContent = 'The collection could not be loaded';
    $('retry').hidden = false; $('retry').textContent = 'Retry'; $('prepare-text').textContent = 'Please retry the connection.';
  }
}
initialize();
