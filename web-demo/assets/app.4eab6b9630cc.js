'use strict';
const RELEASE = 'collection-a82b0f27fc5e';
const COLLECTION_URL = 'assets/collection.6b4fa4570293.json';
const EXPECTED_COUNT = 25;
const $ = id => document.getElementById(id);
const GROUPS = {
  ALL: {title: 'All examples', description: 'Choose a moment. Video and model predictions play together.'},
  REJECT: {title: 'Everyday movements', description: 'Common activities that do not call for a response.'},
  INCOMPLETE: {title: 'Signing in progress', description: 'Follow an utterance as the signer continues.'},
  WAIT: {title: 'A pause within a turn', description: 'Brief holds and pauses before signing continues.'},
  COMPLETE: {title: 'The end of an utterance', description: 'Watch the transition as a signed turn comes to an end.'},
  BARGE: {title: 'Signing during a response', description: 'A new utterance begins while the system is in the responding state.'}
};
const DESCRIPTIONS = {
  REJECT: 'No response is needed for this movement.',
  INCOMPLETE: 'The signed utterance is still in progress.',
  WAIT: 'A brief pause within the current utterance.',
  COMPLETE: 'The model detects the end of the utterance.',
  BARGE: 'The model detects new signing during a response.'
};
const EVENTS = Object.keys(DESCRIPTIONS);
let samples = [], current = null, category = 'ALL', busy = false, renderKey = '';
const states = new Map(), inflight = new Map();
const player = document.createElement('video');
player.controls = true; player.playsInline = true; player.preload = 'auto'; player.muted = true;
player.hidden = true; $('video-stage').append(player);
const prepared = new Map();
const clock = value => { const n = Math.floor(Math.max(0, Number(value) || 0)); return `${String(Math.floor(n / 60)).padStart(2, '0')}:${String(n % 60).padStart(2, '0')}`; };
function message(text) { $('player-status').textContent = text; }
function updateProgress() {
  const n = prepared.size, failed = samples.filter(s => states.get(s.id) === 'error').length;
  $('cache-label').textContent = n === samples.length ? `All ${n} examples ready` : n ? `${n} / ${samples.length} examples ready` : 'Preparing the first example…';
  $('cache-status').className = `cache-status ${n === samples.length ? 'ready' : failed && !n ? 'failed' : ''}`;
  $('retry').hidden = !failed; $('retry').textContent = 'Retry unavailable examples';
  $('prepare-count').textContent = `${n} of ${samples.length} examples ready`;
  $('prepare-progress').style.width = `${samples.length ? n / samples.length * 100 : 0}%`;
  if (!current && failed && failed === samples.length) $('prepare-text').textContent = 'The connection was interrupted. Retry any example below.';
  document.querySelectorAll('.sample-card').forEach(button => {
    const state = states.get(button.dataset.id) || 'queued';
    button.dataset.status = state; button.disabled = state === 'queued' || state === 'loading';
    button.querySelector('.sample-cache-state').textContent = state === 'ready' ? 'Ready to play' : state === 'error' ? 'Retry this example ↻' : 'Preparing…';
  });
  controls();
}
async function requestBytes(path, attempt) {
  const controller = new AbortController(), timer = setTimeout(() => controller.abort(), 16000);
  try {
    const address = attempt ? `${path}${path.includes('?') ? '&' : '?'}retry=${attempt}` : path;
    const response = await fetch(address, {cache: attempt ? 'reload' : 'default', signal: controller.signal});
    if (!response.ok) throw Error(`HTTP ${response.status}`);
    return await response.arrayBuffer();
  } finally { clearTimeout(timer); }
}
async function retryRequest(path, validate) {
  let error;
  for (let attempt = 0; attempt < 3; attempt++) {
    try { return await validate(await requestBytes(path, attempt)); }
    catch (e) { error = e; if (attempt < 2) await new Promise(r => setTimeout(r, 450 * (attempt + 1))); }
  }
  throw error;
}
function validCollection(data) {
  if (data?.schema !== 2 || data.release !== RELEASE || !Array.isArray(data.samples) || data.samples.length !== EXPECTED_COUNT) return false;
  const ids = new Set();
  for (const s of data.samples) {
    if (ids.has(s.id) || !s.id || !DESCRIPTIONS[s.category] || typeof s.title !== 'string' || !s.poster.startsWith('data:image/') || !s.file.includes(s.sha256.slice(0, 12)) || !s.bytes || !Array.isArray(s.record?.records) || !s.record.records.some(r => r.kind === 'prediction')) return false;
    if (s.record.records.some(r => !Number.isFinite(Number(r.time)) || r.kind === 'prediction' && !DESCRIPTIONS[r.event])) return false;
    ids.add(s.id);
  }
  return true;
}
async function getCollection() {
  const inline = $('example-data');
  if (inline) { try { const data = JSON.parse(inline.textContent); if (validCollection(data)) return data; } catch {} }
  return retryRequest(COLLECTION_URL, bytes => {
    const data = JSON.parse(new TextDecoder().decode(bytes));
    if (!validCollection(data)) throw Error('Collection version mismatch');
    return data;
  });
}
async function loadSample(sample) {
  const blob = await retryRequest(sample.file, async bytes => {
    if (bytes.byteLength !== sample.bytes) throw Error('Incomplete video');
    if (globalThis.crypto?.subtle) {
      const digest = await crypto.subtle.digest('SHA-256', bytes);
      const hash = Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, '0')).join('');
      if (hash !== sample.sha256) throw Error('Video version mismatch');
    }
    return new Blob([bytes], {type: 'video/mp4'});
  });
  prepared.set(sample.id, {sample, recording: sample.record, source: URL.createObjectURL(blob)});
  states.set(sample.id, 'ready'); updateProgress();
  if (!current) { selectSample(sample.id, false); message('Choose any ready example to begin.'); }
}
function prepareSample(sample) {
  if (prepared.has(sample.id)) return Promise.resolve();
  if (inflight.has(sample.id)) return inflight.get(sample.id);
  states.set(sample.id, 'loading'); updateProgress();
  const pending = loadSample(sample).catch(() => { states.set(sample.id, 'error'); }).finally(() => { inflight.delete(sample.id); updateProgress(); });
  inflight.set(sample.id, pending); return pending;
}
async function prepareCollection() {
  if (busy) return; busy = true;
  const pending = samples.filter(s => !prepared.has(s.id)); let cursor = 0;
  const worker = async () => { while (cursor < pending.length) await prepareSample(pending[cursor++]); };
  await Promise.all(Array.from({length: Math.min(4, pending.length)}, worker)); busy = false; updateProgress();
}
function controls() {
  $('play').disabled = !current; $('restart').disabled = !current;
  $('play-label').textContent = current && !current.video.paused ? 'Pause' : current?.video.ended ? 'Replay example' : 'Play example';
}
function playCurrent() {
  if (!current) return;
  const video = current.video; if (video.ended) video.currentTime = 0;
  video.play().catch(() => message('Press the video’s play button to begin.')); controls();
}
function selectSample(id, autoplay = true) {
  if (!prepared.has(id)) return;
  player.pause(); current = {...prepared.get(id), video: player}; renderKey = '';
  const {sample, source} = current;
  player.src = source; player.poster = sample.poster; player.hidden = false;
  player.setAttribute('aria-label', sample.title); player.load(); player.playbackRate = Number($('playback-speed').value);
  $('prepare-cover').hidden = true; $('video-name').textContent = sample.title; $('selected-category').textContent = sample.category;
  $('reference-line').hidden = !sample.reference; $('reference-text').textContent = sample.reference || '';
  $('selected-position').textContent = `${String(samples.findIndex(s => s.id === id) + 1).padStart(2, '0')} / ${samples.length}`;
  document.querySelectorAll('.sample-card').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.id === id)));
  render(0, true); controls();
  if (autoplay) { playCurrent(); const rect = $('video-stage').getBoundingClientRect(); if (rect.bottom < 0 || rect.top > innerHeight) $('demo').scrollIntoView({behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth'}); }
}
function render(time, force = false) {
  if (!current) return;
  const {sample, recording, video} = current;
  time = Math.max(0, Math.min(Number(time) || 0, video.currentTime));
  const rows = recording.records.filter(r => Number(r.time) <= time + 1e-7);
  const predictions = rows.filter(r => r.kind === 'prediction'), last = predictions.at(-1);
  const state = rows.at(-1)?.state_after || recording.initial_state || sample.initial_state || 'RECEIVING';
  $('media-clock').textContent = `${clock(time)} / ${clock(video.duration || sample.duration)}`;
  $('prediction-panel').dataset.mediaTime = String(time);
  const key = `${sample.id}|${predictions.length}|${state}`; if (!force && key === renderKey) return; renderKey = key;
  $('prediction-panel').dataset.event = last?.event || ''; $('prediction-panel').dataset.predictionTime = last ? String(last.time) : '';
  $('system-state').textContent = state; $('system-state').classList.toggle('responding', state === 'RESPONDING');
  $('prediction-event').textContent = last?.event || '';
  $('prediction-description').textContent = last ? DESCRIPTIONS[last.event] || '' : '';
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
  for (const name of Object.keys(GROUPS)) {
    const n = name === 'ALL' ? samples.length : samples.filter(s => s.category === name).length;
    const button = document.createElement('button'); button.className = 'category-tab'; button.dataset.category = name; button.setAttribute('aria-pressed', String(name === category));
    const label = document.createElement('span'); label.textContent = name === 'ALL' ? 'All examples' : name; const count = document.createElement('small'); count.textContent = n;
    button.append(label, count); button.addEventListener('click', () => setCategory(name)); $('category-tabs').append(button);
  }
  samples.forEach((sample, index) => {
    const button = document.createElement('button'); button.className = 'sample-card'; button.dataset.id = sample.id; button.dataset.category = sample.category; button.disabled = true; button.setAttribute('aria-pressed', 'false'); button.setAttribute('aria-label', `Play ${sample.category}: ${sample.title}`);
    const frame = document.createElement('div'); frame.className = 'sample-image'; const image = document.createElement('img'); image.src = sample.poster; image.alt = ''; image.decoding = 'async'; const icon = document.createElement('span'); icon.className = 'sample-play'; icon.textContent = '▶'; icon.setAttribute('aria-hidden', 'true'); const duration = document.createElement('span'); duration.className = 'sample-duration'; duration.textContent = clock(sample.duration); frame.append(image, icon, duration);
    const copy = document.createElement('div'); copy.className = 'sample-copy'; const tag = document.createElement('span'); tag.textContent = `${sample.category} / ${String(index + 1).padStart(2, '0')}`; const title = document.createElement('strong'); title.textContent = sample.title; copy.append(tag, title); const cacheState = document.createElement('small'); cacheState.className = 'sample-cache-state'; cacheState.textContent = 'Preparing…'; copy.append(cacheState); button.append(frame, copy); button.addEventListener('click', () => { if (prepared.has(sample.id)) selectSample(sample.id); else prepareSample(sample); }); $('sample-grid').append(button);
  });
  setCategory('ALL');
}
$('play').addEventListener('click', () => { if (!current) return; current.video.paused ? playCurrent() : current.video.pause(); });
$('restart').addEventListener('click', () => { if (!current) return; current.video.currentTime = 0; render(0, true); playCurrent(); });
$('playback-speed').addEventListener('change', () => { if (current) current.video.playbackRate = Number($('playback-speed').value); });
$('retry').addEventListener('click', () => {
  if (!samples.length) initialize();
  else samples.filter(s => states.get(s.id) === 'error').forEach(prepareSample);
});
for (const name of ['timeupdate', 'seeking', 'seeked', 'pause', 'ended', 'ratechange', 'loadedmetadata']) {
  player.addEventListener(name, () => { if (current) { if (name === 'loadedmetadata') player.playbackRate = Number($('playback-speed').value); render(player.currentTime); controls(); } });
}
player.addEventListener('play', () => { controls(); message(''); });
player.addEventListener('error', () => {
  if (!current || !player.error) return;
  const {sample, source} = current; current = null; renderKey = '';
  prepared.delete(sample.id); states.set(sample.id, 'error');
  player.pause(); player.hidden = true; player.removeAttribute('src'); player.removeAttribute('poster'); player.load();
  URL.revokeObjectURL(source);
  $('prepare-cover').hidden = false; $('prepare-text').textContent = 'Choose another ready example, or retry this one.';
  $('prediction-panel').dataset.event = ''; $('prediction-panel').dataset.predictionTime = ''; $('prediction-panel').dataset.mediaTime = '0';
  $('prediction-event').textContent = ''; $('prediction-description').textContent = '';
  $('event-index').textContent = ''; $('event-sequence').replaceChildren();
  $('system-state').textContent = '—'; $('system-state').classList.remove('responding');
  $('media-clock').textContent = '00:00 / 00:00'; $('reference-line').hidden = true;
  document.querySelectorAll('.sample-card').forEach(b => b.setAttribute('aria-pressed', 'false'));
  message('This video could not be opened. Choose another ready example or retry this one.'); updateProgress();
});
function followPlayback() { if (current && !player.paused) render(player.currentTime); requestAnimationFrame(followPlayback); }
requestAnimationFrame(followPlayback);
async function initialize() {
  try {
    const data = await getCollection(); samples = data.samples;
    document.documentElement.dataset.release = RELEASE;
    $('category-tabs').replaceChildren(); $('sample-grid').replaceChildren();
    buildLibrary(); await prepareCollection();
  } catch {
    $('cache-status').className = 'cache-status failed'; $('cache-label').textContent = 'The collection could not be loaded';
    $('retry').hidden = false; $('retry').textContent = 'Retry'; $('prepare-text').textContent = 'Please retry the connection.';
  }
}
initialize();
