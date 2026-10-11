'use strict';
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
let samples = [], current = null, category = 'ALL', ready = false, busy = false, renderKey = '';
const prepared = new Map();
const clock = value => { const n = Math.floor(Math.max(0, Number(value) || 0)); return `${String(Math.floor(n / 60)).padStart(2, '0')}:${String(n % 60).padStart(2, '0')}`; };
function message(text) { $('player-status').textContent = text; }
function showPreparationError() {
  $('cache-status').className = 'cache-status failed'; $('cache-label').textContent = 'Some examples could not be prepared';
  $('prepare-text').textContent = 'Please retry to finish preparing the collection.'; $('retry').hidden = false;
  message('A connection interrupted preparation. Retry once to make the collection ready.');
}
function updateProgress() {
  const count = prepared.size, percent = samples.length ? count / samples.length * 100 : 0;
  $('cache-label').textContent = `Preparing examples · ${count} / ${samples.length}`;
  $('prepare-count').textContent = `${count} of ${samples.length} examples ready`;
  $('prepare-progress').style.width = `${percent}%`;
}
async function fetchChecked(path, type) {
  const response = await fetch(path, {cache: 'force-cache'});
  if (!response.ok) throw Error(`Could not prepare ${path}`);
  return type === 'json' ? response.json() : response.blob();
}
function prepareVideo(video) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => finish(Error('Video preparation timed out')), 45000);
    const loaded = () => { if (video.readyState >= 2) finish(); };
    const failed = () => finish(Error('This video could not be decoded'));
    function finish(error) { clearTimeout(timer); video.removeEventListener('loadeddata', loaded); video.removeEventListener('canplay', loaded); video.removeEventListener('error', failed); error ? reject(error) : resolve(); }
    video.addEventListener('loadeddata', loaded); video.addEventListener('canplay', loaded); video.addEventListener('error', failed); video.load(); loaded();
  });
}
async function prepareSample(sample) {
  if (prepared.has(sample.id)) return;
  const [blob, recording] = await Promise.all([fetchChecked(sample.file, 'blob'), fetchChecked(sample.recording, 'json')]);
  if (!Array.isArray(recording.records) || !recording.records.some(r => r.kind === 'prediction')) throw Error('Prediction record is incomplete');
  const video = document.createElement('video');
  video.controls = true; video.playsInline = true; video.preload = 'auto'; video.muted = true;
  video.setAttribute('aria-label', sample.title); video.poster = sample.poster; const source = URL.createObjectURL(blob); video.src = source;
  const item = {sample, recording, video, source}; $('preload-bin').append(video);
  try { await prepareVideo(video); } catch (error) { video.remove(); URL.revokeObjectURL(source); throw error; }
  const sync = () => { if (current === item) { render(video.currentTime); controls(); } };
  for (const name of ['timeupdate', 'seeking', 'seeked', 'pause', 'ended', 'ratechange']) video.addEventListener(name, sync);
  video.addEventListener('play', () => { if (current !== item) { video.pause(); return; } controls(); message(''); followFrames(item); });
  prepared.set(sample.id, item); updateProgress();
}
function followFrames(item) {
  const v = item.video;
  if (current !== item || v.paused || v.ended || item.framePending) return;
  item.framePending = true;
  const next = (_, meta) => {
    item.framePending = false;
    if (current !== item || v.paused) return;
    const t = meta && !v.seeking && Math.abs(meta.mediaTime - v.currentTime) < .12 ? meta.mediaTime : v.currentTime;
    render(Math.min(t, v.currentTime)); followFrames(item);
  };
  if ('requestVideoFrameCallback' in v) v.requestVideoFrameCallback(next); else requestAnimationFrame(next);
}
async function prepareCollection() {
  if (busy) return; busy = true; $('retry').hidden = true; $('cache-status').className = 'cache-status'; updateProgress();
  const pending = samples.filter(s => !prepared.has(s.id)); let cursor = 0, failed = false;
  const worker = async () => { while (cursor < pending.length) { const sample = pending[cursor++]; try { await prepareSample(sample); } catch { failed = true; } } };
  await Promise.all(Array.from({length: Math.min(4, pending.length)}, worker)); busy = false;
  if (failed || prepared.size !== samples.length) { showPreparationError(); return; }
  ready = true; $('prepare-cover').hidden = true; $('cache-status').className = 'cache-status ready'; $('cache-label').textContent = `All ${samples.length} examples ready`;
  document.querySelectorAll('.sample-card').forEach(button => { button.disabled = false; });
  selectSample(samples[0].id, false); controls(); message('Choose any example to start playback.');
}
function controls() {
  $('play').disabled = !ready || !current; $('restart').disabled = !ready || !current;
  $('play-label').textContent = current && !current.video.paused ? 'Pause' : current?.video.ended ? 'Replay example' : 'Play example';
}
function playCurrent() {
  if (!ready || !current) return;
  const video = current.video; if (video.ended) video.currentTime = 0;
  video.play().catch(() => message('Press the video’s play button to begin.')); controls();
}
function selectSample(id, autoplay = true) {
  if (!ready || !prepared.has(id)) return;
  if (current) { current.video.pause(); $('preload-bin').append(current.video); }
  current = prepared.get(id); renderKey = ''; const {sample, video} = current;
  video.currentTime = 0; video.playbackRate = Number($('playback-speed').value); $('video-stage').append(video);
  $('video-name').textContent = sample.title; $('selected-category').textContent = sample.category;
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
  $('prediction-event').textContent = last?.event || 'Observing';
  $('prediction-description').textContent = last ? DESCRIPTIONS[last.event] || '' : 'Follow the video to the first event.';
  $('event-index').textContent = last ? `${String(EVENTS.indexOf(last.event) + 1).padStart(2, '0')} / 05` : '— / 05';
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
    const copy = document.createElement('div'); copy.className = 'sample-copy'; const tag = document.createElement('span'); tag.textContent = `${sample.category} / ${String(index + 1).padStart(2, '0')}`; const title = document.createElement('strong'); title.textContent = sample.title; copy.append(tag, title); button.append(frame, copy); button.addEventListener('click', () => selectSample(sample.id)); $('sample-grid').append(button);
  });
  setCategory('ALL');
}
$('play').addEventListener('click', () => { if (!current) return; current.video.paused ? playCurrent() : current.video.pause(); });
$('restart').addEventListener('click', () => { if (!current) return; current.video.currentTime = 0; render(0, true); playCurrent(); });
$('playback-speed').addEventListener('change', () => { if (current) current.video.playbackRate = Number($('playback-speed').value); });
$('retry').addEventListener('click', () => { samples.length ? prepareCollection() : initialize(); });
async function initialize() {
  try { const data = await fetchChecked('samples.json', 'json'); samples = Array.isArray(data) ? data : data.samples; if (!samples?.length) throw Error(); buildLibrary(); await prepareCollection(); }
  catch { showPreparationError(); }
}
initialize();
