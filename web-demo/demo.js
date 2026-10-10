'use strict';
const config = window.SIGNDUPLEX_DEMO || {};
const API = (config.apiBase || '').replace(/\/$/, '');
const $ = id => document.getElementById(id);
const video = $('demo-video');
const EVENTS = ['REJECT', 'INCOMPLETE', 'WAIT', 'COMPLETE', 'BARGE'];
const descriptions = {
  REJECT: 'Movement that does not call for a response.',
  INCOMPLETE: 'The signed utterance is still in progress.',
  WAIT: 'A pause within an unfinished utterance.',
  COMPLETE: 'The current utterance is predicted to be complete.',
  BARGE: 'New signing is predicted during a system response.'
};
const actions = {continue: 'Continue observing', start_response: 'Start response', interrupt_response: 'Interrupt response', response_finished: 'Response finished'};
let samples = [], selected = null, asset = null, runId = null, result = null;
let generation = 0, phase = 'empty', ready = false, token = null, tokenRequest = null, objectURL = null, lastRenderKey = '';
const wait = ms => new Promise(resolve => setTimeout(resolve, ms));
// The static host does not serve byte ranges. A local Blob keeps the native
// video controls seekable; the ten small public clips are cached per page.
const publicMedia = new Map();
function publicVideoURL(path) {
  if (!publicMedia.has(path)) publicMedia.set(path, fetch(path).then(response => {
    if (!response.ok) throw Error('This sample video could not be loaded. Please try again.');
    return response.blob();
  }).then(blob => URL.createObjectURL(blob)).catch(error => { publicMedia.delete(path); throw error; }));
  return publicMedia.get(path);
}

const url = path => new URL(path, `${API}/`).href;
const clock = seconds => { const n = Math.max(0, Math.floor(Number(seconds) || 0)); return `${String(Math.floor(n / 60)).padStart(2, '0')}:${String(n % 60).padStart(2, '0')}`; };
const eventAt = (recording, time) => (recording?.records || []).filter(r => Number(r.time) <= time + 1e-7);
function status(message, error = false) { $('demo-status').textContent = message; $('demo-status').classList.toggle('error', error); }
function service(message, state) { $('service-status').className = `service-indicator ${state || ''}`; $('service-status').querySelector('span').textContent = message; $('retry-connection').hidden = state !== 'unavailable'; }
function buttons() {
  const working = ['uploading', 'analyzing', 'cancelling'].includes(phase);
  $('analyze').disabled = !selected || !ready || working;
  $('analyze').textContent = result ? (video.paused ? (video.ended ? 'Replay video' : 'Play video') : 'Playing') : 'Analyze video';
  $('stop').disabled = !selected || (!working && video.paused);
  $('stop').textContent = working ? 'Cancel analysis' : 'Pause';
  $('initial-state').disabled = working;
  $('analysis-progress').hidden = !['uploading', 'analyzing'].includes(phase);
}
async function readJSON(response) {
  let body;
  try { body = await response.json(); } catch { throw Error('The demo service did not return a valid response. Please try again.'); }
  if (!response.ok) {
    if (response.status === 409) throw Error('The model is serving another visitor. Please try again shortly.');
    if (response.status === 429) throw Error('The demo is receiving too many requests. Please wait before trying again.');
    if (response.status === 413) throw Error('Please choose a video smaller than 50 MB.');
    if (response.status === 401) { token = null; throw Error('This demo session has expired. Select your video and try again.'); }
    throw Error(typeof body.detail === 'string' ? body.detail : 'This request could not be completed. Please try another video.');
  }
  return body;
}
async function ensureSession() {
  if (token) return token;
  if (!API) throw Error('The online demo is temporarily unavailable. You can still watch the recorded examples below.');
  if (!tokenRequest) tokenRequest = fetch(url('/api/session'), {method: 'POST', cache: 'no-store'})
    .then(readJSON).then(body => { token = body.token; if (!token) throw Error('Could not open a demo session.'); return token; }).finally(() => { tokenRequest = null; });
  return tokenRequest;
}
async function api(path, options = {}) {
  const session = await ensureSession();
  try { return await readJSON(await fetch(url(path), {...options, cache: 'no-store', headers: {...options.headers, 'X-Demo-Session': session}})); }
  catch (error) { if (error instanceof TypeError) throw Error('The online service could not be reached. Please try again shortly; recorded examples remain available below.'); throw error; }
}
const post = (path, body = {}) => api(path, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
async function health() {
  if (!API) { service('Online service unavailable', 'unavailable'); ready = false; buttons(); return; }
  try {
    const h = await readJSON(await fetch(url('/api/health'), {cache: 'no-store', signal: AbortSignal.timeout(12000)}));
    ready = h.stage === 'ready' || h.ready === true;
    service(ready ? (h.busy ? 'Model online · currently busy' : 'Model online · ready to analyze') : 'Model warming up', ready ? 'ready' : '');
  } catch { ready = false; service('Online service unavailable', 'unavailable'); }
  buttons();
}
async function cancelRun(id) { if (id) { try { await post(`/api/runs/${encodeURIComponent(id)}/stop`); } catch {} } }
function clearResult() { result = null; asset = null; lastRenderKey = ''; $('download-results').hidden = true; $('run-mode').textContent = 'VIDEO PREVIEW'; render(0, true); }
function releaseObjectURL() { if (objectURL) { URL.revokeObjectURL(objectURL); objectURL = null; } }
function selectSample(sample) {
  const version = ++generation; const old = runId; runId = null; video.pause(); releaseObjectURL(); selected = {type: 'sample', sample}; phase = 'selected';
  $('initial-state').value = sample.initial_state || 'RECEIVING'; clearResult();
  video.removeAttribute('src'); video.poster = sample.poster || ''; video.load(); $('video-empty').hidden = true;
  publicVideoURL(sample.file).then(source => { if (version === generation) { video.src = source; video.load(); } }).catch(error => { if (version === generation) status(error.message, true); });
  $('video-name').textContent = sample.title; $('reference-line').hidden = !sample.reference; $('reference-text').textContent = sample.reference || '';
  document.querySelectorAll('.sample-card').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.id === sample.id)));
  status('Video selected. Choose Analyze video to compute predictions, then play them in sync.'); buttons(); cancelRun(old);
}
function selectUpload(file) {
  if (!file) return;
  if (file.size > 50 * 1024 * 1024) { status('Please choose a video smaller than 50 MB.', true); return; }
  generation++; const old = runId; runId = null; video.pause(); releaseObjectURL(); selected = {type: 'upload', file}; phase = 'selected';
  $('initial-state').value = 'RECEIVING'; clearResult(); objectURL = URL.createObjectURL(file); video.src = objectURL; video.removeAttribute('poster'); video.load(); $('video-empty').hidden = true;
  $('video-name').textContent = file.name; $('reference-line').hidden = true; $('reference-text').textContent = '';
  document.querySelectorAll('.sample-card').forEach(b => b.setAttribute('aria-pressed', 'false'));
  status('Upload selected. Choose Analyze video to send it to the model.'); buttons(); cancelRun(old);
}
async function startAnalysis() {
  if (!selected || ['uploading', 'analyzing', 'cancelling'].includes(phase)) return;
  if (!ready) { await health(); if (!ready) { status('The model service is currently unavailable. Recorded examples remain playable below.', true); return; } }
  if (result) { if (video.ended) video.currentTime = 0; await play(); return; }
  if (Number.isFinite(video.duration) && video.duration > 120.05) { status('Please select a video no longer than two minutes.', true); return; }
  const version = ++generation, source = selected, initial = $('initial-state').value;
  phase = 'uploading'; video.pause(); video.currentTime = 0; clearResult(); $('analysis-meter').removeAttribute('value'); $('analysis-count').textContent = 'Preparing video'; buttons();
  status('Preparing your video for the model…');
  try {
    let file;
    if (source.type === 'upload') file = source.file;
    else {
      const response = await fetch(source.sample.file); if (!response.ok) throw Error('The sample video could not be loaded. Please select another sample.');
      file = new File([await response.blob()], `${source.sample.id}.mp4`, {type: 'video/mp4'});
    }
    if (version !== generation) return;
    const form = new FormData(); form.append('file', file);
    asset = await api('/api/upload', {method: 'POST', body: form});
    if (version !== generation) return;
    if (!asset.media_url) throw Error('The prepared video is not available. Please try again.');
    video.src = url(asset.media_url); video.load(); phase = 'analyzing'; buttons();
    let run = await post('/api/runs', {asset_id: asset.id, initial_state: initial, response_seconds: 6});
    if (version !== generation) { await cancelRun(run.id); return; }
    runId = run.id;
    while (['starting', 'running', 'queued'].includes(run.status)) {
      const n = Number(run.progress) || 0, total = Number(run.total) || 0;
      $('analysis-meter').value = total ? 100 * n / total : 0; $('analysis-count').textContent = `${n} / ${total || '…'} decisions`;
      status(`Analyzing the video: ${n} / ${total || '…'} decisions. Playback will be ready when analysis finishes.`);
      await wait(750); if (version !== generation) return;
      run = await api(`/api/runs/${encodeURIComponent(run.id)}`); if (version !== generation) return;
    }
    if (run.status !== 'complete') throw Error(run.error || 'Analysis was stopped. Choose Analyze video to try again.');
    result = run; phase = 'prepared'; lastRenderKey = ''; $('run-mode').textContent = 'MODEL PREDICTIONS';
    if (run.download_url) { $('download-results').href = url(run.download_url); $('download-results').hidden = false; }
    video.currentTime = 0; render(0, true); buttons(); await play();
  } catch (error) { if (version !== generation) return; phase = selected ? 'selected' : 'empty'; result = null; status(error.message, true); buttons(); }
}
async function play() { try { await video.play(); status(result ? 'Predictions are synchronized to playback. Pause or seek to inspect a moment.' : 'Video preview. Analyze the clip to see model predictions.'); } catch { status('Predictions are ready. Press the video’s play button to begin.'); } buttons(); }
async function stop() {
  video.pause(); render(video.currentTime, true);
  if (['uploading', 'analyzing'].includes(phase)) {
    const version = ++generation, old = runId; runId = null; phase = 'cancelling'; buttons(); status('Stopping this analysis…'); await cancelRun(old);
    if (version !== generation) return; phase = selected ? 'selected' : 'empty'; status('Analysis cancelled. Choose Analyze video to start again.'); buttons();
  } else { status('Playback paused. Drag the timeline or choose Play video to continue.'); buttons(); }
}
function reset() {
  generation++; const old = runId; runId = null; selected = null; video.pause(); video.removeAttribute('src'); video.removeAttribute('poster'); video.load(); releaseObjectURL(); phase = 'empty';
  $('video-empty').hidden = false; $('video-name').textContent = 'Choose a video to begin'; $('reference-line').hidden = true; $('initial-state').value = 'RECEIVING';
  document.querySelectorAll('.sample-card').forEach(b => b.setAttribute('aria-pressed', 'false')); clearResult(); buttons(); status('Choose a sample, or upload another video.'); cancelRun(old);
}
function render(time, force = false) {
  time = Number.isFinite(time) ? Math.max(0, time) : 0;
  const records = eventAt(result, time), predictions = records.filter(r => r.kind === 'prediction');
  const p = predictions.at(-1), last = records.at(-1), state = last?.state_after || result?.initial_state || $('initial-state').value;
  const key = `${runId}|${p?.seq ?? p?.time ?? -1}|${last?.seq ?? last?.time ?? -1}|${state}`;
  $('media-clock').textContent = `${clock(time)} / ${clock(video.duration || selected?.sample?.duration)}`;
  if (force || key !== lastRenderKey) {
    lastRenderKey = key; $('prediction-panel').dataset.event = p?.event || ''; $('prediction-panel').dataset.predictionTime = p ? String(p.time) : '';
    $('system-state').textContent = state; $('system-state').classList.toggle('responding', state === 'RESPONDING');
    $('prediction-event').textContent = p?.event || '—'; $('prediction-description').textContent = p ? descriptions[p.event] || '' : result ? 'Play to the first decision to see a prediction.' : 'Analyze a video to see the interaction events.';
    const confidence = p?.probabilities?.[p.event]; $('prediction-confidence').textContent = typeof confidence === 'number' ? `${(confidence * 100).toFixed(1)}%` : '—';
    $('decision-time').textContent = p ? `${Number(Number(p.time).toFixed(3))} s` : '—';
    $('system-action').textContent = last ? actions[last.action] || last.action || 'Continue observing' : 'Awaiting analysis';
    for (const label of EVENTS) { const prob = Number(p?.probabilities?.[label]) || 0; const row = $('probabilities').querySelector(`[data-event="${label}"]`); row.querySelector('i').style.width = `${Math.min(1, Math.max(0, prob)) * 100}%`; row.querySelector('output').textContent = p ? `${(prob * 100).toFixed(1)}%` : '—'; }
    $('timeline-count').textContent = predictions.length ? `${predictions.length} decisions shown` : '';
    $('event-timeline').replaceChildren(...predictions.map((r, i) => { const tr = document.createElement('tr'); if (i === predictions.length - 1) tr.className = 'current'; for (const text of [`${Number(Number(r.time).toFixed(3))} s`, r.event, actions[r.action] || r.action]) { const td = document.createElement('td'); td.textContent = text; tr.append(td); } return tr; }));
  }
  $('prediction-panel').dataset.mediaTime = String(time);
  const responding = !!result && state === 'RESPONDING' && !video.paused && !video.ended;
  $('response-status').textContent = responding ? 'Responding · simulated' : (state === 'RESPONDING' && result ? 'Paused' : 'Silent');
  let d = ''; for (let x = 0; x <= 360; x += 4) { const y = 26 + (responding ? Math.sin(Math.PI * x / 360) ** 1.4 * (10 * Math.sin(x * .10 - time * 12) + 5 * Math.sin(x * .17 + time * 7)) : 0); d += `${x ? 'L' : 'M'}${x} ${y.toFixed(2)} `; } $('response-wave').setAttribute('d', d);
}
function buildSamples() {
  $('sample-count').textContent = `${samples.length} sample videos`;
  $('sample-grid').replaceChildren(...samples.map((sample, index) => {
    const b = document.createElement('button'); b.className = 'sample-card'; b.dataset.id = sample.id; b.setAttribute('aria-pressed', 'false'); b.setAttribute('aria-label', `Select sample ${index + 1}: ${sample.title}`);
    const frame = document.createElement('div'); frame.className = 'sample-image'; const img = document.createElement('img'); img.src = sample.poster; img.alt = ''; img.loading = 'lazy'; frame.append(img); const duration = document.createElement('span'); duration.className = 'sample-duration'; duration.textContent = clock(sample.duration); frame.append(duration);
    const copy = document.createElement('div'); copy.className = 'sample-copy'; const id = document.createElement('span'); id.textContent = `SAMPLE ${String(index + 1).padStart(2, '0')}`; const title = document.createElement('strong'); title.textContent = sample.title; copy.append(id, title); b.append(frame, copy); b.addEventListener('click', () => selectSample(sample)); return b;
  }));
}
async function buildShowcases() {
  const featured = samples.filter(s => s.featured && s.recording);
  for (const [index, sample] of featured.entries()) {
    const card = document.createElement('article'); card.className = 'showcase-card';
    const player = document.createElement('video'); player.poster = sample.poster; player.controls = true; player.playsInline = true; player.preload = 'metadata'; player.setAttribute('aria-label', `${sample.title}: recorded model run`);
    const copy = document.createElement('div'); copy.className = 'showcase-copy'; const tag = document.createElement('span'); tag.textContent = `RECORDED RUN / ${String(index + 1).padStart(2, '0')}`;
    const title = document.createElement('h3'); title.textContent = sample.title; const detail = document.createElement('p'); detail.textContent = sample.description || (sample.initial_state === 'RESPONDING' ? 'Starts in the responding state to test interruption.' : 'Follow the model’s events as the clip plays.');
    const badge = document.createElement('div'); badge.className = 'showcase-event'; const event = document.createElement('strong'); event.textContent = 'Press play'; const time = document.createElement('span'); time.textContent = 'Recorded predictions'; badge.append(event, time);
    const tryButton = document.createElement('button'); tryButton.className = 'showcase-try'; tryButton.textContent = 'Try this video yourself ↗'; tryButton.addEventListener('click', () => { player.pause(); selectSample(sample); $('demo').scrollIntoView({behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth'}); });
    copy.append(tag, title, detail, badge, tryButton); card.append(player, copy); $('showcase-grid').append(card);
    publicVideoURL(sample.file).then(source => { player.src = source; player.load(); }).catch(() => { time.textContent = 'Video unavailable · try refreshing'; });
    let recording = null;
    try { const response = await fetch(sample.recording); if (!response.ok) throw Error(); recording = await response.json(); }
    catch { event.textContent = 'Video preview'; time.textContent = 'Recording unavailable'; }
    const sync = () => { if (!recording) return; const rows = eventAt(recording, player.currentTime).filter(r => r.kind === 'prediction'); const p = rows.at(-1); event.textContent = p?.event || 'Before first decision'; time.textContent = p ? `${Number(Number(p.time).toFixed(3))} s · ${p.state_after === 'RESPONDING' ? 'Responding' : 'Receiving'}` : clock(player.currentTime); card.dataset.predictionTime = p ? String(p.time) : ''; };
    for (const e of ['timeupdate', 'seeking', 'seeked', 'ended']) player.addEventListener(e, sync);
    player.addEventListener('play', () => { video.pause(); document.querySelectorAll('.showcase-card video').forEach(v => { if (v !== player) v.pause(); }); sync(); });
  }
}
for (const event of EVENTS) { const row = document.createElement('div'); row.className = 'probability-row'; row.dataset.event = event; const label = document.createElement('span'); label.textContent = event; const track = document.createElement('span'); track.className = 'probability-track'; track.append(document.createElement('i')); const val = document.createElement('output'); val.textContent = '—'; row.append(label, track, val); $('probabilities').append(row); }
$('retry-connection').addEventListener('click', health);
$('analyze').addEventListener('click', startAnalysis); $('stop').addEventListener('click', stop); $('reset').addEventListener('click', reset);
$('upload').addEventListener('change', () => { selectUpload($('upload').files[0]); $('upload').value = ''; });
$('initial-state').addEventListener('change', () => { generation++; const old = runId; runId = null; video.pause(); clearResult(); phase = selected ? 'selected' : 'empty'; status('Initial state changed. Analyze the video again to apply it.'); buttons(); cancelRun(old); });
$('playback-speed').addEventListener('change', () => { video.playbackRate = Number($('playback-speed').value); render(video.currentTime, true); });
for (const e of ['pause', 'play', 'seeking', 'seeked', 'ended', 'ratechange', 'loadedmetadata']) video.addEventListener(e, () => { if (e === 'loadedmetadata') video.playbackRate = Number($('playback-speed').value); if (['uploading', 'analyzing'].includes(phase) && e === 'play') video.pause(); if (e === 'play') document.querySelectorAll('.showcase-card video').forEach(v => v.pause()); render(video.ended ? Math.max(video.duration || 0, result?.duration || 0) : video.currentTime, true); buttons(); });
video.addEventListener('ended', () => status(result ? 'Playback complete. Replay or seek to inspect the predictions.' : 'Preview complete. Analyze this video to see the model’s predictions.'));
video.addEventListener('error', () => { if (selected && video.getAttribute('src')) status('The video could not be loaded. Select it again or choose another clip.', true); });
if ('requestVideoFrameCallback' in video) { const frame = (now, meta) => { const t = video.paused || video.seeking || Math.abs(meta.mediaTime - video.currentTime) > .12 ? video.currentTime : meta.mediaTime; render(t); video.requestVideoFrameCallback(frame); }; video.requestVideoFrameCallback(frame); }
else { const frame = () => { if (!video.paused) render(video.currentTime); requestAnimationFrame(frame); }; requestAnimationFrame(frame); }
async function initialize() {
  render(0, true); buttons(); const connecting = health();
  try { const response = await fetch(config.samples || 'samples.json'); if (!response.ok) throw Error(); const data = await response.json(); samples = Array.isArray(data) ? data : data.samples; if (!Array.isArray(samples) || !samples.length) throw Error(); buildSamples(); await buildShowcases(); const requested = new URLSearchParams(location.search).get('sample'); selectSample(samples.find(s => s.id === requested) || samples[0]); }
  catch { status('Sample videos could not be loaded. Please refresh the page or upload your own clip.', true); }
  await connecting;
}
initialize();
