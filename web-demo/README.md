# SignDuplex video examples

[Open the example collection](https://x-agentic.github.io/SignDuplex/?v=playback-20261012#demo).

This static website includes 25 examples across five interaction events: 5 each for REJECT, INCOMPLETE, WAIT, COMPLETE, and BARGE. Each clip plays alongside its recorded model-event timeline. The five BARGE examples join clips from the same source recording and start in RECEIVING: a recorded COMPLETE is followed by BARGE when signing resumes, then further predictions. Two WAIT examples show a hold followed by continued signing.

## Hosting

The public demo is served directly from this repository on GitHub Pages. The `Publish SignDuplex demo` workflow publishes `web-demo/` when its files change on `main`; it can also be run manually. Visitors do not need a ChatGPT account or a running inference server.

## Playback and release binding

The entry page embeds one validated collection containing the example metadata, posters, and recorded predictions. JavaScript, CSS, and videos use content-addressed filenames, so a release selects matching resources. The runtime does not request the legacy `samples.json` URL. Previously published hash assets and the original video aliases are retained for cached entry pages.

Every example is selectable as soon as the page renders. An uncached selection starts native MP4 playback immediately, using byte-range requests, without waiting for the collection or a complete-file JavaScript download. All videos and event records are generated before publication. First-time visitors still need to receive video bytes over their connection.

A single background downloader saves complete, length- and SHA-256-verified videos in Cache Storage for reuse across visits. Cached clips play from local Blob URLs. Storage denial or a failed background download does not disable playback. A new selection cancels lower-priority downloads; playback receives bandwidth first. Background transfers time out only after receiving no data, so a progressing slow transfer is not discarded after 16 seconds.

Each media hash has an immutable jsDelivr mirror pinned to the source commit. The player switches sources automatically on an error or a stalled transfer and resumes at the current timestamp. Mirror candidates live in `samples.json` under `media_mirrors`, keyed by the complete SHA-256; changing a video cannot silently reuse an obsolete mirror. After publishing new videos, add mirrors from the commit containing those exact bytes and rebuild.

The page uses one visible video element. Its clock drives recorded events and state through pause, seek, replay, and speed changes. Completing a background cache never replaces the source of a video already playing.

## Preview

```sh
python -m http.server 8792 --directory web-demo
```

Open `http://localhost:8792`. The checked-in release works directly as a static site, with no inference server or external tunnel.

## Rebuild

The build script requires Python and `beautifulsoup4`. From the repository root:

```sh
python web-demo/scripts/build_release.py --static-dir web-demo
```

Update an example's video, poster, recording, and `samples.json` entry together before rebuilding. Preserve the video timebase and original recorded predictions. The builder embeds the collection, creates content-addressed assets, and updates the entry page without removing earlier hash assets or video aliases.

## Files

- `index.html` embeds the version-bound collection and selects hashed app and style assets.
- `samples.json` is the editable build input; `assets/collection.*.json` is the matching release metadata.
- `media/` contains H.264 videos and How2Sign posters; `posters/` contains everyday-activity posters.
- `recordings/` preserves the recorded prediction timelines embedded in the release.
- `scripts/build_release.py` and `scripts/demo-runtime.js` reproduce the static release.
- `LICENSE-media.txt` records media provenance, attribution, and applicable licenses.
