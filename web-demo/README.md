# SignDuplex video examples

[Open the example collection](https://signduplex-demo.brown-charm-3727.chatgpt.site/?v=cachefix-20261011#demo).

This static website includes 17 examples across five interaction events: 5 REJECT clips and 3 each for INCOMPLETE, WAIT, COMPLETE, and BARGE. Each clip plays alongside its recorded model-event timeline.

## Playback and release binding

The entry page embeds one validated collection containing the example metadata, posters, and recorded predictions. JavaScript, CSS, and videos use content-addressed filenames, so a release selects matching resources. The runtime does not request the legacy `samples.json` URL. Previously published hash assets and the original video aliases are retained for cached entry pages.

Videos prebuffer independently in four concurrent workers. Complete video bytes are checked against their expected length and, where Web Crypto is available, SHA-256. Each ready example can play immediately while other examples are still loading. An unavailable clip affects only that clip; playback does not wait for every clip or for hidden decoded frames.

The page uses a single visible video element. Its video clock drives the recorded events and system-state display through pause, seek, replay, and speed changes. Each resource request has a 16-second timeout and up to three attempts with a short backoff. Transient failures recover automatically; a failed example can be retried from its card or the retry button.

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
