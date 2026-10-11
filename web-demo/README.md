# SignDuplex video examples

[Open the example collection](https://signduplex-demo.brown-charm-3727.chatgpt.site/#demo).

The static website includes 17 examples across five interaction events: 5 REJECT clips and 3 each for INCOMPLETE, WAIT, COMPLETE, and BARGE. Each video has a recorded model-event timeline. The page prepares the full video bytes, event records, and first decoded frame before enabling playback.

## Preview

```sh
python -m http.server 8792 --directory web-demo
```

Open `http://localhost:8792`. No build, inference server, or external tunnel is needed for this collection. Once preparation finishes, selecting an example uses only the local video buffer and event record. Pause, seek, replay, and speed changes share the video clock.

## Files

- `samples.json` lists each example and its event category.
- `media/` contains H.264 videos and How2Sign posters; `posters/` contains everyday-activity posters.
- `recordings/` contains the recorded prediction timelines.
- `LICENSE-media.txt` contains media provenance, attribution, and applicable license information.

To replace or add an example, update its media, recording, and manifest entry together, preserving the video timebase. The controller state for each example comes from its recorded run.
