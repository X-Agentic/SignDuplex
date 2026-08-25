from __future__ import annotations

from pathlib import Path


APP_SOURCE = (Path(__file__).parents[1] / "app.py").read_text(encoding="utf-8")


def waveform_javascript() -> str:
    start = APP_SOURCE.index("          const waveform = (() => {")
    end = APP_SOURCE.index(
        '          if (signal && signal.startsWith("PLAY_VIDEO:"))', start
    )
    return APP_SOURCE[start:end]


def test_prediction_action_mapping_is_completely_removed_from_main_ui():
    assert "Prediction-Action Mapping" not in APP_SOURCE
    assert "public-mapping" not in APP_SOURCE
    assert "| Prediction | System Action |" not in APP_SOURCE
    assert "| IDLE | KEEP_LISTENING |" not in APP_SOURCE


def test_initial_listening_waveform_is_a_flat_baseline():
    assert 'gr.Markdown("### Simulated TTS Waveform")' in APP_SOURCE
    assert 'id="tts-waveform-panel"' in APP_SOURCE
    assert 'id="tts-waveform-path" data-mode="QUIET"' in APP_SOURCE
    assert 'd="M 0 70 L 640 70"' in APP_SOURCE
    assert "Waveform is synchronized with the system reply duration." in APP_SOURCE


def test_system_response_start_activates_stationary_smooth_waveform():
    script = waveform_javascript()
    assert 'signal.startsWith("TTS_START:")' in script
    assert "waveform.startFullReply(duration, token);" in script
    assert "Math.random()" not in script
    assert "const anchorCount = 64;" in script
    assert "point.currentAmplitude +=" in script
    assert "0.55 * Math.sin" in script
    assert "draw(stationarySamples(timestamp), state.mode)" in script
    assert "requestAnimationFrame(tick)" in script


def test_waveform_uses_existing_reply_lifecycle_without_duration_timer():
    script = waveform_javascript()
    full_reply = script[script.index("startFullReply:"):script.index("startShortFeedback:")]
    assert "setTimeout(" not in full_reply
    assert 'f"TTS_START:{run_id}:{lifecycle_event.response_id}:"' in APP_SOURCE
    assert 'f"{response.tts_duration:.6f}"' in APP_SOURCE
    assert 'f"STOP:NORMAL:{run_id}:{lifecycle_event.response_id}"' in APP_SOURCE
    assert 'lifecycle_event.event == "SYSTEM_RESPONSE_STARTED"' in APP_SOURCE
    assert 'lifecycle_event.event == "SYSTEM_RESPONSE_FINISHED"' in APP_SOURCE


def test_normal_finish_smoothly_restores_flat_waveform():
    script = waveform_javascript()
    assert 'signal.startsWith("STOP:")' in APP_SOURCE
    assert "waveform.stopNormal();" in APP_SOURCE
    assert "stopNormal: () => stopWithFade(185)" in script
    assert 'state.mode = "QUIET";' in script
    assert "cancelAnimationFrame(state.frame)" in script
    assert 'draw(Array(anchorCount).fill(0), "QUIET")' in script


def test_barge_emits_stop_and_returns_waveform_to_flat():
    barge_branch = APP_SOURCE.index('if aligned.prediction == "BARGE":')
    next_section = APP_SOURCE.index("                last_effective", barge_branch)
    branch = APP_SOURCE[barge_branch:next_section]
    assert 'f"STOP:BARGE:{run_id}:{aligned.record[\'response_id\']}"' in branch
    assert 'signal.startsWith("STOP:")' in APP_SOURCE
    assert "waveform.stopForBarge();" in APP_SOURCE
    assert "stopForBarge: () => stopWithFade(65)" in APP_SOURCE


def test_multiple_reply_rounds_can_restart_after_each_flat_reset():
    script = waveform_javascript()
    assert "startFullReply:" in script
    assert "startShortFeedback:" in script
    assert "stopNormal:" in script
    assert "stopForBarge:" in script
    assert "resetQuiet:" in script
    assert 'state.mode = "QUIET";' in script
    assert "state.lastFrameAt = 0;" in script
    assert "state.points = buildPoints(token);" in script


def test_waveform_panel_sits_in_right_column_without_mapping_placeholder():
    prediction = APP_SOURCE.index('gr.Markdown("### Current Prediction")')
    waveform = APP_SOURCE.index('gr.Markdown("### Simulated TTS Waveform")')
    hidden_states = APP_SOURCE.index("raw_prediction_debug = gr.State", waveform)
    timeline = APP_SOURCE.index("    timeline = gr.Dataframe(", hidden_states)
    assert prediction < waveform < hidden_states < timeline
    assert "public-mapping" not in APP_SOURCE
