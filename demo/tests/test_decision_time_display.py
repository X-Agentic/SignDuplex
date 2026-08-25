from __future__ import annotations

from pathlib import Path

from app import format_model_decision_time


def test_periodic_model_decisions_use_integer_seconds():
    assert format_model_decision_time(1.0, is_final_model_decision=False) == "1 s"
    assert format_model_decision_time(6.0, is_final_model_decision=False) == "6 s"
    assert format_model_decision_time(25.0, is_final_model_decision=False) == "25 s"


def test_non_integer_final_model_decision_keeps_real_time():
    assert (
        format_model_decision_time(25.4667, is_final_model_decision=True)
        == "25.47 s"
    )


def test_controller_events_cannot_overwrite_decision_time():
    source = (Path(__file__).parents[1] / "app.py").read_text(encoding="utf-8")
    lifecycle_start = source.index("for lifecycle_event in tts_simulator.poll")
    lifecycle_end = source.index(
        "            while (\n                worker is not None", lifecycle_start
    )
    lifecycle_branch = source[lifecycle_start:lifecycle_end]

    assert 'f"{event_time:.2f} s"' not in source
    assert "format_model_decision_time(" not in lifecycle_branch
    assert "                    gr.update(),\n                    last_action," in lifecycle_branch


def test_only_aligned_model_decision_formats_decision_time():
    source = (Path(__file__).parents[1] / "app.py").read_text(encoding="utf-8")
    aligned_start = source.index("            while (\n                pending_displays")
    aligned_end = source.index("            playback_done =", aligned_start)
    aligned_branch = source[aligned_start:aligned_end]

    assert "format_model_decision_time(" in aligned_branch
    assert 'aligned.record["decision_index"] == len(decisions)' in aligned_branch
