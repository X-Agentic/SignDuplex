from __future__ import annotations

import ast
from pathlib import Path

import pytest

from runtime.subtitle_timeline import (
    CSL_DAILY_ANNOTATIONS,
    CONCAT_METADATA_DIR,
    SUBTITLE_LEARNED,
    SUBTITLE_SOURCE,
    SubtitleTimelineManager,
    SubtitleTimelineRegistry,
)


VIDEO_NAME = "muted_concat_011_P0002.mp4"
TEXT_1 = "这家餐厅便宜又好吃，怪不得这么多人排队。"
TEXT_2 = "他从来不吸取教训，怪不得总是失败。"
TEXT_3 = "这种主意你都能想出来，怪不得大家都说你聪明！"
REFERENCE_FIXTURES_AVAILABLE = (
    (CONCAT_METADATA_DIR / "concat_011.json").is_file()
    and CSL_DAILY_ANNOTATIONS.is_file()
)
requires_reference_fixtures = pytest.mark.skipif(
    not REFERENCE_FIXTURES_AVAILABLE,
    reason="external reference annotations are intentionally not included",
)


@requires_reference_fixtures
def test_concat_011_uses_three_original_csl_daily_annotations():
    manager = SubtitleTimelineManager.from_video_path(VIDEO_NAME)
    assert manager.enabled
    assert len(manager.segments) == 3
    assert [segment.text for segment in manager.segments] == [
        TEXT_1,
        TEXT_2,
        TEXT_3,
    ]
    assert [segment.semantic_id for segment in manager.segments] == [
        "S003059",
        "S003060",
        "S003061",
    ]
    assert manager.annotation_source == CSL_DAILY_ANNOTATIONS
    assert manager.metadata_source.name == "concat_011.json"
    assert SUBTITLE_SOURCE == "REFERENCE_ANNOTATION"
    assert SUBTITLE_LEARNED is False


@requires_reference_fixtures
def test_segment_boundaries_and_static_pauses_are_exactly_separated():
    manager = SubtitleTimelineManager.from_video_path(VIDEO_NAME)
    assert manager.get_subtitle(0.0) == TEXT_1
    assert manager.get_subtitle(5.899999) == TEXT_1
    assert manager.get_subtitle(5.9) == ""
    assert manager.get_subtitle(10.439999) == ""
    assert manager.get_subtitle(10.44) == TEXT_2
    assert manager.get_subtitle(16.199999) == TEXT_2
    assert manager.get_subtitle(16.2) == ""
    assert manager.get_subtitle(20.339999) == ""
    assert manager.get_subtitle(20.34) == TEXT_3


@requires_reference_fixtures
def test_idle_pause_is_blank_and_barge_overlap_keeps_user_subtitle():
    manager = SubtitleTimelineManager.from_video_path(VIDEO_NAME)
    assert manager.get_subtitle(9.0) == ""
    assert manager.get_subtitle(21.0) == TEXT_3


def test_unknown_video_has_disabled_empty_timeline():
    manager = SubtitleTimelineManager.from_video_path("unrelated_video.mp4")
    assert not manager.enabled
    assert manager.get_subtitle(1.0) == ""
    assert manager.as_records() == []


def test_subtitle_registry_is_separate_and_clears_old_upload():
    registry = SubtitleTimelineRegistry()
    manager = SubtitleTimelineManager.from_video_path(VIDEO_NAME)
    registry.put("video-a", manager)
    assert registry.get("video-a") is manager
    registry.clear()
    try:
        registry.get("video-a")
    except KeyError:
        pass
    else:
        raise AssertionError("old subtitle timeline survived registry clear")


def test_subtitle_has_no_model_or_controller_input_path():
    root = Path(__file__).parents[1]
    subtitle_source = (root / "runtime/subtitle_timeline.py").read_text(
        encoding="utf-8"
    )
    subtitle_tree = ast.parse(subtitle_source)
    imported_modules = {
        node.module
        for node in ast.walk(subtitle_tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "runtime.model_runtime" not in imported_modules
    assert "runtime.interaction_controller" not in imported_modules
    assert "runtime.causal_sign_evidence" not in imported_modules

    app_tree = ast.parse((root / "app.py").read_text(encoding="utf-8"))
    controller_calls = [
        node
        for node in ast.walk(app_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "controller"
        and node.func.attr == "evaluate"
    ]
    model_submissions = [
        node
        for node in ast.walk(app_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "submit"
    ]
    assert len(controller_calls) == 1
    assert all("subtitle" not in ast.unparse(call) for call in controller_calls)
    assert all("subtitle" not in ast.unparse(call) for call in model_submissions)


def test_reference_sentence_clear_uses_playback_clock_without_decision_time_update():
    source = (Path(__file__).parents[1] / "app.py").read_text(encoding="utf-8")
    update_start = source.index("if current_subtitle_key != last_subtitle_key")
    update_end = source.index("            lifecycle_clock =", update_start)
    update_branch = source[update_start:update_end]

    assert "wall_elapsed - display_delay" in source
    assert "format_model_decision_time(" not in update_branch
    assert "global_decision_time" not in update_branch
    assert "gr.update()," in update_branch
    assert '"event": "REFERENCE_SENTENCE_CLEARED"' in update_branch


def test_ui_removes_note_while_logs_keep_annotation_provenance():
    source = (Path(__file__).parents[1] / "app.py").read_text(encoding="utf-8")
    assert 'gr.Markdown("### Reference Sentence")' in source
    assert 'label="Reference Subtitle (Debug)"' not in source
    assert "subtitle_debug = gr.State" in source
    assert "not a model translation" not in source
    assert '"subtitle_source": SUBTITLE_SOURCE' in source
    assert '"subtitle_learned": SUBTITLE_LEARNED' in source
    assert '"subtitle_segment_count": len(subtitle_manager.segments)' in source
    for forbidden in (
        "Model Translation",
        "Predicted Translation",
        "Streaming Translation",
    ):
        assert forbidden not in source


def test_all_stream_yields_match_extended_ui_output_count():
    tree = ast.parse(
        (Path(__file__).parents[1] / "app.py").read_text(encoding="utf-8")
    )
    stream = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "stream_demo"
    )
    lengths = {
        len(node.value.elts)
        for node in ast.walk(stream)
        if isinstance(node, ast.Yield) and isinstance(node.value, ast.Tuple)
    }
    assert lengths == {20}
