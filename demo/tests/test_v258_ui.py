from __future__ import annotations

from pathlib import Path

from app import (
    DEBUG_MODE,
    format_prediction_display,
    prediction_to_chinese_explanation,
    prediction_to_simple_action,
    public_state_for_prediction,
    public_system_state,
    simple_action_for,
)


EXPECTED_EXPLANATIONS = {
    "IDLE": "用户未开始表达，系统继续监听",
    "INCOMPLETE": "用户仍在表达，尚未完成",
    "BACKCHANNEL": "用户仍在表达，系统给出简短反馈",
    "COMPLETE": "用户表达完成，系统开始回应",
    "NO_BARGE": "系统正在说话，用户未打断",
    "BARGE": "用户打断系统，系统切回监听",
}
EXPECTED_ACTIONS = {
    "IDLE": "KEEP_LISTENING",
    "INCOMPLETE": "KEEP_LISTENING",
    "BACKCHANNEL": "SHORT_FEEDBACK",
    "COMPLETE": "SWITCH_TO_SPEAKING",
    "NO_BARGE": "CONTINUE_SPEAKING",
    "BARGE": "SWITCH_TO_LISTENING",
}


def test_prediction_cards_always_include_fixed_chinese_explanation():
    assert prediction_to_chinese_explanation == EXPECTED_EXPLANATIONS
    for prediction, explanation in EXPECTED_EXPLANATIONS.items():
        card = format_prediction_display(prediction)
        assert card == f"## {prediction}\n\n{explanation}"


def test_public_system_actions_use_only_five_simplified_names():
    assert prediction_to_simple_action == EXPECTED_ACTIONS
    assert {simple_action_for(label) for label in EXPECTED_ACTIONS} == {
        "KEEP_LISTENING",
        "SHORT_FEEDBACK",
        "SWITCH_TO_SPEAKING",
        "CONTINUE_SPEAKING",
        "SWITCH_TO_LISTENING",
    }


def test_public_state_has_only_listening_and_speaking():
    assert public_system_state("LISTENING") == "LISTENING"
    assert public_system_state("TURN_READY") == "SPEAKING"
    assert public_system_state("SPEAKING") == "SPEAKING"
    assert {
        public_state_for_prediction(label) for label in EXPECTED_ACTIONS
    } == {"LISTENING", "SPEAKING"}
    assert public_state_for_prediction("COMPLETE") == "SPEAKING"
    assert public_state_for_prediction("NO_BARGE") == "SPEAKING"
    assert public_state_for_prediction("BARGE") == "LISTENING"


def test_waiting_is_removed_and_idle_is_the_initial_public_prediction():
    source = (Path(__file__).parents[1] / "app.py").read_text(encoding="utf-8")
    assert "Waiting" not in source
    assert '"WAITING"' not in source
    assert 'format_prediction_display("IDLE")' in source
    assert 'simple_action_for("IDLE")' in source


def test_debug_panels_and_internal_transition_timeline_are_hidden_by_default():
    source = (Path(__file__).parents[1] / "app.py").read_text(encoding="utf-8")
    assert DEBUG_MODE is False
    assert "with gr.Row(visible=DEBUG_MODE):" not in source
    assert "visible=DEBUG_MODE" not in source
    assert 'label="Controller Override"' not in source
    assert 'label="Motion Score"' not in source
    assert 'label="Interaction Timeline"' in source
    interaction_start = source.index('label="Interaction Timeline"')
    interaction_end = source.index("    )", interaction_start)
    assert "visible=False" in source[interaction_start:interaction_end]


def test_public_mapping_table_is_removed_in_v330():
    source = (Path(__file__).parents[1] / "app.py").read_text(encoding="utf-8")
    assert "Prediction-Action Mapping" not in source
    assert "public-mapping" not in source
    for prediction, action in EXPECTED_ACTIONS.items():
        assert f"| {prediction} | {action} |" not in source


def test_reference_sentence_is_published_only_for_accepted_complete():
    source = (Path(__file__).parents[1] / "app.py").read_text(encoding="utf-8")
    publish_start = source.index('"event": "REFERENCE_SENTENCE_PUBLISHED"')
    guard = source.rfind("if (", 0, publish_start)
    publish_branch = source[guard:publish_start]
    assert 'aligned.prediction == "COMPLETE"' in publish_branch
    assert "current_reference_segment is not None" in publish_branch
    assert '"event": "REFERENCE_SENTENCE_CLEARED"' in source


def test_internal_actions_remain_logged_but_public_timeline_is_simplified():
    source = (Path(__file__).parents[1] / "app.py").read_text(encoding="utf-8")
    assert '"system_action": state_transition.action' in source
    assert "simple_action_for(aligned.prediction)" in source
    assert "last_action = simple_action_for(aligned.prediction)" in source
