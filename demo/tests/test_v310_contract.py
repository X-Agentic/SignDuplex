from __future__ import annotations

from pathlib import Path

from config import BARGE_LABELS, CHECKPOINT, TASK_BY_STATE, TURN_LABELS
from runtime.interaction_state import InteractionStateManager
from runtime.state_controller import action_for


ROOT = Path(__file__).parents[1]


def test_v18_checkpoint_and_factorized_4x2_router():
    assert CHECKPOINT.name == "signduplex_v1_8.pt"
    assert CHECKPOINT.parent.name == "checkpoints"
    assert TURN_LABELS == ("IDLE", "INCOMPLETE", "BACKCHANNEL", "COMPLETE")
    assert BARGE_LABELS == ("BARGE", "NO_BARGE")
    assert TASK_BY_STATE == {
        "LISTENING": "turn_state",
        "SPEAKING": "barge_in",
    }


def test_model_prompt_matches_v18_training_label_order():
    source = (ROOT / "runtime/model_runtime.py").read_text(encoding="utf-8")
    assert "IDLE / INCOMPLETE / BACKCHANNEL / COMPLETE" in source
    assert "BARGE / NO_BARGE" in source
    assert "unified" not in source.lower()


def test_required_prediction_action_state_scenarios():
    manager = InteractionStateManager()
    assert manager.handle_prediction("IDLE").action == "KEEP_LISTENING"
    assert manager.current_system_state == "LISTENING"
    assert manager.handle_prediction("INCOMPLETE").action == "KEEP_LISTENING"
    assert manager.handle_prediction("BACKCHANNEL").action == "SHORT_FEEDBACK"
    complete = manager.handle_prediction("COMPLETE")
    assert complete.action == "SWITCH_TO_SPEAKING"

    manager.start_response()
    no_barge = manager.handle_prediction("NO_BARGE")
    assert no_barge.action == "CONTINUE_SPEAKING"
    assert no_barge.state_after == "SPEAKING"
    barge = manager.handle_prediction("BARGE")
    assert barge.action == "SWITCH_TO_LISTENING"
    assert barge.state_after == "LISTENING"


def test_only_required_public_action_vocabulary_is_mapped():
    actions = {
        label: action_for(label)[0]
        for label in TURN_LABELS + BARGE_LABELS
    }
    assert actions == {
        "IDLE": "KEEP_LISTENING",
        "INCOMPLETE": "KEEP_LISTENING",
        "BACKCHANNEL": "SHORT_FEEDBACK",
        "COMPLETE": "SWITCH_TO_SPEAKING",
        "BARGE": "SWITCH_TO_LISTENING",
        "NO_BARGE": "CONTINUE_SPEAKING",
    }


def test_app_preserves_raw_result_and_logs_controller_result_separately():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "controller_result = controller.evaluate(" in source
    assert "controller_prediction = controller_result.controller_prediction" in source
    for field in (
        '"raw_prediction": raw_prediction',
        '"controller_prediction": controller_prediction',
        '"controller_override_reason": (',
        '"system_action": action',
        '"current_state": public_system_state(',
    ):
        assert field in source


def test_public_ui_contract_and_reference_sentence_guard():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    for heading in (
        "### Video Player",
        "### Reference Sentence",
        "### Current Prediction",
        "### Simulated TTS Waveform",
    ):
        assert heading in source
    for label in ("Current System State", "Decision Time", "System Action"):
        assert f'label="{label}"' in source
    for removed in (
        'label="Controller Override"',
        'label="Controller Override Reason"',
        'label="Controller-level IDLE',
        'label="Motion Score"',
        'label="Hand Motion Score"',
        'label="Raw Prediction',
    ):
        assert removed not in source
    publish = source.index('"event": "REFERENCE_SENTENCE_PUBLISHED"')
    guard = source.rfind("if (", 0, publish)
    assert 'aligned.prediction == "COMPLETE"' in source[guard:publish]


def test_turn_context_remains_state_relative_and_causal():
    source = (ROOT / "runtime/turn_context.py").read_text(encoding="utf-8")
    assert 'if system_state == "LISTENING":\n            return self.turn_start_frame' in source
    assert 'if system_state == "SPEAKING":\n            return self.state_enter_frame' in source
