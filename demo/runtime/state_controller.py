"""State-routed class and action mapping."""

from __future__ import annotations


ACTIONS = {
    "IDLE": ("KEEP_LISTENING", "用户未开始表达，系统继续监听"),
    "INCOMPLETE": ("KEEP_LISTENING", "用户仍在表达，尚未完成"),
    "BACKCHANNEL": ("SHORT_FEEDBACK", "用户仍在表达，系统给出简短反馈"),
    "COMPLETE": ("SWITCH_TO_SPEAKING", "用户表达完成，系统开始回应"),
    "NO_BARGE": ("CONTINUE_SPEAKING", "系统正在说话，用户未打断"),
    "BARGE": ("SWITCH_TO_LISTENING", "用户打断系统，系统切回监听"),
}


def action_for(label: str) -> tuple[str, str]:
    return ACTIONS[label]


def assert_routed_label(system_state: str, label: str) -> None:
    allowed = {
        "LISTENING": {"IDLE", "COMPLETE", "INCOMPLETE", "BACKCHANNEL"},
        "SPEAKING": {"NO_BARGE", "BARGE"},
    }
    if label not in allowed[system_state]:
        raise RuntimeError(f"Invalid routed result: {system_state}/{label}")
