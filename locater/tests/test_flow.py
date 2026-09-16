"""Tests for FlowExecutor (v6)."""

import pytest
from locater.flow import parse_step_string, FlowExecutor


def test_parse_step_string():
    s1 = parse_step_string("nav:https://example.com")
    assert s1 == {"action": "nav", "url": "https://example.com"}

    s2 = parse_step_string("click:text=Start a post,settle=0.2")
    assert s2["action"] == "click"
    assert s2["text"] == "Start a post"
    assert s2["settle"] == 0.2

    s3 = parse_step_string("type:Hello world!,enter=true")
    assert s3["action"] == "type"
    assert s3["text"] == "Hello world!"
    assert s3["enter"] is True

    s4 = parse_step_string("wait:text=Submit,timeout=5,interval=0.2")
    assert s4["action"] == "wait"
    assert s4["text"] == "Submit"
    assert s4["timeout"] == 5
    assert s4["interval"] == 0.2


def test_flow_executor_mock_run():
    executor = FlowExecutor(default_window="W1")
    # Execute simple non-destructive key and sleep steps
    steps = [
        {"action": "sleep", "duration": 0.05},
        {"action": "key", "combo": "shift"},
    ]
    res = executor.run_flow(steps, post_dump=False)
    assert res["ok"] is True
    assert res["steps_executed"] == 2
    assert len(res["step_results"]) == 2
