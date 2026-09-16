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


def test_parse_desktop_actions():
    s_launch = parse_step_string("launch:gnome-calculator")
    assert s_launch == {"action": "launch", "cmd": "gnome-calculator"}

    s_focus = parse_step_string("focus:OpenCode")
    assert s_focus == {"action": "focus", "window": "OpenCode"}

    s_hotkey = parse_step_string("hotkey:ctrl+shift+p")
    assert s_hotkey == {"action": "key", "combo": "ctrl+shift+p"}

    s_drag = parse_step_string("drag:100,200->450,550")
    assert s_drag == {"action": "drag", "x1": 100, "y1": 200, "x2": 450, "y2": 550}

    s_rclick = parse_step_string("right_click:text=Properties")
    assert s_rclick["action"] == "right_click"
    assert s_rclick["text"] == "Properties"

    s_dclick = parse_step_string("double_click:text=Documents")
    assert s_dclick["action"] == "double_click"
    assert s_dclick["text"] == "Documents"

