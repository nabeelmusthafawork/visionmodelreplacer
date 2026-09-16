"""Tests for visiond background daemon IPC and request handling."""

import json
import socket
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from screendump.daemon import VisionDaemon, get_socket_path, send_daemon_request


def test_daemon_ping_and_lifecycle():
    with tempfile.TemporaryDirectory() as tmp:
        sock_path = Path(tmp) / "test_visiond.sock"
        daemon = VisionDaemon(sock_path)

        # Thread to run daemon
        th = threading.Thread(target=daemon.start, daemon=True)
        th.start()
        time.sleep(0.1)

        assert sock_path.exists()
        assert VisionDaemon.ping(sock_path)

        # Send custom ping
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.connect(str(sock_path))
            s.sendall(json.dumps({"cmd": "ping"}).encode("utf-8") + b"\n")
            res = json.loads(s.recv(1024).decode("utf-8"))
            assert res.get("ok") is True

        daemon.stop()
        th.join(timeout=1.0)
        assert not sock_path.exists()


def test_daemon_handle_request():
    with tempfile.TemporaryDirectory() as tmp:
        sock_path = Path(tmp) / "test_visiond.sock"
        daemon = VisionDaemon(sock_path)

        # 1. ping
        res = daemon.handle_request({"cmd": "ping"})
        assert res["ok"] is True

        # 2. invalid cmd
        res_err = daemon.handle_request({"cmd": "unknown_cmd"})
        assert res_err["ok"] is False
