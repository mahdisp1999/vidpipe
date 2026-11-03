import os
import sys
import threading
import time

import pytest

from vidpipe.controls import KeyboardControls


@pytest.mark.skipif(os.name == "nt", reason="PTY-based keyboard test is POSIX-only")
def test_single_key_controls_do_not_require_enter():
    import pty

    master_fd, slave_fd = pty.openpty()
    stream = os.fdopen(slave_fd, "r", buffering=1)
    received = []
    events = {key: threading.Event() for key in "prs"}

    def callback(key):
        received.append(key)
        events[key].set()

    listener = KeyboardControls(
        on_pause=lambda: callback("p"),
        on_resume=lambda: callback("r"),
        on_stop=lambda: callback("s"),
        stream=stream,
    )
    try:
        assert listener.start()
        for key in "prs":
            os.write(master_fd, key.encode())
            assert events[key].wait(1.0), f"key {key!r} was not received"
        assert received == ["p", "r", "s"]
    finally:
        listener.close()
        stream.close()
        os.close(master_fd)
