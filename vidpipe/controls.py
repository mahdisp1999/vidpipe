"""Read P/R/S/Q directly from the controlling terminal without pressing Enter.

Only Python's standard library is used. On POSIX, /dev/tty is preferred when
stdin was redirected, so keyboard controls still work in a normal interactive
shell. The terminal mode is restored on every exit path.
"""
from __future__ import annotations

import os
import select
import sys
import threading
from typing import Callable


class KeyboardControls:
    """Listen for single-key controls while the pipeline runs."""

    def __init__(self, on_pause: Callable[[], None], on_resume: Callable[[], None],
                 on_stop: Callable[[], None], stream=None) -> None:
        self.stream = stream
        self.on_pause = on_pause
        self.on_resume = on_resume
        self.on_stop = on_stop
        self._finished = threading.Event()
        self._thread: threading.Thread | None = None
        self._terminal = None
        self._fd: int | None = None
        self._old_terminal_settings = None
        self._termios = None
        self._msvcrt = None
        self.available = False

    def start(self) -> bool:
        """Open an interactive terminal and enable immediate character input."""
        if os.name == "nt":
            terminal = self.stream or sys.stdin
            if not terminal.isatty():
                return False
            try:
                import msvcrt
                self._msvcrt = msvcrt
            except ImportError:
                return False
            self._terminal = terminal
        else:
            try:
                import termios
                import tty
                self._termios = termios
                terminal = self.stream
                if terminal is None or not terminal.isatty():
                    # stdin may be redirected while the shell's controlling
                    # terminal is still available through /dev/tty.
                    terminal = open("/dev/tty", "r", buffering=1)
                self._terminal = terminal
                self._fd = terminal.fileno()
                self._old_terminal_settings = termios.tcgetattr(self._fd)
                tty.setcbreak(self._fd)
                # cbreak keeps ISIG enabled, so Ctrl+C still raises SIGINT.
                settings = termios.tcgetattr(self._fd)
                settings[3] &= ~termios.ECHO
                termios.tcsetattr(self._fd, termios.TCSADRAIN, settings)
            except (ImportError, OSError, ValueError):
                self._restore_terminal()
                return False

        self._thread = threading.Thread(
            target=self._listen, daemon=True, name="vidpipe-keyboard-controls"
        )
        self._thread.start()
        self.available = True
        return True

    def _listen(self) -> None:
        try:
            while not self._finished.is_set():
                key = self._read_key()
                if not key:
                    continue
                key = key.lower()
                if key == "p":
                    self._safe_call(self.on_pause)
                elif key == "r":
                    self._safe_call(self.on_resume)
                elif key in ("s", "q"):
                    self._safe_call(self.on_stop)
                    break
        finally:
            self._restore_terminal()

    def _read_key(self) -> str | None:
        if os.name == "nt":
            if self._msvcrt.kbhit():
                return self._msvcrt.getwch()
            self._finished.wait(0.05)
            return None
        try:
            ready, _, _ = select.select([self._fd], [], [], 0.1)
            if ready:
                value = os.read(self._fd, 1)
                if not value:
                    self._finished.set()
                    return None
                return value.decode(errors="ignore")
        except (OSError, ValueError):
            self._finished.set()
        return None

    @staticmethod
    def _safe_call(callback: Callable[[], None]) -> None:
        # Repeated or invalid keys are harmless and must not kill the listener.
        try:
            callback()
        except Exception:
            pass

    def _restore_terminal(self) -> None:
        if self._old_terminal_settings is not None and self._fd is not None:
            try:
                self._termios.tcsetattr(
                    self._fd, self._termios.TCSADRAIN, self._old_terminal_settings
                )
            except (OSError, ValueError):
                pass
            self._old_terminal_settings = None
        if self._terminal is not None and self._terminal is not self.stream:
            try:
                self._terminal.close()
            except OSError:
                pass
            self._terminal = None

    def close(self) -> None:
        self._finished.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=0.5)
        self._restore_terminal()
        self.available = False
