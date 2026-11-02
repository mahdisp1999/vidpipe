"""Run controller: state machine with pause/resume/stop using threading.Event.

# کنترل‌کنندهٔ اجرا: ماشین حالت IDLE/RUNNING/PAUSED/STOPPING/STOPPED/FINISHED/FAILED.
# pause با Event (نه kill پردازش)؛ stop = اتمام فریم جاری + بستن encoder stdin.
"""

from __future__ import annotations

import threading
from enum import Enum, auto
from typing import Callable, Optional

from .pipeline import Pipeline
from .video_io import VideoReader, VideoWriter


class State(Enum):
    IDLE = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPING = auto()
    STOPPED = auto()
    FINISHED = auto()
    FAILED = auto()


class StateError(Exception):
    """Raised on invalid state transition."""


class Controller:
    def __init__(self) -> None:
        self.state = State.IDLE
        self._lock = threading.Lock()
        self.pause_event = threading.Event()
        self.stop_event = threading.Event()
        self.pipeline: Optional[Pipeline] = None
        self.reader: Optional[VideoReader] = None
        self.writer: Optional[VideoWriter] = None
        self._thread: Optional[threading.Thread] = None
        self._config: dict = {}
        self._on_done: Optional[Callable[[State], None]] = None

    def _set(self, new: State) -> None:
        with self._lock:
            self.state = new

    def _require(self, *allowed: State) -> None:
        with self._lock:
            if self.state not in allowed:
                raise StateError(f"invalid transition from {self.state.name}")

    def start(self, reader: VideoReader, writer: VideoWriter, queue_depth: int,
              stages: Optional[list] = None,
              on_done: Optional[Callable[[State], None]] = None) -> None:
        # شروع: فقط از IDLE مجاز است.
        self._require(State.IDLE)
        self.reader = reader
        self.writer = writer
        self.pause_event.set()
        self.stop_event.clear()
        self.pipeline = Pipeline(reader, writer, queue_depth, self.pause_event,
                                 self.stop_event, stages=stages)
        self._on_done = on_done
        self._set(State.RUNNING)
        self._thread = threading.Thread(target=self._run, daemon=True, name="controller-run")
        self._thread.start()

    def _run(self) -> None:
        # اجرای pipeline در رشتهٔ جدا؛ در پایان state را FINISHED/STOPPED/FAILED می‌کند.
        try:
            self.pipeline.start()
            self.pipeline.join()
            with self._lock:
                if self.stop_event.is_set():
                    self.state = State.STOPPED
                else:
                    self.state = State.FINISHED
        except BaseException:  # noqa: BLE001
            self._set(State.FAILED)
        finally:
            if self._on_done is not None:
                try:
                    self._on_done(self.state)
                except Exception:  # noqa: BLE001
                    pass

    def pause(self) -> None:
        # مکث: فقط در RUNNING. pause_event را پاک می‌کند (پردازش نمی‌کشد).
        self._require(State.RUNNING)
        self.pause_event.clear()
        self._set(State.PAUSED)

    def resume(self) -> None:
        # ازسرگیری: فقط در PAUSED.
        self._require(State.PAUSED)
        self.pause_event.set()
        self._set(State.RUNNING)

    def stop(self) -> None:
        # توقف: از RUNNING یا PAUSED مجاز. stop_event را set می‌کند و pause را آزاد می‌کند.
        self._require(State.RUNNING, State.PAUSED)
        # اگر مکث بود، اول resume تا reader بتواند stop را ببیند
        if self.state == State.PAUSED:
            self.pause_event.set()
        with self._lock:
            self.state = State.STOPPING
        self.stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=120)

    def wait(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    @property
    def processed_frames(self) -> int:
        return self.pipeline.processed if self.pipeline else 0
