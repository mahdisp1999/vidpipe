"""Start, pause, resume and stop a Pipeline without killing worker threads."""

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


class StateError(RuntimeError):
    pass


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
        self._on_done: Optional[Callable[[State], None]] = None
        self._error: Optional[BaseException] = None

    def _set(self, state: State) -> None:
        with self._lock:
            self.state = state

    def _require(self, *allowed: State) -> None:
        with self._lock:
            if self.state not in allowed:
                raise StateError(f"invalid state transition from {self.state.name}")

    def start(
        self,
        reader: VideoReader,
        writer: VideoWriter,
        queue_depth: int,
        stages: Optional[list] = None,
        on_done: Optional[Callable[[State], None]] = None,
    ) -> None:
        self._require(State.IDLE)
        self.reader = reader
        self.writer = writer
        self.pause_event.set()
        self.stop_event.clear()
        self.pipeline = Pipeline(
            reader, writer, queue_depth, self.pause_event, self.stop_event, stages=stages
        )
        self._on_done = on_done
        self._set(State.RUNNING)
        self._thread = threading.Thread(target=self._run, daemon=True, name="vidpipe-controller")
        self._thread.start()

    def _run(self) -> None:
        try:
            self.pipeline.start()
            self.pipeline.join()
            self._set(State.STOPPED if self.stop_event.is_set() else State.FINISHED)
        except BaseException as error:
            self._error = error
            self._set(State.FAILED)
        finally:
            if self._on_done is not None:
                try:
                    self._on_done(self.state)
                except Exception:
                    pass

    def pause(self) -> None:
        self._require(State.RUNNING)
        self.pause_event.clear()
        self._set(State.PAUSED)

    def resume(self) -> None:
        self._require(State.PAUSED)
        self.pause_event.set()
        self._set(State.RUNNING)

    def stop(self) -> None:
        self._require(State.RUNNING, State.PAUSED)
        if self.state == State.PAUSED:
            self.pause_event.set()
        self._set(State.STOPPING)
        self.stop_event.set()

    def wait(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    @property
    def error(self) -> Optional[BaseException]:
        if self._error is not None:
            return self._error
        return self.pipeline.error if self.pipeline else None

    @property
    def processed_frames(self) -> int:
        return self.pipeline.processed if self.pipeline else 0
