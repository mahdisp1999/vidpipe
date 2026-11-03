"""Three-thread video pipeline with bounded queues and per-stage timing."""

from __future__ import annotations

import queue
import threading
import time
from typing import Callable, Optional

from .stages.base import Frame, FrameSpec, Stage
from .video_io import VideoReader, VideoWriter

_SENTINEL = None
_TICK = 0.1


class Pipeline:
    def __init__(
        self,
        reader: VideoReader,
        writer: VideoWriter,
        queue_depth: int,
        pause_event: threading.Event,
        stop_event: threading.Event,
        stages: Optional[list[Stage]] = None,
    ) -> None:
        self.reader = reader
        self.writer = writer
        self.q_rp: queue.Queue = queue.Queue(maxsize=queue_depth)
        self.q_pw: queue.Queue = queue.Queue(maxsize=queue_depth)
        self.pause_event = pause_event
        self.stop_event = stop_event
        self.stages = stages or []

        self.processed = 0
        self.frames_read = 0
        self.read_seconds = 0.0
        self.stage_seconds = 0.0
        self.write_seconds = 0.0
        self.stage_timings: dict[str, float] = {}
        self._stats_lock = threading.Lock()
        self._error: Optional[BaseException] = None
        self._error_lock = threading.Lock()
        self._abort = threading.Event()
        self._writer_failed = threading.Event()
        self._threads: list[threading.Thread] = []
        self._source_spec = FrameSpec(
            width=reader.width,
            height=reader.height,
            fps=reader.fps,
            frame_size=reader.frame_size,
        )

    def start(self) -> None:
        self._threads = [
            threading.Thread(target=self._reader_loop, daemon=True, name="vidpipe-reader"),
            threading.Thread(target=self._pipeline_loop, daemon=True, name="vidpipe-pipeline"),
            threading.Thread(target=self._writer_loop, daemon=True, name="vidpipe-writer"),
        ]
        for thread in self._threads:
            thread.start()

    def join(self) -> None:
        for thread in self._threads:
            thread.join()
        if self._error is not None:
            raise self._error

    @property
    def error(self) -> Optional[BaseException]:
        return self._error

    def status_snapshot(self) -> dict:
        """Return a small, thread-safe snapshot for the live CLI reporter."""
        with self._stats_lock:
            values = {
                "frames_read": self.frames_read,
                "processed": self.processed,
                "read_seconds": self.read_seconds,
                "stage_seconds": self.stage_seconds,
                "write_seconds": self.write_seconds,
                "stage_timings": dict(self.stage_timings),
            }
        values.update({
            "total_frames": self.reader.total_frames,
            "exact_frame_count": self.reader.exact_frame_count,
            "input_queue": self.q_rp.qsize(),
            "output_queue": self.q_pw.qsize(),
            "queue_depth": self.q_rp.maxsize,
        })

        tile_stage = next(
            (stage for stage in self.stages if getattr(stage, "is_processing", False)), None
        )
        if tile_stage is None:
            tile_stage = next((stage for stage in self.stages if hasattr(stage, "tiles_total")), None)
        if tile_stage is not None:
            values["tiles_completed"] = tile_stage.tiles_completed
            values["tiles_total"] = tile_stage.tiles_total
            values["tile_size"] = tile_stage.tile_size
            values["tile_batch_size"] = tile_stage.last_batch_size
            values["oom_retries"] = tile_stage.oom_retries
            values["model_name"] = tile_stage.model_name
        return values

    def _fail(self, error: BaseException) -> None:
        with self._error_lock:
            if self._error is None:
                self._error = error
        self._abort.set()

    def _put(self, target: queue.Queue, item, should_halt: Callable[[], bool]) -> bool:
        while not should_halt():
            try:
                target.put(item, timeout=_TICK)
                return True
            except queue.Full:
                continue
        return False

    def _reader_should_halt(self) -> bool:
        return self.stop_event.is_set() or self._abort.is_set()

    def _reader_loop(self) -> None:
        try:
            while not self._reader_should_halt():
                if not self.pause_event.is_set():
                    self.pause_event.wait(timeout=_TICK)
                    continue

                started = time.perf_counter()
                data = self.reader.read_frame()
                elapsed = time.perf_counter() - started
                with self._stats_lock:
                    self.read_seconds += elapsed
                if data is None:
                    break

                with self._stats_lock:
                    self.frames_read += 1
                if not self._put(self.q_rp, data, self._reader_should_halt):
                    break
        except BaseException as error:  # Keep errors from worker threads visible to the main thread.
            self._fail(error)
        finally:
            self._put(self.q_rp, _SENTINEL, self._reader_should_halt)

    def _pipeline_loop(self) -> None:
        try:
            while not self._reader_should_halt():
                if not self.pause_event.is_set():
                    self.pause_event.wait(timeout=_TICK)
                    continue
                try:
                    data = self.q_rp.get(timeout=_TICK)
                except queue.Empty:
                    continue
                if data is _SENTINEL:
                    break
                if not self._emit(self._run_stages(data)):
                    return

            self._emit(self._flush_stages())
        except BaseException as error:
            self._fail(error)
        finally:
            self._put(self.q_pw, _SENTINEL, self._writer_failed.is_set)

    def _run_stages(self, data: bytes) -> list[Frame]:
        frames = [Frame(data=data, spec=self._source_spec)]
        for stage_index, stage in enumerate(self.stages, start=1):
            next_frames: list[Frame] = []
            for frame in frames:
                started = time.perf_counter()
                try:
                    next_frames.extend(stage.push(frame))
                finally:
                    elapsed = time.perf_counter() - started
                    label = getattr(stage, "output_label", stage.__class__.__name__)
                    name = f"{stage_index}:{label}"
                    with self._stats_lock:
                        self.stage_seconds += elapsed
                        self.stage_timings[name] = self.stage_timings.get(name, 0.0) + elapsed
            frames = next_frames
        return frames

    def _flush_stages(self) -> list[Frame]:
        frames: list[Frame] = []
        for stage_index, stage in enumerate(self.stages, start=1):
            next_frames: list[Frame] = []
            for frame in frames:
                next_frames.extend(stage.push(frame))
            next_frames.extend(stage.flush())
            frames = next_frames
        return frames

    def _emit(self, frames: list[Frame]) -> bool:
        for frame in frames:
            if not self._put(self.q_pw, frame.data, self._abort.is_set):
                return False
        return True

    def _writer_loop(self) -> None:
        try:
            while True:
                data = self.q_pw.get()
                if data is _SENTINEL:
                    break
                started = time.perf_counter()
                self.writer.write_frame(data)
                elapsed = time.perf_counter() - started
                with self._stats_lock:
                    self.write_seconds += elapsed
                    self.processed += 1
        except BaseException as error:
            self._writer_failed.set()
            self._fail(error)
        finally:
            try:
                self.writer.close()
            except BaseException as error:
                self._writer_failed.set()
                self._fail(error)
