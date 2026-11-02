"""Reader/pipeline/writer threads with two bounded queues.

# لولهٔ پردازش: رشتهٔ خواننده -> صف کران‌دار reader->pipeline -> رشتهٔ pipeline (مراحل) ->
# صف کران‌دار pipeline->writer -> رشتهٔ نویسنده. حافظه محدود و مستقل از طول ویدیو.
"""

from __future__ import annotations

import queue
import threading
from typing import Callable, List, Optional

from .stages.base import Frame, FrameSpec, Stage
from .video_io import VideoReader, VideoWriter

_SENTINEL = None  # نشانهٔ پایان جریان


class Pipeline:
    def __init__(
        self,
        reader: VideoReader,
        writer: VideoWriter,
        queue_depth: int,
        pause_event: threading.Event,
        stop_event: threading.Event,
        stages: Optional[List[Stage]] = None,
        on_frame: Optional[Callable[[bytes], bytes]] = None,
    ) -> None:
        self.reader = reader
        self.writer = writer
        self.q_rp = queue.Queue(maxsize=queue_depth)  # reader -> pipeline
        self.q_pw = queue.Queue(maxsize=queue_depth)  # pipeline -> writer
        self.pause_event = pause_event
        self.stop_event = stop_event
        # مراحل به‌ترتیب اجرا می‌شوند؛ پیش‌فرض یک مرحلهٔ هویت.
        self.stages: List[Stage] = stages if stages is not None else []
        self.on_frame = on_frame  # هویت به‌طور پیش‌فرض
        self.processed = 0
        self._error: Optional[BaseException] = None
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        # شروع سه رشتهٔ خواننده، pipeline، نویسنده.
        self._threads = [
            threading.Thread(target=self._reader, daemon=True, name="reader"),
            threading.Thread(target=self._pipeline, daemon=True, name="pipeline"),
            threading.Thread(target=self._writer, daemon=True, name="writer"),
        ]
        for t in self._threads:
            t.start()

    def join(self) -> None:
        for t in self._threads:
            t.join()
        if self._error is not None:
            raise self._error

    @property
    def error(self) -> Optional[BaseException]:
        return self._error

    def _reader(self) -> None:
        # خواننده: تا stop یا پایان ورودی، فریم می‌خواند و در صف محدود می‌گذارد.
        # pause: منتظر pause_event می‌ماند (بدون kill پردازش).
        try:
            while not self.stop_event.is_set():
                if not self.pause_event.is_set():
                    # مکث: تا resume یا stop منتظر می‌ماند؛ حافظه ثابت می‌ماند.
                    self.pause_event.wait(timeout=0.1)
                    continue
                frame = self.reader.read_frame()
                if frame is None:
                    break
                # backpressure: اگر صف پر باشد، منتظر می‌ماند (حافظه محدود).
                # stop-aware: در صورت stop، منتظر نمی‌ماند.
                while not self.stop_event.is_set():
                    try:
                        self.q_rp.put(frame, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                else:
                    # stop درخواست شده: جریان را قطع می‌کند.
                    self.q_rp.put(_SENTINEL)
                    return
            self.q_rp.put(_SENTINEL)
        except BaseException as e:  # noqa: BLE001
            self._error = e
            self.q_rp.put(_SENTINEL)

    def _pipeline(self) -> None:
        # مراحل را به‌ترتیب روی هر فریم اجرا می‌کند؛ خروجی هر مرحله ورودی بعدی است.
        # فریم‌های خام (bytes) به Frame با مشخصات خواننده بسته‌بندی می‌شوند.
        spec = FrameSpec(
            width=self.reader.width,
            height=self.reader.height,
            fps=self.reader.fps,
            frame_size=self.reader.frame_size,
        )
        try:
            while True:
                frame = self.q_rp.get()
                if frame is _SENTINEL:
                    # پایان جریان: flush همهٔ مراحل و ارسال به صف بعدی.
                    for st in self.stages:
                        for out in st.flush():
                            self.q_pw.put(out.data)
                    self.q_pw.put(_SENTINEL)
                    break
                if self.stop_event.is_set():
                    # stop درخواست شده: جریان را قطع می‌کند.
                    self.q_pw.put(_SENTINEL)
                    break
                if self.stages:
                    # اجرای زنجیرهٔ مراحل روی فریم ورودی.
                    frames: List[Frame] = [Frame(data=frame, spec=spec)]
                    for st in self.stages:
                        nxt: List[Frame] = []
                        for f in frames:
                            nxt.extend(st.push(f))
                        frames = nxt
                    for out in frames:
                        self.q_pw.put(out.data)
                else:
                    out = frame if self.on_frame is None else self.on_frame(frame)
                    self.q_pw.put(out)
        except BaseException as e:  # noqa: BLE001
            self._error = e
            self.q_pw.put(_SENTINEL)

    def _writer(self) -> None:
        # نویسنده: فریم‌ها را به encoder می‌نویسد؛ در پایان stdin را می‌بندد.
        try:
            while True:
                frame = self.q_pw.get()
                if frame is _SENTINEL:
                    break
                self.writer.write_frame(frame)
                self.processed += 1
        except BaseException as e:  # noqa: BLE001
            self._error = e
        finally:
            # بستن graceful encoder: stdin بسته و منتظر خروج ffmpeg.
            self.writer.close()
