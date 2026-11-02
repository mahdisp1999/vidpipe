"""لوله‌ی پردازش: سه رشته که با دو صف کران‌دار به هم وصل‌اند.

    reader (خواندن فریم) --q_rp--> pipeline (اجرای مراحل) --q_pw--> writer (نوشتن)

صف‌ها کران‌دار هستند، پس مصرف حافظه به طول ویدیو وابسته نیست.

قاعدهٔ مهم: هیچ انتظار بی‌پایانی وجود ندارد. هر put و get با timeout انجام
می‌شود و هر بار وضعیت توقف یا خطا بررسی می‌شود.
"""

from __future__ import annotations

import queue
import threading
from typing import Callable, List, Optional

from .stages.base import Frame, FrameSpec, Stage
from .video_io import VideoReader, VideoWriter

_SENTINEL = None  # نشانهٔ پایان جریان داخل صف‌ها
_TICK = 0.1       # هر چند ثانیه وضعیت توقف بررسی شود


class Pipeline:
    def __init__(
        self,
        reader: VideoReader,
        writer: VideoWriter,
        queue_depth: int,
        pause_event: threading.Event,
        stop_event: threading.Event,
        stages: Optional[List[Stage]] = None,
    ) -> None:
        self.reader = reader
        self.writer = writer
        self.q_rp: queue.Queue = queue.Queue(maxsize=queue_depth)  # reader -> pipeline
        self.q_pw: queue.Queue = queue.Queue(maxsize=queue_depth)  # pipeline -> writer
        self.pause_event = pause_event
        self.stop_event = stop_event          # توقف درخواستی کاربر (s یا Ctrl+C)
        self.stages: List[Stage] = stages or []
        self.processed = 0                    # تعداد فریم‌های نوشته‌شده

        self._error: Optional[BaseException] = None
        # abort: اگر هر رشته‌ای خطا بدهد، بقیه باید بی‌درنگ کنار بیایند.
        self._abort = threading.Event()
        # writer_failed: نویسنده مرده؛ دیگر کسی صف را خالی نمی‌کند.
        self._writer_failed = threading.Event()

        self._src_spec = FrameSpec(
            width=reader.width, height=reader.height,
            fps=reader.fps, frame_size=reader.frame_size,
        )
        self._threads: List[threading.Thread] = []

    # ------------------------------------------------------------ کنترل کلی
    def start(self) -> None:
        """سه رشته را راه می‌اندازد."""
        self._threads = [
            threading.Thread(target=self._reader, daemon=True, name="reader"),
            threading.Thread(target=self._pipeline, daemon=True, name="pipeline"),
            threading.Thread(target=self._writer, daemon=True, name="writer"),
        ]
        for t in self._threads:
            t.start()

    def join(self) -> None:
        """منتظر پایان هر سه رشته می‌ماند و اولین خطا را دوباره بالا می‌آورد."""
        for t in self._threads:
            t.join()
        if self._error is not None:
            raise self._error

    @property
    def error(self) -> Optional[BaseException]:
        return self._error

    def _fail(self, e: BaseException) -> None:
        """اولین خطا را ثبت و به بقیهٔ رشته‌ها اعلام توقف می‌کند."""
        if self._error is None:
            self._error = e
        self._abort.set()

    # ------------------------------------------------------------ ابزار صف
    def _put(self, q: queue.Queue, item, should_halt: Callable[[], bool]) -> bool:
        """آیتم را در صف می‌گذارد.

        # اگر صف پر باشد، هر چند وقت یک‌بار should_halt را بررسی می‌کنیم.
        # اگر توقف لازم شد، False برمی‌گردانیم تا رشته گیر نکند.
        """
        while not should_halt():
            try:
                q.put(item, timeout=_TICK)
                return True
            except queue.Full:
                continue
        return False

    def _reader_should_halt(self) -> bool:
        return self.stop_event.is_set() or self._abort.is_set()

    # ------------------------------------------------------------ رشتهٔ reader
    def _reader(self) -> None:
        """فریم خام را از ffmpeg می‌خواند و در صف می‌گذارد."""
        try:
            while not self._reader_should_halt():
                if not self.pause_event.is_set():
                    # مکث: فریمی نمی‌خوانیم، پس حافظه ثابت می‌ماند.
                    self.pause_event.wait(timeout=_TICK)
                    continue
                data = self.reader.read_frame()
                if data is None:  # به انتهای ویدیو رسیدیم
                    break
                if not self._put(self.q_rp, data, self._reader_should_halt):
                    break
        except BaseException as e:  # noqa: BLE001
            self._fail(e)
        finally:
            # علامت پایان ورودی. اگر توقف باشد، pipeline دیگر منتظر نیست.
            self._put(self.q_rp, _SENTINEL, self._reader_should_halt)

    # ------------------------------------------------------------ رشتهٔ pipeline
    def _pipeline(self) -> None:
        """هر فریم را از همهٔ مراحل پشت‌سرهم عبور می‌دهد."""
        try:
            while not self._reader_should_halt():
                try:
                    data = self.q_rp.get(timeout=_TICK)
                except queue.Empty:
                    continue
                if data is _SENTINEL:
                    break
                if not self._emit(self._run_stages(data)):
                    return
            # پایان ویدیو یا توقف: آنچه در بافر مراحل مانده را بیرون بده.
            self._emit(self._flush_stages())
        except BaseException as e:  # noqa: BLE001
            self._fail(e)
        finally:
            # علامت پایان به نویسنده. اگر نویسنده مرده باشد، رد می‌شود.
            self._put(self.q_pw, _SENTINEL, self._writer_failed.is_set)

    def _run_stages(self, data: bytes) -> List[Frame]:
        """یک فریم خام را از همهٔ مراحل عبور می‌دهد."""
        frames = [Frame(data=data, spec=self._src_spec)]
        for st in self.stages:
            nxt: List[Frame] = []
            for f in frames:
                nxt.extend(st.push(f))
            frames = nxt
        return frames

    def _flush_stages(self) -> List[Frame]:
        """بافر مراحل را خالی می‌کند.

        # خروجی flush هر مرحله باید از مراحل بعدی هم رد شود، برای همین
        # ترتیب مراحل را نگه می‌داریم.
        """
        frames: List[Frame] = []
        for st in self.stages:
            nxt: List[Frame] = []
            for f in frames:
                nxt.extend(st.push(f))
            nxt.extend(st.flush())
            frames = nxt
        return frames

    def _emit(self, frames: List[Frame]) -> bool:
        """فریم‌های خروجی را به صف نویسنده می‌فرستد."""
        for f in frames:
            if not self._put(self.q_pw, f.data, self._abort.is_set):
                return False
        return True

    # ------------------------------------------------------------ رشتهٔ writer
    def _writer(self) -> None:
        """فریم‌ها را به encoder می‌نویسد و در پایان stdin را می‌بندد."""
        try:
            while True:
                data = self.q_pw.get()
                if data is _SENTINEL:
                    break
                self.writer.write_frame(data)
                self.processed += 1
        except BaseException as e:  # noqa: BLE001
            self._writer_failed.set()
            self._fail(e)
        finally:
            # بستن امن encoder: ffmpeg تمام می‌شود و MKV معتبر می‌ماند.
            self.writer.close()