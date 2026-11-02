"""نقطهٔ ورود خط فرمان: اجرای زنجیرهٔ مراحل روی یک ویدیو.

کنترل‌ها (در همان ترمینال، بعد از زدن Enter):
    p = مکث
    r = ادامه
    s = توقف و ذخیره
Ctrl+C هم مثل s عمل می‌کند. خروجی همیشه MKV است و حتی بعد از توقف پخش می‌شود.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading

from .config import load_config
from .controller import Controller, State, StateError
from .memory import derive_memory_plan
from .registry import resolve
from .stages.base import FrameSpec
from .video_io import VideoReader, VideoWriter


def _build_stages(cfg: dict, device: str):
    """هر مرحلهٔ پیکربندی‌شده را به ترتیب می‌سازد و لود می‌کند."""
    stages = []
    for entry in cfg["stages"]:
        cls = resolve(entry["name"])
        st = cls()
        st.load(device, entry)
        stages.append(st)
    return stages


def _final_spec(stages, src_spec: FrameSpec) -> FrameSpec:
    """مشخصات نهایی خروجی را با زنجیره‌کردن plan() روی مراحل به دست می‌آورد.

    # این اندازه برای ساخت VideoWriter لازم است.
    """
    spec = src_spec
    for st in stages:
        spec = st.plan(spec)
    return spec


def _try(action) -> None:
    """یک دستور کنترلی را اجرا می‌کند.

    # اگر حالت فعلی اجازهٔ آن دستور را ندهد (مثلاً مکث وقتی در حال توقف است)،
    # خطا را بی‌صدا رد می‌کنیم تا برنامه کرش نکند.
    """
    try:
        action()
    except StateError:
        pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="vidpipe local video enhancement pipeline")
    parser.add_argument("input", help="مسیر ویدیوی ورودی")
    parser.add_argument("output", help="مسیر ویدیوی خروجی (MKV پیشنهاد می‌شود)")
    parser.add_argument("--config", default="config.yaml", help="مسیر فایل پیکربندی")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    reader = VideoReader(args.input)
    src_spec = FrameSpec(
        width=reader.width, height=reader.height,
        fps=reader.fps, frame_size=reader.frame_size,
    )

    # انتخاب دستگاه و لاگ برنامهٔ حافظه (مدل‌ها خودشان لود می‌شوند).
    plan = derive_memory_plan(cfg, src_spec, src_spec)
    stages = _build_stages(cfg, plan.device)
    out_spec = _final_spec(stages, src_spec)

    writer = VideoWriter(
        args.output, out_spec.width, out_spec.height, out_spec.fps,
        cfg["codec"], cfg["crf"], cfg["preset"], args.input,
    )
    controller = Controller()

    # Ctrl+C = توقف و ذخیره، نه kill پردازش.
    def on_sigint(_sig, _frame):
        _try(controller.stop)

    signal.signal(signal.SIGINT, on_sigint)

    def on_done(state: State) -> None:
        print(f"[vidpipe] state: {state.name}")

    controller.start(reader, writer, cfg["queue_depth"], stages=stages, on_done=on_done)

    # رشتهٔ پس‌زمینه برای خواندن دستورها از ترمینال.
    def stdin_loop() -> None:
        for line in sys.stdin:
            cmd = line.strip().lower()
            if cmd == "p":
                _try(controller.pause)
            elif cmd == "r":
                _try(controller.resume)
            elif cmd == "s":
                _try(controller.stop)
                break
            else:
                continue
            print(f"[vidpipe] state: {controller.state.name}")

    threading.Thread(target=stdin_loop, daemon=True, name="stdin-cmds").start()

    # منتظر حالت نهایی (FINISHED / STOPPED / FAILED) می‌مانیم.
    controller.wait()
    reader.close()
    for st in stages:
        st.release()

    return 1 if controller.state == State.FAILED else 0


if __name__ == "__main__":
    sys.exit(main())