"""CLI entry point: run the configured stage chain on one video.

# نقطهٔ ورود خط فرمان: اجرای زنجیرهٔ مراحل پیکربندی‌شده روی یک ویدیو.
# کنترل‌ها: p=مکث، r=ازسرگیری، s=توقف و ذخیره، Ctrl+C=توقف و ذخیره.
# خروجی همیشه MKV است و حتی بعد از توقف قابل پخش است.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading

from .config import load_config
from .controller import Controller, State
from .memory import derive_memory_plan
from .registry import resolve
from .stages.base import FrameSpec
from .video_io import VideoReader, VideoWriter


def _build_stages(cfg: dict, device: str):
    """Instantiate and load each configured stage in order.

    # هر مرحلهٔ پیکربندی‌شده را به‌ترتیب می‌سازد و بارگذاری می‌کند.
    """
    stages = []
    for entry in cfg["stages"]:
        cls = resolve(entry["name"])
        st = cls()
        st.load(device, entry)
        stages.append(st)
    return stages


def _final_spec(stages, src_spec: FrameSpec) -> FrameSpec:
    """Chain plan() over stages to get the final output spec.

    # با زنجیره‌کردن plan() روی مراحل، مشخصات نهایی خروجی به‌دست می‌آید.
    # این اندازه برای ساخت VideoWriter لازم است تا output_scale درست کار کند.
    """
    spec = src_spec
    for st in stages:
        spec = st.plan(spec)
    return spec


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="vidpipe local video enhancement pipeline")
    parser.add_argument("input", help="input video path")
    parser.add_argument("output", help="output video path (MKV recommended)")
    parser.add_argument("--config", default="config.yaml", help="config file path")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    reader = VideoReader(args.input)
    src_spec = FrameSpec(
        width=reader.width, height=reader.height,
        fps=reader.fps, frame_size=reader.frame_size,
    )

    # انتخاب دستگاه و برنامهٔ حافظه (فقط برای لاگ؛ مدل‌ها خودشان بارگذاری می‌شوند).
    plan = derive_memory_plan(cfg, src_spec, src_spec)
    device = plan.device

    # ساخت و بارگذاری مراحل؛ اندازهٔ نهایی خروجی از آخرین مرحله می‌آید.
    stages = _build_stages(cfg, device)
    out_spec = _final_spec(stages, src_spec)

    writer = VideoWriter(
        args.output, out_spec.width, out_spec.height, out_spec.fps,
        cfg["codec"], cfg["crf"], cfg["preset"], args.input,
    )

    controller = Controller()

    # Ctrl+C -> controller.stop() (توقف و ذخیره)، نه kill پردازش.
    def on_sigint(_sig, _frame):
        controller.stop()

    signal.signal(signal.SIGINT, on_sigint)

    # چاپ تغییر حالت فقط یک بار (RUNNING/PAUSED/STOPPING/STOPPED/FINISHED).
    last_state = None

    def on_state_change(state: State) -> None:
        nonlocal last_state
        if state is not last_state:
            print(f"[vidpipe] state: {state.name}")
            last_state = state

    controller.start(reader, writer, cfg["queue_depth"], stages=stages,
                     on_done=on_state_change)

    # رشتهٔ پس‌زمینه برای خواندن دستورهای صفحه‌کلید از stdin.
    # p=مکث، r=ازسرگیری، s=توقف و ذخیره (مثل Ctrl+C).
    def _stdin_loop() -> None:
        for line in sys.stdin:
            cmd = line.strip().lower()
            if cmd == "p":
                try:
                    controller.pause()
                except Exception:  # noqa: BLE001 - اگر در حالت مجاز نباشد نادیده بگیر
                    pass
            elif cmd == "r":
                try:
                    controller.resume()
                except Exception:  # noqa: BLE001
                    pass
            elif cmd == "s":
                controller.stop()
                break

    stdin_thread = threading.Thread(target=_stdin_loop, daemon=True, name="stdin-cmds")
    stdin_thread.start()

    # منتظر رسیدن به حالت نهایی (FINISHED/STOPPED/FAILED) می‌مانیم.
    controller.wait()
    reader.close()
    for st in stages:
        st.release()

    if controller.state == State.FAILED:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
