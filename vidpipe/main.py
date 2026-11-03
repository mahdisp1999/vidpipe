"""vidpipe command-line interface."""

from __future__ import annotations

import argparse
import logging
import signal
import subprocess
import sys
from pathlib import Path

from . import __version__
from .config import load_config
from .controller import Controller, State, StateError
from .controls import KeyboardControls
from .memory import configure_vram_limit, derive_memory_plan, gpu_memory_stats, select_device
from .naming import make_output_path
from .progress import ProgressReporter
from .registry import resolve
from .stages.base import FrameSpec
from .video_io import VideoReader, VideoWriter

log = logging.getLogger("vidpipe")

_FORMAT_EXTENSIONS = {
    "matroska": ".mkv",
    "mkv": ".mkv",
    "mp4": ".mp4",
    "mov": ".mov",
    "webm": ".webm",
    "avi": ".avi",
    "flv": ".flv",
    "mpeg": ".mpeg",
    "mpegts": ".ts",
}


def _parse_option(value: str) -> tuple[str, str]:
    """Parse a repeated FFmpeg option written as KEY=VALUE."""
    if "=" not in value:
        raise argparse.ArgumentTypeError("option must use KEY=VALUE, for example cq=19")
    key, item = value.split("=", 1)
    key = key.strip().lstrip("-")
    item = item.strip()
    if not key or not item or any(char.isspace() for char in key):
        raise argparse.ArgumentTypeError("option key and value must not be empty")
    return key, item


def _build_parser() -> argparse.ArgumentParser:
    examples = """Examples:
  vidpipe input.mp4
  vidpipe input.mkv --output-dir ./enhanced --codec libx265 --crf 20 --preset slow
  vidpipe input.mp4 --codec h264_nvenc --video-option rc=vbr --video-option cq=19
  vidpipe input.mp4 --tile-size 160 --batch-size 2 --device gpu
  vidpipe input.mp4 --output ./output/final.mp4 --format mp4 --audio-codec aac
  vidpipe --list-encoders

During processing, press P to pause, R to resume, S or Q to stop.
Keys work immediately; Enter is not required. Ctrl+C requests a clean stop.
"""
    parser = argparse.ArgumentParser(
        prog="vidpipe",
        description=(
            "Enhance video frames with a configurable chain of image models, "
            "then encode the result with FFmpeg."
        ),
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", nargs="?", help="input video file (FFmpeg detects its format by default)")
    parser.add_argument("-o", "--output", help="exact output file path; generated model-based name is used when omitted")
    parser.add_argument("--output-dir", default="output", help="directory for generated output names (default: output)")
    default_config = Path.cwd() / "config.yaml"
    if not default_config.is_file():
        default_config = Path(__file__).resolve().parents[1] / "config.yaml"
    parser.add_argument("--config", default=str(default_config), help="YAML configuration file")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    encode = parser.add_argument_group("video encoding")
    encode.add_argument("--codec", "--video-codec", dest="codec", help="FFmpeg video encoder, e.g. libx264, libx265, h264_nvenc")
    crf_group = encode.add_mutually_exclusive_group()
    crf_group.add_argument("--crf", type=int, help="quality value for encoders that support CRF (common range: 0-51)")
    crf_group.add_argument("--no-crf", dest="disable_crf", action="store_true", help="do not pass -crf; useful with encoders such as NVENC")
    encode.add_argument("--preset", help="encoder speed/quality preset; available values depend on the encoder")
    encode.add_argument("--pixel-format", help="output pixel format, e.g. yuv420p, yuv444p, p010le")
    encode.add_argument("--audio-codec", help="audio encoder (copy, aac, opus, ...); use none to omit audio")
    encode.add_argument("--format", "--output-format", dest="output_format", help="FFmpeg output container/muxer, e.g. matroska, mp4, webm")
    encode.add_argument("--video-option", action="append", type=_parse_option, default=[], metavar="KEY=VALUE", help="extra FFmpeg video encoder option; repeatable")

    decode = parser.add_argument_group("video decoding")
    decode.add_argument("--input-format", help="force an FFmpeg input demuxer; normally detected automatically")
    decode.add_argument("--decoder", help="force a video decoder; normally selected automatically by FFmpeg")

    processing = parser.add_argument_group("processing and memory")
    processing.add_argument("--device", choices=("auto", "gpu", "cpu"), help="inference device (overrides config.yaml)")
    processing.add_argument("--tile-size", type=int, help="model tile width/height in pixels; minimum 64")
    processing.add_argument("--tile-pad", type=int, help="neighbor padding around every tile")
    processing.add_argument("--batch-size", type=int, help="number of same-sized tiles sent to the model together")
    processing.add_argument("--queue-depth", type=int, help="maximum number of prefetched frames in each queue")
    half_group = processing.add_mutually_exclusive_group()
    half_group.add_argument("--half", dest="half", action="store_true", default=None, help="use FP16 on a supported CUDA model")
    half_group.add_argument("--no-half", dest="half", action="store_false", help="force FP32 model input/weights")

    display = parser.add_argument_group("progress display")
    display.add_argument("--progress-interval", type=float, default=1.0, help="seconds between progress updates (default: 1)")
    display.add_argument("--no-progress", action="store_true", help="disable the live progress line")
    display.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    display.add_argument("--overwrite", action="store_true", help="allow replacing an existing --output file")

    listing = parser.add_argument_group("FFmpeg capability lists")
    listing.add_argument("--list-encoders", action="store_true", help="show video/audio encoders available in this FFmpeg build")
    listing.add_argument("--list-decoders", action="store_true", help="show decoders available in this FFmpeg build")
    listing.add_argument("--list-formats", action="store_true", help="show input demuxers and output muxers available in this FFmpeg build")
    listing.add_argument("--list-pixel-formats", action="store_true", help="show pixel formats available in this FFmpeg build")
    return parser


def _list_ffmpeg(option: str) -> int:
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", option],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        print("vidpipe: ffmpeg was not found in PATH", file=sys.stderr)
        return 127
    print(result.stdout, end="")
    return 0 if result.returncode == 0 else result.returncode


def _apply_cli_overrides(config: dict, args: argparse.Namespace) -> None:
    if args.disable_crf:
        config["crf"] = None
    if args.crf is not None and not 0 <= args.crf <= 63:
        raise ValueError("--crf must be between 0 and 63")
    for argument, key in (("codec", "codec"), ("crf", "crf"), ("preset", "preset"),
                          ("pixel_format", "pixel_format"), ("audio_codec", "audio_codec")):
        value = getattr(args, argument, None)
        if value is not None:
            config[key] = value

    if args.device is not None:
        config["device"] = args.device
    if args.queue_depth is not None:
        if args.queue_depth < 1:
            raise ValueError("--queue-depth must be at least 1")
        config["queue_depth"] = args.queue_depth
    if args.batch_size is not None:
        if args.batch_size < 1:
            raise ValueError("--batch-size must be at least 1")
        config["memory"]["batch_size"] = args.batch_size
        config["memory"]["batch_size_max"] = max(config["memory"]["batch_size_max"], args.batch_size)
    if args.tile_size is not None and args.tile_size < 64:
        raise ValueError("--tile-size must be at least 64")
    if args.tile_pad is not None and args.tile_pad < 0:
        raise ValueError("--tile-pad cannot be negative")
    if args.progress_interval <= 0:
        raise ValueError("--progress-interval must be greater than zero")


def _build_stages(config: dict, device: str, args: argparse.Namespace):
    """Load stages and pass shared batch defaults to model stages."""
    stages = []
    for entry in config["stages"]:
        stage_class = resolve(entry["name"])
        params = dict(entry)
        if entry["name"] == "model":
            params.setdefault("tile_size", config["memory"]["tile_size"] or 192)
            params.setdefault("batch_size", config["memory"]["batch_size"])
            params.setdefault("batch_size_max", config["memory"]["batch_size_max"])
            if args.tile_size is not None:
                params["tile_size"] = args.tile_size
            if args.tile_pad is not None:
                params["tile_pad"] = args.tile_pad
            if args.batch_size is not None:
                params["batch_size"] = args.batch_size
                params["batch_size_max"] = max(params["batch_size_max"], args.batch_size)
            if args.half is not None:
                params["half"] = args.half
        stage = stage_class()
        stage.load(device, params)
        stages.append(stage)
    return stages


def _final_spec(stages, source_spec: FrameSpec) -> FrameSpec:
    spec = source_spec
    for stage in stages:
        spec = stage.plan(spec)
    return spec


def _output_path(args: argparse.Namespace, stages, source_spec: FrameSpec, output_spec: FrameSpec) -> str:
    if args.output:
        path = Path(args.output).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"output already exists: {path}; use --overwrite to replace it")
        return str(path)

    output_format = args.output_format or "matroska"
    normalized_format = output_format.lower().lstrip(".")
    extension = _FORMAT_EXTENSIONS.get(normalized_format, f".{normalized_format}")
    return make_output_path(
        args.input,
        args.output_dir,
        stages,
        source_spec,
        output_spec,
        extension=extension,
    )


def _run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    _apply_cli_overrides(config, args)
    reader = VideoReader(args.input, input_format=args.input_format, decoder=args.decoder)
    source_spec = FrameSpec(reader.width, reader.height, reader.fps, reader.frame_size)

    device, _reason = ("cpu", "")
    stages = []
    writer = None
    try:
        # Choose the actual device before loading model weights.
        device, _reason = select_device(config["device"])
        configure_vram_limit(device, config["memory"])
        stages = _build_stages(config, device, args)
        output_spec = _final_spec(stages, source_spec)
        memory_plan = derive_memory_plan(config, source_spec, output_spec)
        effective_queue_depth = min(config["queue_depth"], memory_plan.depth)
        log.info(
            "effective queue depth=%d (configured cap=%d, memory-plan ceiling=%d)",
            effective_queue_depth, config["queue_depth"], memory_plan.depth,
        )
        output_path = _output_path(args, stages, source_spec, output_spec)

        writer = VideoWriter(
            output_path,
            output_spec.width,
            output_spec.height,
            output_spec.fps,
            config["codec"],
            config["crf"],
            config["preset"],
            args.input,
            pixel_format=config["pixel_format"],
            audio_codec=config["audio_codec"],
            output_format=args.output_format,
            video_options=args.video_option,
            overwrite=args.overwrite,
        )

        print(f"[vidpipe] Input : {args.input}", file=sys.stderr)
        print(f"[vidpipe] Output: {output_path}", file=sys.stderr)
        print(
            f"[vidpipe] Video: {reader.width}x{reader.height} @ {reader.fps:.3f} FPS; "
            f"frames={'~' if not reader.exact_frame_count else ''}{reader.total_frames or 'unknown'}; "
            f"{reader.codec} -> {config['codec']}; device={device}; "
            f"queue={effective_queue_depth}",
            file=sys.stderr,
        )

        controller = Controller()
        controller.start(reader, writer, effective_queue_depth, stages=stages)

        def request_stop(_signum, _frame):
            try:
                controller.stop()
            except StateError:
                pass

        previous_sigint = signal.signal(signal.SIGINT, request_stop)
        keyboard = KeyboardControls(
            on_pause=controller.pause,
            on_resume=controller.resume,
            on_stop=controller.stop,
        )
        controls_ready = keyboard.start()
        if controls_ready:
            print("[controls] P: pause | R: resume | S/Q: stop | Ctrl+C: stop", file=sys.stderr)
        elif sys.stdin.isatty():
            print("[controls] Interactive single-key input is unavailable; use Ctrl+C to stop.", file=sys.stderr)

        # Prefer stdout when it is a terminal; otherwise use stderr if that is
        # the interactive terminal (some IDEs redirect stdout separately).
        if sys.stdout.isatty():
            progress_stream = sys.stdout
        elif sys.stderr.isatty():
            progress_stream = sys.stderr
        else:
            progress_stream = sys.stdout

        reporter = ProgressReporter(
            interval=args.progress_interval,
            color=False if args.no_color else None,
            stream=progress_stream,
        )
        try:
            while True:
                controller.wait(timeout=args.progress_interval)
                state_name = controller.state.name
                snapshot = controller.pipeline.status_snapshot() if controller.pipeline else {}
                if not args.no_progress:
                    line = reporter.render(
                        snapshot, state_name, gpu_memory_stats(),
                        final=state_name in {"STOPPED", "FINISHED", "FAILED"},
                    )
                    reporter.print_update(
                        line, final=state_name in {"STOPPED", "FINISHED", "FAILED"}
                    )
                if state_name in {"STOPPED", "FINISHED", "FAILED"}:
                    break
        finally:
            keyboard.close()
            signal.signal(signal.SIGINT, previous_sigint)
        reader.close(allow_incomplete=controller.state in {State.STOPPED, State.FAILED})
        if controller.state == State.FAILED:
            if controller.error:
                raise RuntimeError(f"pipeline failed: {controller.error}") from controller.error
            raise RuntimeError("pipeline failed; see the error message above")

        print(
            f"[vidpipe] {controller.state.name}: {controller.processed_frames} frames written "
            f"to {output_path}",
            file=sys.stderr,
        )
        return 0
    finally:
        # Model weights and CUDA cache are released even when a later stage fails.
        if writer is not None and writer.proc.poll() is None:
            try:
                writer.close()
            except Exception:
                pass
        try:
            reader.close()
        except Exception:
            pass
        for stage in stages:
            try:
                stage.release()
            except Exception:
                log.exception("failed to release stage %s", stage.__class__.__name__)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    list_flags = (
        (args.list_encoders, "-encoders"),
        (args.list_decoders, "-decoders"),
        (args.list_formats, "-formats"),
        (args.list_pixel_formats, "-pix_fmts"),
    )
    for enabled, option in list_flags:
        if enabled:
            return _list_ffmpeg(option)

    if not args.input:
        parser.error("an input video is required unless a --list-* option is used")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        return _run(args)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"vidpipe: error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
