# vidpipe

`vidpipe` is a local, frame-based video enhancement pipeline. It decodes a video with FFmpeg, sends frames through an ordered chain of image-processing stages, and encodes the result. Frames move through bounded in-memory queues; the application does not create intermediate image files.

The current inference backend is **PyTorch + Spandrel**. ONNX and TensorRT are not implemented in this release. The project adds no Python dependencies beyond the existing `requirements.txt`.

## Features

- Live, single-line CLI progress with percentage, frame counts, remaining work, FPS, elapsed time and ETA.
- Colored and bold status fields when the progress output is an interactive terminal.
- Immediate keyboard controls: `P` pause, `R` resume, `S` or `Q` stop. No Enter key is required.
- `Ctrl+C` requests a clean stop rather than abruptly killing worker threads.
- Tile batching and CUDA out-of-memory recovery.
- Reader, inference and writer workers connected by bounded queues.
- Per-stage timing, queue occupancy and PyTorch VRAM telemetry.
- Output names include the model chain, overall scale and a denoise tag when applicable.
- A direct `vidpipe` command with `--help` and FFmpeg capability-list options.

## Requirements

- Python 3.10 or newer (Python 3.12 is recommended).
- FFmpeg and FFprobe available in `PATH`.
- For model inference: PyTorch, Spandrel and NumPy.
- For YAML configuration: PyYAML.
- An NVIDIA driver and CUDA-enabled PyTorch build for GPU inference. CPU inference is also supported.

Install FFmpeg on Ubuntu if needed:

```bash
sudo apt install ffmpeg
```

## Installation

Create and activate a virtual environment, then install the project's existing requirements. Install the PyTorch build that matches your CUDA or CPU environment separately.

Using `uv`:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

Using `pip`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install the CLI launcher:

```bash
./scripts/install-cli.sh
```

The installer creates `~/.local/bin/vidpipe` as a symlink to this checkout. Ensure `~/.local/bin` is in `PATH` and activate the same virtual environment before running the command.

```bash
vidpipe --help
vidpipe --version
```

## Quick start

Put model weights in `models/` (this directory is ignored by Git), then edit `config.yaml` and point each `model:` entry to a real model file.

```bash
vidpipe input.mp4
```

If `--output` is omitted, the output is written under `output/` with a generated name that records the processing chain. For example:

```text
input__1xDeJPG_OmniSR__then__realesr-general-wdn-x4v3__scale-4x__denoised.mkv
```

Use `--output` when an exact output path is preferred.

## Runtime keyboard controls

When `stdin` is an interactive terminal, these keys are read immediately:

| Key | Action |
|---|---|
| `P` | Pause processing |
| `R` | Resume processing |
| `S` | Stop processing cleanly |
| `Q` | Stop processing cleanly |
| `Ctrl+C` | Request a clean stop |

Do not press Enter after a control key. The terminal is temporarily placed in a character-at-a-time input mode while the job runs. Its original settings are restored when processing ends.

If standard input is not an interactive terminal (for example, when input is piped), single-key controls are unavailable. `Ctrl+C` remains available when the process is attached to a terminal.

## Progress display

After a short header (input, output, resolution, models, device, precision), the run shows a **six-line live panel**. The panel is redrawn in place: the cursor moves up by the panel height (`ESC[nA`) and each line is cleared before being written (`ESC[2K`), so it never scrolls down the terminal.

Illustrative example:

```text
RUNNING    [██████████░░░░░░░░░░░░░░░░░░░░]  31.50% done   68.50% left
Frames     315 / 1,000                Remaining  685 frames
Speed      0.35 fps                   Elapsed    00:15:00    ETA 00:32:37
Model      realesr-general-x4v3       Tiles      4 / 12 @192px b2
Queue      in 2/4  out 1/4            VRAM       5.42 / 8.00 GB
GPU        96%
Per frame  decode 8 ms | process 2700 ms | encode 12 ms
```

Notes:

- The done and left percentages are computed from the same frame count, so they always add up to 100%.
- A `~` (and `(approx.)`) appears when the frame count is an estimate (for example, for variable-frame-rate input).
- `Elapsed` is wall-clock time since the run started and includes paused time. `ETA` uses the smoothed speed.
- `GPU` needs the `pynvml` package (listed in `requirements.txt`). Without it, the field shows `n/a`.
- The `OOM` field appears only when CUDA out-of-memory recovery has run.
- When output is redirected to a file or pipe, each refresh is one line with the panel's fields joined by ` | `.

Options:

```text
--progress-interval SECONDS   Refresh interval (default: 1.0)
--no-progress                Disable the live panel (the header is still printed)
--no-color                   Disable ANSI colors
```

## Command reference

Show all options:

```bash
vidpipe --help
```

### Input and output

```bash
vidpipe INPUT
vidpipe INPUT --output ./output/enhanced.mp4
vidpipe INPUT --output-dir ./output
vidpipe INPUT --config ./config.yaml
vidpipe INPUT --overwrite
```

- `INPUT`: source video. FFmpeg normally detects its container automatically.
- `-o, --output`: exact output path.
- `--output-dir`: directory used for automatically generated names.
- `--config`: YAML configuration file.
- `--overwrite`: allow replacing an existing explicit output file.

### Encoding options

```bash
vidpipe INPUT --codec libx264 --crf 18 --preset slow
vidpipe INPUT --codec libx265 --crf 22 --preset medium
vidpipe INPUT --codec h264_nvenc --no-crf --video-option rc=vbr --video-option cq=19
vidpipe INPUT --pixel-format yuv420p --audio-codec aac --format mp4
```

| Option | Purpose |
|---|---|
| `--codec`, `--video-codec` | FFmpeg video encoder, such as `libx264`, `libx265` or `h264_nvenc` |
| `--crf N` | Quality factor for encoders that support CRF; common range is encoder-dependent |
| `--no-crf` | Do not pass `-crf`; useful when an encoder uses another quality control |
| `--preset NAME` | Encoder speed/quality preset; valid names depend on the selected encoder |
| `--pixel-format FORMAT` | Output pixel format, such as `yuv420p`, `yuv444p` or `p010le` |
| `--audio-codec NAME` | Audio encoder or `copy`; use `none` to omit audio |
| `--format NAME` | Output container/muxer, such as `matroska`, `mp4` or `webm` |
| `--video-option KEY=VALUE` | Additional video-encoder option; repeat to pass multiple options |

`--crf` is not universal. For example, NVENC commonly uses options such as `cq` or `rc` instead. Check the selected encoder's FFmpeg documentation and available options.

### Decoding options

```bash
vidpipe INPUT --input-format mov --decoder h264
```

| Option | Purpose |
|---|---|
| `--input-format NAME` | Force an FFmpeg input demuxer; normally auto-detected |
| `--decoder NAME` | Force a decoder; normally selected automatically by FFmpeg |

### Processing and memory

```bash
vidpipe INPUT --device gpu --tile-size 192 --tile-pad 16 --batch-size 2
vidpipe INPUT --device cpu --queue-depth 2 --no-half
```

| Option | Purpose |
|---|---|
| `--device {auto,gpu,cpu}` | Select inference device |
| `--tile-size N` | Tile width and height in pixels; minimum 64 |
| `--tile-pad N` | Neighbor padding around each tile |
| `--batch-size N` | Number of same-shaped tiles sent to the model in one inference call |
| `--queue-depth N` | Maximum number of frames held in each pipeline queue |
| `--half` | Request FP16 where the CUDA model supports it |
| `--no-half` | Force FP32 model execution |

Tile batching can increase throughput but consumes more VRAM. If CUDA reports an out-of-memory error, the model stage reduces the tile batch size first and then reduces tile size before retrying the current frame.

### FFmpeg capability lists

These commands show the capabilities of the FFmpeg executable found in `PATH`:

```bash
vidpipe --list-encoders
vidpipe --list-decoders
vidpipe --list-formats
vidpipe --list-pixel-formats
```

They include the available input demuxers and output muxers, audio/video encoders and decoders, and pixel formats as reported by FFmpeg.

## Configuration

`config.yaml` contains the default processing stages, device, memory limits, queue depth and FFmpeg encoding settings. Model stages are applied in the order in which they appear in `stages`.

A model stage entry has the following general shape:

```yaml
stages:
  - name: model
    model: models/your-model.pth
    tile_size: 192
    tile_pad: 16
    batch_size: 1
    batch_size_max: 4
    half: true
```

Use the model's own documentation to select valid tile sizes, padding and precision. Do not assume every model supports FP16 or every tile size.

## Output naming

When an output path is generated automatically, `naming.py` derives a readable name from:

- The input filename.
- The ordered model chain.
- The net scale factor across the stages.
- A `denoised` suffix when a configured model name indicates denoising.

Providing `--output` bypasses automatic naming and uses the exact path supplied.

## Tests

Run the test suite from the repository root:

```bash
python -m pytest -q
```

The tests cover configuration validation, controller state transitions, memory planning, frame pipeline behavior, model tiling and OOM recovery, progress formatting, keyboard input, output naming and video metadata handling.

## Documentation

- [Architecture (English)](ARCHITECTURE.md)
- [معماری (فارسی)](ARCHITECTURE_FA.md)
- [راهنمای فارسی](README_FA.md)
