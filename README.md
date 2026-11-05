# vidpipe

`vidpipe` is a local video toolkit with two modes:

- **Enhancement:** decodes a video with FFmpeg, sends frames through an ordered chain of image-model stages, and encodes the result. Frames move through bounded in-memory queues; the application does not create intermediate image files.
- **Lossless range removal:** cuts one or more time ranges out of a video with FFmpeg stream copy. Nothing is re-encoded.

The enhancement backend is **PyTorch + Spandrel**. ONNX and TensorRT are not implemented in this release. The project adds no Python dependencies beyond the existing `requirements.txt`.

## Features

- Lossless removal of time ranges (`--remove-range`) with automatic pipe, tmpfs and disk backends.
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
- For enhancement: PyTorch, Spandrel and NumPy.
- For YAML configuration: PyYAML.
- For GPU inference: an NVIDIA driver and a CUDA-enabled PyTorch build. CPU inference is also supported.
- Range removal needs only FFmpeg and FFprobe.

Install FFmpeg on Ubuntu if needed:

```
sudo apt install ffmpeg
```

## Installation

Create and activate a virtual environment, then install the project's requirements. Install the PyTorch build that matches your CUDA or CPU environment separately.

Using `uv`:

```
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

Using `pip`:

```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install the CLI launcher:

```
./scripts/install-cli.sh
```

The installer creates `~/.local/bin/vidpipe` as a symlink to this checkout. Make sure `~/.local/bin` is in `PATH`, and activate the same virtual environment before running the command.

```
vidpipe --help
vidpipe --version
```

## Quick start

Enhancement: put model weights in `models/` (this directory is ignored by Git), point each `model:` entry in `config.yaml` to a real model file, then run:

```
vidpipe input.mp4
```

If `--output` is omitted, the output is written under `output/` with a generated name that records the processing chain:

```
input__1xDeJPG_OmniSR__then__realesr-general-wdn-x4v3__scale-4x__denoised.mkv
```

Range removal (no models or `config.yaml` needed):

```
vidpipe input.mkv --remove-range 00:10:00-00:15:00
```

## Runtime keyboard controls

When `stdin` is an interactive terminal, these keys are read immediately during enhancement:

| Key      | Action                  |
|----------|-------------------------|
| `P`      | Pause processing        |
| `R`      | Resume processing       |
| `S`      | Stop processing cleanly |
| `Q`      | Stop processing cleanly |
| `Ctrl+C` | Request a clean stop    |

Do not press Enter after a control key. The terminal is temporarily placed in a character-at-a-time input mode while the job runs, and its original settings are restored when processing ends.

If standard input is not an interactive terminal (for example, when input is piped), single-key controls are unavailable. `Ctrl+C` remains available when the process is attached to a terminal.

## Progress display

After a short header (input, output, resolution, models, device, precision), enhancement shows a **six-line live panel**. The panel is redrawn in place: the cursor moves up by the panel height (`ESC[nA`) and each line is cleared before being written (`ESC[2K`), so it never scrolls down the terminal.

Illustrative example (numbers are made up):

```
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
- A `~` (and `(approx.)`) appears when the frame count is an estimate, for example for variable-frame-rate input.
- `Elapsed` is wall-clock time since the run started and includes paused time. `ETA` uses the smoothed speed.
- `GPU` needs the `pynvml` package (listed in `requirements.txt`). Without it, the field shows `n/a`.
- The `OOM` field appears only when CUDA out-of-memory recovery has run.
- When output is redirected to a file or pipe, each refresh is one line with the panel's fields joined by `|`.

Options:

```
--progress-interval SECONDS   Refresh interval (default: 1.0)
--no-progress                 Disable the live panel (the header is still printed)
--no-color                    Disable ANSI colors
```

Range removal prints its own short status lines and does not use the panel.

## Command reference

Show all options:

```
vidpipe --help
```

### Input and output

```
vidpipe INPUT
vidpipe INPUT --output ./output/enhanced.mp4
vidpipe INPUT --output-dir ./output
vidpipe INPUT --config ./config.yaml
vidpipe INPUT --overwrite
```

- `INPUT`: source video. FFmpeg normally detects its container automatically.
- `-o, --output`: exact output path.
- `--output-dir`: directory used for automatically generated names (enhancement only).
- `--config`: YAML configuration file.
- `--overwrite`: allow replacing an existing output file.

### Removing time ranges

```
vidpipe INPUT --remove-range START-END [--remove-range START-END ...] [-o OUTPUT] [--overwrite]
```

```
vidpipe lecture.mkv --remove-range 00:10:00-00:15:00
vidpipe lecture.mkv --remove-range 90-120 --remove-range 1:02:00-1:02:30 -o lecture_trimmed.mkv
```

- Each range is `START-END`. Each side can be seconds (`90`), `MM:SS` (`1:30`), or `HH:MM:SS.fraction` (`00:01:30.5`).
- Ranges must lie inside the video. Overlapping ranges are merged. Removing the whole video is an error.
- The default output is `<name>_cut<extension>` next to the input. `--output-dir` is not used in this mode.
- Video and audio are copied without re-encoding. Cut points snap to keyframes, so a cut may be a fraction of a second off the exact time. Keyframe spacing (GOP) is the limit: a video with a keyframe every 10 seconds can shift cuts by up to about 10 seconds.
- vidpipe tries three backends in order: `pipe` (one FFmpeg pass, no temporary files), then `tmpfs` (temporary parts in `/dev/shm`), then `disk`. The first one that works is used. The chosen backend is printed.
- Temporary files are removed after success or failure. The output is written under a temporary name and renamed only when it is complete.
- Range removal does not read `config.yaml`, load models, or select a device.

### Encoding options

```
vidpipe INPUT --codec libx264 --crf 18 --preset slow
vidpipe INPUT --codec libx265 --crf 22 --preset medium
vidpipe INPUT --codec h264_nvenc --no-crf --video-option rc=vbr --video-option cq=19
vidpipe INPUT --pixel-format yuv420p --audio-codec aac --format mp4
```

| Option                     | Purpose                                                                         |
|----------------------------|---------------------------------------------------------------------------------|
| `--codec`, `--video-codec` | FFmpeg video encoder, such as `libx264`, `libx265` or `h264_nvenc`              |
| `--crf N`                  | Quality factor for encoders that support CRF; common range is encoder-dependent |
| `--no-crf`                 | Do not pass `-crf`; useful when an encoder uses another quality control         |
| `--preset NAME`            | Encoder speed/quality preset; valid names depend on the selected encoder        |
| `--pixel-format FORMAT`    | Output pixel format, such as `yuv420p`, `yuv444p` or `p010le`                   |
| `--audio-codec NAME`       | Audio encoder or `copy`; use `none` to omit audio                               |
| `--format NAME`            | Output container/muxer, such as `matroska`, `mp4` or `webm`                     |
| `--video-option KEY=VALUE` | Additional video-encoder option; repeat to pass multiple options                |

These options apply to enhancement only. `--crf` is not universal: NVENC commonly uses options such as `cq` or `rc`. Check the selected encoder's FFmpeg documentation.

### Decoding options

```
vidpipe INPUT --input-format mov --decoder h264
```

| Option                | Purpose                                                    |
|-----------------------|------------------------------------------------------------|
| `--input-format NAME` | Force an FFmpeg input demuxer; normally auto-detected      |
| `--decoder NAME`      | Force a decoder; normally selected automatically by FFmpeg |

### Processing and memory

```
vidpipe INPUT --device gpu --tile-size 192 --tile-pad 16 --batch-size 2
vidpipe INPUT --device cpu --queue-depth 2 --no-half
```

| Option                    | Purpose                                                             |
|---------------------------|---------------------------------------------------------------------|
| `--device {auto,gpu,cpu}` | Select inference device                                             |
| `--tile-size N`           | Tile width and height in pixels; minimum 64                         |
| `--tile-pad N`            | Neighbor padding around each tile                                   |
| `--batch-size N`          | Number of same-shaped tiles sent to the model in one inference call |
| `--queue-depth N`         | Maximum number of frames held in each pipeline queue                |
| `--half`                  | Request FP16 where the CUDA model supports it                       |
| `--no-half`               | Force FP32 model execution                                          |

Tile batching can increase throughput but uses more VRAM. If CUDA reports an out-of-memory error, the model stage reduces the tile batch size first, then the tile size, and retries the current frame.

### FFmpeg capability lists

```
vidpipe --list-encoders
vidpipe --list-decoders
vidpipe --list-formats
vidpipe --list-pixel-formats
```

These show what the FFmpeg executable in `PATH` supports: encoders, decoders, demuxers and muxers, and pixel formats.

## Configuration

`config.yaml` contains the enhancement stages, device, memory limits, queue depth and FFmpeg encoding settings. Model stages run in the order they appear in `stages`. Range removal ignores this file.

A model stage entry has this shape:

```
stages:
  - name: model
    model: models/your-model.pth
    tile_size: 192
    tile_pad: 16
    batch_size: 1
    batch_size_max: 4
    half: true
```

Use each model's documentation to choose valid tile sizes, padding and precision. Not every model supports FP16 or every tile size.

## Output naming

Enhancement without `--output` derives a name from:

- The input filename.
- The ordered model chain.
- The net scale factor across the stages.
- A `denoised` suffix when a model name indicates denoising.

Range removal without `--output` uses `<name>_cut<extension>` next to the input. `--output` bypasses automatic naming in both modes.

## Tests

Run the test suite from the repository root:

```
python -m pytest -q
```

The tests cover configuration validation, controller state transitions, memory planning, frame pipeline behavior, model tiling and OOM recovery, progress formatting, keyboard input, output naming, video metadata handling, and range removal (`test_cut.py`). Range removal tests run real FFmpeg on a generated clip, check that decoded frames match the source (no re-encode), and check that no temporary files are left behind. They are skipped when FFmpeg is missing.

## Documentation

- [Architecture (English)](ARCHITECTURE.md)
- [معماری (فارسی)](ARCHITECTURE_FA.md)
- [راهنمای فارسی](README_FA.md)