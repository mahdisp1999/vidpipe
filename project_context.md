# Project Context: Local Video Enhancement Pipeline (vidpipe)

Last updated after the decision round. Status: design decided, no code yet.

## 1. Owner and communication
- Experienced Python developer and Linux user, works hands-on and self-directed.
- Communicates in Persian and English. Explanations in Persian when requested; code, identifiers, and prompts in English.
- Wants conceptual, direct answers. Dislikes over-engineering and says so explicitly. Call out over-engineering when seen.
- Every complex piece of code must have explanatory comments written in Persian.

## 2. Goal
A desktop-capable application that enhances one input video locally, producing one output video through an ordered, configurable chain of stages:
1. Restore / denoise (1x models, and denoise-capable upscale models)
2. Upscale (4x models, output resized to the requested scale when needed)
3. Frame interpolation to a target FPS (integer multiplier or fixed FPS)

The run must be controllable: Start, Pause, Resume, and Stop/Save at any moment, always leaving a playable output file containing everything processed up to that point.

## 3. Environment (confirmed)
- OS: Ubuntu 26.04
- GPU: NVIDIA RTX 4060, 8 GB VRAM
- RAM: 32 GB
- Python: 3.12
- Driver / CUDA: CUDA 13.2 (confirm with `nvidia-smi` before install; the PyTorch wheel must match a CUDA 13.x build that the installed driver supports)
- PyTorch: 2.11

## 4. Models (candidate list, all to be license-checked before use)
| Class | Models | Behavior |
|---|---|---|
| Restore (1x) | 1x-Debandurh-FS-Ultra-lite, 1xDeJPG_OmniSR | Keep resolution. Cheap. Run before upscale. |
| Upscale (4x) | realesr-general-x4v3, realesr-general-wdn-x4v3, 4xFaceUpSharpDAT, BSRGAN, 4xNomos2_realplksr_dysample, 4x-Rybu | Output x4. Resize to requested scale if target is 2x. |
| Denoise + upscale | realesr-general-wdn-x4v3 | Combined. Denoise strength is a parameter. |

Notes:
- 4xFaceUpSharpDAT is a DAT transformer: heavy, needs tiling, invents detail on non-face areas. Do not use for identity-sensitive content.
- BSRGAN and DAT models are heavier on 8 GB VRAM; smaller tiles and lower throughput are expected.
- Licenses differ; some may be non-commercial only. Check each before use.
- Model choice is configuration, not code. Stage class is per architecture family, not per model file.

## 5. Decisions (final)
| Topic | Decision |
|---|---|
| Frame transport | ffmpeg subprocess pipes, raw rgb24 in and out |
| Input FPS | Auto-detect CFR or VFR. If VFR, force CFR on decode. |
| Target FPS | Either integer multiplier (2x, 3x) or fixed value (e.g. 60). Non-integer ratios handled by RIFE at arbitrary t. |
| Output codec | Configurable: libx264, libx265, libsvtav1, etc. CRF and preset configurable. |
| Output container | MKV by default (survives abrupt stop). MP4 only when explicitly requested. |
| Bit depth / HDR | Out of scope for v1 (8-bit SDR only). |
| Stage order | Configurable in YAML. Default: restore -> upscale -> interpolate. |
| Interpolation | RIFE, implemented directly as a Stage (Video2x rejected: file-to-file only). |
| Tiling | Inside the Stage (shared helper). Reflect padding, exact crop. Feathering deferred. |
| Audio | Second ffmpeg input (source file), `-map 1:a? -c:a copy`. No temp files. |
| Device | One device per run. GPU preferred, CPU fallback with warning. Logged at startup. |
| Concurrency | Threads. Two bounded queues (reader to pipeline, pipeline to writer). |
| Pause / Resume | In-session only. Implemented with threading.Event; reader stops pulling frames. |
| Stop / Save | Finish the current frame, flush stages, close encoder pipe gracefully, produce valid MKV. |
| Resume after restart | Out of v1. Deferred to v2 with checkpoint JSON and segment merge. |
| Work directory | The output file's directory is the work directory. Used for the file-listing acceptance test. |
| CUDA/PyTorch | Pin the wheel that matches CUDA 13.x and the installed driver, verified before install. |

## 6. Hard constraints
- No frame is written to disk. Frames flow through RAM and VRAM only.
- No intermediate file. The only file written is the output (MKV by default).
  - Exception: none in v1.
- Each model is loaded once per run.
- Streaming: the whole video is never loaded into memory.
- Memory bounded and independent of video length.
- Audio preserved without temp files.
- Stop and Save must always produce a playable file with all frames processed so far, even if the run is incomplete.
- Pause must keep memory constant and must not kill any process.
- Stop must not kill ffmpeg abruptly. Close its stdin and wait for it to finish.

## 7. Architecture
Primary work: the one GPU-bound stage running at a time. Background work: decode, encode, audio, and queue handling.

Modules (kept minimal):
```
vidpipe/
  main.py          entry point (CLI first, GUI later)
  config.py        load and validate YAML config
  controller.py    state machine: IDLE, RUNNING, PAUSED, STOPPING, STOPPED, FINISHED, FAILED
  pipeline.py      reader thread, bounded queues, writer thread, stage execution
  video_io.py      ffmpeg input and output wrappers (replaceable I/O layer)
  memory.py        auto and manual derivation of queue depth, tile size, batch size
  registry.py      dict: stage name -> stage class
  stages/
    base.py        Stage base class
    identity.py    pass-through stage for testing
    restore.py     1x restore models
    upscale.py     Nx upscale models with tiling
    rife.py        RIFE interpolation with arbitrary t
  tests/
```
Merged on purpose: device selection lives in `memory.py` / `main.py`; tiling lives in `upscale.py` unless a second stage needs it.

State machine:
```
IDLE -> RUNNING <-> PAUSED
          |
          v
       STOPPING -> STOPPED (valid partial file)
          |
          v
       FINISHED / FAILED
```

## 8. Memory (8 GB VRAM, 32 GB RAM)
- VRAM safety margin: 20% of free VRAM never allocated (about 1.6 GB of 8 GB total).
- RAM safety margin: 15% of free RAM never allocated.
- Auto mode: probe at startup, derive queue depth, tile size, batch size, log values.
- Manual mode: config values are hard caps; `torch.cuda.set_per_process_memory_fraction` applied.
- Values fixed for the whole run in v1 (no dynamic re-probing).
- Estimates must be replaced by measured values during the identity and upscale steps.

## 9. Acceptance criteria (measurable)
1. Peak RAM stays below the configured or auto-derived cap on a 10-minute 1080p clip.
2. At least 20% of VRAM stays free during the primary stage, unless a manual cap is lower.
3. Output audio matches the source (same codec, duration within 50 ms).
4. No file other than the output is created in the work directory during a run (listing before and after).
5. Each model is loaded exactly once per run (log line).
6. With prefetch enabled, total run time is measurably lower than with it disabled.
7. Stop at start, middle, and end each produce a playable MKV whose duration matches the processed frame count.
8. Pause holds memory constant and resumes without frame loss or duplication (frame index check).
9. Adding a model changes only: one file, one registry entry, one config block.
10. Every dependency is justified; no feature is added that was not requested.

## 10. Priorities when trade-offs conflict
1. Correctness (right output, audio preserved, valid file on Stop)
2. Simplicity (ponytail principle, minimal files)
3. Throughput (GPU stays busy in the heavy stage)
4. Memory efficiency (within limits with safety margin)

## 11. Roadmap
1. Identity pipeline with Controller (Start, Pause, Resume, Stop), MKV output, and Stop/Pause tests.
2. Config, registry, device selection, and memory derivation (auto and manual).
3. Restore stage (1x models), first with 1xDeJPG_OmniSR.
4. Upscale stage with tiling, first with realesr-general-x4v3, then the remaining models.
5. RIFE interpolation: integer multiplier first, then fixed target FPS with arbitrary t and frame dropping.
6. Acceptance run on the 10-minute 1080p clip, including Stop and Pause at several points.

Deferred to v2: resume after restart (checkpoint and segment merge), multi-GPU, multiprocessing, dynamic re-probing, bit depth and HDR.

## 12. Open items before coding
- Confirm the CUDA 13.x wheel for PyTorch 2.11 against the installed driver.
- Provide or choose a 10-minute 1080p test clip.
- License check for each candidate model.

## 13. Working rules for AI assistance
- Use prompt chaining with a review checkpoint between steps.
- Give a short plan before any code. Mark primary versus background work. Respect the hard constraints and call out violations.
- Keep the first version minimal; flag over-engineering.