import os
import re
import numpy as np
import torch
from PIL import Image
from spandrel import ModelLoader, ImageModelDescriptor

# ===================== تنظیمات =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# هر مرحله یک لیست از (مسیر مدل، وزن) است.
# مراحل به ترتیب اجرا می‌شوند؛ خروجی هر مرحله ورودی مرحلهٔ بعد است.
# داخل یک مرحله فقط مدل‌های هم‌معماری و هم‌scale را می‌توان با وزن ترکیب کرد.
STAGES = [
    # مرحله ۱: رفع blocking (1x، کیفیت را بالا می‌برد بدون بزرگ‌نمایی)
    [("./models/1xDeJPG_OmniSR.pth", 1.0)],
    # [("./models/1x-Debandurh-FS-Ultra-lite.pth", 1.0)],
    # [("./models/1x-Debandurh-FS-Ultra-lite.pth", 1.0)],
    # مرحله ۲: بزرگ‌نمایی ۴x (دو مدل هم‌معماری، ترکیب وزنی)
    [
        ("./models/realesr-general-x4v3.pth", 0.7),
        ("./models/realesr-general-wdn-x4v3.pth", 0.3),
    ],
    # [("./models/4xFaceUpSharpDAT.pth", 1.0)],
]

INPUT_PATH = "./Input/file"
OUTPUT_DIR = "./output"
TILE = 192
OVERLAP = 16


# ===================== نام‌گذاری =====================
def _stage_tag(models):
    """
    برچسب یک مرحله. مثال تک‌مدل: '1xDeJPG_OmniSR'
    مثال چند‌مدل: 'realesr-general-x4v3-0.3+realesr-general-wdn-x4v3-0.7'
    """
    parts = []
    for path, w in models:
        name = re.sub(r"[^\w\-]+", "-", os.path.splitext(os.path.basename(path))[0])
        parts.append(name if len(models) == 1 else f"{name}-{w:g}")
    return "+".join(parts)


def _chain_tag(stages):
    """برچسب کل زنجیره: مراحل با '__then__' از هم جدا می‌شوند."""
    return "__then__".join(_stage_tag(s) for s in stages)


def make_unique_output_path(input_path, output_dir, stages, ext=".png"):
    """
    مسیر خروجی یکتا می‌سازد. اگر فایل وجود داشته باشد، پسوند _2، _3، ... اضافه می‌شود.
    """
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]
    stem = f"{base}__{_chain_tag(stages)}"

    candidate = os.path.join(output_dir, stem + ext)
    counter = 2
    while os.path.exists(candidate):
        candidate = os.path.join(output_dir, f"{stem}_{counter}{ext}")
        counter += 1
    return candidate


# ===================== لود مدل =====================
def load_model(models):
    """
    یک یا چند مدل هم‌معماری را لود می‌کند.
    اگر بیشتر از یک مدل باشد، وزن‌هایشان به صورت میانگین وزنی ترکیب می‌شود.
    """
    if not models:
        raise ValueError("لیست مدل‌ها خالی است")

    descs = []
    for path, _ in models:
        d = ModelLoader().load_from_file(path)
        if not isinstance(d, ImageModelDescriptor):
            raise TypeError(f"{path} یک مدل تصویری معتبر نیست")
        descs.append(d)

    base = descs[0]
    for d in descs[1:]:
        if type(d.model) is not type(base.model) or d.scale != base.scale:
            raise ValueError(
                "معماری یا scale مدل‌های این مرحله یکی نیست؛ "
                "مدل‌های با معماری متفاوت را در یک مرحله ترکیب نکن"
            )

    if len(descs) == 1:
        return base.to(DEVICE).eval()

    total = sum(w for _, w in models)
    if total <= 0:
        raise ValueError("مجموع وزن‌ها باید مثبت باشد")

    blended = None
    for (_, w), d in zip(models, descs):
        sd = d.model.state_dict()
        coef = w / total
        if blended is None:
            blended = {k: coef * v.float() for k, v in sd.items()}
        else:
            for k in blended:
                blended[k] += coef * sd[k].float()

    base.model.load_state_dict(blended)
    return base.to(DEVICE).eval()


# ===================== تایلینگ =====================
def _starts(size, tile, step):
    """موقعیت شروع کاشی‌ها طوری که همیشه داخل تصویر بمانند."""
    if size <= tile:
        return [0]
    starts = list(range(0, size - tile, step))
    starts.append(size - tile)
    return sorted(set(starts))


def _axis_weight(n, overlap, device, dtype):
    """وزن خطی برای لبه‌های کاشی، تا محل هم‌پوشانی بدون درز ترکیب شود."""
    w = torch.ones(n, device=device, dtype=dtype)
    o = min(overlap, n // 2)
    if o > 0:
        ramp = torch.arange(1, o + 1, device=device, dtype=dtype) / (o + 1)
        w[:o] = ramp
        w[-o:] = ramp.flip(0)
    return w


@torch.inference_mode()
def upscale_tiled(desc, x, tile=256, overlap=16):
    """
    هر مدلی را (1x یا 4x یا هر scale دیگر) روی تصویر کاشی‌بندی‌شده اجرا می‌کند.
    scale از خود مدل خوانده می‌شود.
    """
    scale = desc.scale
    _, c, h, w = x.shape
    out = torch.zeros(1, c, h * scale, w * scale, device=x.device, dtype=x.dtype)
    weights = torch.zeros(1, 1, h * scale, w * scale, device=x.device, dtype=x.dtype)

    step = tile - overlap
    for y0 in _starts(h, tile, step):
        for x0 in _starts(w, tile, step):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            patch = x[:, :, y0:y1, x0:x1]
            res = desc(patch)

            wy = _axis_weight((y1 - y0) * scale, overlap * scale, x.device, x.dtype)
            wx = _axis_weight((x1 - x0) * scale, overlap * scale, x.device, x.dtype)
            wgt = (wy[:, None] * wx[None, :])[None, None]

            out[:, :, y0 * scale:y1 * scale, x0 * scale:x1 * scale] += res * wgt
            weights[:, :, y0 * scale:y1 * scale, x0 * scale:x1 * scale] += wgt

    return out / weights.clamp_min(1e-8)


# ===================== اجرای زنجیره =====================
def run_chain(x, stages):
    """
    مراحل را یکی‌یکی اجرا می‌کند. هر مرحله بعد از اتمام، مدلش از حافظه آزاد می‌شود
    تا VRAM برای مرحلهٔ بعد خالی شود.
    """
    y = x
    for i, models in enumerate(stages, start=1):
        names = [os.path.basename(p) for p, _ in models]
        print(f"[Stage {i}] Loading: {names}")

        desc = load_model(models)
        y = upscale_tiled(desc, y, tile=TILE, overlap=OVERLAP).clamp(0, 1)
        print(f"[Stage {i}] Completed, Output Dimensions: {tuple(y.shape[-2:])}")

        del desc
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    return y


# ===================== اجرا =====================
def main():
    img = Image.open(INPUT_PATH).convert("RGB")
    x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().div(255).unsqueeze(0).to(DEVICE)

    y = run_chain(x, STAGES)
    y = y.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()

    output_path = make_unique_output_path(INPUT_PATH, OUTPUT_DIR, STAGES, ext=".jpg")
    # Image.fromarray((y * 255).round().astype(np.uint8)).save(output_path, optimize=True, quality=95)
    Image.fromarray((y * 255).round().astype(np.uint8)).save(output_path, format="JPEG", quality=93, subsampling=0)
    # Image.fromarray((y * 255).round().astype(np.uint8)).save(output_path, format="WEBP", quality=92, method=5)
    print(f"ذخیره شد: {output_path}")


if __name__ == "__main__":
    main()