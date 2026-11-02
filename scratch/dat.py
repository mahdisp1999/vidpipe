import numpy as np
import torch
from PIL import Image
from spandrel import ModelLoader, ImageModelDescriptor

device = "cuda" if torch.cuda.is_available() else "cpu"
# desc = ModelLoader().load_from_file("./models/realesr-general-wdn-x4v3.pth")
# assert isinstance(desc, ImageModelDescriptor)
# desc.to(device).eval()

def load_denoise_blended(t: float):
    """
    مدل را با ترکیب وزنی دو مدل Real-ESRGAN می‌سازد.
    t = وزن مدل wdn (۰ = denoise کم، ۱ = denoise زیاد)
    """
    # دو مدل هم‌معماری هستند، پس کلیدهای state_dict یکسان‌اند
    base = ModelLoader().load_from_file("./models/realesr-general-x4v3.pth")
    wdn = ModelLoader().load_from_file("./models/realesr-general-wdn-x4v3.pth")
    assert isinstance(base, ImageModelDescriptor)
    assert isinstance(wdn, ImageModelDescriptor)

    sd_base = base.model.state_dict()
    sd_wdn = wdn.model.state_dict()

    # میانگین وزنی روی هر تنسور: (1-t)*base + t*wdn
    blended = {k: (1.0 - t) * sd_base[k] + t * sd_wdn[k] for k in sd_base}
    base.model.load_state_dict(blended)

    return base

desc = load_denoise_blended(t=0.7)  # <- شدت denoise را اینجا تنظیم کن
desc.to(device).eval()


# مدل DAT از fp16 پشتیبانی نمی‌کند، پس مدل و ورودی هر دو float32 می‌مانند
scale = desc.scale
input_image_path= "./Input/frame00000001.jpg"
output_image_path = "./output/frame00000001_denoised_upscaled.png"
# output_image_path.removeprefix()
img = Image.open(input_image_path).convert("RGB")
x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)

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
def upscale_tiled(x, tile=256, overlap=16):
    _, c, h, w = x.shape
    out = torch.zeros(1, c, h * scale, w * scale, device=x.device, dtype=x.dtype)
    weights = torch.zeros(1, 1, h * scale, w * scale, device=x.device, dtype=x.dtype)

    step = tile - overlap
    for y0 in _starts(h, tile, step):
        for x0 in _starts(w, tile, step):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            patch = x[:, :, y0:y1, x0:x1]
            res = desc(patch)

            # وزن‌ها باید هم‌اندازهٔ خروجی مدل (یعنی مقیاس‌شده) باشند
            wy = _axis_weight((y1 - y0) * scale, overlap * scale, x.device, x.dtype)
            wx = _axis_weight((x1 - x0) * scale, overlap * scale, x.device, x.dtype)
            wgt = (wy[:, None] * wx[None, :])[None, None]

            out[:, :, y0 * scale:y1 * scale, x0 * scale:x1 * scale] += res * wgt
            weights[:, :, y0 * scale:y1 * scale, x0 * scale:x1 * scale] += wgt

    return out / weights.clamp_min(1e-8)

y = upscale_tiled(x)
y = y.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
Image.fromarray((y * 255).round().astype(np.uint8)).save(output_image_path)