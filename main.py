import gc

import numpy as np
import torch
from PIL import Image
from spandrel import ModelLoader

# ---------------------------------------------------------------- تنظیمات
SCALE = 4             # ضریب بزرگ‌نمایی مدل
TILE = 256             # اندازه هر بلوک ورودی (پیکسل). اگر باز هم OOM داد کمش کن (مثلاً 128)
PAD = 32               # حاشیه همپوشانی برای جلوگیری از خطوط درز بین بلوک‌ها
MODEL_PATH = "./4xFaceUpSharpDAT.pth"
INPUT_PATH = "./Image_profile_telegram_instagram_portrait8croped.webp"
OUTPUT_PATH = "output_4x_4xFaceUpSharpDATscr.png"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")


# ---------------------------------------------------------------- آزادسازی حافظه
def free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


free_memory()

model = ModelLoader().load_from_file(MODEL_PATH).to(device).eval()


# ---------------------------------------------------------------- پردازش بلوکی
def upscale_tiled(img: Image.Image) -> Image.Image:
    arr = np.array(img)                       # (H, W, 3) uint8
    h, w = arr.shape[:2]
    out = np.zeros((h * SCALE, w * SCALE, arr.shape[2]), dtype=np.float32)
    weight = np.zeros((h * SCALE, w * SCALE, 1), dtype=np.float32)

    for y0 in range(0, h, TILE):
        for x0 in range(0, w, TILE):
            y1 = min(y0 + TILE, h)
            x1 = min(x0 + TILE, w)

            # محدوده‌ی گسترده‌شده با حاشیه
            py0 = max(0, y0 - PAD)
            px0 = max(0, x0 - PAD)
            py1 = min(h, y1 + PAD)
            px1 = min(w, x1 + PAD)

            tile = arr[py0:py1, px0:px1]
            x = (
                torch.from_numpy(tile)
                .permute(2, 0, 1)
                .float()
                .div(255)
                .unsqueeze(0)
                .to(device)
            )

            with torch.no_grad():
                y = model(x)

            y = y.squeeze(0).clamp(0, 1).permute(1, 2, 0).float().cpu().numpy()

            # برش حاشیه اضافه از خروجی
            top = (y0 - py0) * SCALE
            left = (x0 - px0) * SCALE
            th = (y1 - y0) * SCALE
            tw = (x1 - x0) * SCALE
            y = y[top:top + th, left:left + tw]

            oy, ox = y0 * SCALE, x0 * SCALE
            out[oy:oy + th, ox:ox + tw] += y
            weight[oy:oy + th, ox:ox + tw] += 1.0

            del x, y
            free_memory()
            print(f"  tile ({x0},{y0}) -> ({x1},{y1})")

    out /= np.maximum(weight, 1e-8)
    return Image.fromarray((out * 255).round().astype(np.uint8))


# ---------------------------------------------------------------- اجرا
img = Image.open(INPUT_PATH).convert("RGB")
print(f"input: {img.width}x{img.height}")

result = upscale_tiled(img)
result.save(OUTPUT_PATH)
print(f"ذخیره شد: {OUTPUT_PATH} ({result.width}x{result.height})")

del model
free_memory()
