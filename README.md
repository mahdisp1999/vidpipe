# vidpipe 🎬

ابزار محلی برای بهبود کیفیت ویدیو با زنجیره‌ای از مدل‌های هوش مصنوعی.
ویدیو را می‌خواند، هر فریم را از مراحل پردازش عبور می‌دهد و خروجی MKV می‌سازد.
طراحی کامل در [ARCHITECTURE.md](ARCHITECTURE.md) آمده است.

## ✅ پیش‌نیازها

- Python 3.12
- ffmpeg و ffprobe (در PATH باشند)
- برای GPU: درایور NVIDIA و نسخهٔ CUDA سازگار PyTorch (اختیاری، بدونش روی CPU اجرا می‌شود)

## 📦 نصب

### روش ۱: با uv (پیشنهادی، سریع‌تر)

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

### روش ۲: با pip استاندارد

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> نسخهٔ PyTorch را جداگانه و با CUDA مناسب GPU نصب کن. راهنما در pytorch.org است.
> مثال با uv:
> ```bash
> uv pip install torch --index-url https://download.pytorch.org/whl/cu130
> ```
> (آدرس index را با نسخهٔ CUDA درایور خودت از pytorch.org تطبیق بده.)

## ▶️ اجرا

از ریشهٔ پروژه:

```bash
python -m vidpipe.main input.mp4 output.mkv --config config.yaml
```

### کنترل در حین اجرا

| کلید | کار |
|---|---|
| `p` + Enter | مکث |
| `r` + Enter | ادامه |
| `s` + Enter یا Ctrl+C | توقف و ذخیره (فایل قابل پخش می‌ماند) |

## ⚙️ پیکربندی

فایل `config.yaml` را ویرایش کن. مراحل پردازش در بخش `stages` به ترتیب اجرا می‌شوند:

```yaml
stages:
  - name: model
    model: ./models/realesr-general-x4v3.pth
    tile_size: 256
    tile_pad: 16
```

مدل‌های `.pth` را در پوشهٔ `models/` بگذار. این پوشه در git نادیده گرفته می‌شود.
قبل از استفاده از هر مدل، لایسنس آن را بررسی کن.

## ➕ افزودن مدل جدید

اگر معماری مدل از نوع موجود باشد، فقط یک بلوک `model` در `config.yaml` اضافه کن.
برای نوع کاملاً جدید، یک کلاس در `vidpipe/stages/` بساز و آن را در `vidpipe/registry.py` ثبت کن.

## 🧪 تست

```bash
pytest
```

## 🚧 وضعیت فعلی

- ✅ لوله‌ی هویت (identity) با Start، Pause، Resume و Stop
- ✅ مرحلهٔ مدل با تایلینگ
- ⏳ برنامهٔ حافظهٔ خودکار هنوز به صف‌ها وصل نشده است
- ⏳ مرحلهٔ RIFE (درون‌یابی فریم) هنوز پیاده‌سازی نشده است