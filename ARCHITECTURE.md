# معماری vidpipe

## ۱. ایده‌ی کلی

ویدیو فریم‌به‌فریم خوانده می‌شود، از زنجیرهٔ مراحل (مثلاً مدل‌های بهبود کیفیت) عبور داده می‌شود و دوباره به یک فایل MKV نوشته می‌شود.

- هیچ فریمی روی دیسک نوشته نمی‌شود.
- هیچ فایل موقتی ساخته نمی‌شود.
- صدا مستقیم از فایل اصلی کپی می‌شود.

## ۲. نمودار مفهومی

```mermaid
flowchart LR
    IN[📼 فایل ورودی] --> R

    subgraph Background [پس‌زمینه: I/O و صف]
        direction LR
        R[Reader<br/>ffmpeg decode] -- q_rp<br/>صف کران‌دار --> P
        W[Writer<br/>ffmpeg encode] 
    end

    subgraph Primary [کار اصلی: پردازش فریم]
        direction LR
        P[Pipeline<br/>Stage 1 → Stage 2 → ...]
    end

    P -- q_pw<br/>صف کران‌دار --> W
    W --> OUT[🎞️ فایل MKV<br/>+ صدای اصلی]

    C[🎛️ Controller<br/>IDLE · RUNNING · PAUSED · ...] -. دستور p/r/s .-> P
```

## ۳. سه رشته و دو صف

| رشته | کار |
|---|---|
| **reader** | فریم خام `rgb24` را از ffmpeg می‌خواند |
| **pipeline** | هر فریم را از مراحل به ترتیب عبور می‌دهد |
| **writer** | فریم‌های آماده را به encoder ffmpeg می‌دهد |

صف‌ها **کران‌دار** هستند. وقتی صفی پر شود، رشتهٔ قبلی منتظر می‌ماند. به همین دلیل مصرف حافظه به طول ویدیو وابسته نیست.

## ۴. حالت‌های اجرا

```mermaid
stateDiagram-v2
    [*] --> IDLE
    IDLE --> RUNNING: start
    RUNNING --> PAUSED: pause (p)
    PAUSED --> RUNNING: resume (r)
    RUNNING --> STOPPING: stop (s / Ctrl+C)
    PAUSED --> STOPPING: stop (s / Ctrl+C)
    STOPPING --> STOPPED: فایل ناقص ولی قابل پخش
    RUNNING --> FINISHED: ویدیو تمام شد
    RUNNING --> FAILED: خطا
    STOPPED --> [*]
    FINISHED --> [*]
    FAILED --> [*]
```

- **مکث (pause):** reader دیگر فریم نمی‌خواند. هیچ پردازشی kill نمی‌شود و حافظه ثابت می‌ماند.
- **توقف (stop):** reader متوقف می‌شود، بافر مراحل خالی می‌شود و ffmpeg با بستن stdin تمام می‌شود.

## ۵-الف. مسیر یک فریم

```mermaid
sequenceDiagram
    participant FF as ffmpeg (decode)
    participant R as Reader
    participant P as Pipeline
    participant S as Stage(s)
    participant W as Writer
    participant FE as ffmpeg (encode)

    FF->>R: بایت‌های rgb24
    R->>P: q_rp (کران‌دار)
    P->>S: push(frame)
    S-->>P: فریم(های) خروجی
    P->>W: q_pw (کران‌دار)
    W->>FE: stdin
```

هر مرحله یک کلاس است با این متدها:

| متد | کار |
|---|---|
| `load` | یک‌بار در شروع، منابع (مثل مدل) را بارگذاری می‌کند |
| `plan` | اندازهٔ خروجی را از روی ورودی حساب می‌کند |
| `push` | یک فریم می‌گیرد و صفر یا چند فریم برمی‌گرداند |
| `flush` | در پایان، فریم‌های بافرشده را بیرون می‌دهد |
| `release` | منابع را آزاد می‌کند |

ترتیب مراحل در `config.yaml` تعیین می‌شود. مرحلهٔ `model` یک مدل `.pth` را با تایلینگ اجرا می‌کند؛ کاشی‌ها با پیکسل‌های همسایه کار می‌کنند تا درز دیده نشود.


## ۶. دستگاه و حافظه

- `device: auto` اگر CUDA باشد از GPU استفاده می‌کند، وگرنه CPU با هشدار.
- اندازهٔ صف‌ها از `queue_depth` در config خوانده می‌شود.
- `memory.py` فعلاً فقط مقادیر پیشنهادی را لاگ می‌کند و هنوز به صف‌ها وصل نشده است.

## ۷. فایل‌ها

| فایل | مسئولیت |
|---|---|
| `main.py` | ورودی خط فرمان، دستورهای کنترلی |
| `config.py` | خواندن و اعتبارسنجی YAML |
| `controller.py` | ماشین حالت و کنترل اجرا |
| `pipeline.py` | سه رشته و دو صف |
| `video_io.py` | ورودی و خروجی ffmpeg |
| `memory.py` | انتخاب دستگاه و برنامهٔ حافظه |
| `registry.py` | نام مرحله ← کلاس مرحله |
| `stages/base.py` | رابط پایهٔ مرحله |
| `stages/identity.py` | مرحلهٔ عبور مستقیم (برای تست) |
| `stages/model.py` | مرحلهٔ مدل با تایلینگ |

## ۸. اصول

1. **Stop و Pause امن:** هیچ پردازشی kill نمی‌شود.
2. **فایل همیشه قابل پخش:** MKV حتی بعد از توقف معتبر می‌ماند.
3. **حافظهٔ ثابت:** صف‌های کران‌دار، مستقل از طول ویدیو.
4. **سادگی:** هر فایل یک مسئولیت دارد و افزودن مدل جدید فقط شامل یک کلاس یا یک بلوک config است.