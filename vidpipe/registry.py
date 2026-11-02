"""Stage registry: plain dict mapping stage name -> stage class.

# ثبات مرحله: نگاشت سادهٔ نام مرحله به کلاس آن. بدون کشف خودکار یا پلاگین.
"""

from __future__ import annotations

from typing import Dict, Type

from .stages.base import Stage
from .stages.identity import IdentityStage
from .stages.model import ModelStage

STAGES: Dict[str, Type[Stage]] = {
    "identity": IdentityStage,
    "model": ModelStage,
}


def resolve(name: str) -> Type[Stage]:
    """Return the stage class for a name, raising a clear error if unknown.

    # کلاس مرحله را برای یک نام برمی‌گرداند؛ اگر ناشناس باشد خطای واضح می‌دهد.
    """
    if name not in STAGES:
        raise ValueError(f"unknown stage: {name!r} (known: {sorted(STAGES)})")
    return STAGES[name]