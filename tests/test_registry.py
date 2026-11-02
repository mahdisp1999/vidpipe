"""Stage registry tests.

# تست‌های ثبات مرحله: resolve نام‌های شناخته‌شده و ناشناخته.
"""

import pytest

from vidpipe.registry import STAGES, resolve
from vidpipe.stages.identity import IdentityStage


def test_identity_resolves_to_class():
    # identity باید به کلاس IdentityStage برسد.
    assert resolve("identity") is IdentityStage


def test_identity_in_registry():
    # identity باید در ثبات موجود باشد.
    assert "identity" in STAGES


def test_unknown_stage_raises():
    # نام ناشناخته باید خطای واضح بدهد.
    with pytest.raises(ValueError) as ei:
        resolve("does-not-exist")
    assert "unknown stage" in str(ei.value)
    assert "identity" in str(ei.value)