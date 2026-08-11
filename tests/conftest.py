"""Local test runtime defaults shared by the bundled regression fixtures."""

from __future__ import annotations

import os
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_BUNDLED_HF_CACHE = _REPO_ROOT / "csj" / "artifacts" / "hf_cache"

# The repository ships this cache for deterministic Kronos regression tests.
# Do not override an explicit user/CI cache choice; a checkout without the
# bundled cache retains Hugging Face's normal resolution behavior.
if _BUNDLED_HF_CACHE.is_dir():
    os.environ.setdefault("HF_HUB_CACHE", str(_BUNDLED_HF_CACHE))
