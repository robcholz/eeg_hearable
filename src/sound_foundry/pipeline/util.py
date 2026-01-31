from __future__ import annotations

import tempfile
from pathlib import Path

_BUILD_CACHE: tempfile.TemporaryDirectory | None = None


def get_build_cache_dir() -> Path:
    global _BUILD_CACHE
    if _BUILD_CACHE is None:
        _BUILD_CACHE = tempfile.TemporaryDirectory(prefix="sound_foundry_buildcache_")
    return Path(_BUILD_CACHE.name)


def cleanup_buildcache() -> None:
    global _BUILD_CACHE
    if _BUILD_CACHE is not None:
        _BUILD_CACHE.cleanup()
        _BUILD_CACHE = None
