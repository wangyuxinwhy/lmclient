from __future__ import annotations

from pathlib import Path
from typing import Generic, TypeVar, cast

import diskcache

T = TypeVar('T')


class DiskCache(Generic[T]):
    _diskcache: diskcache.Cache | None
    _cache_dir: Path | None

    def __init__(self, cache_dir: Path | str) -> None:
        self.cache_dir = Path(cache_dir)

    def save(self, key: str, value: T) -> None:
        if self._diskcache is not None:
            self._diskcache[key] = value
        else:
            raise RuntimeError('Cache is not enabled')

    def get(self, key: str) -> T | None:
        if self._diskcache is not None and key in self._diskcache:
            model_output = cast(T, self._diskcache[key])
            return model_output

    @property
    def use_cache(self) -> bool:
        return self._diskcache is not None

    @property
    def cache_dir(self) -> Path | None:
        return self._cache_dir

    @cache_dir.setter
    def cache_dir(self, value: Path) -> None:
        if value.exists() and not value.is_dir():
            raise ValueError(f'Cache directory {value} is not a directory')
        value.mkdir(parents=True, exist_ok=True)
        self._diskcache = diskcache.Cache(value)
