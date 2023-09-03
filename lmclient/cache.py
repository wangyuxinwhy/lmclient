from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import diskcache

from lmclient.types import ChatModelOutput

DEFAULT_CACHE_DIR = Path(os.getenv('LMCLIENT_CACHE_DIR', '~/.cache/lmclient')).expanduser().resolve()


class ChatCacheMixin:
    _cache: diskcache.Cache | None
    _cache_dir: Path | None

    def __init__(self, use_cache: Path | str | bool = False) -> None:
        if isinstance(use_cache, (str, Path)):
            self.cache_dir = Path(use_cache)
        elif use_cache:
            self.cache_dir = DEFAULT_CACHE_DIR
        else:
            self.cache_dir = None

    def cache_model_output(self, key: str, model_output: ChatModelOutput) -> None:
        if self._cache is not None:
            self._cache[key] = model_output
        else:
            raise RuntimeError('Cache is not enabled')

    def try_load_model_output(self, key: str):
        if self._cache is not None and key in self._cache:
            model_output = cast(ChatModelOutput, self._cache[key])
            return model_output

    @property
    def use_cache(self) -> bool:
        return self._cache is not None

    @property
    def cache_dir(self) -> Path | None:
        return self._cache_dir

    @cache_dir.setter
    def cache_dir(self, value: Path | None) -> None:
        if value is not None:
            if value.exists() and not value.is_dir():
                raise ValueError(f'Cache directory {value} is not a directory')
            value.mkdir(parents=True, exist_ok=True)
            self._cache = diskcache.Cache(value)
        else:
            self._cache = None
