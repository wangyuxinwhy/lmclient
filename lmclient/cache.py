from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import cast

import diskcache

from lmclient.types import Messages, ModelResponse, Prompt, ModelParameters
from lmclient.utils import to_dict
from lmclient.version import __cache_version__

DEFAULT_CACHE_DIR = Path(os.getenv('LMCLIENT_CACHE_DIR', '~/.cache/lmclient')).expanduser().resolve()


class ChatCacheMixin:
    identifier: str
    _cache: diskcache.Cache | None
    _cache_dir: Path | None

    def __init__(self, use_cache: Path | str | bool = False) -> None:
        if isinstance(use_cache, (str, Path)):
            self.cache_dir = Path(use_cache)
        elif use_cache:
            self.cache_dir = DEFAULT_CACHE_DIR
        else:
            self.cache_dir = None

    def cache_response(self, key: str, response: ModelResponse) -> None:
        if self._cache is not None:
            self._cache[key] = response
        else:
            raise RuntimeError('Cache is not enabled')

    def try_load_response(self, key: str):
        if self._cache is not None and key in self._cache:
            response = self._cache[key]
            response = cast(ModelResponse, response)
            return response

    def generate_hash_key(self, messages: Messages, parameters: ModelParameters) -> str:
        if isinstance(prompt, str):
            hash_text = prompt
        else:
            hash_text = '---'.join([f'{k}={v}' for message in prompt for k, v in to_dict(message).items()])
        items = sorted([f'{key}={value}' for key, value in parameters.model_dump()])
        items += [f'__cache_version__={__cache_version__}']
        items = [hash_text, self.identifier] + items
        task_string = '---'.join(items)
        return self.md5_hash(task_string)

    @staticmethod
    def md5_hash(string: str):
        return hashlib.md5(string.encode()).hexdigest()

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
