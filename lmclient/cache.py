from __future__ import annotations

from pathlib import Path
from typing import Generic, Optional, TypeVar, cast

import diskcache

from lmclient.types import PathOrStr

T = TypeVar('T')


class BaseCache(Generic[T]):
    def set(self, key: str, value: T) -> None:
        raise NotImplementedError

    def get(self, key: str, default: T | None = None) -> T | None:
        raise NotImplementedError

    def delete(self, key: str) -> None:
        raise NotImplementedError


class DiskCache(BaseCache[T], Generic[T]):
    """
    A disk-based cache implementation using the `diskcache` library.

    Args:
        cache_dir (PathOrStr): The directory to use for storing cached data.

    Properties:
        cache_dir (Path): The directory to use for storing cached data.
        diskcache (diskcache.Cache): The underlying `diskcache.Cache` instance.

    Raises:
        ValueError: If the specified cache directory is not a directory.
    """

    _disk_cache: diskcache.Cache | None
    _cache_dir: Path | None

    def __init__(self, cache_dir: PathOrStr) -> None:
        self.cache_dir = Path(cache_dir)

    def set(self, key: str, value: T) -> None:
        """
        Set the value of a key in the cache.

        Args:
            key (str): The key to set the value for.
            value (T): The value to set for the key.

        Returns:
            None
        """
        self.disk_cache[key] = value

    def get(self, key: str, default: T | None = None) -> T | None:
        """
        Retrieve the value for the given key from the cache.

        Args:
            key (str): The key to retrieve the value for.
            default (Optional[T]): The default value to return if the key is not found.

        Returns:
            Optional[T]: The value for the given key, or the default value if the key is not found.
        """
        value = self.disk_cache.get(key, default=default)   # type: ignore
        value = cast(Optional[T], value)
        return value

    def delete(self, key: str) -> None:
        """
        Deletes the value associated with the given key from the cache.

        Args:
            key (str): The key to delete from the cache.

        Returns:
            None
        """
        if key in self.disk_cache:
            del self.disk_cache[key]

    @property
    def cache_dir(self) -> Path | None:
        return self._cache_dir

    @property
    def disk_cache(self) -> diskcache.Cache:
        if self._disk_cache is None:
            raise RuntimeError('DiskCache is not enabled, set cache_dir first')
        return self._disk_cache

    @cache_dir.setter
    def cache_dir(self, value: Path) -> None:
        if value.exists() and not value.is_dir():
            raise ValueError(f'Cache directory {value} is not a directory')
        value.mkdir(parents=True, exist_ok=True)
        self._cache_dir = value
        self._disk_cache = diskcache.Cache(value)
