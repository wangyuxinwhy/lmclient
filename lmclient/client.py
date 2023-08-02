from __future__ import annotations

import hashlib
import os
import time
from enum import Enum
from pathlib import Path
from typing import Callable, ClassVar, Sequence, cast

import anyio
import asyncer
import diskcache
import tqdm

from lmclient.types import ChatModel, Messages, ModelResponse, TaskResult
from lmclient.version import __cache_version__

DEFAULT_CACHE_DIR = Path(os.getenv('LMCLIENT_CACHE_DIR', '~/.cache/lmclient')).expanduser().resolve()


class ErrorMode(str, Enum):
    RAISE = 'raise'
    IGNORE = 'ignore'


class ProgressBarMode(str, Enum):
    AUTO = 'auto'
    ALWAYS = 'always'
    NEVER = 'never'


class LMClient:
    NUM_SECONDS_PER_MINUTE: ClassVar[int] = 60
    PROGRESS_BAR_THRESHOLD: ClassVar[int] = 20

    def __init__(
        self,
        model: ChatModel,
        max_requests_per_minute: int = 20,
        async_capacity: int = 3,
        timeout: int | None = 20,
        error_mode: ErrorMode | str = ErrorMode.RAISE,
        cache_dir: Path | str | None = DEFAULT_CACHE_DIR,
        progress_bar: ProgressBarMode | str = ProgressBarMode.AUTO,
        postprocess_function: Callable[[ModelResponse], ModelResponse] | None = None,
    ):
        self.model = model
        self.async_capacity = async_capacity
        self.max_requests_per_minute = max_requests_per_minute
        self.error_mode = ErrorMode(error_mode)
        self.progress_bar_mode = ProgressBarMode(progress_bar)
        self._task_created_time_list: list[int] = []

        cache_dir = Path(cache_dir) if cache_dir is not None else None
        if cache_dir is not None:
            if cache_dir.exists() and not cache_dir.is_dir():
                raise ValueError(f'Cache directory {cache_dir} is not a directory')
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache = diskcache.Cache(cache_dir)
        else:
            self.cache = None

        self.postprocess_function = postprocess_function or self.model.default_postprocess_function

        if timeout is not None:
            self.model.timeout = timeout

    def run(self, prompts: Sequence[str | Messages], **kwargs) -> list[TaskResult]:
        progress_bar = self._get_progress_bar(num_tasks=len(prompts))
        task_results: list[TaskResult] = []
        for prompt in prompts:
            task_result = self._run_single_task(prompt=prompt, progress_bar=progress_bar, **kwargs)
            task_results.append(task_result)
        progress_bar.close()
        return task_results

    @asyncer.runnify
    async def async_run(self, prompts: Sequence[str | Messages], **kwargs) -> list[TaskResult]:
        limiter = anyio.CapacityLimiter(self.async_capacity)
        task_created_lock = anyio.Lock()
        progress_bar = self._get_progress_bar(num_tasks=len(prompts))

        soon_values: list[asyncer.SoonValue[TaskResult]] = []
        async with asyncer.create_task_group() as task_group:
            soon_func = task_group.soonify(self._async_run_single_task)
            for prompt in prompts:
                soon_value = soon_func(
                    prompt=prompt,
                    limiter=limiter,
                    task_created_lock=task_created_lock,
                    progress_bar=progress_bar,
                    **kwargs,
                )
                soon_values.append(soon_value)

        progress_bar.close()
        values = [soon_value.value for soon_value in soon_values]
        return values

    async def _async_run_single_task(
        self,
        prompt: str | Messages,
        limiter: anyio.CapacityLimiter,
        task_created_lock: anyio.Lock,
        progress_bar: tqdm.tqdm,
        **kwargs,
    ) -> TaskResult:
        async with limiter:
            task_key = self._gen_task_key(prompt=prompt, **kwargs)

            response = self.read_from_cache(task_key)

            if response is None:
                async with task_created_lock:
                    sleep_time = self._calculate_sleep_time()
                    if sleep_time > 0:
                        await anyio.sleep(sleep_time)
                    self._task_created_time_list.append(int(time.time()))

                try:
                    response = await self.model.async_chat(prompt=prompt, **kwargs)
                    if self.cache is not None:
                        self.cache[task_key] = response
                except BaseException as e:
                    if self.error_mode is ErrorMode.RAISE:
                        raise
                    elif self.error_mode is ErrorMode.IGNORE:
                        return TaskResult(error_message=str(e))
                    else:
                        raise ValueError(f'Unknown error mode: {self.error_mode}')

            try:
                output = self.postprocess_function(response)
                result = TaskResult(response=response, output=output)
            except BaseException as e:
                if self.error_mode is ErrorMode.RAISE:
                    raise
                elif self.error_mode is ErrorMode.IGNORE:
                    result = TaskResult(error_message=str(e), response=response)
                else:
                    raise ValueError(f'Unknown error mode: {self.error_mode}')

            progress_bar.update(1)
            return result

    def _run_single_task(self, prompt: str | Messages, progress_bar: tqdm.tqdm, **kwargs) -> TaskResult:
        task_key = self._gen_task_key(prompt=prompt, **kwargs)

        response = self.read_from_cache(task_key)
        if response is None:
            sleep_time = self._calculate_sleep_time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            self._task_created_time_list.append(int(time.time()))

            try:
                response = self.model.chat(prompt=prompt, **kwargs)
                if self.cache is not None:
                    self.cache[task_key] = response
            except BaseException as e:
                if self.error_mode is ErrorMode.RAISE:
                    raise
                elif self.error_mode is ErrorMode.IGNORE:
                    return TaskResult(error_message=str(e))
                else:
                    raise ValueError(f'Unknown error mode: {self.error_mode}')

        try:
            output = self.postprocess_function(response)
            result = TaskResult(response=response, output=output)
        except BaseException as e:
            if self.error_mode is ErrorMode.RAISE:
                raise
            elif self.error_mode is ErrorMode.IGNORE:
                result = TaskResult(error_message=str(e), response=response)
            else:
                raise ValueError(f'Unknown error mode: {self.error_mode}')

        progress_bar.update(1)
        return result

    def read_from_cache(self, key: str) -> ModelResponse | None:
        if self.cache is not None and key in self.cache:
            response = self.cache[key]
            response = cast(ModelResponse, response)
            return response
        return

    def _calculate_sleep_time(self):
        idx = 0
        current_time = time.time()
        for i, task_created_time in enumerate(self._task_created_time_list):
            if current_time - task_created_time < self.NUM_SECONDS_PER_MINUTE:
                idx = i
                break
        self._task_created_time_list = self._task_created_time_list[idx:]

        if len(self._task_created_time_list) < self.max_requests_per_minute:
            return 0
        else:
            return max(self.NUM_SECONDS_PER_MINUTE - int(current_time - self._task_created_time_list[0]) + 1, 0)

    def _gen_task_key(self, prompt: str | Messages, **kwargs) -> str:
        if not isinstance(prompt, str):
            prompt = '---'.join([f'{message["role"]}={message["content"]}' for message in prompt])
        items = sorted([f'{key}={value}' for key, value in kwargs.items()])
        items += [f'__cache_version__={__cache_version__}']
        items = [prompt, self.model.identifier] + items
        task_string = '---'.join(items)
        return self.md5_hash(task_string)

    @staticmethod
    def md5_hash(string: str):
        return hashlib.md5(string.encode()).hexdigest()

    def _get_progress_bar(self, num_tasks: int) -> tqdm.tqdm:
        use_progress_bar = (self.progress_bar_mode is ProgressBarMode.ALWAYS) or (
            self.progress_bar_mode is ProgressBarMode.AUTO and num_tasks > self.PROGRESS_BAR_THRESHOLD
        )
        progress_bar = tqdm.tqdm(desc=f'{self.model.__class__.__name__}', total=num_tasks, disable=not use_progress_bar)
        return progress_bar
