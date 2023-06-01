from __future__ import annotations

import time
from enum import Enum
from typing import ClassVar

import anyio
import asyncer
import diskcache
import tqdm

from lmclient.protocols import CompletionModel


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
        completion_model: CompletionModel,
        max_requests_per_minute: int = 20,
        async_capacity: int = 3,
        error_mode: ErrorMode | str = ErrorMode.RAISE,
        cache_dir: str | None = None,
        progress_bar: ProgressBarMode | str = ProgressBarMode.AUTO,
    ):
        self.completion_model = completion_model
        self.async_capacity = async_capacity
        self.max_requests_per_minute = max_requests_per_minute
        self.error_mode = ErrorMode(error_mode)
        self.progress_bar_mode = ProgressBarMode(progress_bar)
        self._task_created_time_list: list[int] = []
        self.cache = diskcache.Cache(cache_dir) if cache_dir is not None else None

    def run(self, prompts: list[str], **kwargs) -> list[str]:
        progress_bar = self._get_progress_bar(num_tasks=len(prompts))
        completions: list[str] = []
        for prompt in prompts:
            completion = self._run_single_task(prompt=prompt, progress_bar=progress_bar, **kwargs)
            completions.append(completion)
        progress_bar.close()
        return completions

    @asyncer.runnify
    async def async_run(self, prompts: list[str], **kwargs) -> list[str]:
        limiter = anyio.CapacityLimiter(self.async_capacity)
        task_created_lock = anyio.Lock()
        progress_bar = self._get_progress_bar(num_tasks=len(prompts))

        soon_values: list[asyncer.SoonValue[str]] = []
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
        prompt: str,
        limiter: anyio.CapacityLimiter,
        task_created_lock: anyio.Lock,
        progress_bar: tqdm.tqdm,
        **kwargs,
    ) -> str:
        async with limiter:
            task_key = self._gen_task_key(prompt=prompt, **kwargs)
            if self.cache is not None and task_key in self.cache:
                completion = self.cache[task_key]  # type: ignore
                progress_bar.update(1)
                return completion

            async with task_created_lock:
                sleep_time = self._calculate_sleep_time()
                if sleep_time > 0:
                    await anyio.sleep(sleep_time)
                self._task_created_time_list.append(int(time.time()))

            try:
                completion = await self.completion_model.async_complete(prompt=prompt, **kwargs)
                if self.cache is not None:
                    self.cache[task_key] = completion
            except BaseException as e:
                if self.error_mode is ErrorMode.RAISE:
                    raise
                elif self.error_mode is ErrorMode.IGNORE:
                    completion: str = f'Error: {e}'
                else:
                    raise ValueError(f'Unknown error mode: {self.error_mode}')

            progress_bar.update(1)
            return completion

    def _run_single_task(self, prompt: str, progress_bar: tqdm.tqdm, **kwargs) -> str:
        task_key = self._gen_task_key(prompt=prompt, **kwargs)
        if self.cache is not None and task_key in self.cache:
            completion = self.cache[task_key]  # type: ignore
            progress_bar.update(1)
            return completion

        sleep_time = self._calculate_sleep_time()
        if sleep_time > 0:
            time.sleep(sleep_time)
        self._task_created_time_list.append(int(time.time()))

        try:
            completion = self.completion_model.complete(prompt=prompt, **kwargs)
            if self.cache is not None:
                self.cache[task_key] = completion
        except BaseException as e:
            if self.error_mode is ErrorMode.RAISE:
                raise
            elif self.error_mode is ErrorMode.IGNORE:
                completion: str = f'Error: {e}'
            else:
                raise ValueError(f'Unknown error mode: {self.error_mode}')

        progress_bar.update(1)
        return completion

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

    def _gen_task_key(self, prompt: str, **kwargs) -> str:
        items = sorted([f'{key}={value}' for key, value in kwargs.items()])
        items = [prompt, self.completion_model.identifier] + items
        return '---'.join(items)

    def _get_progress_bar(self, num_tasks: int) -> tqdm.tqdm:
        use_progress_bar = (self.progress_bar_mode is ProgressBarMode.ALWAYS) or (
            self.progress_bar_mode is ProgressBarMode.AUTO and num_tasks > self.PROGRESS_BAR_THRESHOLD
        )
        progress_bar = tqdm.tqdm(
            desc=f'{self.completion_model.__class__.__name__}', total=num_tasks, disable=not use_progress_bar
        )
        return progress_bar
