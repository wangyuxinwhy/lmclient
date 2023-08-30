from __future__ import annotations

import os
import time
from enum import Enum
from pathlib import Path
from typing import ClassVar, Generic, Sequence, TypeVar

import anyio
import asyncer
import tqdm

from lmclient.models import BaseChatModel
from lmclient.openai_schema import OpenAISchema
from lmclient.types import ChatModelOutput, Prompt

DEFAULT_CACHE_DIR = Path(os.getenv('LMCLIENT_CACHE_DIR', '~/.cache/lmclient')).expanduser().resolve()

T = TypeVar('T')
T_O = TypeVar('T_O', bound=OpenAISchema)


class ErrorMode(str, Enum):
    RAISE = 'raise'
    IGNORE = 'ignore'


class ProgressBarMode(str, Enum):
    AUTO = 'auto'
    ALWAYS = 'always'
    NEVER = 'never'


class LMClient(Generic[T]):
    error_mode: ErrorMode
    NUM_SECONDS_PER_MINUTE: ClassVar[int] = 60
    PROGRESS_BAR_THRESHOLD: ClassVar[int] = 20

    def __init__(
        self,
        chat_model: BaseChatModel[T],
        async_capacity: int = 3,
        max_requests_per_minute: int = 20,
        error_mode: ErrorMode | str = ErrorMode.RAISE,
        progress_bar: ProgressBarMode | str = ProgressBarMode.AUTO,
    ):
        self.chat_model = chat_model
        self.async_capacity = async_capacity
        self.max_requests_per_minute = max_requests_per_minute
        self.error_mode = ErrorMode(error_mode)
        self.progress_bar_mode = ProgressBarMode(progress_bar)
        self._task_created_time_list: list[int] = []

    def run(self, prompts: Sequence[Prompt], **kwargs) -> list[ChatModelOutput[T]]:
        progress_bar = self._get_progress_bar(num_tasks=len(prompts))
        task_results: list[ChatModelOutput[T]] = []
        for prompt in prompts:
            task_result = self._run_single_task(prompt=prompt, progress_bar=progress_bar, **kwargs)
            task_results.append(task_result)
        progress_bar.close()
        return task_results

    async def _async_run(self, prompts: Sequence[Prompt], **kwargs) -> list[ChatModelOutput[T]]:
        limiter = anyio.CapacityLimiter(self.async_capacity)
        task_created_lock = anyio.Lock()
        progress_bar = self._get_progress_bar(num_tasks=len(prompts))

        soon_values: list[asyncer.SoonValue[ChatModelOutput[T]]] = []
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

    def async_run(self, prompts: Sequence[Prompt], **kwargs) -> list[ChatModelOutput[T]]:
        return asyncer.runnify(self._async_run)(prompts, **kwargs)

    async def _async_run_single_task(
        self,
        prompt: Prompt,
        limiter: anyio.CapacityLimiter,
        task_created_lock: anyio.Lock,
        progress_bar: tqdm.tqdm,
        **kwargs,
    ) -> ChatModelOutput:
        async with limiter:
            task_key = self.chat_model.generate_hash_key(prompt=prompt, **kwargs)
            response = self.chat_model.try_load_response(task_key)

            if response is None:
                async with task_created_lock:
                    sleep_time = self._calculate_sleep_time()
                    if sleep_time > 0:
                        await anyio.sleep(sleep_time)
                    self._task_created_time_list.append(int(time.time()))

            try:
                output = await self.chat_model.async_chat(prompt=prompt, **kwargs)
                progress_bar.update(1)
                return output
            except BaseException as e:
                if self.error_mode is ErrorMode.RAISE:
                    raise
                elif self.error_mode is ErrorMode.IGNORE:
                    return ChatModelOutput(error_message=str(e))
                else:
                    raise ValueError(f'Unknown error mode: {self.error_mode}') from e

    def _run_single_task(self, prompt: Prompt, progress_bar: tqdm.tqdm, **kwargs) -> ChatModelOutput:
        task_key = self.chat_model.generate_hash_key(prompt=prompt, **kwargs)
        response = self.chat_model.try_load_response(task_key)

        if response is None:
            sleep_time = self._calculate_sleep_time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            self._task_created_time_list.append(int(time.time()))

        try:
            output = self.chat_model.chat(prompt=prompt, **kwargs)
            progress_bar.update(1)
            return output
        except BaseException as e:
            if self.error_mode is ErrorMode.RAISE:
                raise
            elif self.error_mode is ErrorMode.IGNORE:
                return ChatModelOutput(output=f'Response Error: {e}', response={})
            else:
                raise ValueError(f'Unknown error mode: {self.error_mode}') from e

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

    def _get_progress_bar(self, num_tasks: int) -> tqdm.tqdm:
        use_progress_bar = (self.progress_bar_mode is ProgressBarMode.ALWAYS) or (
            self.progress_bar_mode is ProgressBarMode.AUTO and num_tasks > self.PROGRESS_BAR_THRESHOLD
        )
        progress_bar = tqdm.tqdm(desc=f'{self.chat_model.__class__.__name__}', total=num_tasks, disable=not use_progress_bar)
        return progress_bar
