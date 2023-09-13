from __future__ import annotations

import time
from enum import Enum
from typing import AsyncGenerator, ClassVar, Generator, Generic, NoReturn, Sequence, cast

import anyio
import asyncer
import tqdm

from lmclient.models import load_from_model_id
from lmclient.models.base import T_O, T_P, BaseChatModel
from lmclient.types import ChatModelOutput, Message, Messages, Prompt


class ErrorMode(str, Enum):
    RAISE = 'raise'
    IGNORE = 'ignore'


class ProgressBarMode(str, Enum):
    AUTO = 'auto'
    ALWAYS = 'always'
    NEVER = 'never'


def ensure_messages(prompt: Prompt) -> Messages:
    if isinstance(prompt, str):
        return [Message(role='user', content=prompt)]
    elif isinstance(prompt, Message):
        return [prompt]
    elif isinstance(prompt, dict):
        return [Message(**prompt)]
    else:
        messages: list[Message] = []
        for message in prompt:
            if isinstance(message, dict):
                messages.append(Message(**message))
            else:
                messages.append(message)
        return messages


class CompletionEngine(Generic[T_P, T_O]):
    error_mode: ErrorMode
    NUM_SECONDS_PER_MINUTE: ClassVar[int] = 60
    PROGRESS_BAR_THRESHOLD: ClassVar[int] = 20

    def __init__(
        self,
        chat_model: BaseChatModel[T_P, T_O] | str,
        async_capacity: int = 3,
        max_requests_per_minute: int = 20,
        error_mode: ErrorMode | str = ErrorMode.RAISE,
        progress_bar: ProgressBarMode | str = ProgressBarMode.AUTO,
    ):
        if isinstance(chat_model, str):
            chat_model = load_from_model_id(chat_model)  # type: ignore
            self.chat_model = cast(BaseChatModel[T_P, T_O], chat_model)
        else:
            self.chat_model = chat_model
        self.async_capacity = async_capacity
        self.max_requests_per_minute = max_requests_per_minute
        self.error_mode = ErrorMode(error_mode)
        self.progress_bar_mode = ProgressBarMode(progress_bar)
        self._task_created_time_list: list[int] = []

    def run(self, prompts: Sequence[Prompt], override_parameters: T_P | None = None) -> Generator[ChatModelOutput, None, None]:
        progress_bar = self._get_progress_bar(num_tasks=len(prompts))
        for prompt in prompts:
            task_result = self._run_single_task(
                prompt=prompt, progress_bar=progress_bar, override_parameters=override_parameters
            )
            yield task_result

        progress_bar.close()

    def _run_single_task(
        self, prompt: Prompt, progress_bar: tqdm.tqdm[NoReturn], override_parameters: T_P | None = None
    ) -> ChatModelOutput:
        messages = ensure_messages(prompt)

        try:
            output = self.chat_model.chat_completion(messages=messages, override_parameters=override_parameters)
            if not output.is_cache:
                sleep_time = self._calculate_sleep_time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                self._task_created_time_list.append(int(time.time()))
            progress_bar.update(1)
            return output
        except BaseException as e:
            if self.error_mode is ErrorMode.RAISE:
                raise
            elif self.error_mode is ErrorMode.IGNORE:
                return ChatModelOutput(messages=[Message(role='error', content=str(e))])
            else:
                raise ValueError(f'Unknown error mode: {self.error_mode}') from e

    async def async_run(
        self, prompts: Sequence[Prompt], override_parameters: T_P | None = None
    ) -> AsyncGenerator[ChatModelOutput, None]:
        limiter = anyio.CapacityLimiter(self.async_capacity)
        task_created_lock = anyio.Lock()
        progress_bar = self._get_progress_bar(num_tasks=len(prompts))

        async with asyncer.create_task_group() as task_group:
            soon_values: list[asyncer.SoonValue[ChatModelOutput]] = []
            soon_func = task_group.soonify(self._async_run_single_task)
            for prompt in prompts:
                soon_value = soon_func(
                    prompt=prompt,
                    limiter=limiter,
                    task_created_lock=task_created_lock,
                    progress_bar=progress_bar,
                    override_parameters=override_parameters,
                )
                soon_values.append(soon_value)
            for soon_value in soon_values:
                while not soon_value.ready:
                    await anyio.sleep(0.01)
                yield soon_value.value

        progress_bar.close()

    def encapsulated_async_run(
        self, prompts: Sequence[Prompt], override_parameters: T_P | None = None
    ) -> list[ChatModelOutput]:
        async def proxy_function():
            results = [i async for i in self.async_run(prompts, override_parameters=override_parameters)]
            return results

        return anyio.run(proxy_function)

    async def _async_run_single_task(
        self,
        prompt: Prompt,
        limiter: anyio.CapacityLimiter,
        task_created_lock: anyio.Lock,
        progress_bar: tqdm.tqdm[NoReturn],
        override_parameters: T_P | None = None,
    ) -> ChatModelOutput:
        messages = ensure_messages(prompt)

        async with limiter:
            try:
                output = await self.chat_model.async_chat_completion(messages=messages, override_parameters=override_parameters)
                if not output.is_cache:
                    async with task_created_lock:
                        sleep_time = self._calculate_sleep_time()
                        if sleep_time > 0:
                            await anyio.sleep(sleep_time)
                        self._task_created_time_list.append(int(time.time()))
                progress_bar.update(1)
                return output
            except BaseException as e:
                if self.error_mode is ErrorMode.RAISE:
                    raise
                elif self.error_mode is ErrorMode.IGNORE:
                    return ChatModelOutput(messages=[Message(role='error', content=f'Error: {e}')])
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

    def _get_progress_bar(self, num_tasks: int) -> tqdm.tqdm[NoReturn]:
        use_progress_bar = (self.progress_bar_mode is ProgressBarMode.ALWAYS) or (
            self.progress_bar_mode is ProgressBarMode.AUTO and num_tasks > self.PROGRESS_BAR_THRESHOLD
        )
        progress_bar = tqdm.tqdm(desc=f'{self.chat_model.__class__.__name__}', total=num_tasks, disable=not use_progress_bar)
        return progress_bar
