from __future__ import annotations

import time
from typing import (
    Any,
    AsyncGenerator,
    ClassVar,
    Generator,
    Generic,
    Literal,
    NoReturn,
    TypedDict,
    TypeVar,
)

import anyio
import asyncer
import tqdm
from typing_extensions import Self, Unpack

from lmclient.chat_completion import ChatCompletionModel, ChatCompletionModelOutput, ModelParameters, load_from_model_id
from lmclient.chat_completion.message import Prompt, Prompts, ensure_messages

P = TypeVar('P', bound=ModelParameters)
ErrorMode = Literal['raise', 'ignore']
ProgressBarMode = Literal['auto', 'never', 'always']


class CompletionEngineKwargs(TypedDict):
    async_capacity: int
    max_requests_per_minute: int
    error_mode: ErrorMode
    progress_bar_mode: ProgressBarMode


class CompletionEngine(Generic[P]):
    """
    Args:
        chat_model (BaseChatModel[T_P]): The chat model to use for generating completions.
        async_capacity (int, optional): The maximum number of asynchronous requests that can be made at once. Defaults to 3.
        max_requests_per_minute (int, optional): The maximum number of requests that can be made per minute. Defaults to 20.
        error_mode (ErrorMode, optional): The error handling mode. Defaults to 'raise'.
        progress_bar_mode (ProgressBarMode, optional): The progress bar mode. Defaults to 'auto'.
    """

    NUM_SECONDS_PER_MINUTE: ClassVar[int] = 60
    PROGRESS_BAR_THRESHOLD: ClassVar[int] = 20

    def __init__(
        self,
        chat_model: ChatCompletionModel[P],
        async_capacity: int = 3,
        max_requests_per_minute: int = 20,
        error_mode: ErrorMode = 'raise',
        progress_bar_mode: ProgressBarMode = 'auto',
    ) -> None:
        self.chat_model = chat_model
        self.async_capacity = async_capacity
        self.max_requests_per_minute = max_requests_per_minute
        self.error_mode = error_mode
        self.progress_bar_mode = progress_bar_mode
        self._task_created_time_list: list[int] = []

    @classmethod
    def from_model_id(cls, model_id: str, **kwargs: Unpack[CompletionEngineKwargs]) -> Self:
        chat_model = load_from_model_id(model_id)
        return cls(chat_model, **kwargs)

    def run(self, prompts: Prompts, **kwargs: Any) -> Generator[ChatCompletionModelOutput, None, None]:
        progress_bar = self._get_progress_bar(num_tasks=len(prompts))
        for prompt in prompts:
            task_result = self._run_single_task(prompt=prompt, progress_bar=progress_bar, **kwargs)
            yield task_result
        progress_bar.close()

    def _run_single_task(
        self,
        prompt: Prompt,
        progress_bar: tqdm.tqdm[NoReturn],
        **kwargs: Any,
    ) -> ChatCompletionModelOutput:
        messages = ensure_messages(prompt)
        sleep_time = self._calculate_sleep_time()
        if sleep_time > 0:
            time.sleep(sleep_time)
        self._task_created_time_list.append(int(time.time()))

        try:
            output = self.chat_model.completion(prompt=messages, **kwargs)
        except Exception as e:
            if self.error_mode == 'raise':
                raise
            if self.error_mode == 'ignore':
                return ChatCompletionModelOutput(chat_model_id=self.chat_model.model_id, extra={'error': str(e)})
            raise ValueError(f'Unknown error mode: {self.error_mode}') from e
        else:
            progress_bar.update(1)
            return output

    async def async_run(self, prompts: Prompts, **kwargs: Any) -> AsyncGenerator[ChatCompletionModelOutput, None]:
        limiter = anyio.CapacityLimiter(self.async_capacity)
        task_created_lock = anyio.Lock()
        progress_bar = self._get_progress_bar(num_tasks=len(prompts))

        async with asyncer.create_task_group() as task_group:
            soon_values: list[asyncer.SoonValue[ChatCompletionModelOutput]] = []
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
            for soon_value in soon_values:
                while not soon_value.ready:
                    await anyio.sleep(0.01)
                yield soon_value.value

        progress_bar.close()

    async def _async_run_single_task(
        self,
        prompt: Prompt,
        limiter: anyio.CapacityLimiter,
        task_created_lock: anyio.Lock,
        progress_bar: tqdm.tqdm[NoReturn],
        **kwargs: Any,
    ) -> ChatCompletionModelOutput:
        messages = ensure_messages(prompt)

        async with limiter:
            try:
                async with task_created_lock:
                    sleep_time = self._calculate_sleep_time()
                    if sleep_time > 0:
                        await anyio.sleep(sleep_time)
                    self._task_created_time_list.append(int(time.time()))
                output = await self.chat_model.async_completion(messages, **kwargs)
            except Exception as e:
                if self.error_mode == 'raise':
                    raise
                if self.error_mode == 'ignore':
                    return ChatCompletionModelOutput(chat_model_id=self.chat_model.model_id, extra={'error': str(e)})

                raise ValueError(f'Unknown error mode: {self.error_mode}') from e
            else:
                progress_bar.update(1)
                return output

    def _calculate_sleep_time(self) -> int:
        idx = 0
        current_time = time.time()
        for i, task_created_time in enumerate(self._task_created_time_list):
            if current_time - task_created_time < self.NUM_SECONDS_PER_MINUTE:
                idx = i
                break
        self._task_created_time_list = self._task_created_time_list[idx:]

        if len(self._task_created_time_list) < self.max_requests_per_minute:
            return 0

        return max(self.NUM_SECONDS_PER_MINUTE - int(current_time - self._task_created_time_list[0]) + 1, 0)

    def _get_progress_bar(self, num_tasks: int) -> tqdm.tqdm[NoReturn]:
        use_progress_bar = (self.progress_bar_mode == 'always') or (
            self.progress_bar_mode == 'auto' and num_tasks > self.PROGRESS_BAR_THRESHOLD
        )
        return tqdm.tqdm(desc=f'{self.chat_model.__class__.__name__}', total=num_tasks, disable=not use_progress_bar)
