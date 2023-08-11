from __future__ import annotations

import hashlib
import os
import time
from enum import Enum
from pathlib import Path
from typing import ClassVar, Generic, Sequence, Type, TypeVar, cast

import anyio
import asyncer
import diskcache
import tqdm

from lmclient.models import AzureChat, BaseChatModel, OpenAIChat
from lmclient.parsers import MinimaxTextParser, ModelResponseParser, OpenAIParser, OpenAISchema
from lmclient.types import Messages, ModelResponse, TaskResult
from lmclient.version import __cache_version__

DEFAULT_CACHE_DIR = Path(os.getenv('LMCLIENT_CACHE_DIR', '~/.cache/lmclient')).expanduser().resolve()
DEFAULT_MODEL_PARSER_MAP: dict[str, Type[ModelResponseParser]] = {
    'OpenAIChat': OpenAIParser,
    'AzureChat': OpenAIParser,
    'MinimaxChat': MinimaxTextParser,
}

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
        model: BaseChatModel,
        max_requests_per_minute: int = 20,
        async_capacity: int = 3,
        timeout: int | None = None,
        error_mode: ErrorMode | str = ErrorMode.RAISE,
        cache_dir: Path | str | None = DEFAULT_CACHE_DIR,
        progress_bar: ProgressBarMode | str = ProgressBarMode.AUTO,
        output_parser: ModelResponseParser[T] | None = None,
    ):
        self.model = model
        self.async_capacity = async_capacity
        self.max_requests_per_minute = max_requests_per_minute
        self.error_mode = ErrorMode(error_mode)
        self.progress_bar_mode = ProgressBarMode(progress_bar)
        self._task_created_time_list: list[int] = []

        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.output_parser = output_parser or DEFAULT_MODEL_PARSER_MAP[self.model.__class__.__name__]()

        if timeout is not None:
            if hasattr(self.model, 'timeout'):
                setattr(self.model, 'timeout', timeout)
            else:
                raise ValueError(f'Model {self.model} does not support timeout')

    @property
    def cache_dir(self) -> Path | None:
        return getattr(self, '_cache_dir')

    @cache_dir.setter
    def cache_dir(self, value: Path | None) -> None:
        if value is not None:
            if value.exists() and not value.is_dir():
                raise ValueError(f'Cache directory {value} is not a directory')
            value.mkdir(parents=True, exist_ok=True)
            self._cache = diskcache.Cache(value)
        else:
            self._cache = None

    def run(self, prompts: Sequence[str | Messages], **kwargs) -> list[TaskResult[T]]:
        progress_bar = self._get_progress_bar(num_tasks=len(prompts))
        task_results: list[TaskResult] = []
        for prompt in prompts:
            task_result = self._run_single_task(prompt=prompt, progress_bar=progress_bar, **kwargs)
            task_results.append(task_result)
        progress_bar.close()
        return task_results

    async def _async_run(self, prompts: Sequence[str | Messages], **kwargs) -> list[TaskResult[T]]:
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

    def async_run(self, prompts: Sequence[str | Messages], **kwargs) -> list[TaskResult[T]]:
        return asyncer.runnify(self._async_run)(prompts, **kwargs)

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
                    if self._cache is not None:
                        self._cache[task_key] = response
                except BaseException as e:
                    if self.error_mode is ErrorMode.RAISE:
                        raise
                    elif self.error_mode is ErrorMode.IGNORE:
                        return TaskResult(error_message=str(e))
                    else:
                        raise ValueError(f'Unknown error mode: {self.error_mode}')

            try:
                output = self.output_parser(response)
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
                if self._cache is not None:
                    self._cache[task_key] = response
            except BaseException as e:
                if self.error_mode is ErrorMode.RAISE:
                    raise
                elif self.error_mode is ErrorMode.IGNORE:
                    return TaskResult(error_message=str(e))
                else:
                    raise ValueError(f'Unknown error mode: {self.error_mode}')

        try:
            output = self.output_parser(response)
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
        if self._cache is not None and key in self._cache:
            response = self._cache[key]
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
            prompt = '---'.join([f'{k}={v}' for message in prompt for k, v in message.items()])
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


class LMClientForStructuredData(LMClient[T_O]):
    SupportedModels = [OpenAIChat, AzureChat]

    def __init__(
        self,
        model: BaseChatModel,
        schema: Type[T_O],
        system_prompt: str = 'Generate structured data from a given text',
        max_requests_per_minute: int = 20,
        async_capacity: int = 3,
        timeout: int | None = None,
        error_mode: ErrorMode | str = ErrorMode.RAISE,
        cache_dir: Path | str | None = DEFAULT_CACHE_DIR,
        progress_bar: ProgressBarMode | str = ProgressBarMode.AUTO,
    ):
        if not any([isinstance(model, supported_model) for supported_model in self.SupportedModels]):
            raise ValueError(f'Unsupported model: {model.__class__.__name__}. Supported models: {self.SupportedModels}')
        self.system_prompt = system_prompt
        self.default_kwargs = {
            'functions': [schema.openai_schema()],
            'function_call': {'name': schema.openai_schema()['name']},
        }

        super().__init__(
            model=model,
            max_requests_per_minute=max_requests_per_minute,
            async_capacity=async_capacity,
            timeout=timeout,
            error_mode=error_mode,
            cache_dir=cache_dir,
            progress_bar=progress_bar,
            output_parser=schema.from_response,
        )

    def run(self, prompts: Sequence[str], **kwargs) -> list[TaskResult[T_O]]:
        assembled_prompts = []
        for prompt in prompts:
            messages = [
                {'role': 'system', 'text': self.system_prompt},
                {'role': 'user', 'text': prompt},
            ]
            assembled_prompts.append(messages)
        kwargs = {**self.default_kwargs, **kwargs}
        print(kwargs)
        return super().run(prompts, **kwargs)

    async def _async_run(self, prompts: Sequence[str | Messages], **kwargs) -> list[TaskResult[T_O]]:
        assembled_prompts = []
        for prompt in prompts:
            messages = [
                {'role': 'system', 'text': self.system_prompt},
                {'role': 'user', 'text': prompt},
            ]
            assembled_prompts.append(messages)
        kwargs = {**self.default_kwargs, **kwargs}
        return await super()._async_run(prompts, **kwargs)
