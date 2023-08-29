from __future__ import annotations

from tenacity import retry, stop_after_attempt, wait_random_exponential

from lmclient.types import ModelResponse, Prompt


class BaseChatModel:
    def chat(self, prompt: Prompt, **kwargs) -> ModelResponse:
        ...

    async def async_chat(self, prompt: Prompt, **kwargs) -> ModelResponse:
        ...

    def chat_with_retry(self, prompt: Prompt, max_wait: int = 20, max_attempt: int = 3, **kwargs) -> ModelResponse:
        return retry(wait=wait_random_exponential(min=1, max=max_wait), stop=stop_after_attempt(max_attempt))(self.chat)(
            prompt=prompt, **kwargs
        )

    async def async_chat_with_retry(self, prompt: Prompt, max_wait: int = 20, max_attempt: int = 3, **kwargs) -> ModelResponse:
        return await retry(wait=wait_random_exponential(min=1, max=max_wait), stop=stop_after_attempt(max_attempt))(
            self.async_chat
        )(prompt=prompt, **kwargs)

    @property
    def identifier(self) -> str:
        self_dict_string = ', '.join(f'{k}={v}' for k, v in self.__dict__.items())
        return f'{self.__class__.__name__}({self_dict_string})'
