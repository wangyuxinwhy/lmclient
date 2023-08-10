from __future__ import annotations

from lmclient.types import Messages, ModelResponse


class BaseChatModel:
    def chat(self, prompt: Messages | str, **kwargs) -> ModelResponse:
        ...

    async def async_chat(self, prompt: Messages | str, **kwargs) -> ModelResponse:
        ...

    @property
    def identifier(self) -> str:
        self_dict_string = ', '.join(f'{k}={v}' for k, v in self.__dict__.items())
        return f'{self.__class__.__name__}({self_dict_string})'
