from typing import Protocol

from lmclient.types import Message, Stream
from lmclient.utils import is_function_call_message, is_text_message


class Printer(Protocol):
    def print_message(self, message: Message) -> None:
        ...

    def print_stream(self, stream: Stream) -> None:
        ...


class SimplePrinter(Printer):
    def print_message(self, message: Message) -> None:
        if is_text_message(message):
            print(f'{message["role"]}: {message["content"]}')
        elif is_function_call_message(message):
            print(f'Function call: {message["content"]["name"]}\nArguments: {message["content"]["arguments"]}')
        else:
            raise RuntimeError(f'Invalid message type: {type(message)}')

    def print_stream(self, stream: Stream) -> None:
        if stream.control == 'start':
            print('assistant: ', end='', flush=True)
        print(stream.delta, end='', flush=True)
        if stream.control == 'finish':
            print()
