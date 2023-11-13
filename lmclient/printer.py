# ruff: noqa: T201
import time
from typing import Protocol

import rich

from lmclient.chat_completion.message import (
    AssistantMessage,
    FunctionCallMessage,
    FunctionMessage,
    Message,
    ToolCallsMessage,
    ToolMessage,
    UserMessage,
)
from lmclient.chat_completion.model_output import Stream


class Printer(Protocol):
    def print_message(self, message: Message) -> None:
        ...

    def print_stream(self, stream: Stream) -> None:
        ...


class SimplePrinter(Printer):
    """
    A simple printer that prints messages and streams to the console.

    Args:
        smooth (bool, optional): Whether to use smooth printing. Defaults to True.
        interval (float, optional): The interval between each print. Defaults to 0.03.
    """

    def __init__(self, smooth: bool = True, interval: float = 0.03) -> None:
        self.smooth = smooth
        self.interval = interval

    def print_message(self, message: Message) -> None:
        if isinstance(message, (UserMessage, AssistantMessage, FunctionMessage, ToolMessage)):
            print(f'{message.role}: {message.content}')
        elif isinstance(message, FunctionCallMessage):
            print(f'Function call: {message.content.name}\nArguments: {message.content.arguments}')
        elif isinstance(message, ToolCallsMessage):
            for tool_call in message.content:
                print(
                    f'Tool call: {tool_call.id}\nFunction: {tool_call.function.name}\nArguments: {tool_call.function.arguments}'
                )
        else:
            raise TypeError(f'Invalid message type: {type(message)}')

    def print_stream(self, stream: Stream) -> None:
        if stream.control == 'start':
            print('assistant: ', end='', flush=True)
        if self.smooth:
            for char in stream.delta:
                print(char, end='', flush=True)
                time.sleep(self.interval)
        else:
            print(stream.delta, end='', flush=True)
        if stream.control == 'finish':
            print()


class RichPrinter(Printer):
    """
    A rich printer that prints messages and streams to the console.

    Args:
        smooth (bool, optional): Whether to use smooth printing. Defaults to True.
        interval (float, optional): The interval between each print. Defaults to 0.03.
    """

    def __init__(self, smooth: bool = True, interval: float = 0.03) -> None:
        self.smooth = smooth
        self.interval = interval

    def print_message(self, message: Message) -> None:
        if isinstance(message, UserMessage):
            rich.print(f'🤠 : [green]{message.content}[/green]')
        elif isinstance(message, AssistantMessage):
            rich.print(f'🤖 : {message.content}')
        elif isinstance(message, FunctionCallMessage):
            rich.print(f'🔀 : [blue] Name:{message.content.name}\nArguments: {message.content.arguments}[/blue]')
        elif isinstance(message, (FunctionMessage, ToolMessage)):
            rich.print(f'🔀 : [blue]{message.content}[/blue]')
        elif isinstance(message, ToolCallsMessage):
            for tool_call in message.content:
                rich.print(
                    f'🔀 : [blue]Tool call: {tool_call.id}\nFunction: {tool_call.function.name}\nArguments: {tool_call.function.arguments}[/blue]'
                )
        else:
            raise TypeError(f'Invalid message type: {type(message)}')

    def print_stream(self, stream: Stream) -> None:
        if stream.control == 'start':
            print('🤖 : ', end='', flush=True)
        if self.smooth:
            for char in stream.delta:
                print(char, end='', flush=True)
                time.sleep(self.interval)
        else:
            print(stream.delta, end='', flush=True)
        if stream.control == 'finish':
            print()
