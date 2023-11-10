import json
from typing import Any, Sequence, Type

from lmclient.message import Message


class MessageError(Exception):
    ...


class MessageTypeError(MessageError):
    def __init__(self, invalid_message: Message, allowed_message_type: Sequence[Type[Message]], *args: object) -> None:
        message = f'invalid message type: {type(invalid_message)}, only {tuple(allowed_message_type)} is allowed'
        super().__init__(message, *args)


class MessageValueError(MessageError):
    def __init__(self, invalid_message: Message, *args: object) -> None:
        message = f'invalid message value: {invalid_message}'
        super().__init__(message, *args)


class UnexpectedResponseError(Exception):
    """
    Exception raised when an unexpected response is received from the server.

    Attributes:
        response (dict): The response from the server.
    """

    def __init__(self, response: dict, *args: Any) -> None:
        try:
            message = json.dumps(response, indent=4, ensure_ascii=False)
        except TypeError:
            message = str(response)
        super().__init__(message, *args)
