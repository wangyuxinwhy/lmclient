import json
from typing import Any


class MessageError(Exception):
    pass


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
