from lmclient.exceptions import ParserError
from lmclient.parsers.base import ModelResponseParser
from lmclient.types import ModelResponse


class MinimaxTextParser(ModelResponseParser):
    def __call__(self, response: ModelResponse) -> str:
        try:
            output = response['choices'][0]['message']['text']
        except (KeyError, IndexError) as e:
            raise ParserError('Parse response failed') from e
        return output
