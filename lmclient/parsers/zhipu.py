from lmclient.exceptions import ParserError
from lmclient.parsers.base import ModelResponseParser
from lmclient.types import ModelResponse


class ZhiPuParser(ModelResponseParser):
    def __call__(self, response: ModelResponse) -> str:
        try:
            output = response['data']['choices'][0]['content'].strip('"').strip()
        except (KeyError, IndexError) as e:
            raise ParserError('Parse response failed') from e
        return output
