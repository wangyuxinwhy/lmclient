from lmclient import ChatEngine, OpenAIChat, OpenAIChatParameters, function


@function
def get_weather(loc: str) -> str:
    """
    获取指定地区的天气信息

    Parameters:
        loc: 地区，比如北京，上海等
    """
    return f'{loc}，晴朗，27度'


@function
def google(keyword: str) -> str:
    """
    搜索谷歌

    Parameters:
        keyword: 搜索关键词
    """
    return '没有内容'


def test_function() -> None:
    model = OpenAIChat(parameters=OpenAIChatParameters(functions=[get_weather.json_schema, google.json_schema], temperature=0))
    engine = ChatEngine(model, functions=[get_weather, google], stream=False, function_call_raise_error=True)
    reply = engine.chat('今天北京天气怎么样？')
    assert '27' in reply
