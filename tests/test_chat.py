from lmclient import ChatEngine, OpenAIChat, function


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
    model = OpenAIChat()
    engine = ChatEngine(model, temperature=0, functions=[get_weather, google], stream=False)
    reply = engine.chat('今天北京天气怎么样？')
    assert '27' in reply
