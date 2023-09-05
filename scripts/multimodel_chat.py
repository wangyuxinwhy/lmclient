import asyncio

from lmclient import ChatEngine, MinimaxProChat, OpenAIChat, WenxinChat, ZhiPuChat

chat_models = {
    'wenxin': WenxinChat(timeout=20),
    'llama2-70b': WenxinChat(model='llama_2_70b', timeout=20),
    'gpt4': OpenAIChat(model='gpt-4'),
    'gpt3.5': OpenAIChat(model='gpt-3.5-turbo'),
    'minimax': MinimaxProChat(),
    'zhipu': ZhiPuChat(),
}
engines = {model_name: ChatEngine(chat_model=chat_model) for model_name, chat_model in chat_models.items()}   # type: ignore


async def multimodel_chat(user_input: str):
    reply_list = await asyncio.gather(*[engine.async_chat(user_input) for engine in engines.values()])
    for model_name, reply in zip(engines.keys(), reply_list):
        print(f'{model_name}: {reply}')


if __name__ == '__main__':
    asyncio.run(multimodel_chat('你好'))
