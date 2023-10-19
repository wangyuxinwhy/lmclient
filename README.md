# lmclient

面向于大规模异步请求 OpenAI 接口设计的客户端，使用场景 self-instruct, 大规模翻译等

## Features

1. 支持大规模异步请求 openai 接口
2. 支持进度条
3. 支持限制每分钟最大请求次数
4. 支持限制异步容量 （类似于线程池的大小）
5. 支持磁盘缓存
6. 100% type hints
7. 非常易用
8. 支持 OpenAI, Azure, Minimax, MinimaxPro, 智谱, 百度文心, 腾讯混元
9. 支持 FunctionCall

## 安装方式
支持 python3.8 及以上
```shell
pip install lmclient-core
```

## 使用方法

1. CompletionEngine
```python
from lmclient import CompletionEngine, OpenAIChat, OpenAIChatParameters

model = OpenAIChat('gpt-3.5-turbo',  parameters=OpenAIChatParameters(temperature=0))
# 控制每分钟最大请求次数为 20， 异步容量为 5
client = CompletionEngine(model, async_capacity=5, max_requests_per_minute=20)
prompts = [
    'Hello, my name is',
    'can you please tell me your name?',
    [{'role': 'user', 'content': 'hello, who are you?'}],
    'what is your name?',
]
outputs = client.async_run(prompts=prompts)
for output in outputs:
    print(output.reply)
```

2. ChatEngine
```python
from lmclient import ChatEngine, OpenAIChat

model = OpenAIChat('gpt-3.5-turbo')
chat_engine = ChatEngine(model)
print(chat_engine.chat('你好，我是 chat_engine'))
print(chat_engine.chat('我上一句话是什么？'))
```
