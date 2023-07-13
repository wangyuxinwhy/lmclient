# lmclient

LM Async Client, OpenAI, Azure ...

## Features

1. support asynchronous request openai interface
2. support progress bar
3. support limit max requests per minute
4. support limit async capacity
5. support disk cache

## Install

```shell
pip install lmclient-core
```

## Usage

```python
from lmclient import LMClient, OpenAIChat

model = OpenAIChat('gpt-3.5-turbo')
# 控制每分钟最大请求次数为 20， 异步容量为 5
client = LMClient(model, async_capacity=5, max_requests_per_minute=20)
prompts = [
    'Hello, my name is',
    'can you please tell me your name?',
    [{'role': 'system', 'content': 'your are lmclient demo assistant'}, {'role': 'user', 'content': 'hello, who are you?'}],
    'what is your name?',
]
values = client.async_run(prompts=prompts, temperature=0)
print(values)
```

## Advanced Usage

```python
# limit max_requests_per_minute to 20
# limit async_capacity to 5 (max 5 async requests at the same time)
# use cache
# set error_mode to ignore (ignore or raise)

from lmclient import LMClient, OpenAIChat
model = OpenAIChat('gpt-3.5-turbo')
client = LMClient(model, max_requests_per_minute=20, async_capacity=5, cache_dir='openai_cache', error_mode='ignore')
```