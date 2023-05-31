# lmclient

支持异步请求 OpenAI 接口的客户端

## Features

1. 支持异步请求 openai 接口
2. 支持进度条
3. 支持限制每分钟最大请求次数
4. 支持限制异步容量 （类似于线程池的大小）
5. 支持磁盘缓存

## Install

```shell
pip install lmclient-core
```

## Usage

```python
from lmclient import LMClient, AzureCompletion, OpenAICompletion

openai_completion = OpenAICompletion(model='gpt-3.5-turbo')
# azure_completion = AzureCompletion()
client = LMClient(openai_completion, async_capacity=5, max_requests_per_minute=20)
prompts = [
    'Hello, my name is',
    'can you please tell me your name?',
    'i want to know your name',
    'what is your name?',
]
values = client.async_run(prompts=prompts)
print(values)
```

## Advanced Usage

```python
# limit max_requests_per_minute to 20
# limit async_capacity to 5 (max 5 async requests at the same time)
# use cache
# set error_mode to ignore (ignore or raise)

from lmclient import LMClient, OpenAICompletion
openai_completion = OpenAICompletion(model='gpt-3.5-turbo', max_requests_per_minute=20, async_capacity=5, cache_dir='openai_cache', error_mode='ignore')
```