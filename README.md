# lmclient

[english](./README-en.md)

面向于大规模异步请求 OpenAI 接口设计的客户端，使用场景 self-instruct, 大规模翻译等

## Features

1. 支持大规模异步请求 openai 接口
2. 支持进度条
3. 支持限制每分钟最大请求次数
4. 支持限制异步容量 （类似于线程池的大小）
5. 支持磁盘缓存
6. 100% type hints
7. 非常易用

## 安装方式
支持 python3.8 及以上
```shell
pip install lmclient-core
```

## 使用方法

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

## 使用样例： 大规模翻译

项目作者已经使用了此脚本通过 OpenAI 翻译了 10W+ 的句对数据集，运行非常稳定和流畅。

### 准备工作
您需要进入 `scripts` 目录下，并安装 `typer`, 执行 `pip install "typer[all]"` 即可。

### 查看帮助
通过 `python translate.py --help` 可以查看帮助

![](https://yuxin-wang.oss-cn-beijing.aliyuncs.com/uPic/AxbBw5.png)

### 执行脚本

通过如下命令执行翻译脚本，此脚本将会把 `translate_input.jsonl` 文件中每一行翻译成中文，并且输出到 `output.jsonl` 文件中。当然，在实际使用时，您需要指定成自己的输入文件，格式相同就可以了。

```shell
python translate.py data/input.jsonl output.jsonl
```

#### input.jsonl
```json
{"text": "players who have scored 5 goals in world cup finals"}
{"text": "where was christianity most strongly established by ad 325"}
{"text": "when was the last time turkey was in the world cup"}
```

#### ouptut.jsonl
```json
{"text": "players who have scored 5 goals in world cup finals", "translation": "在世界杯决赛中打进5个进球的球员"}
{"text": "where was christianity most strongly established by ad 325", "translation": "在325年前，基督教在哪个地方最为稳固？"}
{"text": "when was the last time turkey was in the world cup", "translation": "土耳其上一次参加世界杯是什么时间？"}
```

### 核心代码

翻译脚本可以在 [translate.py](./scripts/translate.py) 中找到，核心代码如下

```python
# 核心代码
client = LMClient(
        model,
        max_requests_per_minute=max_requests_per_minute,
        async_capacity=async_capacity,
        error_mode=error_mode,
        cache_dir=cache_dir,
    )

texts = read_from_jsonl(input_josnl_file)
prompts = []
for text in texts:
    prompt = f'translate following sentece to chinese\nsentence: {text}\ntranslation: '
    prompts.append(prompt)
results = client.async_run(prompts)
```

## 使用样例： self-instruct

TODO: 待补充 self-instruct 脚本

## 进阶使用

```python
# limit max_requests_per_minute to 20
# limit async_capacity to 5 (max 5 async requests at the same time)
# use cache
# set error_mode to ignore (ignore or raise)

from lmclient import LMClient, OpenAIChat
model = OpenAIChat('gpt-3.5-turbo')
client = LMClient(model, max_requests_per_minute=20, async_capacity=5, cache_dir='openai_cache', error_mode='ignore', timeout=20)
```
