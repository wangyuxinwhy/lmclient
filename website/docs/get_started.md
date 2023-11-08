---
sidebar_position: 1
---

# 快速开始

## 简介

LMClient 让你用最简单的方式调用最强大的语言模型！

LMClient 统一了不同平台的消息格式，推理参数，接口封装，返回解析。提供流式，非流式，同步，异步多种调用方式，支持输入检查，参数检查，添加了 *ChatEngine*, *CompletionEngine*, *function* 等多种辅助功能...

目前已支持：[OpenAI](https://openai.com/) | [Azure](https://azure.microsoft.com/) | [Minimax](https://api.minimax.chat/) | [百度文心](https://cloud.baidu.com/product/wenxinworkshop) | [百川](https://platform.baichuan-ai.com/docs/api) | [智谱](https://www.zhipuai.cn/) | [腾讯混元](https://cloud.tencent.com/product/hunyuan)


## 基础使用
open in colab
### 安装
```bash
pip install lmclient-core
```

### 初始化模型

```python
from lmclient import OpenAIChat


gpt35 = OpenAIChat(api_key='<your openai api key>')
# gpt4 = OpenAIChat(model='gpt-4')
model_response = gpt35.chat_completion('你好')
```

当你没有指定 api_key 时，LMClient 会尝试从环境变量中导入。

除了显式的引入 `OpenAIChat` 类外，您还可以通过 model_id 来初始化模型，详情参考 [模型列表](./model_list.md)

```python
from lmclient import load_from_model_id

model = load_from_model_id('openai/gpt-4')
model_response = model.chat_completion('你好')
```

### 修改推理参数

- 在模型初始化时，设置默认的推理参数

```python
from lmclient import OpenAIChat, OpenAIChatParameters

model = OpenAIChat(api_key='<your openai api key>', parameters=OpenAIChatParameters(temperature=0))
model_response = model.chat_completion('你好')
```

- 在方法调用时，设置动态的推理参数

```python
model_response = model.chat_completion('你好', temperature=0)
```

### 流式推理

使用 `stream_chat_completion` 方法进行流式推理

```python
for stream_response in model.stream_chat_completion('你好'):
    print(stream_response.stream.delta, end='', flush=True)
    if stream_response.is_finish:
        response = stream_response
```

此外，LMClient 还支持异步推理和异步流式推理，详情参考 [进阶使用](./advanced_usage.md)


### ChatEngine 聊天引擎

`chat_completion` 方法是无状态的，即每次调用都是独立的，不会记住上一次的对话内容。

ChatEngine 聊天引擎，为了聊天场景而设计，添加了对话历史管理，处理 *function call* ，流式打印等功能

```python
from lmclient import MinimaxChat, ChatEngine, function


bot = ChatEngine(MinimaxChat())
bot.chat('你好，我叫 lmclient')
bot.chat('我是谁？')
```

输出

```
user: 你好，我叫 lmclient
assistant: 你好，lmclient，我是MM智能助理，很高兴认识你！
user: 我是谁？
assistant: 你是lmclient，一个由MiniMax自研的大型语言模型。
```

### CompletionEngine 补全引擎

CompletionEngine 补全引擎，为批量生成或补全任务而设计，在 `chat_completion` 的功能之上，添加了异步并发控制，每分钟最大请求控制，进度条等功能。

- `run` 方法同步批量运行
- `async_run` 方法异步批量运行

```python
import asyncio

from lmclient import CompletionEngine, MinimaxChat

async def main():
    model = MinimaxChat()
    # 控制每分钟最大请求次数为 20， 异步并发为 5
    completion_engine = CompletionEngine(model, async_capacity=5, max_requests_per_minute=20)
    prompts = [
        'Hello, my name is',
        'can you please tell me your name?',
        'what is your name?',
    ]
    async for response in completion_engine.async_run(prompts=prompts):
        print(response.reply)

asyncio.run(main())
```

### function call 函数调用

LMClient 提供了 function call 的集成，并提供了 `@function` 装饰器，经过装饰的 Python 函数可以自动的生成 jsonschema ，进而帮助简化 function call 工作流。


```python
from lmclient import OpenAIChat, ChatEngine, function

@function
def get_weather(city: str) -> str:
    """Get weather of the city."""
    return f'{city}天气晴朗, 气温20度'

bot = ChatEngine(OpenAIChat(), functions=[get_weather])
bot.chat('今天北京天气怎么样？')
```

输出

```
user: 今天北京天气怎么样？
Function call: get_weather
Arguments: {
    "city": "北京"
}
function: "北京天气晴朗, 气温20度"
assistant: 今天北京天气晴朗，气温20度。
```

:::warning
不是所有平台的模型都支持 function call，目前只有 百度文心，MinimaxPro， OpenAI 支持
:::


### 多模型调用

只调用一个模型显示不出统一接口的魅力，我们来调用多个模型，看看 LMClient 是如何统一不同平台的接口的。

```python
from lmclient import MinimaxChat, MinimaxProChat, OpenAIChat, WenxinChat, ZhiPuChat, HunyuanChat, BaichuanChat, ZhiPuCharacterChat

chat_models = [
    MinimaxChat(),
    MinimaxProChat(),
    OpenAIChat(),
    WenxinChat(),
    ZhiPuChat(),
    ZhiPuCharacterChat(),
    HunyuanChat(),
    BaichuanChat(),
]

for model in chat_models:
    model_response = model.chat_completion('你好，你是谁？请介绍一下你自己', temperature=0)
    print(f'{model.model_id}: {model_response.reply}')
```

输出

```
minimax/abab5.5-chat: 你好！有什么我可以帮助你的吗？
minimax_pro/abab5.5-chat: 你好！有什么我可以帮助你的吗？
openai/gpt-3.5-turbo: 你好！有什么我可以帮助你的吗？
wenxin/ERNIE-Bot: 你好，很高兴和你交流。有什么我可以帮助你的吗？
zhipu/chatglm_turbo: " 你好👋！我是人工智能助手智谱清言，可以叫我小智🤖，很高兴见到你，欢迎问我任何问题。"
zhipu-character/characterglm: "（有点尴尬）你好，不好意思，我好像走错化妆间了\n"
hunyuan/v1: 你好！很高兴见到你。请问有什么我可以帮助你的？
zhipu/Baichuan2-53B: 你好！很高兴和你交流。请问有什么问题我可以帮助你解决吗？
```

## 之后呢？

- 如果你对 LMClient 支持的模型感兴趣，可以查看 [模型列表](./model_list.md)
- 如果你对 ChatEngine，CompletionEngine，function call 等功能感兴趣，可以查看 [进阶使用](./advanced_usage.md)
- 如果你对 LMClient 的诞生，设计以及愿景感兴趣，可以查看 [LMClient Blog](/blog)