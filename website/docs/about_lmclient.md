# LMClient 从何而来

国内的大模型厂家百花齐放，国外也是层出不穷。但不同平台之间的信息格式，调用方式，推理参数也是五花八门，缺少统一。如果你想要在业务或者应用中支持不同的平台，你需要安装各种 SDK，再开发不同的适配代码！这样的工作，繁琐，易错，也很无聊。

那么，开源社区中有没有现成的工具来解决这个问题呢？经过一番调研之后，只找到了一个没有活跃开发的项目 [LLM-Fusion_API](https://github.com/ninehills/LLM-Fusion-API) ，以及一个只覆盖了国外主流大模型平台的项目 [litellm](https://github.com/BerriAI/litellm)。

`litellm` 看起来相当不错，那直接给 litellm 提几个 PR 来覆盖国内的大模型平台吧！但是当我看到 [litellm.completion](https://github.com/BerriAI/litellm/blob/v0.11.1/litellm/main.py#L158) 这个 function 包含 1000+ 行代码后，我放弃了这个天真的想法 😄。

现在只剩下最后一条路了，既然目前开源社区中没有满足我需求的项目，那我自己写一个吧！

LMClient 由此诞生 🎉

## LMClient 设计

在决定自己写 LMClient 后，面对的第一个问题，就是要搞清楚一个好用的 **大语言模型客户端** 应该具备哪些性质？

我简单的罗列一下：

1. **简单**：无需关心细节，只需要一个统一的接口
2. **易用**：提供开箱即用的功能，如 *ChatEngine*, *CompletionEngine*, *function call* 等
3. **高效**：支持流式/非流式推理，以及异步/同步推理
4. **灵活**：支持模型参数，*functions*，*system prompt* 等设置
5. **可靠**：输入检查，参数检查，异常处理
6. **丰富**：尽可能多的支持各种平台

LMClient 的设计也由此展开~

### 简单

LMClient 提供的模型都使用统一的 `chat_completion` 进行调用，LMClient 在内部统一了不同平台之间的消息格式，推理参数，接口封装，返回解析等！

现在，你可以写出像下方一样的代码，是不是很简单？

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

### 易用

LMClient 提供了开箱即用的功能，如 ChatEngine, CompletionEngine, function 等

- **ChatEngine**: 聊天引擎，`chat_completion` 是无状态的，即每次调用都是独立的，不会记住上一次的对话内容。而 **ChatEngine** 则添加了管理对话历史，处理 *function call* ，流式打印等功能
- **CompletionEngine**: 补全引擎，在 `chat_completion` 的功能之上，添加了异步并发控制，每分钟最大请求控制，进度条等功能
- **function**: 一个装饰器，可以将任意 Python 函数转换为的 jsonschema ，方便 *function call* 的使用
- ...

### 高效

LMClient 支持流式/非流式推理，以及异步/同步推理，注意这里的 api 名称有所变化，而命名是非常规律的。*async* 前缀代表异步，*stream* 前缀代表流式。



- 非流式同步推理
    ```python {4}
    from lmclient import MinimaxChat

    model = MinimaxChat()
    response = model.chat_completion('你好')
    print(response.reply)
    ```

- 流式同步推理 
    ```python {4}
    from lmclient import MinimaxChat

    model = MinimaxChat()
    stream_responses = model.stream_chat_completion('你好')
    for stream_response in stream_responses:
        print(stream_response.stream.delta, end='', flush=True)
        if stream_response.is_finish:
            response = stream_response
    ```

- 非流式异步推理
    ```python {6}
    import asyncio

    from lmclient import MinimaxChat

    model = MinimaxChat()
    co_response = model.async_chat_completion('你好')
    response = asyncio.run(co_response)
    print(response)
    ```

- 流式异步推理
    ```python {7}
    import asyncio

    from lmclient import MinimaxChat

    model = MinimaxChat()
    async def async_chat():
        async for stream_response in model.async_stream_chat_completion('你好'):
            if stream_response.is_finish:
                response = stream_response
                return response
            else:
                print(stream_response.stream.delta, end='', flush=True)
    response = asyncio.run(async_chat())
    print(response)
    ```


### 灵活

LMClient 支持推理参数的设置，这方面的细节很多。例如 **max_tokens** 在不同的平台有不同的名称，**temperature** 在有些平台不能设置为 0 等等。不过，作为用户你无需关心这些细节，LMClient 会为你处理所有麻烦。

```python {5}
from lmclient import MinimaxChat, OpenAIChat

models = [MinimaxChat(), OpenAIChat()]
for model in models:
    model_response = model.chat_completion('你好', temperature=0, max_tokens=10)
    print(f'{model.model_id}: {model_response.reply}')
```

输出

```
minimax/abab5.5-chat: 你好！有什么我可以帮助你的吗？
openai/gpt-3.5-turbo: 你好！有什么我可以
```

### 可靠

LMClient 内部对输入，模型参数，异常等进行了检查和处理，保证了代码的可靠性。

当你错误的将 **temperature** 设置为负数时，你将会得到如下的错误提示。

```python
from lmclient import OpenAIChat

model = OpenAIChat()
model.chat_completion('你好', temperature=-1)
```

输出

```
ValidationError: 1 validation error for OpenAIChatParameters
temperature
  Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]
    For further information visit https://errors.pydantic.dev/2.4/v/greater_than_equal
```

### 丰富

LMClient 当前支持以下 **7** 个平台的 **9** 种模型，注意有同一平台会提供不同种类的模型，比如 `ZhiPuCharacterChat` 就专门为角色扮演而生。

当然同种类型模型还有不同的版本，比如百度有 *Ernie-Bot-4* 和 *Ernie-Bot-3.5* 等。

完成的列表如下：

- MinimaxChat
- MinimaxProChat
- AzureChat
- OpenAIChat
- ZhiPuChat
- ZhiPuCharacterChat
- WenxinChat
- HunyuanChat
- BaichuanChat

LMClient 还在丰富中，欢迎大家提 PR 或者 Issue 来支持更多的平台！


## LMClient 代码

### 可读性

LMClient 的代码使用 [ruff](https://docs.astral.sh/ruff/) 进行详尽的代码风格检查和格式化，使用 [pyright](https://github.com/microsoft/pyright) 做静态类型检查，并做到了 100% 的类型注解 (type hints)。因此，当您使用 LMClient 进行开发时，你将获得完美的开发体验。

总之，LMClient 的代码在可读性方面做了最大的努力，代码的设计也是简洁易懂的，哪怕深入源码，也不会让你感到困惑。

### 测试

LMClient 的代码使用 `pytest` 进行单元测试，覆盖了 LMClient 所有公开提供的接口。

