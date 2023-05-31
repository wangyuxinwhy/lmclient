# lmclient

LM Async Client, OpenAI, Azure ...


## Install

```shell
pip install lmclient
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
