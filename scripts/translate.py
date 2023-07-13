from __future__ import annotations

import json
from pathlib import Path

import typer

from lmclient import AzureChat, LMClient, OpenAIChat
from lmclient.client import ErrorMode


def read_from_jsonl(file: str | Path):
    texts: list[str] = []
    with open(file, 'r') as f:
        for line in f:
            texts.append(json.loads(line.strip())['text'])
    return texts


def main(
    input_josnl_file: Path,
    output_file: Path,
    model_name: str = 'gpt-3.5-turbo',
    max_requests_per_minute: int = 20,
    async_capacity: int = 3,
    error_mode: ErrorMode = ErrorMode.IGNORE,
    cache_dir: str = 'lmclient-translate-cache',
):

    if model_name == 'azure':
        model = AzureChat()
    else:
        model = OpenAIChat(model_name)

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
    completions = client.async_run(prompts)

    with open(output_file, 'w') as f:
        for text, completion in zip(texts, completions):
            f.write(json.dumps({'text': text, 'translation': completion}, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    typer.run(main)
