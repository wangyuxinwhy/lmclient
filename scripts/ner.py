from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import List, Optional

import typer

from lmclient import AzureChat, LMClient, OpenAIChat
from lmclient.client import ErrorMode
from lmclient.parsers.openai import OpenAISchema


class ModelType(str, Enum):
    openai = 'openai'
    azure = 'azure'


class NerInfo(OpenAISchema):
    person: Optional[List[str]] = None
    location: Optional[List[str]] = None
    organization: Optional[List[str]] = None


def read_from_jsonl(file: str | Path):
    texts: list[str] = []
    with open(file, 'r') as f:
        for line in f:
            texts.append(json.loads(line.strip())['text'])
    return texts


def main(
    input_josnl_file: Path,
    output_file: Path,
    model_type: ModelType = ModelType.openai,
    max_requests_per_minute: int = 20,
    async_capacity: int = 3,
    error_mode: ErrorMode = ErrorMode.IGNORE,
    cache: bool = False,
):

    if model_type is ModelType.azure:
        model = AzureChat(
            'gpt-35-turbo-16k',
            api_version='2023-07-01-preview',
        )
    else:
        model = OpenAIChat('gpt-3.5-turbo')

    client = LMClient(
        model,
        max_requests_per_minute=max_requests_per_minute,
        async_capacity=async_capacity,
        error_mode=error_mode,
        output_parser=NerInfo.from_response,
    )
    if not cache:
        client.cache_dir = None

    texts = read_from_jsonl(input_josnl_file)
    prompts = []
    for text in texts:
        system_prompt = f'You are a NER model, extract entity information from the text.'
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': text},
        ]
        prompts.append(messages)
    schema = NerInfo.openai_schema()
    results = client.async_run(prompts, functions=[schema], function_call='auto')
    with open(output_file, 'w') as f:
        for text, result in zip(texts, results):
            if result.output is None:
                output = None
            else:
                output = result.output.model_dump()
            output_dict = {'text': text, 'output': output}
            f.write(json.dumps(output_dict, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    typer.run(main)
