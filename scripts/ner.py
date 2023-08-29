from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import List

import typer

from lmclient import AzureChat, Field, LMClientForStructuredData, OpenAIChat, OpenAISchema
from lmclient.client import ErrorMode


class ModelType(str, Enum):
    openai = 'openai'
    azure = 'azure'


class NerInfo(OpenAISchema):
    """命名实体信息，包括人名，地名和组织名"""

    person: List[str] = Field(default_factory=list)
    location: List[str] = Field(default_factory=list)
    organization: List[str] = Field(default_factory=list)


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

    client = LMClientForStructuredData(
        model,
        schema=NerInfo,
        system_prompt='You are a NER model, extract entity information from the text.',
        max_requests_per_minute=max_requests_per_minute,
        async_capacity=async_capacity,
        error_mode=error_mode,
    )
    if not cache:
        client.cache_dir = None

    texts = read_from_jsonl(input_josnl_file)
    results = client.async_run(texts)
    with open(output_file, 'w') as f:
        for text, result in zip(texts, results):
            if result.output is None:
                output = None
            else:
                try:
                    output = result.output.model_dump()
                except AttributeError:
                    output = result.output.dict()
            output_dict = {'text': text, 'output': output}
            f.write(json.dumps(output_dict, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    typer.run(main)
