from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import List

import typer

from lmclient import LMClient, OpenAIExtract
from lmclient.client import ErrorMode
from lmclient.openai_schema import Field, OpenAISchema


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
    max_requests_per_minute: int = 20,
    async_capacity: int = 3,
    error_mode: ErrorMode = ErrorMode.RAISE,
    use_cache: bool = False,
):

    model = OpenAIExtract(
        schema=NerInfo,
        use_cache=use_cache,
    )

    client = LMClient(
        model,
        max_requests_per_minute=max_requests_per_minute,
        async_capacity=async_capacity,
        error_mode=error_mode,
    )
    texts = read_from_jsonl(input_josnl_file)
    model_outputs = client.async_run(texts)
    with open(output_file, 'w') as f:
        for text, output in zip(texts, model_outputs):
            output = output.message.dict() if output.message else None
            output_dict = {'text': text, 'output': output}
            f.write(json.dumps(output_dict, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    typer.run(main)
