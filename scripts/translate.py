from __future__ import annotations

import json
from pathlib import Path

import typer
from anyio import open_file
from asyncer import runnify

from lmclient import CompletionEngine
from lmclient.completion_engine import ErrorMode
from lmclient.models import load_from_model_id


def read_from_text_file(file: str | Path) -> list[str]:
    file = Path(file)
    texts: list[str] = file.read_text().split('\n')
    return texts


@runnify
async def main(
    input_josnl_file: Path,
    output_file: Path,
    model_id: str = 'openai',
    max_requests_per_minute: int = 10,
    async_capacity: int = 3,
    error_mode: ErrorMode = 'raise',
) -> None:
    model = load_from_model_id(model_id=model_id)
    client = CompletionEngine(
        model,  # type: ignore
        max_requests_per_minute=max_requests_per_minute,
        async_capacity=async_capacity,
        error_mode=error_mode,
    )

    texts = read_from_text_file(input_josnl_file)
    prompts: list[str] = []
    for text in texts:
        prompt = f'translate following sentece to chinese\nsentence: {text}\ntranslation: '
        prompts.append(prompt)

    index = 0
    async with await open_file(output_file, 'w') as f:
        async for result in client.async_run(prompts):
            text = texts[index]
            await f.write(json.dumps({'text': text, 'translation': result.reply}, ensure_ascii=False) + '\n')
            await f.flush()
            index += 1


if __name__ == '__main__':
    typer.run(main)
