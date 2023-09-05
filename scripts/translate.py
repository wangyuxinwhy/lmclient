from __future__ import annotations

import json
from pathlib import Path

import typer

from lmclient import CompletionEngine
from lmclient.completion_engine import ErrorMode
from lmclient.models import load_from_model_id


def read_from_text_file(file: str | Path):
    file = Path(file)
    texts: list[str] = file.read_text().split('\n')
    return texts


def main(
    input_josnl_file: Path,
    output_file: Path,
    model_id: str = 'openai',
    max_requests_per_minute: int = 5,
    async_capacity: int = 3,
    error_mode: ErrorMode = ErrorMode.RAISE,
    use_cache: bool = True,
) -> None:
    model = load_from_model_id(model_id=model_id, use_cache=use_cache)
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
    results = client.async_run(prompts)

    with open(output_file, 'w') as f:
        for text, result in zip(texts, results):
            f.write(json.dumps({'text': text, 'translation': result.reply}, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    typer.run(main)
