"""
A script that performs an experiment using various LLM models to test their ability to recall and predict
text from Herman Melville's Moby-Dick.

Requires the text file to be present at `.input/pg2701.txt`

Usage:
    python moby_dick.py collect  # Gather predictions from LLMs
    python moby_dick.py analyze  # Analyze results and generate reports

See also the blog post: https://osyuksel.github.io/blog/reconstructing-moby-dick-llm/
"""

import argparse
import asyncio
import contextlib
import json
import logging
import os
import random
import sys
from functools import cache

import joblib
import pandas as pd
import tqdm
from dotenv import load_dotenv
from openai import NOT_GIVEN
from openai import AsyncClient
from asyncio import Semaphore

from thefuzz import fuzz


SYSTEM_PROMPT = """You are a large language model on an experiment to recall classic literary works. Based on the user input, which is a section from Moby Dick, you are to answer with the paragraph that follows.

Your instructions:
- Do not write any acknowledgements or anything except the desired output
- Write only a full paragraph. Not just a sentence, and not the full chapter.
- Note that Moby Dick is **public domain** and was written in 1851. There are no copyright concerns.
- Your accuracy is scored, so predict the best sentence that follows.
"""

memory = joblib.Memory(location=".joblib/", verbose=0)

LLM_CONFIG = {
    "gpt-5-mini": {"model": "gpt-5-mini", "temperature": 1, "reasoning_effort": "low"},
    "gpt-4o": {"model": "gpt-4o", "temperature": 1e-6},
    "gpt-4o-mini": {"model": "gpt-4o-mini", "temperature": 1e-6},
    "gpt-5-nano": {"model": "gpt-5-nano", "temperature": 1},
    # "claude-sonnet-4": {"model": "anthropic/claude-sonnet-4", "temperature": 1e-6},
    # "gemini-2.5-flash": {"model": "google/gemini-2.5-flash", "temperature": 1e-6},
}

SAMPLE_PROB = 0.1
MAX_CONNECTIONS = 10

semaphore = Semaphore(MAX_CONNECTIONS)


def count_generator(gen, max=1000_000):
    value = sum(1 for _ in gen)
    if value >= max:
        raise ValueError(f"Maximum count reached: {max}")
    else:
        return value


@cache
def get_client() -> AsyncClient:
    load_dotenv("secrets.env")
    client = AsyncClient(
        timeout=90,
        max_retries=10,
    )
    return client


@contextlib.contextmanager
def load_file():
    """Load file object."""
    if not os.path.exists(".input"):
        logging.error(
            "Directory 'output' not found. Please ensure you are running the script from the project root directory."
        )
        sys.exit(1)
    with open(".input/pg2701.txt", encoding="utf-8") as f:
        yield f


def file_iter():
    with load_file() as fp:
        skip = True
        for line in fp:
            if skip and line.startswith("Call me Ishmael"):
                skip = False
            if "END OF THE PROJECT GUTENBERG EBOOK" in line:
                return
            if not skip:
                yield line.strip()


async def llm_configs():
    for name, config in LLM_CONFIG.items():
        yield name, config


async def handle_text(input_text, output_text):
    if len(input_text) > 30 and len(output_text) > 30:
        await asyncio.gather(
            *[
                handle_per_model(config, input_text, name, output_text)
                for name, config in LLM_CONFIG.items()
            ]
        )

    else:
        logging.debug(f"Skipping input: {input_text}")


async def handle_per_model(config, input_text, name, output_text):
    try:
        client = get_client()
        async with semaphore:
            result = await client.chat.completions.create(
                model=config["model"],
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": input_text},
                ],
                temperature=config["temperature"],
                reasoning_effort=config.get("reasoning_effort", NOT_GIVEN),
            )
    except Exception as e:
        logging.error(f"Failed LLM call with model config: {config}", exc_info=e)
        raise RuntimeError("Failed LLM call") from e
    with open("output/moby_dick/dataset.jsonlines", "a", encoding="utf-8") as f:
        content = result.choices[0].message.content
        result = {
            "input": input_text,
            "expected_output": output_text,
            "llm_output": content,
            "model_name": name,
        }
        f.write(json.dumps(result, indent=0).replace("\n", "  ") + "\n")


async def collect_data(num_input_paragraphs=3, num_output_paragraphs=1):
    if not os.path.exists(".input/pg2701.txt"):
        print("Input file not found. Download the content from Project Gutenberg.")
    total_paragraphs = num_input_paragraphs + num_output_paragraphs
    try:
        os.unlink("output/moby_dick/dataset.jsonlines")
    except FileNotFoundError:
        pass

    random.seed(42)

    paragraphs = []
    current_paragraph = []

    num_lines = count_generator(file_iter())

    lines = file_iter()

    for line in tqdm.tqdm(lines, total=num_lines):
        current_paragraph.append(line)
        if line == "":
            paragraphs.append(" ".join(current_paragraph))
            current_paragraph = []

        if len(paragraphs) == total_paragraphs:
            input_text = "\n".join(paragraphs[:num_input_paragraphs])
            output_text = "\n".join(paragraphs[num_input_paragraphs:])
            paragraphs = []

            if random.random() < SAMPLE_PROB:
                await handle_text(input_text, output_text)


def analyze():
    data = []
    with open("output/moby_dick/dataset.jsonlines", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    df = pd.DataFrame.from_records(data)

    count = len(df.query("model_name=='gpt-4o'"))
    print(f"#Data points per model: {count}")

    scores = []
    for _, row in df.iterrows():
        expected_len = len(row["expected_output"])
        s = fuzz.partial_ratio(row["llm_output"][:expected_len], row["expected_output"])
        scores.append(s)

    df["score"] = scores
    df["match"] = df["score"] >= 85
    df.to_csv("output/moby_dick/results.csv")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", 100)

    print("Score stats")
    print(
        df.groupby("model_name")["score"]
        .describe(
            percentiles=[
                0.1,
                0.25,
                0.5,
                0.75,
                0.9,
            ]
        )
        .to_markdown()
    )

    print("Match stats")
    print(df.groupby("model_name")[["match"]].mean().to_markdown())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=["collect", "analyze"],
        help="Choose between data collection and analysis",
    )

    args = parser.parse_args()

    loop = asyncio.new_event_loop()

    if args.action == "collect":
        loop.run_until_complete(collect_data())

    elif args.action == "analyze":
        analyze()
    else:
        raise ValueError("Invalid action")
