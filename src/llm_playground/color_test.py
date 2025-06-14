import logging
import os
import random
import sys
from dataclasses import dataclass

import dotenv
import joblib
import pandas as pd
from dotenv import find_dotenv
from openai import OpenAI

memory = joblib.Memory(location=".joblib", verbose=0)

UNUSUAL_EYE_COLORS = ["pink", "red", "gray", "black", "white", "copper"]

UNUSUAL_HAIR_COLORS = ["blue", "green", "amber", "violet"]

NAMES = [
    "Ali",
    "John",
    "Karl",
    "Juan",
    "Dimitri",
    "Wei",
    "Riku",
    "Pranav",
    "Zoe",
    "Lucy",
    "Marta",
    "Maryam",
    "Isha",
    "Nagisa",
]

dotenv.load_dotenv(find_dotenv("secrets.env"), verbose=True)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    # api_key="<OPENROUTER_API_KEY>",
)

MODELS = [
    "anthropic/claude-sonnet-4",
    "openai/gpt-4o",
    "google/gemini-2.5-flash-preview-05-20",
    "x-ai/grok-3-beta",
    "meta-llama/llama-4-scout",
    "openai/gpt-4o-mini",
    "anthropic/claude-3.5-haiku",
    "deepseek/deepseek-r1-0528",
    "meta-llama/llama-4-maverick",
    "google/gemma-3-4b-it",
    "microsoft/phi-4-multimodal-instruct",
    "mistral/ministral-8b",
    "mistralai/ministral-3b",
    "mistralai/mistral-tiny",
    "sentientagi/dobby-mini-unhinged-plus-llama-3.1-8b",
    "inception/mercury-coder-small-beta",
    "meta-llama/llama-3.2-1b-instruct",
    "qwen/qwen3-8b",
    "qwen/qwen-turbo",
    "liquid/lfm-3b",
    "amazon/nova-micro-v1",
    "openai/gpt-4.1-nano",
]

PROMPT_TPL = """
{names[0]} has {ecolors[0]} eyes and {hcolors[0]} hair.
{names[1]} has {ecolors[1]} eyes and {hcolors[1]} hair.
{names[2]} has {ecolors[2]} eyes and {hcolors[2]} hair.
{names[3]} has green eyes and brown hair.
{names[4]} has brown eyes and black hair.
===
Given the above information, answer the following:
1. What is {names[0]}'s eye color?
2. What is {names[1]}'s hair color?
3. What is {names[2]}'s hair color?

Give your answer in the format:
1. <answer>
2. <answer>
...

Only give the answer to the question. Do not give any other information or acknowledgements.
""".strip()

rng = random.Random()
rng.seed(2025)


@dataclass(frozen=True)
class Sample:
    names: list[str]
    eye_colors: list[str]
    hair_colors: list[str]

    @property
    def text(self):
        return f"{self.names} | {self.eye_colors} | {self.hair_colors}"


@dataclass
class Result:
    raw: str | None
    value: list[str] | None
    success: bool
    fail_reason: str | None


def create_sample() -> Sample:
    names = rng.sample(NAMES, k=5)
    eye_colors = rng.sample(UNUSUAL_EYE_COLORS, k=3)
    hair_colors = rng.sample(UNUSUAL_HAIR_COLORS, k=3)
    return Sample(names=names, eye_colors=eye_colors, hair_colors=hair_colors)


def construct_prompt(sample: Sample):
    text = PROMPT_TPL.format(
        names=sample.names, ecolors=sample.eye_colors, hcolors=sample.hair_colors
    )
    return text


def expected_answer(sample: Sample) -> list[str]:
    return [
        sample.eye_colors[0],
        sample.hair_colors[1],
        sample.hair_colors[2],
    ]


@memory.cache
def execute_api_call(model: str, prompt: str) -> str | None:
    logging.info(f"Making fresh API call to {model}....")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.01,
    )
    return response.choices[0].message.content


def parse_response(text, length=3):
    """Parse response and pad if needed, returning a list of length 6"""
    answers = text.split("\n")
    answers = [a.split(".")[1].strip() if "." in a else "" for a in answers]
    answers = answers[:6]

    for _ in range(length - len(answers)):
        answers.append("")

    return answers


def get_response(model, prompt) -> Result:
    content = None
    try:
        content = execute_api_call(model, prompt)
        value = parse_response(content)
        return Result(raw=content, value=value, success=True, fail_reason=None)
    except Exception as e:
        return Result(raw=content, value=None, success=False, fail_reason=str(e))


def evaluate(sample: Sample, result: Result) -> float:
    if not result.success or not result.value:
        return 0.0
    expected = expected_answer(sample)
    correct = sum(
        1 for exp, res in zip(expected, result.value) if exp.lower() == res.lower()
    )
    return correct / len(expected)


def run_experiment(n_samples: int = 10):
    results = []
    for _ in range(n_samples):
        sample = create_sample()
        prompt = construct_prompt(sample)
        for model in MODELS:
            result = get_response(model=model, prompt=prompt)
            accuracy = evaluate(sample, result)
            results.append((sample, model, result, accuracy))
    return results


def create_detailed_df(results):
    records = []
    for sample, model, result, accuracy in results:
        record = {
            "model": model,
            "success": result.success,
            "accuracy": accuracy,
            "expected": expected_answer(sample) if result.success else None,
            "prompt": construct_prompt(sample),
            "actual": result.value if result.success else None,
            "fail_reason": result.fail_reason,
            "raw_response": result.raw,
        }
        records.append(record)
    return pd.DataFrame(records)


def create_summary_df(results):
    summary_data = {}
    for _, model, result, accuracy in results:
        if model not in summary_data:
            summary_data[model] = {"error_rate": 0, "avg_accuracy": 0, "count": 0}
        summary_data[model]["count"] += 1
        summary_data[model]["error_rate"] += 1 if not result.success else 0
        summary_data[model]["avg_accuracy"] += accuracy

    summary_records = []
    for model, stats in summary_data.items():
        count = stats["count"]
        summary_records.append(
            {
                "model": model,
                "error_rate": stats["error_rate"] / count,
                "avg_accuracy": stats["avg_accuracy"] / count,
                "total_samples": count,
            }
        )
    return pd.DataFrame(summary_records)


def main():
    if not os.path.exists("output"):
        logging.error(
            "Directory 'output' not found. Please ensure you are running the script from the project root directory."
        )
        sys.exit(1)

    results = run_experiment()

    detailed_df = create_detailed_df(results)
    summary_df = create_summary_df(results).sort_values(by=["model", "avg_accuracy"], ascending=[True, False])

    os.makedirs("output/color_test", exist_ok=True)
    detailed_df.to_csv("output/color_test/detailed.csv", index=False)
    summary_df.to_csv("output/color_test/summary.csv", index=False)

    print("\nSummary Results:")
    print(summary_df.to_string())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
