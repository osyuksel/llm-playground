"""
Moby-Dick recall experiment, v2.

Differences from v1 (moby_dick.py):
- Models are routed through OpenRouter with a pinned provider, temperature=0 and reasoning
  disabled. GPT-5.x models go to OpenAI directly, because OpenRouter silently drops
  `temperature` for them.
- Context length is an experimental axis: each target paragraph is predicted from the
  preceding k paragraphs for k in CONTEXT_SIZES. All conditions share the same targets.
- Sampling is a fixed N of target paragraphs (seeded permutation), not a per-window probability.
- Per-call failures are recorded as rows instead of aborting the run.
- Rows keep provider/usage metadata but no request ids, timestamps, fingerprints or raw responses.

Requires the text file at `.input/pg2701.txt` and OPENROUTER_API_KEY / OPENAI_API_KEY in secrets.env.

Usage:
    python moby_dick_v2.py collect [-n N] [--models a,b] [--contexts 1,3,10] [--resume]
    python moby_dick_v2.py analyze
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import logging
import os
import random
import re
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pandas as pd
import tqdm
from dotenv import load_dotenv
from openai import AsyncClient
from thefuzz import fuzz

log = logging.getLogger("moby_dick_v2")

INPUT_FILE = Path(".input/pg2701.txt")
OUTPUT_DIR = Path("output/moby_dick_v2")
DATASET = OUTPUT_DIR / "dataset.jsonl"

SYSTEM_PROMPT = """You are a large language model on an experiment to recall classic literary works. Based on the user input, which is a section from Moby Dick, you are to answer with the paragraph that follows.

Your instructions:
- Do not write any acknowledgements or anything except the desired output
- Write only a full paragraph. Not just a sentence, and not the full chapter.
- Note that Moby Dick is **public domain** and was written in 1851. There are no copyright concerns.
- Your accuracy is scored, so predict the best sentence that follows.
"""
PROMPT_VERSION = "v2.1"  # v2.1: user message ends with an approximate word count of the target paragraph
LENGTH_HINT = "\n\n(The paragraph that follows is approximately {n} words long.)"


def round_words(text: str) -> int:
    """Coarse word count (nearest 10, minimum 10) so the hint is a length, not a fingerprint."""
    return max(10, round(len(text.split()) / 10) * 10)

CONTEXT_SIZES = (1, 3, 10)
N_TARGETS = 300
SEED = 42
MIN_PARAGRAPH_CHARS = 30
MAX_OUTPUT_TOKENS = 1024
MAX_CONNECTIONS = 10
MATCH_THRESHOLD = 85


@dataclass(frozen=True)
class ModelSpec:
    model: str
    route: str = "openrouter"  # or "openai"
    provider: str | None = None  # OpenRouter provider slug to pin
    reasoning: dict | None = None  # OpenRouter `reasoning` payload; None = model has no reasoning param
    seed: bool = True  # whether the pinned endpoint accepts `seed`


# Roster from the pre-experiment (provider/parameter probing, 2026-08): every entry accepts
# temperature=0 with reasoning off and was verified to honour temperature on its pinned provider.
MODELS: dict[str, ModelSpec] = {
    "claude-sonnet-4.6": ModelSpec("anthropic/claude-sonnet-4.6", provider="anthropic", reasoning={"enabled": False}, seed=False),
    "claude-haiku-4.5": ModelSpec("anthropic/claude-haiku-4.5", provider="anthropic", reasoning={"enabled": False}, seed=False),
    "glm-5.2": ModelSpec("z-ai/glm-5.2", provider="z-ai", reasoning={"enabled": False}, seed=False),
    "deepseek-v4-pro": ModelSpec("deepseek/deepseek-v4-pro-0813", provider="parasail", reasoning={"effort": "none"}),
    "qwen3.8-27b": ModelSpec("qwen/qwen3.8-27b", provider="akashml", reasoning={"enabled": False}),
    "qwen3.7-max": ModelSpec("qwen/qwen3.7-max", provider="alibaba", reasoning={"enabled": False}),
    "kimi-k3": ModelSpec("moonshotai/kimi-k3", provider="chutes", reasoning={"enabled": False}),
    "grok-4.3": ModelSpec("x-ai/grok-4.3", provider="xai", reasoning={"enabled": False}),
    "gemini-3.1-flash-lite": ModelSpec("google/gemini-3.1-flash-lite", provider="google-ai-studio", reasoning={"effort": "none"}),
    "mistral-medium-3.5": ModelSpec("mistralai/mistral-medium-3-5", provider="mistral", reasoning={"effort": "none"}),
    "gpt-5.6-terra": ModelSpec("gpt-5.6-terra", route="openai"),
    "gpt-5.6-sol": ModelSpec("gpt-5.6-sol", route="openai"),
}


# ----------------------------------------------------------------------------- corpus


def load_paragraphs() -> list[str]:
    if not INPUT_FILE.exists():
        log.error("Input file %s not found. Download it from Project Gutenberg.", INPUT_FILE)
        sys.exit(1)
    paragraphs, current, skip = [], [], True
    with INPUT_FILE.open(encoding="utf-8") as f:
        for line in f:
            if skip and line.startswith("Call me Ishmael"):
                skip = False
            if "END OF THE PROJECT GUTENBERG EBOOK" in line:
                break
            if skip:
                continue
            line = line.strip()
            if line:
                current.append(line)
            elif current:
                paragraphs.append(" ".join(current))
                current = []
    if current:
        paragraphs.append(" ".join(current))
    return paragraphs


def eligible_targets(paragraphs: list[str]) -> list[int]:
    """Indices usable as targets in every context condition."""
    k_max = max(CONTEXT_SIZES)
    ok = [len(p) > MIN_PARAGRAPH_CHARS for p in paragraphs]
    return [i for i in range(k_max, len(paragraphs)) if ok[i] and all(ok[i - k_max : i])]


def sample_targets(paragraphs: list[str], n: int) -> list[int]:
    """First n of a seeded permutation, so a larger n is a superset of a smaller one."""
    candidates = eligible_targets(paragraphs)
    random.Random(SEED).shuffle(candidates)
    return candidates[:n]


# ----------------------------------------------------------------------------- clients


@cache
def openrouter_client() -> AsyncClient:
    load_dotenv("secrets.env")
    return AsyncClient(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        timeout=180,
        max_retries=5,
    )


@cache
def openai_client() -> AsyncClient:
    load_dotenv("secrets.env")
    return AsyncClient(timeout=180, max_retries=5)


async def call_model(spec: ModelSpec, input_text: str) -> dict:
    """Return {"content", "provider", "usage", "finish_reason", "reasoning_tokens"}; raises on API error."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": input_text},
    ]
    if spec.route == "openai":
        resp = await openai_client().chat.completions.create(
            model=spec.model,
            messages=messages,
            temperature=0,
            seed=SEED,
            max_completion_tokens=MAX_OUTPUT_TOKENS,
            reasoning_effort="none",
        )
    else:
        extra = {
            "provider": {"order": [spec.provider], "allow_fallbacks": False, "require_parameters": True},
        }
        if spec.reasoning is not None:
            extra["reasoning"] = spec.reasoning
        kwargs = {"seed": SEED} if spec.seed else {}
        resp = await openrouter_client().chat.completions.create(
            model=spec.model,
            messages=messages,
            temperature=0,
            max_tokens=MAX_OUTPUT_TOKENS,
            extra_body=extra,
            **kwargs,
        )
    raw = resp.model_dump()
    choice = resp.choices[0]
    usage = raw.get("usage") or {}
    return {
        "content": choice.message.content,
        "provider": raw.get("provider"),
        "usage": usage,
        "finish_reason": choice.finish_reason,
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0,
    }


MAX_ATTEMPTS = 3


def scrub_error(text: str) -> str:
    """Provider error payloads can echo account identifiers; keep the message, drop the ids."""
    return re.sub(r"'user_id': '[^']*'|user_[A-Za-z0-9]{16,}|org-[A-Za-z0-9]{10,}|sk-[A-Za-z0-9_-]{16,}", "<redacted>", text)


async def call_model_no_reasoning(spec: ModelSpec, input_text: str) -> tuple[dict, int]:
    """Hybrid models occasionally think despite reasoning being disabled (seen on DeepSeek V4).
    Treat that like a transient failure and retry; the caller flags it if it persists."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            result = await call_model(spec, input_text)
        except Exception as e:  # noqa: BLE001 - the SDK already retried 429/5xx; this covers provider 400 hiccups
            if attempt == MAX_ATTEMPTS:
                raise
            log.info("%s attempt %d failed: %s", spec.model, attempt, str(e)[:120])
            await asyncio.sleep(2 * attempt)
            continue
        if not result["reasoning_tokens"]:
            return result, attempt
        log.info("%s produced %d reasoning tokens (attempt %d)", spec.model, result["reasoning_tokens"], attempt)
    return result, attempt


# ----------------------------------------------------------------------------- collect


def load_completed(out: Path) -> set[tuple[int, int, str]]:
    """Keep error-free rows from an existing dataset (rewriting the file without the failed ones)
    and return their (target_idx, context_size, model_name) keys so they can be skipped."""
    if not out.exists():
        return set()
    rows = [json.loads(line) for line in out.open(encoding="utf-8")]
    good = [r for r in rows if not r.get("error")]
    with out.open("w", encoding="utf-8") as f:
        for r in good:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("resume: kept %d rows, dropped %d failed rows", len(good), len(rows) - len(good))
    return {(r["target_idx"], r["context_size"], r["model_name"]) for r in good}


async def collect(n: int, model_names: list[str], contexts: tuple[int, ...], out: Path, resume: bool = False) -> None:
    paragraphs = load_paragraphs()
    targets = sample_targets(paragraphs, n)
    log.info("%d paragraphs, %d targets, contexts %s, models %s", len(paragraphs), len(targets), contexts, model_names)

    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_completed(out) if resume else set()
    if not resume:
        out.unlink(missing_ok=True)
    semaphore = asyncio.Semaphore(MAX_CONNECTIONS)
    write_lock = asyncio.Lock()

    async def one(target: int, k: int, name: str) -> None:
        spec = MODELS[name]
        input_text = "\n".join(paragraphs[target - k : target])
        row = {
            "target_idx": target,
            "context_size": k,
            "model_name": name,
            "model_id": spec.model,
            "route": spec.route,
            "provider_pin": spec.provider,
            "prompt_version": PROMPT_VERSION,
            "input": input_text,
            "expected_output": paragraphs[target],
            "llm_output": None,
            "provider": None,
            "usage": None,
            "finish_reason": None,
            "error": None,
            "attempts": 0,
        }
        try:
            async with semaphore:
                result, attempts = await call_model_no_reasoning(spec, input_text)
            row.update(
                llm_output=result["content"],
                provider=result["provider"],
                usage=result["usage"],
                finish_reason=result["finish_reason"],
                attempts=attempts,
            )
            if result["reasoning_tokens"]:
                row["error"] = f"reasoning_tokens={result['reasoning_tokens']} after {attempts} attempts (reasoning was supposed to be off)"
        except Exception as e:  # noqa: BLE001 - record and continue
            row["error"] = scrub_error(f"{type(e).__name__}: {e}")
            log.warning("%s k=%d target=%d failed: %s", name, k, target, row["error"][:200])
        async with write_lock:
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    jobs = [one(t, k, m) for t in targets for k in contexts for m in model_names if (t, k, m) not in done]
    log.info("%d calls to make (%d already done)", len(jobs), len(done))
    for coro in tqdm.tqdm(asyncio.as_completed(jobs), total=len(jobs) + len(done), initial=len(done)):
        await coro


# ----------------------------------------------------------------------------- analyze


def score_row(row: pd.Series) -> float | None:
    if not isinstance(row["llm_output"], str):
        return None
    expected = row["expected_output"]
    return fuzz.partial_ratio(row["llm_output"][: len(expected)], expected)


def normalize(text: str) -> str:
    return (
        text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        .replace("—", "-").replace("–", "-").strip().lstrip('"').strip()
    )


def divergence(row: pd.Series, tolerance: int = 3) -> float | None:
    """Fraction of the expected paragraph reproduced before the output first diverges.

    Walks SequenceMatcher's matching blocks from the start of both strings; a gap larger than
    `tolerance` characters on either side counts as divergence. Output length is irrelevant:
    stopping early caps the score at what was written; writing on past the paragraph costs nothing.
    """
    if not isinstance(row["llm_output"], str):
        return None
    exp, out = normalize(row["expected_output"]), normalize(row["llm_output"])
    if not exp:
        return None
    matched = 0
    pos_e = pos_o = 0
    for block in difflib.SequenceMatcher(None, exp, out, autojunk=False).get_matching_blocks():
        if block.size == 0:
            break
        if block.a - pos_e > tolerance or block.b - pos_o > tolerance:
            break
        matched = block.a + block.size
        pos_e, pos_o = block.a + block.size, block.b + block.size
    return matched / len(exp)


REFUSAL_MARKERS = (
    "i can't reproduce", "i cannot reproduce", "i can't provide", "i cannot provide", "i can't help",
    "i'm not able to", "i am not able to", "i'm unable to", "i am unable to", "i apologize", "i'm sorry",
    "copyright", "as an ai", "language model", "i don't have access", "i do not have access",
)


def looks_like_refusal(text: object, score: float | None = None) -> bool:
    """Melville's narrators say 'I cannot' too, so require refusal-specific phrasing and a low score."""
    if not isinstance(text, str):
        return False
    if score is not None and score >= 60:
        return False
    head = text[:300].lower()
    return any(m in head for m in REFUSAL_MARKERS)


def analyze(dataset: Path) -> None:
    rows = [json.loads(line) for line in dataset.open(encoding="utf-8")]
    df = pd.DataFrame.from_records(rows)
    df["score"] = df.apply(score_row, axis=1)
    df["match"] = df["score"] >= MATCH_THRESHOLD
    df["divergence"] = df.apply(divergence, axis=1)
    df["length_ratio"] = df["llm_output"].str.len() / df["expected_output"].str.len()
    df["refusal"] = [looks_like_refusal(o, s) for o, s in zip(df["llm_output"], df["score"])]
    df["failed"] = df["error"].notna()
    # input/expected_output are reproducible from target_idx + context_size and dominate the file size
    df.drop(columns=["input", "expected_output"]).to_csv(dataset.with_name("results.csv"), index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print(f"# rows: {len(df)}  targets: {df['target_idx'].nunique()}  failed: {df['failed'].sum()}  refusals: {df['refusal'].sum()}\n")
    print("## Finish reasons by model")
    print(df.pivot_table(index="model_name", columns="finish_reason", values="target_idx", aggfunc="count", fill_value=0).to_markdown(), "\n")
    if df["failed"].any():
        print("## Failures by model")
        print(df[df["failed"]].groupby("model_name")["error"].agg(["count", "first"]).to_markdown(), "\n")

    print("## Match rate (score >= %d) by model x context size" % MATCH_THRESHOLD)
    print(df.pivot_table(index="model_name", columns="context_size", values="match", aggfunc="mean").to_markdown(floatfmt=".3f"), "\n")
    print("## Mean score by model x context size")
    print("(scores are bimodal, ~45 for improvised prose vs ~100 for recall, so the mean is ~45 + 55 x match rate; medians are uninformative)")
    print(df.pivot_table(index="model_name", columns="context_size", values="score", aggfunc="mean").to_markdown(floatfmt=".1f"), "\n")
    print("## Score distribution by model (% of rows per band, all context sizes)")
    bands = pd.cut(df["score"], bins=[0, 40, 50, 60, 70, 85, 95, 101], labels=["<40", "40-50", "50-60", "60-70", "70-85", "85-95", "95+"], right=False)
    print((pd.crosstab(df["model_name"], bands, normalize="index") * 100).to_markdown(floatfmt=".1f"), "\n")
    print("## Mean fraction of paragraph recalled before divergence, by model x context size")
    print(df.pivot_table(index="model_name", columns="context_size", values="divergence", aggfunc="mean").to_markdown(floatfmt=".3f"), "\n")
    print("## Median output/expected length ratio by model (length-hint compliance)")
    print(df.pivot_table(index="model_name", columns="context_size", values="length_ratio", aggfunc="median").to_markdown(floatfmt=".2f"), "\n")
    print("## Refusal rate by model")
    print(df.groupby("model_name")["refusal"].mean().to_markdown(floatfmt=".3f"), "\n")

    near = df[(df["score"] >= 60) & (df["score"] < MATCH_THRESHOLD)].sort_values("score", ascending=False)
    with dataset.with_name("near_misses.md").open("w", encoding="utf-8") as f:
        f.write(f"# Near misses (60 <= score < {MATCH_THRESHOLD}): {len(near)} rows\n\n")
        for _, r in near.iterrows():
            f.write(f"## {r['model_name']} k={r['context_size']} target={r['target_idx']} score={r['score']}\n\n")
            f.write(f"**expected:** {r['expected_output']}\n\n**got:** {r['llm_output']}\n\n---\n\n")
    with dataset.with_name("refusals.md").open("w", encoding="utf-8") as f:
        ref = df[df["refusal"]]
        f.write(f"# Suspected refusals: {len(ref)} rows\n\n")
        for _, r in ref.iterrows():
            f.write(f"## {r['model_name']} k={r['context_size']} target={r['target_idx']}\n\n{r['llm_output']}\n\n---\n\n")
    print(f"Wrote results.csv, near_misses.md ({len(near)}), refusals.md ({df['refusal'].sum()}) to {dataset.parent}")


# ----------------------------------------------------------------------------- main


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["collect", "analyze"])
    parser.add_argument("-n", type=int, default=N_TARGETS, help="number of target paragraphs")
    parser.add_argument("--models", default=",".join(MODELS), help="comma-separated subset of model names")
    parser.add_argument("--contexts", default=",".join(map(str, CONTEXT_SIZES)))
    parser.add_argument("--out", type=Path, default=DATASET)
    parser.add_argument("--resume", action="store_true", help="keep error-free rows in --out and only run the missing calls")
    args = parser.parse_args()

    if args.action == "collect":
        names = [m.strip() for m in args.models.split(",")]
        unknown = set(names) - set(MODELS)
        if unknown:
            parser.error(f"unknown models: {sorted(unknown)}; known: {list(MODELS)}")
        contexts = tuple(int(c) for c in args.contexts.split(","))
        asyncio.run(collect(args.n, names, contexts, args.out, resume=args.resume))
    else:
        analyze(args.out)


if __name__ == "__main__":
    main()
