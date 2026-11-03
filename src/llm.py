"""Shared Groq call with retry/backoff for rate limits."""

import os
import time

from dotenv import load_dotenv
from groq import BadRequestError, Groq, RateLimitError

load_dotenv()
_client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "openai/gpt-oss-20b"


def call_llm(prompt, temperature=0.0, max_retries=6, reasoning_effort="low"):
    kwargs = dict(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort

    for attempt in range(max_retries):
        try:
            resp = _client.chat.completions.create(**kwargs)
            return (resp.choices[0].message.content or "").strip()
        except RateLimitError:
            wait = 5 * (attempt + 1)
            print(f"    rate limited, waiting {wait}s...")
            time.sleep(wait)
        except (TypeError, BadRequestError):
            # SDK/model doesn't accept reasoning_effort: retry without it
            if "reasoning_effort" in kwargs:
                kwargs.pop("reasoning_effort")
                continue
            raise
    raise RuntimeError("LLM call failed after retries")
