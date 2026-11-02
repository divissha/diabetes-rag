"""Shared Groq call with retry/backoff for rate limits."""

import os
import time

from dotenv import load_dotenv
from groq import Groq, RateLimitError

load_dotenv()
_client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "openai/gpt-oss-20b"


def call_llm(prompt, temperature=0.0, max_retries=6):
    for attempt in range(max_retries):
        try:
            resp = _client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return resp.choices[0].message.content.strip()
        except RateLimitError:
            wait = 5 * (attempt + 1)
            print(f"    rate limited, waiting {wait}s...")
            time.sleep(wait)
    raise RuntimeError("LLM call failed after retries")
