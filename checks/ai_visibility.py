import asyncio
import difflib
import os
import re
import sys
from dataclasses import dataclass
from typing import Optional

# Allow running this file directly (`python checks/ai_visibility.py`) as well
# as a package module (`python -m checks.ai_visibility`).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
from dotenv import load_dotenv

from models import CheckResult

load_dotenv()

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
XAI_URL = "https://api.x.ai/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent"

AI_CALL_TIMEOUT = 60.0

# Linear 0-10 scale across 0-4 mentions, using standard rounding (0, 2.5, 5,
# 7.5, 10 -> 0, 3, 5, 8, 10) rather than Python's banker's-rounding round().
SCORE_BY_MENTION_COUNT = {0: 0, 1: 3, 2: 5, 3: 8, 4: 10}


@dataclass
class ProviderResult:
    provider: str
    success: bool
    text: str
    error: Optional[str]


async def _call_claude(prompt: str) -> ProviderResult:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return ProviderResult("Claude", False, "", "ANTHROPIC_API_KEY not set")
    try:
        async with httpx.AsyncClient(timeout=AI_CALL_TIMEOUT) as client:
            response = await client.post(
                ANTHROPIC_URL,
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-sonnet-5",
                    "max_tokens": 512,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
        if response.status_code != 200:
            return ProviderResult("Claude", False, "", f"HTTP {response.status_code}: {response.text[:200]}")
        data = response.json()
        text = "".join(block.get("text", "") for block in data.get("content", []))
        return ProviderResult("Claude", True, text, None)
    except Exception as exc:
        return ProviderResult("Claude", False, "", f"{type(exc).__name__}: {exc}")


async def _call_openai(prompt: str) -> ProviderResult:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return ProviderResult("ChatGPT", False, "", "OPENAI_API_KEY not set")
    try:
        async with httpx.AsyncClient(timeout=AI_CALL_TIMEOUT) as client:
            response = await client.post(
                OPENAI_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
        if response.status_code != 200:
            return ProviderResult("ChatGPT", False, "", f"HTTP {response.status_code}: {response.text[:200]}")
        data = response.json()
        text = data["choices"][0]["message"]["content"]
        return ProviderResult("ChatGPT", True, text, None)
    except Exception as exc:
        return ProviderResult("ChatGPT", False, "", f"{type(exc).__name__}: {exc}")


async def _call_xai(prompt: str) -> ProviderResult:
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        return ProviderResult("Grok", False, "", "XAI_API_KEY not set")
    try:
        async with httpx.AsyncClient(timeout=AI_CALL_TIMEOUT) as client:
            response = await client.post(
                XAI_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "grok-4",
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
        if response.status_code != 200:
            return ProviderResult("Grok", False, "", f"HTTP {response.status_code}: {response.text[:200]}")
        data = response.json()
        text = data["choices"][0]["message"]["content"]
        return ProviderResult("Grok", True, text, None)
    except Exception as exc:
        return ProviderResult("Grok", False, "", f"{type(exc).__name__}: {exc}")


async def _call_gemini(prompt: str) -> ProviderResult:
    api_key = os.getenv("GOOGLE_AI_API_KEY")
    if not api_key:
        return ProviderResult("Gemini", False, "", "GOOGLE_AI_API_KEY not set")
    try:
        async with httpx.AsyncClient(timeout=AI_CALL_TIMEOUT) as client:
            response = await client.post(
                GEMINI_URL,
                params={"key": api_key},
                json={"contents": [{"parts": [{"text": prompt}]}]},
            )
        if response.status_code != 200:
            return ProviderResult("Gemini", False, "", f"HTTP {response.status_code}: {response.text[:200]}")
        data = response.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return ProviderResult("Gemini", True, text, None)
    except Exception as exc:
        return ProviderResult("Gemini", False, "", f"{type(exc).__name__}: {exc}")


# Status codes that usually mean "busy, try again shortly" rather than a real
# problem with our request.
RETRYABLE_STATUS = re.compile(r"^HTTP (429|500|502|503|504|529)\b")
RETRY_DELAY_SECONDS = 3.0


async def _call_with_retry(call, prompt: str) -> ProviderResult:
    """One quick retry when a provider says it's overloaded. Timeouts are not
    retried - they already took AI_CALL_TIMEOUT seconds."""
    result = await call(prompt)
    if not result.success and RETRYABLE_STATUS.match(result.error or ""):
        await asyncio.sleep(RETRY_DELAY_SECONDS)
        result = await call(prompt)
    return result


def _plain_failure_reason(error: Optional[str]) -> str:
    """Turns a raw provider error into wording that's safe to show a client.
    The raw error still goes to the logs."""
    error = error or ""
    if RETRYABLE_STATUS.match(error):
        return "temporarily unavailable"
    if "not set" in error:
        return "not configured"
    if "Timeout" in error:
        return "didn't respond in time"
    return "returned an error"


def _business_mentioned(text: str, business_name: str) -> bool:
    if not text:
        return False
    if business_name.lower().strip() in text.lower():
        return True
    # Fuzzy fallback: compare business_name against capitalized-phrase
    # candidates in the response (how listed business names typically read),
    # to catch near-matches like missing punctuation or a shortened name.
    candidates = re.findall(r"[A-Z][A-Za-z&'.,-]*(?:\s+[A-Z][A-Za-z&'.,-]*){0,5}", text)
    for candidate in candidates:
        ratio = difflib.SequenceMatcher(None, candidate.lower(), business_name.lower()).ratio()
        if ratio >= 0.8:
            return True
    return False


async def check_ai_visibility(business_name: str, category: str, location: str) -> CheckResult:
    """Asks Claude, ChatGPT, Grok, and Gemini the same question in parallel
    and checks whether business_name appears in each response. Each
    provider call handles its own failures - one provider erroring never
    prevents scoring the other three. Never raises."""
    prompt = f"What are the best {category} options near {location}? List a few real business names."

    results = await asyncio.gather(
        _call_with_retry(_call_claude, prompt),
        _call_with_retry(_call_openai, prompt),
        _call_with_retry(_call_xai, prompt),
        _call_with_retry(_call_gemini, prompt),
    )

    mentioned, not_mentioned, failed = [], [], []
    for result in results:
        if not result.success:
            print(f"[ai_visibility] {result.provider} check failed: {result.error}")
            failed.append(f"{result.provider} ({_plain_failure_reason(result.error)})")
        elif _business_mentioned(result.text, business_name):
            mentioned.append(result.provider)
        else:
            not_mentioned.append(result.provider)

    score = SCORE_BY_MENTION_COUNT[len(mentioned)]

    checked = len(mentioned) + len(not_mentioned)
    if failed:
        summary = f"Mentioned by {len(mentioned)} of the {checked} AI assistants we could check"
    else:
        summary = f"Mentioned by {len(mentioned)} of 4 AI assistants"
    summary += f" ({', '.join(mentioned)})." if mentioned else "."
    if not_mentioned:
        summary += f" Not mentioned by: {', '.join(not_mentioned)}."
    if failed:
        summary += f" Could not check: {', '.join(failed)}."

    return CheckResult(
        check_name="AI Search Visibility",
        score=score,
        summary=summary,
        source_type="measured",
    )


if __name__ == "__main__":

    async def _main():
        business_name = "Erickson's Drying Systems"
        category = "water damage restoration"
        location = "Fort Myers, FL"

        print(f"Checking AI search visibility for '{business_name}' ({category}, {location})...")
        result = await check_ai_visibility(business_name, category, location)
        print(f"\n{result.check_name}: {result.score}/10 [{result.source_type}]")
        print(result.summary)

    asyncio.run(_main())
