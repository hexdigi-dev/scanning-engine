import asyncio
import difflib
import json
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
OPENAI_URL = "https://api.openai.com/v1/responses"
XAI_URL = "https://api.x.ai/v1/responses"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# Newest model first. If a provider rejects a model name (retired or not on
# this account), the next one is tried, so a model rename never silently
# turns the whole check into "not checked". The model actually used is logged.
CLAUDE_MODELS = ["claude-sonnet-5-5", "claude-sonnet-5"]
OPENAI_MODELS = ["gpt-6-astra", "gpt-5.5"]
XAI_MODELS = ["grok-4.7", "grok-4.5"]
GEMINI_MODELS = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-flash-latest"]

# Web search makes each answer slower than a plain chat reply.
AI_CALL_TIMEOUT = 90.0

# Linear 0-10 scale across 0-4 mentions, using standard rounding (0, 2.5, 5,
# 7.5, 10 -> 0, 3, 5, 8, 10) rather than Python's banker's-rounding round().
SCORE_BY_MENTION_COUNT = {0: 0, 1: 3, 2: 5, 3: 8, 4: 10}


@dataclass
class ProviderResult:
    provider: str
    success: bool
    text: str
    error: Optional[str]
    model: Optional[str] = None


def _model_rejected(response: httpx.Response) -> bool:
    """True when the error is about the model name, so the next model in the
    list is worth trying."""
    if response.status_code == 404:
        return True
    body = response.text.lower()
    return response.status_code == 400 and "model" in body and (
        "not found" in body or "does not exist" in body or "invalid" in body or "not supported" in body
    )


async def _post_with_model_fallback(provider: str, models: list, send) -> ProviderResult:
    """send(client, model) -> httpx.Response. Tries each model in order."""
    last_error = "no models configured"
    async with httpx.AsyncClient(timeout=AI_CALL_TIMEOUT) as client:
        for model in models:
            response = await send(client, model)
            if response.status_code == 200:
                return ProviderResult(provider, True, response.text, None, model)
            # One line, so the whole reason shows up in the logs.
            last_error = f"HTTP {response.status_code} ({model}): {' '.join(response.text.split())[:400]}"
            if not _model_rejected(response):
                break
    return ProviderResult(provider, False, "", last_error)


def _responses_api_text(data: dict) -> str:
    """Final answer text from an OpenAI/xAI Responses API reply."""
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    parts = []
    for item in data.get("output", []) or []:
        if item.get("type") == "message":
            for content in item.get("content", []) or []:
                if content.get("type") in ("output_text", "text"):
                    parts.append(content.get("text", ""))
    return "\n".join(parts)


async def _call_claude(prompt: str, place: dict) -> ProviderResult:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return ProviderResult("Claude", False, "", "ANTHROPIC_API_KEY not set")

    async def send(client, model):
        tool = {"type": "web_search_20250305", "name": "web_search", "max_uses": 5}
        if place.get("city"):
            tool["user_location"] = {
                "type": "approximate", "city": place["city"], "region": place.get("state") or "", "country": "US",
            }
        return await client.post(
            ANTHROPIC_URL,
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={
                "model": model,
                "max_tokens": 1500,
                "tools": [tool],
                "messages": [{"role": "user", "content": prompt}],
            },
        )

    try:
        result = await _post_with_model_fallback("Claude", CLAUDE_MODELS, send)
        if result.success:
            data = json.loads(result.text)
            result.text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        return result
    except Exception as exc:
        return ProviderResult("Claude", False, "", f"{type(exc).__name__}: {exc}")


async def _call_responses_api(
    provider: str, url: str, key_env: str, models: list, prompt: str, place: dict, extra: dict
) -> ProviderResult:
    """OpenAI and xAI share the Responses API shape and its web_search tool."""
    api_key = os.getenv(key_env)
    if not api_key:
        return ProviderResult(provider, False, "", f"{key_env} not set")

    async def send(client, model):
        tool = {"type": "web_search"}
        if provider == "ChatGPT" and place.get("city"):
            tool["user_location"] = {
                "type": "approximate", "city": place["city"], "region": place.get("state") or "", "country": "US",
            }
        return await client.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "input": prompt, "tools": [tool], **extra},
        )

    try:
        result = await _post_with_model_fallback(provider, models, send)
        if result.success:
            result.text = _responses_api_text(json.loads(result.text))
        return result
    except Exception as exc:
        return ProviderResult(provider, False, "", f"{type(exc).__name__}: {exc}")


async def _call_openai(prompt: str, place: dict) -> ProviderResult:
    # "required" makes it actually search, the way ChatGPT does for local
    # recommendations, instead of answering from memory.
    return await _call_responses_api(
        "ChatGPT", OPENAI_URL, "OPENAI_API_KEY", OPENAI_MODELS, prompt, place,
        {"tool_choice": "required", "reasoning": {"effort": "low"}},
    )


async def _call_xai(prompt: str, place: dict) -> ProviderResult:
    return await _call_responses_api("Grok", XAI_URL, "XAI_API_KEY", XAI_MODELS, prompt, place, {})


async def _call_gemini(prompt: str, place: dict) -> ProviderResult:
    api_key = os.getenv("GOOGLE_AI_API_KEY")
    if not api_key:
        return ProviderResult("Gemini", False, "", "GOOGLE_AI_API_KEY not set")

    def sender(tools):
        async def send(client, model):
            return await client.post(
                GEMINI_URL.format(model=model),
                headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                json={"contents": [{"role": "user", "parts": [{"text": prompt}]}], "tools": tools},
            )
        return send

    try:
        # Google Search + Google Maps grounding: the same sources the Gemini
        # app uses for "best plumber near me" questions. Maps grounding can
        # be unavailable on some keys/plans - then fall back to Search only.
        result = await _post_with_model_fallback(
            "Gemini", GEMINI_MODELS, sender([{"google_search": {}}, {"google_maps": {}}])
        )
        if not result.success and not RETRYABLE_STATUS.match(result.error or ""):
            print(f"[ai_visibility] Gemini with Maps grounding failed, retrying with Search only: {result.error}")
            result = await _post_with_model_fallback("Gemini", GEMINI_MODELS, sender([{"google_search": {}}]))
        if result.success:
            data = json.loads(result.text)
            candidate = (data.get("candidates") or [{}])[0]
            texts = [p.get("text", "") for p in candidate.get("content", {}).get("parts", []) or []]
            # Business names from the Maps/web sources it grounded on count too.
            for chunk in candidate.get("groundingMetadata", {}).get("groundingChunks", []) or []:
                for kind in ("maps", "web"):
                    title = (chunk.get(kind) or {}).get("title")
                    if title:
                        texts.append(title)
            result.text = "\n".join(texts)
        return result
    except Exception as exc:
        return ProviderResult("Gemini", False, "", f"{type(exc).__name__}: {exc}")


# Status codes that usually mean "busy, try again shortly" rather than a real
# problem with our request.
RETRYABLE_STATUS = re.compile(r"^HTTP (429|500|502|503|504|529)\b")
RETRY_DELAY_SECONDS = 3.0


async def _call_with_retry(call, prompt: str, place: dict) -> ProviderResult:
    """One quick retry when a provider says it's overloaded. Timeouts are not
    retried - they already took AI_CALL_TIMEOUT seconds."""
    result = await call(prompt, place)
    if not result.success and RETRYABLE_STATUS.match(result.error or ""):
        await asyncio.sleep(RETRY_DELAY_SECONDS)
        result = await call(prompt, place)
    return result


def _plain_failure_reason(error: Optional[str]) -> str:
    """Turns a raw provider error into wording that's safe to show a client.
    The raw error still goes to the logs."""
    error = error or ""
    if RETRYABLE_STATUS.match(error.split(" (")[0]):
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
    """Asks Claude, ChatGPT, Grok, and Gemini - each with live web search on,
    the way their consumer apps answer local questions - what a new customer
    would ask, and checks whether business_name appears in each answer.
    category should be the Google category ("Plumber") when known. Each
    provider handles its own failures. Never raises."""
    city, _, state = (location or "").partition(",")
    place = {"city": city.strip() or None, "state": state.strip() or None}
    prompt = (
        f"I need a {category.lower()} in {location}. Who are the best ones to call? "
        "Give me a few specific local business names."
    )

    results = await asyncio.gather(
        _call_with_retry(_call_claude, prompt, place),
        _call_with_retry(_call_openai, prompt, place),
        _call_with_retry(_call_xai, prompt, place),
        _call_with_retry(_call_gemini, prompt, place),
    )
    print(
        "[ai_visibility] "
        + "; ".join(
            f"{r.provider}: {r.model or 'FAILED'}"
            + (f" -> {'mentioned' if _business_mentioned(r.text, business_name) else 'not mentioned'}" if r.success else "")
            for r in results
        )
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
