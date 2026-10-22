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

from models import CheckResult, short_business_type

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


# Grok's reasoning + web search regularly runs past 90 seconds.
PROVIDER_TIMEOUTS = {"Grok": 140.0}


async def _post_with_model_fallback(provider: str, models: list, send) -> ProviderResult:
    """send(client, model) -> httpx.Response. Tries each model in order."""
    last_error = "no models configured"
    async with httpx.AsyncClient(timeout=PROVIDER_TIMEOUTS.get(provider, AI_CALL_TIMEOUT)) as client:
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
    parts = [data["output_text"]] if isinstance(data.get("output_text"), str) else []
    for item in data.get("output", []) or []:
        if item.get("type") == "message":
            for content in item.get("content", []) or []:
                if content.get("type") in ("output_text", "text"):
                    if not isinstance(data.get("output_text"), str):
                        parts.append(content.get("text", ""))
                    # Cited sources (URL + title) count too: an answer that
                    # links to their website is a recommendation.
                    for note in content.get("annotations", []) or []:
                        parts.append(f"{note.get('title') or ''} {note.get('url') or ''}")
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
            texts = []
            for block in data.get("content", []) or []:
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))
                    for cite in block.get("citations", []) or []:
                        texts.append(f" [{cite.get('title') or ''} {cite.get('url') or ''}] ")
            result.text = "".join(texts)
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
        # Google rejects Search + Maps together on some models, so Maps
        # (closest to how Gemini answers "best plumber near me") goes alone.
        result = await _post_with_model_fallback("Gemini", GEMINI_MODELS, sender([{"google_maps": {}}]))
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
                    source = chunk.get(kind) or {}
                    if source.get("title") or source.get("uri"):
                        texts.append(f"{source.get('title') or ''} {source.get('uri') or ''}")
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


# Words that are part of a legal or trading name but not how people (or AI
# assistants) actually refer to a business: "Xavier Financial & Mortgage
# Group, LLC" is usually written "Xavier Financial".
NAME_SUFFIX_WORDS = {
    "llc", "inc", "incorporated", "co", "company", "corp", "corporation", "ltd",
    "pllc", "pc", "lp", "llp", "group", "the",
}
PHONE_PATTERN = re.compile(r"\(?\b\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b")


def _name_words(text: str) -> list:
    text = (text or "").lower().replace("&", " and ").replace("'", "").replace("\u2019", "")
    return re.sub(r"[^a-z0-9]+", " ", text).split()


def _core_name_words(business_name: str) -> list:
    words = _name_words(business_name)
    while words and words[-1] in NAME_SUFFIX_WORDS | {"and"}:
        words.pop()
    while words and words[0] in NAME_SUFFIX_WORDS:
        words.pop(0)
    return words


def _website_host(website_url: Optional[str]) -> Optional[str]:
    host = re.sub(r"^[a-z]+://", "", (website_url or "").strip().lower()).split("/")[0].split(":")[0]
    host = host[4:] if host.startswith("www.") else host
    return host if "." in host else None


def _business_mentioned(
    text: str, business_name: str, website_url: Optional[str] = None, phone: Optional[str] = None
) -> bool:
    """True when an AI answer names the business: its name (with or without
    LLC / Inc / Group etc., or just the first two words of a longer name),
    its website, or its phone number."""
    if not text:
        return False
    if business_name.lower().strip() in text.lower():
        return True

    padded_text = " " + " ".join(_name_words(text)) + " "
    core = _core_name_words(business_name)
    if core and (" " + " ".join(core) + " ") in padded_text:
        return True
    # "Xavier Financial and Mortgage" -> "Xavier Financial". Only for names
    # of 3+ words, and only when the shortened name still has some length,
    # so a two-letter or one-word fragment can't match by accident.
    if len(core) >= 3:
        short = core[:2]
        if short[-1] != "and" and len("".join(short)) >= 8 and (" " + " ".join(short) + " ") in padded_text:
            return True

    host = _website_host(website_url)
    if host and host in text.lower():
        return True

    phone_digits = re.sub(r"\D", "", phone or "")[-10:]
    if len(phone_digits) == 10:
        for found in PHONE_PATTERN.findall(text):
            if re.sub(r"\D", "", found)[-10:] == phone_digits:
                return True

    # Fuzzy fallback: compare against capitalized-phrase candidates in the
    # response (how listed business names typically read), to catch
    # near-matches like missing punctuation.
    targets = [business_name.lower()] + ([" ".join(core)] if core else [])
    candidates = re.findall(r"[A-Z][A-Za-z&'.,-]*(?:\s+[A-Z][A-Za-z&'.,-]*){0,5}", text)
    for candidate in candidates:
        for target in targets:
            if difflib.SequenceMatcher(None, candidate.lower(), target).ratio() >= 0.8:
                return True
    return False


def _visibility_prompts(category: str, location: str, service: Optional[str]) -> list:
    """Three differently worded questions a real customer might ask. One
    question can't prove a business is never recommended - answers vary with
    wording, so we ask a few and report how often they come up."""
    category = category.lower()
    city = (location or "").split(",")[0].strip() or location
    prompts = [
        f"I need a {category} in {location}. Who are the best ones to call? "
        "Give me a few specific local business names.",
        f"Can you recommend a trustworthy, well-reviewed {category} near {city}? "
        "I'd like the names of a few local businesses I could contact.",
    ]
    service = short_business_type(service)
    if service and service != category:
        prompts.append(f"Who do people in {location} use for {service}? Name a few specific local businesses.")
    else:
        prompts.append(f"Which {category} in {location} has the best reputation? Name a few local options.")
    return prompts


def _and_list(items: list) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


PROVIDER_CALLS = (("Claude", _call_claude), ("ChatGPT", _call_openai), ("Grok", _call_xai), ("Gemini", _call_gemini))


async def check_ai_visibility(
    business_name: str,
    category: str,
    location: str,
    website_url: Optional[str] = None,
    phone: Optional[str] = None,
    service: Optional[str] = None,
) -> CheckResult:
    """Asks Claude, ChatGPT, Grok, and Gemini - each with live web search on,
    the way their consumer apps answer local questions - three differently
    worded questions a new customer would ask, and counts how many answers
    name the business (by name, website or phone). category should be a
    specific search category ("Plumber", "Mortgage broker"); service is what
    the business wrote it does on the form. Each provider handles its own
    failures. Never raises."""
    city, _, state = (location or "").partition(",")
    place = {"city": city.strip() or None, "state": state.strip() or None}
    prompts = _visibility_prompts(category, location, service)

    jobs = [(provider, prompt, call) for prompt in prompts for provider, call in PROVIDER_CALLS]
    results = await asyncio.gather(*(_call_with_retry(call, prompt, place) for _, prompt, call in jobs))

    per_provider = {provider: {"asked": 0, "mentioned": 0, "errors": []} for provider, _ in PROVIDER_CALLS}
    for (provider, prompt, _), result in zip(jobs, results):
        tally = per_provider[provider]
        if not result.success:
            print(f"[ai_visibility] {provider} check failed: {result.error}")
            tally["errors"].append(result.error)
            continue
        tally["asked"] += 1
        hit = _business_mentioned(result.text, business_name, website_url, phone)
        tally["mentioned"] += int(hit)
        print(f"[ai_visibility] {provider} ({result.model}) -> {'mentioned' if hit else 'not mentioned'}: {prompt!r}")

    answered = sum(t["asked"] for t in per_provider.values())
    mentions = sum(t["mentioned"] for t in per_provider.values())
    failed = [
        f"{provider} ({_plain_failure_reason(t['errors'][0])})"
        for provider, t in per_provider.items()
        if t["asked"] == 0
    ]

    if answered == 0:
        return CheckResult(
            check_name="AI Search Visibility",
            score=0,
            summary="Not checked - none of the AI assistants answered this time.",
            source_type="not_checked",
        )

    # Share of answers that named them, 0-10, standard rounding.
    score = int(10 * mentions / answered + 0.5)
    providers_checked = [p for p, t in per_provider.items() if t["asked"]]
    example = prompts[0].split(" Who are")[0].rstrip(".")
    if mentions:
        found_by = ", ".join(
            f"{p} {t['mentioned']} of {t['asked']}" for p, t in per_provider.items() if t["mentioned"]
        )
        summary = (
            f"Came up in {mentions} of the {answered} test questions we asked AI assistants ({found_by})."
        )
        missed = [p for p, t in per_provider.items() if t["asked"] and not t["mentioned"]]
        if missed:
            summary += f" Didn't come up in answers from {_and_list(missed)}."
    else:
        summary = (
            f"Didn't come up in any of the {answered} test questions we asked "
            f"{_and_list(providers_checked)}."
        )
    summary += (
        f' We asked questions like "{example}". AI answers vary with the person, their location '
        "and how they word the question, so customers may still find you this way - "
        "this shows how consistently you come up."
    )
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
        result = await check_ai_visibility(business_name, category, location, None, None, category)
        print(f"\n{result.check_name}: {result.score}/10 [{result.source_type}]")
        print(result.summary)

    asyncio.run(_main())
