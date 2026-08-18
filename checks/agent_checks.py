import asyncio
import os
import re
import sys

# Allow running this file directly (`python checks/agent_checks.py`) as well
# as a package module (`python -m checks.agent_checks`).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx

from config import ANTHROPIC_API_KEY
from models import CheckResult

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-sonnet-5"

# Web search tool calls can involve several search round-trips inside a
# single API call, so give this more room than a plain completion.
AGENT_CALL_TIMEOUT = 90.0

RESPONSE_FORMAT_INSTRUCTIONS = """
After researching, respond in EXACTLY this format and nothing else:
SCORE: <integer 0-10>
SUMMARY: <one or two sentences citing what you actually found>
"""


def _parse_score_summary(text: str):
    score_match = re.search(r"SCORE:\s*(\d+)", text)
    if not score_match:
        return None
    score = max(0, min(10, int(score_match.group(1))))

    summary_match = re.search(r"SUMMARY:\s*(.+)", text, re.DOTALL)
    summary = re.sub(r"\s+", " ", summary_match.group(1)).strip() if summary_match else text.strip()[:300]
    return score, summary


async def _run_agent_check(check_name: str, system_prompt: str, user_prompt: str) -> CheckResult:
    """Runs one Claude + web-search-tool call and parses a SCORE/SUMMARY out
    of the response. Never raises - any failure (network, API error, or an
    unparseable response) produces a score-0 CheckResult with a clear
    summary instead."""
    try:
        async with httpx.AsyncClient(timeout=AGENT_CALL_TIMEOUT) as client:
            response = await client.post(
                ANTHROPIC_URL,
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": ANTHROPIC_MODEL,
                    "max_tokens": 1500,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_prompt}],
                    "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}],
                },
            )

        if response.status_code != 200:
            return CheckResult(
                check_name=check_name,
                score=0,
                summary=f"{check_name} check failed (HTTP {response.status_code}): {response.text[:200]}",
                source_type="measured",
            )

        data = response.json()
        full_text = "\n".join(
            block.get("text", "") for block in data.get("content", []) if block.get("type") == "text"
        )

        parsed = _parse_score_summary(full_text)
        if parsed is None:
            excerpt = full_text[:200] or "(empty response)"
            return CheckResult(
                check_name=check_name,
                score=0,
                summary=f"Could not parse a score from the AI's response. Raw excerpt: {excerpt}",
                source_type="measured",
            )

        score, summary = parsed
        return CheckResult(check_name=check_name, score=score, summary=summary, source_type="measured")

    except Exception as exc:
        summary = (
            f"{check_name} check failed: {type(exc).__name__}: {exc}"
            if str(exc)
            else f"{check_name} check failed: {type(exc).__name__}"
        )
        return CheckResult(check_name=check_name, score=0, summary=summary, source_type="measured")


async def check_local_ranking(business_name: str, category: str, location: str) -> CheckResult:
    system_prompt = f"""You are evaluating local search visibility for a small business.

Search the web for "{category} near {location}" and identify which businesses appear
prominently in the results (map pack / local results, top organic results, directory
listings for that search).

Score 0-10 for whether and where "{business_name}" appears:
- 0: Does not appear anywhere in the results you found for this search
- 1-3: Appears only deep in results (e.g. page 2+, or a passing directory mention), not
  among the top local results
- 4-6: Appears in general search results but not in the top local map-pack style results
- 7-8: Appears among the top 10 local results for the search
- 9-10: Appears in the top 3 results / local map pack for the search
{RESPONSE_FORMAT_INSTRUCTIONS}"""
    user_prompt = (
        f'Search for "{category} near {location}" and determine whether and where '
        f"'{business_name}' appears in the results."
    )
    return await _run_agent_check("Local Search Ranking", system_prompt, user_prompt)


async def check_social_presence(business_name: str, website_url: str) -> CheckResult:
    system_prompt = f"""You are evaluating a small business's social media presence.

Search the web for "{business_name}" social media profiles (Facebook, Instagram, LinkedIn,
etc.), and cross-reference what you find against the business's website ({website_url})
where possible, to check existence, recent activity, and consistency (matching name,
contact info, and branding across platforms and the website).

Score 0-10:
- 0: No social media profiles found for this business
- 1-3: One or two profiles found but they appear inactive/abandoned (no recent posts) or
  don't clearly match this specific business
- 4-6: Active profile(s) found, but presence is inconsistent - missing on major platforms,
  sporadic activity, or inconsistent branding/contact info versus the website
- 7-8: Active presence on multiple major platforms with reasonably recent activity and
  consistent branding/contact info
- 9-10: Strong, active, consistent presence across multiple major platforms with recent
  posts and information matching the website
{RESPONSE_FORMAT_INSTRUCTIONS}"""
    user_prompt = (
        f"Search for '{business_name}' social media profiles and check them against "
        f"the business website {website_url} for existence, activity, and consistency."
    )
    return await _run_agent_check("Social Media Presence", system_prompt, user_prompt)


async def check_reputation(business_name: str, location: str) -> CheckResult:
    system_prompt = f"""You are evaluating a small business's online reputation on review
platforms OTHER THAN Google (Google reviews are checked separately elsewhere).

Search the web for "{business_name}" reviews on Yelp, Facebook, BBB, Angi, HomeAdvisor,
and other relevant review or directory platforms. The business is located near {location}.

Score 0-10:
- 0: No reviews or listings found on any non-Google platform
- 1-3: Found on one platform but with very few reviews or a poor/mixed rating
- 4-6: Found on 1-2 platforms with a decent rating (roughly 3.5-4.2 stars equivalent) or
  only a moderate number of reviews
- 7-8: Found on multiple platforms with generally positive ratings (roughly 4.3+ stars
  equivalent) and a reasonable number of reviews
- 9-10: Strong presence across multiple review platforms with consistently excellent
  ratings and a substantial number of reviews
{RESPONSE_FORMAT_INSTRUCTIONS}"""
    user_prompt = (
        f"Search for '{business_name}' (near {location}) reviews on Yelp and other "
        "non-Google review platforms, and assess their online reputation there."
    )
    return await _run_agent_check("Online Reputation Scan", system_prompt, user_prompt)


if __name__ == "__main__":

    async def _main():
        business_name = "Erickson's Drying Systems"
        category = "water damage restoration"
        location = "Fort Myers, FL"
        website_url = "https://www.ericksonsdrying.com"

        results = await asyncio.gather(
            check_local_ranking(business_name, category, location),
            check_social_presence(business_name, website_url),
            check_reputation(business_name, location),
        )
        for result in results:
            print(f"{result.check_name}: {result.score}/10 [{result.source_type}]")
            print(f"  {result.summary}\n")

    asyncio.run(_main())
