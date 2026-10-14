import asyncio
import os
import re
from typing import Optional
import sys

# Allow running this file directly (`python checks/agent_checks.py`) as well
# as a package module (`python -m checks.agent_checks`).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx

from config import ANTHROPIC_API_KEY
from models import CheckResult

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-sonnet-5-5"

# Web search tool calls can involve several search round-trips inside a
# single API call, so give this more room than a plain completion.
AGENT_CALL_TIMEOUT = 90.0

RESPONSE_FORMAT_INSTRUCTIONS = """
After researching, respond in EXACTLY this format and nothing else:
SCORE: <integer 0-10>
SUMMARY: <one or two sentences citing what you actually found>

The SUMMARY is shown to the business owner. Address them as "you"/"your business". Never
write in the first person ("I", "my searches", "we couldn't load") and never describe your
research process or tools - state the finding.
"""


def _parse_score_summary(text: str):
    score_match = re.search(r"SCORE:\s*(\d+)", text)
    if not score_match:
        return None
    score = max(0, min(10, int(score_match.group(1))))

    summary_match = re.search(r"SUMMARY:\s*(.+)", text, re.DOTALL)
    summary = re.sub(r"\s+", " ", summary_match.group(1)).strip() if summary_match else text.strip()[:300]
    return score, summary


# Claude API statuses worth retrying: rate limit, overload, server errors.
RETRY_STATUSES = (429, 500, 502, 503, 529)
MAX_ATTEMPTS = 3


async def post_with_retry(client: httpx.AsyncClient, url: str, label: str, **kwargs) -> httpx.Response:
    """POST with up to MAX_ATTEMPTS tries on rate limits/overloads, waiting
    for the server's retry-after (capped at 20s) or 5s, 10s."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = await client.post(url, **kwargs)
        if response.status_code not in RETRY_STATUSES or attempt == MAX_ATTEMPTS:
            return response
        try:
            wait = min(20.0, float(response.headers.get("retry-after", 0)) or 5.0 * attempt)
        except ValueError:
            wait = 5.0 * attempt
        print(f"[agent] {label}: HTTP {response.status_code}, retrying in {wait:.0f}s (attempt {attempt + 1})")
        await asyncio.sleep(wait)
    return response


async def _run_agent_check(
    check_name: str, system_prompt: str, user_prompt: str, extra_keys: tuple = ()
) -> CheckResult:
    """Runs one Claude + web-search-tool call and parses a SCORE/SUMMARY out
    of the response. Never raises - any failure (network, API error, or an
    unparseable response) produces a score-0 CheckResult with a clear
    summary instead."""
    try:
        async with httpx.AsyncClient(timeout=AGENT_CALL_TIMEOUT) as client:
            response = await post_with_retry(
                client,
                ANTHROPIC_URL,
                check_name,
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
            # Our side failed (credits, outage) - never show the raw error to
            # the client or count it against the business.
            print(f"[agent] {check_name} failed: HTTP {response.status_code}: {' '.join(response.text.split())[:300]}")
            return CheckResult(
                check_name=check_name,
                score=0,
                summary="We couldn't complete this check this time.",
                source_type="not_checked",
            )

        data = response.json()
        full_text = "\n".join(
            block.get("text", "") for block in data.get("content", []) if block.get("type") == "text"
        )

        parsed = _parse_score_summary(full_text)
        if parsed is None:
            excerpt = full_text[:200] or "(empty response)"
            print(f"[agent] {check_name}: couldn't parse a score. Excerpt: {' '.join(excerpt.split())}")
            return CheckResult(
                check_name=check_name,
                score=0,
                summary="We couldn't complete this check this time.",
                source_type="not_checked",
            )

        score, summary = parsed
        details = {}
        # extra_keys: "KEY" (answer yes/no/unknown) or ("KEY", "a|b|c", default).
        for item in extra_keys:
            key, allowed, default = (item, "yes|no|unknown", "unknown") if isinstance(item, str) else item
            match = re.search(rf"(?m)^\s*{key}:\s*({allowed})\b", full_text, re.IGNORECASE)
            details[key.lower()] = match.group(1).lower() if match else default
            # Keep the KEY: lines out of the client-facing summary.
            summary = re.sub(rf"(?m)^\s*{key}:\s*({allowed})\b", "", summary, flags=re.IGNORECASE).strip()
        return CheckResult(
            check_name=check_name, score=score, summary=summary, source_type="measured", details=details
        )

    except Exception as exc:
        print(f"[agent] {check_name} failed: {type(exc).__name__}: {exc}")
        return CheckResult(
            check_name=check_name,
            score=0,
            summary="We couldn't complete this check this time.",
            source_type="not_checked",
        )


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


SOCIAL_PLATFORMS = ("FACEBOOK", "INSTAGRAM", "LINKEDIN", "TIKTOK", "YOUTUBE", "X", "NEXTDOOR")


async def check_social_presence(
    business_name: str,
    website_url: str,
    location: Optional[str] = None,
    phone: Optional[str] = None,
    site_links: Optional[list] = None,
    stated_handles: str = "",
) -> CheckResult:
    where = f" in {location}" if location else ""
    if site_links:
        evidence = (
            "Their own website links to these social profiles - treat each as belonging to this "
            f"business: {', '.join(site_links)}. Check each one you can."
        )
    elif site_links is not None:
        evidence = "Their website has no links to social profiles (that is itself worth noting)."
    else:
        evidence = "We couldn't read their website for social links."
    if stated_handles.strip():
        evidence += (
            f" The business owner told us their profiles are: {stated_handles.strip()}. Treat these as "
            "theirs and check each one. If their website doesn't link to any of them, say so: "
            "customers can't find those profiles from the site."
        )
    system_prompt = f"""You are evaluating a small business's social media presence.
The business is "{business_name}"{where}, website {website_url}{', phone ' + phone if phone else ''}.

{evidence}

Also search for their profiles on Facebook, Instagram, LinkedIn, TikTok, YouTube and Nextdoor.
Social handles often differ from the business name (a marine company might be "@absolutedocks"),
so also search by their website domain, phone number, city, and the services they offer, not
just the exact name. Count a profile as theirs when it matches their website, phone, address or
city. Other businesses share names - ignore profiles that clearly belong to a different business.

Instagram and Facebook often can't be read without logging in. A profile that clearly exists
counts as found even if you can't see its posts. Don't judge posting activity.

Before SCORE, add two lines per platform, exactly in this form:
FACEBOOK: found | not_found
FACEBOOK_FOLLOWERS: <number, e.g. 1200> | unknown
INSTAGRAM: found | not_found
INSTAGRAM_FOLLOWERS: <number> | unknown
LINKEDIN: found | not_found
LINKEDIN_FOLLOWERS: <number> | unknown
TIKTOK: found | not_found
TIKTOK_FOLLOWERS: <number> | unknown
YOUTUBE: found | not_found
YOUTUBE_FOLLOWERS: <number> | unknown
X: found | not_found
X_FOLLOWERS: <number> | unknown
NEXTDOOR: found | not_found
NEXTDOOR_FOLLOWERS: <number> | unknown
Only give a follower (or page like) count you actually saw; otherwise write unknown.
For SCORE, give any number 0-10 (it's recalculated from the lines above).
In the SUMMARY, name the profiles you found (platform and handle) and say whether their website
links to them. Don't mention a score or posting activity.
{RESPONSE_FORMAT_INSTRUCTIONS}"""
    user_prompt = (
        f"Find {business_name}'s social media profiles{where} (website {website_url}) and check "
        "their existence, activity, and consistency with the website."
    )
    return await _run_agent_check(
        "Social Media Presence",
        system_prompt,
        user_prompt,
        extra_keys=tuple(
            item
            for key in SOCIAL_PLATFORMS
            for item in ((key, "found|not_found", "not_found"), (f"{key}_FOLLOWERS", r"[\d][\d,.]*[ \t]*[kKmM]?|unknown", "unknown"))
        ),
    )


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


async def check_nap_consistency(business_name: str, address: str, phone: str) -> CheckResult:
    system_prompt = f"""You are auditing Name/Address/Phone (NAP) consistency for a business
across online directories. This is one of the most failure-prone checks in this system - if
you can't get a clear, confirmed read after a genuine search effort, say so plainly and score
accordingly. Do not guess or assume consistency you haven't actually verified.

The business's reference info, as we have it on file, is:
Name: {business_name}
Address: {address}
Phone: {phone}

Search the web for this business's listing on Yelp, Apple Maps, Bing Places, and Facebook.
For each one you can find, compare the name, address, and phone number shown there against the
reference info above. If you happen to notice the business on other directories or data
aggregators (e.g. ZoomInfo, Yellow Pages, BBB) during your search and spot a mismatch there
too, note that as well - an inconsistency anywhere reflects poorly on NAP consistency, not
just on those four platforms.

Score 0-10:
- 0: Business not found on any of the four directories, OR found but shows wildly
  inconsistent info (multiple different addresses/phone numbers across listings) with no way
  to tell which is correct
- 1-3: Found on only 1-2 of the four directories, and what you found shows a clear mismatch
  (wrong phone number or address) versus the reference info
- 4-6: Found on 2-3 directories with partial consistency - some listings match the reference,
  others show a discrepancy (e.g. an outdated address, or a different phone number)
- 7-8: Found on most (3-4) of the directories with mostly consistent info, at most one minor
  discrepancy
- 9-10: Found on all four directories with name, address, and phone consistently matching the
  reference info

If your search genuinely can't turn up enough information to judge on most of these
platforms, don't guess a middle score out of politeness - say so plainly in the summary
("could not confirm listings on X, Y") and let the score reflect that real uncertainty, since
we can't confirm consistency we can't find.
{RESPONSE_FORMAT_INSTRUCTIONS}"""
    user_prompt = (
        f"Search for '{business_name}' listings on Yelp, Apple Maps, Bing Places, and Facebook, "
        f"and compare the name, address, and phone number shown on each against our reference "
        f"info - address '{address}', phone '{phone}'."
    )
    return await _run_agent_check("Consistency, Everywhere", system_prompt, user_prompt)


async def check_ad_activity(
    business_name: str,
    website_url: str = "",
    location: Optional[str] = None,
    ad_tags: Optional[list] = None,
    category: Optional[str] = None,
) -> CheckResult:
    where = f" in {location}" if location else ""
    if ad_tags:
        evidence = (
            f"The business's own website ({website_url}) has these ad tags installed: {', '.join(ad_tags)}. "
            "They make it possible to run, track and retarget ads on those platforms, but they are not "
            "proof that ads are running right now."
        )
    elif ad_tags is not None:
        evidence = (
            f"The business's website ({website_url}) has NO Google Ads or Meta (Facebook/Instagram) ad "
            "tags installed. Businesses running ads usually install these, so this leans toward no "
            "paid ads (or ads running without any tracking)."
        )
    else:
        evidence = f"Their website is {website_url}; we couldn't read it for ad tracking tags."
    system_prompt = f"""You are checking whether a local business is currently running paid ads.
The business is "{business_name}"{where}, website {website_url}. Other businesses share this
name in other places - only count evidence that clearly belongs to THIS one (matching city,
website, or phone).

{evidence}

Also try Meta's Ad Library and Google's Ads Transparency Center for this exact business, but
expect those tools to often be unreadable - that is normal and not itself a finding.

Also check whether OTHER local businesses of the same type appear as paid/sponsored results
(Google sponsored results or Local Services Ads "Google Guaranteed", Yelp sponsored) for a
search like "{category or 'their service'} in {location or 'their area'}".

Before SCORE, add these two lines exactly:
ADS_RUNNING: yes | no | unknown   (is THIS business currently running paid ads anywhere -
  Google, Meta, Yelp, other? Ad tracking tags alone are not proof of ads running.)
COMPETITORS_ADVERTISING: yes | no | unknown   (are competitors paying to appear in that search?)

Score 0-10 for how confident you are THIS business is running ads (0 = clearly not, 10 = confirmed).
In the SUMMARY, describe what you found for the business owner, including any competitor ads.
{RESPONSE_FORMAT_INSTRUCTIONS}"""
    user_prompt = (
        f"Is {business_name}{where} ({website_url}) running paid ads on Google or Meta right now? "
        "Report what you find."
    )
    return await _run_agent_check(
        "Visible Ad Activity", system_prompt, user_prompt, extra_keys=("ADS_RUNNING", "COMPETITORS_ADVERTISING")
    )


if __name__ == "__main__":

    async def _main():
        business_name = "Erickson's Drying Systems"
        category = "water damage restoration"
        location = "Fort Myers, FL"
        website_url = "https://www.ericksonsdrying.com"
        address = "12651 Metro Pkwy, Fort Myers, FL 33966, USA"
        phone = "(239) 277-7744"

        results = await asyncio.gather(
            check_local_ranking(business_name, category, location),
            check_social_presence(business_name, website_url),
            check_reputation(business_name, location),
            check_nap_consistency(business_name, address, phone),
            check_ad_activity(business_name),
        )
        for result in results:
            print(f"{result.check_name}: {result.score}/10 [{result.source_type}]")
            print(f"  {result.summary}\n")

    asyncio.run(_main())
