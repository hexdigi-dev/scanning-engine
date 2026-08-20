import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import httpx

from config import ANTHROPIC_API_KEY
from models import CheckResult, LeakEstimate

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-sonnet-5"
SYNTHESIS_TIMEOUT = 60.0

# Tone calibration only, pulled from our actual site copy - not a literal
# template to force onto whichever check happens to come first. The third
# example in particular describes a hands-on UX walkthrough we don't have a
# check for yet; it's here for voice, not to be assigned to another check.
VOICE_EXAMPLES = [
    "Claimed, right category, accurate name/address/phone, hours, and photos — rated on its own.",
    "Your name, address, and phone compared across every directory we can find you on — not "
    "just Google — flagging anywhere they don't match.",
    "A real hands-on walkthrough of your site as a visitor would experience it — CTAs, "
    "findability, and whether the right next step is actually in front of them.",
]


@dataclass
class SynthesisResult:
    checks: List[CheckResult]
    leak_narrative: str
    review_note: Optional[str]


def _build_system_prompt() -> str:
    voice_block = "\n".join(f'- "{example}"' for example in VOICE_EXAMPLES)
    return f"""You are the final editing pass on a small-business marketing diagnostic report,
before it goes out to the business owner.

VOICE: direct, plain-language, no jargon. For tone calibration only (these describe specific
checks from our site copy - don't force a line onto a check it doesn't actually describe),
here's the voice we write in:
{voice_block}

HARD RULE, not a style preference: a check or figure with source_type "measured" is stated as
fact - direct, declarative language, no hedging. Anything that comes from an estimate
(self-reported answers, or a stated assumption) must be explicitly framed as an estimate every
time it's mentioned - phrasing like "based on what you told us" or "an estimated $X" - and must
never be stated as a confirmed fact.

REPUTATION GAP RULE, applies whenever supporting_leaks includes an entry labeled "Reputation
Gap" - this is a general rule for any business, not a one-off for whichever business happens to
be scanned:
- Pull the negative-signal specifics (complaints, bad reviews, low platform ratings) from the
  Online Reputation Scan check's summary, but describe them at a CATEGORY level, not verbatim -
  e.g. "an unresolved complaint" or "a negative review about a specific service area," never the
  exact complaint wording or a review quote. This matches how the rest of the report avoids
  overclaiming precision.
- Add a remedy tie-in, the same way headline_leak_monthly ties to the Speed-to-Lead bot and
  dormant_lead_value ties to Lead Revival:
  - If negative reviews appear on standard review platforms (Google, Yelp, Facebook, BBB's
    review section, Yellow Pages, etc.), mention that the review response service can help
    address those directly.
  - If a FORMAL BBB COMPLAINT exists - distinct from a BBB review, a separate formal process -
    explicitly call out that distinction: it requires a direct, personalized response from the
    business through BBB's own portal within a 14-day window, and can't be automated. Offer to
    help think through that response, but don't claim the bot can handle it.
- If neither negative reviews nor a formal complaint show up in the check data, skip the remedy
  tie-in rather than inventing one.

Respond with ONLY a JSON object, no markdown code fences, no commentary before or after, in
exactly this shape:
{{
  "checks": [{{"check_name": "<copied exactly from input>", "summary": "<rewritten summary>"}}],
  "leak_narrative": "<final narrative text for the leak estimate section, see instructions>",
  "review_note": "<string, or null>"
}}
The "checks" array must contain exactly one entry per input check, in the same order, with
check_name copied exactly as given - only the summary text changes."""


def _build_user_prompt(
    business_name: str, all_check_results: List[CheckResult], leak_estimate: LeakEstimate
) -> str:
    checks_payload = [
        {
            "check_name": check.check_name,
            "score": check.score,
            "summary": check.summary,
            "source_type": check.source_type,
        }
        for check in all_check_results
    ]

    # Iterate generically - supporting_leaks may hold any number of entries
    # (currently 0, but this list will grow in a later step) and every one
    # of them needs to be woven into leak_narrative, not just the first few.
    supporting_leaks_payload = [
        {"label": leak.label, "monthly_value": leak.monthly_value, "explanation": leak.explanation}
        for leak in leak_estimate.supporting_leaks
    ]

    leak_payload = {
        "headline_leak_monthly": leak_estimate.headline_leak_monthly,
        "headline_explanation": leak_estimate.headline_explanation,
        "supporting_leaks": supporting_leaks_payload,
        "dormant_lead_value": leak_estimate.dormant_lead_value,
        "dormant_lead_explanation": leak_estimate.dormant_lead_explanation,
        "flagged_for_review": leak_estimate.flagged_for_review,
    }

    return f"""Business: {business_name}

Here are all {len(all_check_results)} check results (source_type "measured" = directly tested,
state as fact; "estimated" = self-reported/assumption-based, must be framed as an estimate):
{json.dumps(checks_payload, indent=2)}

Here is the leak estimate data. Every figure in it is an ESTIMATE (self-reported answers or a
stated assumption), never measured, so headline_leak_monthly and dormant_lead_value must always
be framed as estimates in leak_narrative - never stated as a confirmed fact:
{json.dumps(leak_payload, indent=2)}

Tasks:
1. Rewrite each check's summary in our voice - direct, plain-language, no jargon. Keep
   check_name exactly as given. Keep every fact and number accurate - improve the phrasing and
   consistency, don't invent or drop information.
2. Write leak_narrative: present headline_leak_monthly as the headline number, explicitly framed
   as an estimate based on self-reported answers, and preserve the existing connection to our
   Speed-to-Lead bot as the fix for this specific leak - that connection is already present in
   headline_explanation above, so keep it, don't drop it while rewriting.
3. supporting_leaks currently has {len(supporting_leaks_payload)} entries. If there are any,
   weave every single one into leak_narrative. Treat this as a variable-length list in general -
   more entries may be added in a later step, and all of them need to show up, not just some.
   If one of them is labeled "Reputation Gap," follow the REPUTATION GAP RULE from the system
   prompt - pull specifics from the Online Reputation Scan check's summary above (if present
   among the check results), describe them at a category level, and add the appropriate remedy
   tie-in (review response service, and/or the BBB-complaint-specific caveat if a formal
   complaint is present).
4. End leak_narrative with one closing line connecting dormant_lead_value (also framed as an
   estimate) to our Lead Revival campaign, preserving that connection from
   dormant_lead_explanation above.
5. flagged_for_review is {json.dumps(leak_estimate.flagged_for_review)}. If true, set
   review_note to a short INTERNAL-ONLY note (never shown to the client) saying this report
   needs a human glance before sending, and why. If false, set review_note to null."""


def _extract_json(text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError("No JSON object found in response")


def _fallback_result(
    all_check_results: List[CheckResult], leak_estimate: LeakEstimate, reason: str
) -> SynthesisResult:
    """Used when the synthesis call fails or returns something we can't
    parse. Never crashes /scan over a rewriting failure - falls back to the
    original, unpolished check summaries and the deterministic leak
    explanation text (which already carries the Speed-to-Lead/Lead Revival
    ties from calculation.py), and always surfaces the failure internally
    via review_note so a human knows the output wasn't AI-polished."""
    narrative = f"{leak_estimate.headline_explanation} {leak_estimate.dormant_lead_explanation}"
    note = f"AI synthesis step failed ({reason}); showing unpolished check summaries and the raw leak estimate text instead of a rewritten narrative."
    if leak_estimate.flagged_for_review:
        note += " Additionally, flagged_for_review is True on the underlying leak estimate - please sanity check the figures before sending."
    return SynthesisResult(checks=list(all_check_results), leak_narrative=narrative, review_note=note)


async def synthesize_report(
    business_name: str, all_check_results: List[CheckResult], leak_estimate: LeakEstimate
) -> SynthesisResult:
    """Makes one final Claude call to rewrite every CheckResult summary in a
    consistent voice and produce the finished leak-estimate narrative. Never
    raises - any failure (network, API error, or an unparseable response)
    falls back to the original unpolished content instead."""
    try:
        async with httpx.AsyncClient(timeout=SYNTHESIS_TIMEOUT) as client:
            response = await client.post(
                ANTHROPIC_URL,
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": ANTHROPIC_MODEL,
                    "max_tokens": 4000,
                    "system": _build_system_prompt(),
                    "messages": [
                        {
                            "role": "user",
                            "content": _build_user_prompt(business_name, all_check_results, leak_estimate),
                        }
                    ],
                },
            )

        if response.status_code != 200:
            return _fallback_result(
                all_check_results, leak_estimate, f"HTTP {response.status_code}: {response.text[:200]}"
            )

        data = response.json()
        full_text = "\n".join(
            block.get("text", "") for block in data.get("content", []) if block.get("type") == "text"
        )
        parsed = _extract_json(full_text)

        polished_by_name = {
            item["check_name"]: item["summary"] for item in parsed.get("checks", []) if "check_name" in item
        }
        # Preserve the original list's order, count, score, and source_type
        # exactly - only the summary text is replaced, and only for checks
        # Claude actually returned a rewrite for; anything missing keeps its
        # original summary rather than being dropped.
        polished_checks = [
            check.model_copy(update={"summary": polished_by_name.get(check.check_name, check.summary)})
            for check in all_check_results
        ]

        leak_narrative = parsed.get("leak_narrative") or leak_estimate.headline_explanation
        review_note = parsed.get("review_note")
        if leak_estimate.flagged_for_review and not review_note:
            # Safety net: don't let a missed instruction silently drop this.
            review_note = (
                "flagged_for_review is True on the underlying leak estimate - please sanity "
                "check the figures before sending."
            )

        return SynthesisResult(checks=polished_checks, leak_narrative=leak_narrative, review_note=review_note)

    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        return _fallback_result(all_check_results, leak_estimate, reason)


if __name__ == "__main__":
    from calculation import calculate_leak_estimate
    from models import ScanRequest

    async def _main():
        test_request = ScanRequest(
            contact_name="Test Contact",
            phone="555-000-0000",
            email="test@ericksonsdrying.com",
            business_name="Erickson's Drying Systems",
            website_url="https://www.ericksonsdrying.com",
            business_type="Water damage restoration & mold remediation",
            monthly_leads="25–50",
            avg_job_value="$4,500",
            response_time="Within an hour",
            close_rate="25–50%",
            dormant_leads="100–500",
        )
        leak_estimate = calculate_leak_estimate(
            test_request,
            {"review_count": 22, "reviews_with_owner_response": 0},
            ai_visibility_score=0,
            local_ranking_score=1,
            reputation_score=4,
            lcp_seconds=4.2,
        )

        sample_checks = [
            CheckResult(
                check_name="Google Business Profile",
                score=10,
                summary="Found on Google as 'Erickson's Drying Systems' (category: establishment, hours listed: yes, photos: 10).",
                source_type="measured",
            ),
            CheckResult(
                check_name="Consistency, Everywhere",
                score=2,
                summary="Yellow Pages lists a transposed-digit address and ZoomInfo lists a different phone number than the reference.",
                source_type="measured",
            ),
            CheckResult(
                check_name="AI Search Visibility",
                score=0,
                summary="Mentioned by 0 of 4 AI assistants. Not mentioned by: Claude, ChatGPT, Grok, Gemini.",
                source_type="measured",
            ),
            CheckResult(
                check_name="Online Reputation Scan",
                score=4,
                summary=(
                    "On Birdeye, Erickson's Drying Systems has a 4.1 star rating with 38 reviews, and "
                    "has a Facebook presence with 1,967 likes, but has a much weaker showing "
                    "elsewhere: BBB shows the business as Not BBB Accredited, with a formal BBB "
                    "complaint on file alleging misleading practices around a reconstruction job that "
                    "remains unresolved, Yelp's page has one negative review specifically calling out "
                    "poor reconstruction workmanship, and Trustpilot shows only a 2.8 average "
                    "TrustScore out of 5 with 3 reviews."
                ),
                source_type="measured",
            ),
        ]

        result = await synthesize_report(test_request.business_name, sample_checks, leak_estimate)
        print("review_note:", result.review_note)
        print()
        for check in result.checks:
            print(f"{check.check_name} ({check.score}/10): {check.summary}")
        print()
        print("leak_narrative:")
        print(result.leak_narrative)

    asyncio.run(_main())
