import asyncio
import os
import sys

# Allow running this file directly (`python checks/api_checks.py`) as well as
# as a package module (`python -m checks.api_checks`).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx

from config import GOOGLE_API_KEY
from models import CheckResult

PAGESPEED_URL = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"
FIND_PLACE_URL = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
PLACE_DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"

REQUEST_TIMEOUT = 30.0
PAGESPEED_TIMEOUT = 90.0  # Lighthouse audits (esp. with 2 categories) can run well past 30s


async def check_website_health(website_url: str) -> list[CheckResult]:
    """Runs Google PageSpeed Insights (performance + accessibility) and
    returns two CheckResults, both scaled from the API's 0-100 score to 0-10.
    Never raises - any failure produces score-0 CheckResults instead."""
    params = {
        "url": website_url,
        "key": GOOGLE_API_KEY,
        "category": ["performance", "accessibility"],
        "strategy": "mobile",
    }

    try:
        async with httpx.AsyncClient(timeout=PAGESPEED_TIMEOUT) as client:
            response = await client.get(PAGESPEED_URL, params=params)

        if response.status_code != 200:
            reason = response.json().get("error", {}).get("message", response.text[:200])
            return [
                CheckResult(
                    check_name="Website Health",
                    score=0,
                    summary=f"PageSpeed Insights request failed (HTTP {response.status_code}): {reason}",
                    source_type="measured",
                ),
                CheckResult(
                    check_name="Accessibility Basics",
                    score=0,
                    summary=f"PageSpeed Insights request failed (HTTP {response.status_code}): {reason}",
                    source_type="measured",
                ),
            ]

        data = response.json()
        categories = data.get("lighthouseResult", {}).get("categories", {})

        perf_score_raw = categories.get("performance", {}).get("score")
        a11y_score_raw = categories.get("accessibility", {}).get("score")

        if perf_score_raw is None:
            perf_check = CheckResult(
                check_name="Website Health",
                score=0,
                summary="PageSpeed Insights did not return a performance score for this URL.",
                source_type="measured",
            )
        else:
            perf_100 = round(perf_score_raw * 100)
            perf_check = CheckResult(
                check_name="Website Health",
                score=round(perf_100 / 10),
                summary=f"PageSpeed performance score: {perf_100}/100 (mobile).",
                source_type="measured",
            )

        if a11y_score_raw is None:
            a11y_check = CheckResult(
                check_name="Accessibility Basics",
                score=0,
                summary="PageSpeed Insights did not return an accessibility score for this URL.",
                source_type="measured",
            )
        else:
            a11y_100 = round(a11y_score_raw * 100)
            a11y_check = CheckResult(
                check_name="Accessibility Basics",
                score=round(a11y_100 / 10),
                summary=f"PageSpeed accessibility score: {a11y_100}/100.",
                source_type="measured",
            )

        return [perf_check, a11y_check]

    except Exception as exc:
        summary = f"Website health check failed: {type(exc).__name__}: {exc}" if str(exc) else f"Website health check failed: {type(exc).__name__}"
        return [
            CheckResult(check_name="Website Health", score=0, summary=summary, source_type="measured"),
            CheckResult(check_name="Accessibility Basics", score=0, summary=summary, source_type="measured"),
        ]


async def check_google_presence(
    business_name: str, website_url: str
) -> tuple[list[CheckResult], dict]:
    """Looks up the business via Places API (Find Place -> Place Details) and
    returns two CheckResults plus a raw-data dict with review counts needed
    for a later calculation (e.g. Lead Revival). Never raises - failures and
    "not found" both produce score-0 CheckResults with an honest summary."""
    raw_data = {"review_count": 0, "reviews_with_owner_response": 0}

    not_found_checks = [
        CheckResult(
            check_name="Google Business Profile",
            score=0,
            summary=f"No Google Business Profile found for '{business_name}'.",
            source_type="measured",
        ),
        CheckResult(
            check_name="Google Reviews",
            score=0,
            summary="No reviews available - business profile not found.",
            source_type="measured",
        ),
    ]

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            find_params = {
                "input": business_name,
                "inputtype": "textquery",
                "fields": "place_id,name",
                "key": GOOGLE_API_KEY,
            }
            find_response = await client.get(FIND_PLACE_URL, params=find_params)
            find_data = find_response.json()

            if find_response.status_code != 200 or find_data.get("status") != "OK":
                status = find_data.get("status", f"HTTP {find_response.status_code}")
                checks = [
                    CheckResult(
                        check_name="Google Business Profile",
                        score=0,
                        summary=f"Could not find a Google Business Profile for '{business_name}' (status: {status}).",
                        source_type="measured",
                    ),
                    CheckResult(
                        check_name="Google Reviews",
                        score=0,
                        summary="No reviews available - business profile not found.",
                        source_type="measured",
                    ),
                ]
                return checks, raw_data

            candidates = find_data.get("candidates", [])
            if not candidates:
                return not_found_checks, raw_data

            place_id = candidates[0]["place_id"]

            details_params = {
                "place_id": place_id,
                "fields": "name,type,opening_hours,photo,rating,user_ratings_total,review",
                "key": GOOGLE_API_KEY,
            }
            details_response = await client.get(PLACE_DETAILS_URL, params=details_params)
            details_data = details_response.json()

            if details_response.status_code != 200 or details_data.get("status") != "OK":
                status = details_data.get("status", f"HTTP {details_response.status_code}")
                checks = [
                    CheckResult(
                        check_name="Google Business Profile",
                        score=0,
                        summary=f"Found a matching place but details lookup failed (status: {status}).",
                        source_type="measured",
                    ),
                    CheckResult(
                        check_name="Google Reviews",
                        score=0,
                        summary="No reviews available - place details lookup failed.",
                        source_type="measured",
                    ),
                ]
                return checks, raw_data

            result = details_data.get("result", {})

            # --- Google Business Profile ---
            category = (result.get("types") or ["uncategorized"])[0].replace("_", " ")
            has_hours = "opening_hours" in result
            photo_count = len(result.get("photos", []))

            profile_score = 3  # base score for being found at all
            if has_hours:
                profile_score += 3
            if photo_count >= 3:
                profile_score += 2
            elif photo_count >= 1:
                profile_score += 1
            if category != "uncategorized":
                profile_score += 2
            profile_score = min(profile_score, 10)

            profile_summary = (
                f"Found on Google as '{result.get('name', business_name)}' "
                f"(category: {category}, hours listed: {'yes' if has_hours else 'no'}, "
                f"photos: {photo_count})."
            )
            profile_check = CheckResult(
                check_name="Google Business Profile",
                score=profile_score,
                summary=profile_summary,
                source_type="measured",
            )

            # --- Google Reviews ---
            rating = result.get("rating")
            review_count = result.get("user_ratings_total", 0)
            reviews = result.get("reviews", [])

            # The Places API does not currently expose owner-response data on
            # individual reviews; this checks for it defensively so the count
            # stays accurate if that ever changes, rather than assuming 0.
            reviews_with_response = sum(
                1 for r in reviews if r.get("owner_response") or r.get("author_response")
            )

            raw_data = {
                "review_count": review_count,
                "reviews_with_owner_response": reviews_with_response,
            }

            if rating is None:
                reviews_check = CheckResult(
                    check_name="Google Reviews",
                    score=0,
                    summary="Business is listed but has no rating or reviews yet.",
                    source_type="measured",
                )
            else:
                reviews_score = min(round(rating * 2), 10)
                reviews_summary = f"{rating}/5 average rating across {review_count} reviews."
                reviews_check = CheckResult(
                    check_name="Google Reviews",
                    score=reviews_score,
                    summary=reviews_summary,
                    source_type="measured",
                )

            return [profile_check, reviews_check], raw_data

    except Exception as exc:
        summary = f"Google presence check failed: {type(exc).__name__}: {exc}" if str(exc) else f"Google presence check failed: {type(exc).__name__}"
        checks = [
            CheckResult(check_name="Google Business Profile", score=0, summary=summary, source_type="measured"),
            CheckResult(check_name="Google Reviews", score=0, summary=summary, source_type="measured"),
        ]
        return checks, raw_data


if __name__ == "__main__":

    async def _main():
        business_name = "Blue Bottle Coffee San Francisco"
        website_url = "https://bluebottlecoffee.com"

        print(f"Checking website health for {website_url}...")
        health_checks = await check_website_health(website_url)
        for check in health_checks:
            print(f"  {check.check_name}: {check.score}/10 [{check.source_type}] - {check.summary}")

        print(f"\nChecking Google presence for '{business_name}'...")
        presence_checks, raw_data = await check_google_presence(business_name, website_url)
        for check in presence_checks:
            print(f"  {check.check_name}: {check.score}/10 [{check.source_type}] - {check.summary}")
        print(f"\nRaw data for later calculations: {raw_data}")

    asyncio.run(_main())
