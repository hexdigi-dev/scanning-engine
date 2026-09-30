import asyncio
import os
import sys
from typing import Optional
from urllib.parse import urlparse

# Allow running this file directly (`python checks/api_checks.py`) as well as
# as a package module (`python -m checks.api_checks`).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx

from config import GOOGLE_API_KEY
from models import CheckResult

PAGESPEED_URL = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"
FIND_PLACE_URL = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
PLACE_DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"
TEXT_SEARCH_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"
# How many same-name Google listings to compare against the submitted website.
MAX_CANDIDATES_TO_CHECK = 5

REQUEST_TIMEOUT = 30.0
PAGESPEED_TIMEOUT = 90.0  # Lighthouse audits (esp. with 2 categories) can run well past 30s


async def _fetch_pagespeed_category(
    website_url: str, category: str, check_name: str, label: str
) -> tuple[CheckResult, dict]:
    """Runs a single-category PageSpeed Insights audit and returns one
    CheckResult scaled from the API's 0-100 score to 0-10, plus a raw-data
    dict (only populated for the "performance" category, which is where
    Lighthouse's Largest Contentful Paint audit lives). Never raises - any
    failure produces a score-0 CheckResult and an empty raw-data dict."""
    params = {
        "url": website_url,
        "key": GOOGLE_API_KEY,
        "category": category,
        "strategy": "mobile",
    }

    try:
        async with httpx.AsyncClient(timeout=PAGESPEED_TIMEOUT) as client:
            response = await client.get(PAGESPEED_URL, params=params)

        if response.status_code != 200:
            reason = response.json().get("error", {}).get("message", response.text[:200])
            return (
                CheckResult(
                    check_name=check_name,
                    score=0,
                    summary=f"PageSpeed Insights request failed (HTTP {response.status_code}): {reason}",
                    source_type="measured",
                ),
                {},
            )

        data = response.json()
        score_raw = data.get("lighthouseResult", {}).get("categories", {}).get(category, {}).get("score")

        if score_raw is None:
            return (
                CheckResult(
                    check_name=check_name,
                    score=0,
                    summary=f"PageSpeed Insights did not return a {label} score for this URL.",
                    source_type="measured",
                ),
                {},
            )

        raw_extra = {"score_100": round(score_raw * 100)}
        if category == "performance":
            lcp_ms = (
                data.get("lighthouseResult", {})
                .get("audits", {})
                .get("largest-contentful-paint", {})
                .get("numericValue")
            )
            raw_extra["lcp_seconds"] = round(lcp_ms / 1000, 2) if lcp_ms is not None else None
            # The mobile screenshot of the first screen, used to judge the
            # calls-to-action ("data:image/jpeg;base64,...").
            raw_extra["screenshot"] = (
                data.get("lighthouseResult", {})
                .get("audits", {})
                .get("final-screenshot", {})
                .get("details", {})
                .get("data")
            )

        score_100 = round(score_raw * 100)
        return (
            CheckResult(
                check_name=check_name,
                score=round(score_100 / 10),
                summary=f"PageSpeed {label} score: {score_100}/100 (mobile).",
                source_type="measured",
            ),
            raw_extra,
        )

    except Exception as exc:
        summary = (
            f"{check_name} check failed: {type(exc).__name__}: {exc}"
            if str(exc)
            else f"{check_name} check failed: {type(exc).__name__}"
        )
        return CheckResult(check_name=check_name, score=0, summary=summary, source_type="measured"), {}


async def check_website_health(website_url: str) -> tuple[list[CheckResult], dict]:
    """Runs Google PageSpeed Insights performance and accessibility audits
    as two concurrent single-category requests (smaller audits, run in
    parallel via asyncio.gather rather than one combined call run twice as
    slow, or sequentially) and returns two CheckResults, each scaled from
    the API's 0-100 score to 0-10, plus a raw-data dict with "lcp_seconds"
    (None if it couldn't be measured) for the website speed leak calc."""
    (performance_check, performance_raw), (accessibility_check, accessibility_raw) = await asyncio.gather(
        _fetch_pagespeed_category(website_url, "performance", "Website Health", "performance"),
        _fetch_pagespeed_category(website_url, "accessibility", "Accessibility Basics", "accessibility"),
    )
    raw_data = {
        "lcp_seconds": performance_raw.get("lcp_seconds"),
        "accessibility_score": accessibility_raw.get("score_100"),
        "screenshot": performance_raw.get("screenshot"),
    }
    return [performance_check, accessibility_check], raw_data


def _extract_city_state(result: dict) -> Optional[str]:
    city = None
    state = None
    for component in result.get("address_components", []):
        types = component.get("types", [])
        if "locality" in types:
            city = component.get("long_name")
        if "administrative_area_level_1" in types:
            state = component.get("short_name")
    if city and state:
        return f"{city}, {state}"
    return result.get("formatted_address")


def _site_host(url: Optional[str]) -> Optional[str]:
    """'https://www.Example.com/contact' -> 'example.com'."""
    if not url:
        return None
    if "://" not in url:
        url = f"https://{url}"
    host = (urlparse(url).hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host or None


def _phone_digits(phone: Optional[str]) -> Optional[str]:
    """'(239) 789-3330' / '+1 239-789-3330' -> '2397893330' (last 10 digits)."""
    digits = "".join(ch for ch in (phone or "") if ch.isdigit())
    return digits[-10:] if len(digits) >= 10 else None


def _same_site(a: Optional[str], b: Optional[str]) -> bool:
    if not a or not b:
        return False
    return a == b or a.endswith("." + b) or b.endswith("." + a)


async def _find_matching_place(
    client: httpx.AsyncClient,
    business_name: str,
    website_url: str,
    phone: Optional[str] = None,
    city: Optional[str] = None,
    street_address: Optional[str] = None,
    zip_code: Optional[str] = None,
) -> tuple[Optional[str], int]:
    """Many businesses share a name ("Scott's Plumbing" exists in several
    states), so the first search result can be the wrong company. This checks
    up to MAX_CANDIDATES_TO_CHECK same-name listings (searching with the city
    when we have one) and returns the place_id of the one whose Google listing
    links to the submitted website, lists the submitted phone number, OR
    shows the submitted street address (street number + ZIP), plus how many
    candidates were found. Returns (None, count) when none match."""
    query = f"{business_name} {city}".strip() if city else business_name
    search_response = await client.get(
        TEXT_SEARCH_URL, params={"query": query, "key": GOOGLE_API_KEY}
    )
    search_data = search_response.json()
    if search_response.status_code != 200 or search_data.get("status") not in ("OK", "ZERO_RESULTS"):
        raise RuntimeError(f"Places text search failed (status: {search_data.get('status')})")

    candidates = search_data.get("results", [])[:MAX_CANDIDATES_TO_CHECK]
    target_host = _site_host(website_url)
    target_phone = _phone_digits(phone)
    street_number = (street_address or "").strip().split(" ")[0]
    street_number = street_number if street_number.isdigit() else None
    target_zip = (zip_code or "").strip()[:5] or None

    def same_address(listing_address: Optional[str]) -> bool:
        if not (street_number and target_zip and listing_address):
            return False
        return listing_address.startswith(street_number + " ") and target_zip in listing_address

    async def candidate_contact(place_id: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
        response = await client.get(
            PLACE_DETAILS_URL,
            params={
                "place_id": place_id,
                "fields": "website,formatted_phone_number,international_phone_number,formatted_address",
                "key": GOOGLE_API_KEY,
            },
        )
        result = response.json().get("result", {})
        listing_phone = result.get("international_phone_number") or result.get("formatted_phone_number")
        return (
            _site_host(result.get("website")),
            _phone_digits(listing_phone),
            result.get("formatted_address"),
        )

    contacts = await asyncio.gather(
        *(candidate_contact(c["place_id"]) for c in candidates), return_exceptions=True
    )
    for candidate, contact in zip(candidates, contacts):
        if isinstance(contact, Exception):
            continue
        host, listing_phone, listing_address = contact
        if (
            _same_site(host, target_host)
            or (target_phone and listing_phone == target_phone)
            or same_address(listing_address)
        ):
            return candidate["place_id"], len(candidates)
    return None, len(candidates)


async def check_google_presence(
    business_name: str,
    website_url: str,
    phone: Optional[str] = None,
    city: Optional[str] = None,
    street_address: Optional[str] = None,
    zip_code: Optional[str] = None,
) -> tuple[list[CheckResult], dict]:
    """Looks up the business via Places API (Find Place -> Place Details) and
    returns two CheckResults plus a raw-data dict with review counts (needed
    for a later calculation, e.g. Lead Revival) and a derived location (used
    by the AI visibility check). Never raises - failures and "not found"
    both produce score-0 CheckResults with an honest summary, and raw_data's
    "location"/"address"/"phone" stay None so callers can detect they
    couldn't be determined."""
    raw_data = {
        "review_count": 0,
        "reviews_with_owner_response": 0,
        "location": None,
        "address": None,
        "phone": None,
        # Google profile details for the Online Foundation leak.
        # gbp_found: True (matched), False (no listing exists), None
        # (listings exist but none could be confirmed as this business).
        "gbp_found": False,
        "gbp_has_hours": None,
        "gbp_photo_count": None,
        "gbp_has_category": None,
        "gbp_has_website": None,
        "rating": None,
    }

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
            place_id, candidate_count = await _find_matching_place(
                client, business_name, website_url, phone, city, street_address, zip_code
            )
            if place_id is None:
                if candidate_count == 0:
                    return not_found_checks, raw_data
                # Listings with this name exist, but none link to this
                # website - reporting on one of them would risk describing a
                # different company.
                raw_data["gbp_found"] = None
                checks = [
                    CheckResult(
                        check_name="Google Business Profile",
                        score=0,
                        summary=(
                            f"We found Google listings named '{business_name}', but none of them "
                            f"link to {_site_host(website_url)} or show the phone number or address you "
                            "gave us, so we couldn't confirm which one is yours. Adding your website to your Google Business Profile helps "
                            "customers and search engines connect the two."
                        ),
                        source_type="not_checked",
                    ),
                    CheckResult(
                        check_name="Google Reviews",
                        score=0,
                        summary="No reviews checked - we couldn't confirm which Google listing is yours.",
                        source_type="not_checked",
                    ),
                ]
                return checks, raw_data

            details_params = {
                "place_id": place_id,
                "fields": "name,type,opening_hours,photo,rating,user_ratings_total,review,formatted_address,address_component,formatted_phone_number,website",
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
                "location": _extract_city_state(result),
                "address": result.get("formatted_address"),
                "phone": result.get("formatted_phone_number"),
                "gbp_found": True,
                "gbp_has_hours": has_hours,
                "gbp_photo_count": photo_count,
                "gbp_has_category": category != "uncategorized",
                "gbp_has_website": bool(result.get("website")),
                "rating": rating,
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
        health_checks, health_raw_data = await check_website_health(website_url)
        for check in health_checks:
            print(f"  {check.check_name}: {check.score}/10 [{check.source_type}] - {check.summary}")
        print(f"  Raw data: {health_raw_data}")

        print(f"\nChecking Google presence for '{business_name}'...")
        presence_checks, raw_data = await check_google_presence(business_name, website_url)
        for check in presence_checks:
            print(f"  {check.check_name}: {check.score}/10 [{check.source_type}] - {check.summary}")
        print(f"\nRaw data for later calculations: {raw_data}")

    asyncio.run(_main())


# --- Conversion basics ------------------------------------------------------

BOOKING_MARKERS = (
    "calendly.com", "acuityscheduling", "housecallpro", "servicetitan", "jobber",
    "setmore", "squareup.com/appointments", "booksy", "schedulicity", "vagaro",
    "book online", "book now", "schedule online", "schedule service",
    "schedule an appointment", "request an appointment", "book an appointment",
)
CONTACT_FORM_MARKERS = ("<form", "wpforms", "gform_", "contact-form", "hsforms", "jotform", "typeform")


async def check_conversion_basics(website_url: str) -> dict:
    """Fetches the homepage and looks for the three things that turn a
    visitor into a lead: a tap-to-call phone link, a contact form, and online
    booking. Returns {"tap_to_call": bool, "contact_form": bool,
    "online_booking": bool}, or {} if the page couldn't be fetched - in which
    case nothing is counted against the business."""
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            response = await client.get(
                website_url, headers={"User-Agent": "Mozilla/5.0 (compatible; FoundationScan/1.0)"}
            )
        if response.status_code != 200:
            return {}
        html = response.text.lower()
    except Exception as exc:
        print(f"[conversion_basics] could not fetch {website_url}: {type(exc).__name__}: {exc}")
        return {}
    return {
        "tap_to_call": 'href="tel:' in html or "href='tel:" in html,
        "contact_form": any(marker in html for marker in CONTACT_FORM_MARKERS),
        "online_booking": any(marker in html for marker in BOOKING_MARKERS),
    }
