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
# Places API (New). Requires "Places API (New)" enabled on the Google Cloud
# project that owns GOOGLE_API_KEY (separate from the legacy "Places API").
TEXT_SEARCH_NEW_URL = "https://places.googleapis.com/v1/places:searchText"
PLACE_FIELD_MASK = ",".join(
    "places." + field
    for field in (
        "id", "displayName", "websiteUri", "nationalPhoneNumber", "internationalPhoneNumber",
        "formattedAddress", "addressComponents", "primaryType", "types", "regularOpeningHours",
        "photos", "rating", "userRatingCount", "pureServiceAreaBusiness", "primaryTypeDisplayName",
    )
)
# How many Google listings from the search to compare against the submission.
MAX_CANDIDATES_TO_CHECK = 10

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


SPEED_TEST_RUNS = 3  # Google's speed test varies run to run; use the median


async def check_website_health(website_url: str) -> tuple[list[CheckResult], dict]:
    """Runs Google PageSpeed Insights: the performance audit SPEED_TEST_RUNS
    times in parallel (taking the run with the median load time, so the same
    site doesn't get different dollar figures on different days) plus one
    accessibility audit. Returns two CheckResults, each scaled 0-100 -> 0-10,
    plus a raw-data dict with "lcp_seconds" (None if it couldn't be
    measured), the accessibility score, and the median run's screenshot."""
    results = await asyncio.gather(
        *(
            _fetch_pagespeed_category(website_url, "performance", "Website Health", "performance")
            for _ in range(SPEED_TEST_RUNS)
        ),
        _fetch_pagespeed_category(website_url, "accessibility", "Accessibility Basics", "accessibility"),
    )
    performance_runs, (accessibility_check, accessibility_raw) = results[:-1], results[-1]

    measured = sorted(
        (run for run in performance_runs if run[1].get("lcp_seconds") is not None),
        key=lambda run: run[1]["lcp_seconds"],
    )
    if measured:
        performance_check, performance_raw = measured[(len(measured) - 1) // 2]
    else:
        performance_check, performance_raw = performance_runs[0]
    print(
        f"[speed] {website_url}: load times {[run[1]['lcp_seconds'] for run in measured]}s "
        f"-> using {performance_raw.get('lcp_seconds')}s"
    )
    # Fall back to any run's screenshot if the median run didn't return one.
    screenshot = performance_raw.get("screenshot") or next(
        (run[1].get("screenshot") for run in performance_runs if run[1].get("screenshot")), None
    )
    raw_data = {
        "lcp_seconds": performance_raw.get("lcp_seconds"),
        "accessibility_score": accessibility_raw.get("score_100"),
        "screenshot": screenshot,
    }
    return [performance_check, accessibility_check], raw_data


def _extract_city_state(place: dict) -> Optional[str]:
    """'Fort Myers, FL' from a Places API (New) result's address components.
    Service-area businesses often hide their address, so this can be None."""
    city = None
    state = None
    for component in place.get("addressComponents", []) or []:
        types = component.get("types", [])
        if "locality" in types:
            city = component.get("longText")
        if "administrative_area_level_1" in types:
            state = component.get("shortText")
    if city and state:
        return f"{city}, {state}"
    return None


def _latest_review_days(place: dict) -> Optional[int]:
    """Days since the newest review Google returned, or None if no dates."""
    from datetime import datetime, timezone

    newest = None
    for review in place.get("reviews", []) or []:
        published = review.get("publishTime")
        if not published:
            continue
        try:
            when = datetime.fromisoformat(published.replace("Z", "+00:00"))
        except ValueError:
            continue
        newest = when if newest is None or when > newest else newest
    return None if newest is None else (datetime.now(timezone.utc) - newest).days


def _open_24_7(place: dict) -> bool:
    """True when Google lists the business as open 24 hours, every day."""
    periods = (place.get("regularOpeningHours") or {}).get("periods") or []
    # Google represents "open 24 hours" as a single period with no close.
    return len(periods) == 1 and "close" not in periods[0]


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


def _normalize_name(name: Optional[str]) -> str:
    """"Paul's Plumbing, LLC" -> "pauls plumbing llc" for loose name comparison."""
    return " ".join("".join(ch for ch in (name or "").lower() if ch.isalnum() or ch == " ").split())


async def _search_places(
    client: httpx.AsyncClient, query: str, page_size: int = MAX_CANDIDATES_TO_CHECK, with_reviews: bool = False
) -> list:
    """Places API (New) Text Search. Unlike the legacy API, this can return
    service-area businesses (plumbers, cleaners, mobile services) that hide
    their street address on Google - the legacy API silently leaves them
    out, which made real profiles look like they didn't exist."""
    response = await client.post(
        TEXT_SEARCH_NEW_URL,
        headers={
            "Content-Type": "application/json",
            "X-Goog-Api-Key": GOOGLE_API_KEY,
            # Reviews (with dates) only for the business's own lookup - they
            # make the call pricier, so the competitor search skips them.
            "X-Goog-FieldMask": PLACE_FIELD_MASK + (",places.reviews" if with_reviews else ""),
        },
        json={
            "textQuery": query,
            "includePureServiceAreaBusinesses": True,
            "pageSize": page_size,
            "regionCode": "US",
        },
    )
    if response.status_code != 200:
        try:
            reason = response.json().get("error", {}).get("message", response.text[:300])
        except Exception:
            reason = response.text[:300]
        raise RuntimeError(f"Places text search failed (HTTP {response.status_code}): {reason}")
    return response.json().get("places", []) or []


def _pick_matching_place(
    places: list,
    business_name: str,
    website_url: str,
    phone: Optional[str],
    street_address: Optional[str],
    zip_code: Optional[str],
) -> Optional[dict]:
    """Many businesses share a name, so the first result can be the wrong
    company. Returns the listing that links to the submitted website, lists
    the submitted phone number, or shows the submitted street address
    (street number + ZIP). None when nothing can be confirmed."""
    target_host = _site_host(website_url)
    target_phone = _phone_digits(phone)
    street_number = (street_address or "").strip().split(" ")[0]
    street_number = street_number if street_number.isdigit() else None
    target_zip = (zip_code or "").strip()[:5] or None

    for place in places:
        host = _site_host(place.get("websiteUri"))
        listing_phone = _phone_digits(
            place.get("internationalPhoneNumber") or place.get("nationalPhoneNumber")
        )
        listing_address = place.get("formattedAddress") or ""
        same_address = bool(
            street_number
            and target_zip
            and listing_address.startswith(street_number + " ")
            and target_zip in listing_address
        )
        if (
            _same_site(host, target_host)
            or (target_phone and listing_phone == target_phone)
            or same_address
        ):
            return place
    return None


async def check_google_presence(
    business_name: str,
    website_url: str,
    phone: Optional[str] = None,
    city: Optional[str] = None,
    street_address: Optional[str] = None,
    zip_code: Optional[str] = None,
) -> tuple[list[CheckResult], dict]:
    """Searches Google for "[Business Name] [City]" (Places API (New), with
    service-area businesses included), confirms the listing by website,
    phone, or address, and returns two CheckResults plus a raw-data dict with
    review counts and profile details for the leak calculations. Never
    raises. "No profile" is only reported when Google returned nothing at
    all for the search; any error or unconfirmed match is "Not checked",
    which adds nothing to the leak total."""
    raw_data = {
        "review_count": 0,
        "reviews_with_owner_response": 0,
        "location": None,
        "address": None,
        "phone": None,
        # gbp_found: True (matched), False (Google returned no listings for
        # the search), None (couldn't check, or couldn't confirm which one).
        "gbp_found": False,
        "gbp_has_hours": None,
        "gbp_photo_count": None,
        "gbp_has_category": None,
        "gbp_has_website": None,
        "rating": None,
        "gbp_place_id": None,
        # For the map-pack check: the listing's Google category ("Plumber")
        # and where it ranked when searching its own name + city.
        "gbp_category": None,
        "brand_query": None,
        "brand_rank": None,
        "latest_review_days": None,
        "open_24_7": None,
        "hours_text": [],
    }

    query = f"{business_name} {city}".strip() if city else business_name

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            places = await _search_places(client, query, with_reviews=True)
    except Exception as exc:
        # An error means we couldn't check - never report it as "no profile",
        # which would invent a large foundation leak.
        print(f"[google] presence check failed for '{business_name}' (query: {query!r}): {type(exc).__name__}: {exc}")
        raw_data["gbp_found"] = None
        summary = "We couldn't reach Google to check your Business Profile this time."
        return [
            CheckResult(check_name="Google Business Profile", score=0, summary=summary, source_type="not_checked"),
            CheckResult(check_name="Google Reviews", score=0, summary=summary, source_type="not_checked"),
        ], raw_data

    place = _pick_matching_place(places, business_name, website_url, phone, street_address, zip_code)
    print(
        f"[google] search {query!r}: {len(places)} result(s) "
        f"{[p.get('displayName', {}).get('text') for p in places]}; "
        f"matched: {place.get('displayName', {}).get('text') if place else 'NONE'}"
    )

    if place is None:
        if not places:
            return [
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
            ], raw_data
        # Listings came back but none could be confirmed as this business -
        # reporting on one of them would risk describing a different company.
        raw_data["gbp_found"] = None
        same_name = any(
            _normalize_name(p.get("displayName", {}).get("text")) == _normalize_name(business_name)
            for p in places
        )
        return [
            CheckResult(
                check_name="Google Business Profile",
                score=0,
                summary=(
                    f"We found Google listings{' named ' + repr(business_name) if same_name else ''}, "
                    f"but none of them link to {_site_host(website_url)} or show the phone number or "
                    "address you gave us, so we couldn't confirm which one is yours. Adding your website "
                    "to your Google Business Profile helps customers and search engines connect the two."
                ),
                source_type="not_checked",
            ),
            CheckResult(
                check_name="Google Reviews",
                score=0,
                summary="No reviews checked - we couldn't confirm which Google listing is yours.",
                source_type="not_checked",
            ),
        ], raw_data

    # --- Google Business Profile ---
    primary_type = place.get("primaryType") or (place.get("types") or [None])[0]
    category = (primary_type or "uncategorized").replace("_", " ")
    has_hours = bool(place.get("regularOpeningHours"))
    # Places API (New) returns at most 10 photos, which is plenty for a
    # "has photos" check.
    photo_count = len(place.get("photos", []) or [])
    service_area_only = bool(place.get("pureServiceAreaBusiness"))

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

    display_name = place.get("displayName", {}).get("text") or business_name
    profile_check = CheckResult(
        check_name="Google Business Profile",
        score=profile_score,
        summary=(
            f"Found on Google as '{display_name}' "
            f"(category: {category}, hours listed: {'yes' if has_hours else 'no'}, "
            f"photos: {photo_count}{'+' if photo_count >= 10 else ''}"
            f"{', service-area business' if service_area_only else ''})."
        ),
        source_type="measured",
    )

    # --- Google Reviews ---
    rating = place.get("rating")
    review_count = place.get("userRatingCount", 0) or 0

    raw_data.update(
        {
            "review_count": review_count,
            # The Places API doesn't expose owner replies to reviews.
            "reviews_with_owner_response": 0,
            "location": _extract_city_state(place),
            # Service-area businesses hide their street address; fall back
            # to the form's address rather than a vague "Fort Myers, FL".
            "address": None if service_area_only else place.get("formattedAddress"),
            "phone": place.get("nationalPhoneNumber"),
            "gbp_found": True,
            "gbp_has_hours": has_hours,
            "gbp_photo_count": photo_count,
            "gbp_has_category": category != "uncategorized",
            "gbp_has_website": bool(place.get("websiteUri")),
            "rating": rating,
            "gbp_place_id": place.get("id"),
            "latest_review_days": _latest_review_days(place),
            "open_24_7": _open_24_7(place),
            "hours_text": (place.get("regularOpeningHours") or {}).get("weekdayDescriptions") or [],
            "gbp_category": (place.get("primaryTypeDisplayName") or {}).get("text"),
            "brand_query": query,
            "brand_rank": places.index(place) + 1,
        }
    )

    if rating is None or review_count == 0:
        reviews_check = CheckResult(
            check_name="Google Reviews",
            score=0,
            summary="Business is listed but has no rating or reviews yet.",
            source_type="measured",
        )
    else:
        reviews_check = CheckResult(
            check_name="Google Reviews",
            score=min(round(rating * 2), 10),
            summary=f"{rating}/5 average rating across {review_count} reviews.",
            source_type="measured",
        )

    return [profile_check, reviews_check], raw_data

MAP_PACK_SEARCH_DEPTH = 20  # Places API (New) returns at most 20 per search


async def check_map_pack(
    business_name: str, city: Optional[str], presence: dict, fallback_category: Optional[str]
) -> CheckResult:
    """Measured local-ranking check, replacing the old AI web-search guess.
    Searches Google for "[Google category] in [City]" (e.g. "Plumber in Fort
    Myers") - what a new customer who doesn't know the business would type -
    and finds where the business's own listing ranks. Also reports where it
    ranked for "[Business Name] [City]" from the profile lookup. Google's
    live map results vary with the searcher's location, so this is a close
    approximation of the map 3-pack, not an exact copy of it.

    Scores (calculation.py counts < 9 as "not in the 3-pack"):
    top 3 = 10; 4-10 "there but weak" = 5; 11-20 "hard to find" = 1;
    not in the top 20 or no profile "virtually invisible" = 0."""
    gbp_found = presence.get("gbp_found")
    if gbp_found is None or not city:
        return CheckResult(
            check_name="Local Search Ranking",
            score=0,
            summary=(
                "Not checked - we couldn't confirm the business's Google Business Profile, "
                "so we couldn't look for it in the map results."
                if city
                else "Not checked - no city was provided."
            ),
            source_type="not_checked",
        )

    category = presence.get("gbp_category") or fallback_category or "business"
    query = f"{category} in {city}"

    if gbp_found is False:
        return CheckResult(
            check_name="Local Search Ranking",
            score=0,
            summary=(
                f"Without a Google Business Profile, '{business_name}' can't appear in Google's map "
                f"results for searches like '{query}'."
            ),
            source_type="measured",
        )

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            places = await _search_places(client, query, MAP_PACK_SEARCH_DEPTH)
    except Exception as exc:
        print(f"[google] map-pack search failed for {query!r}: {type(exc).__name__}: {exc}")
        return CheckResult(
            check_name="Local Search Ranking",
            score=0,
            summary="Not checked - we couldn't reach Google to run the local search this time.",
            source_type="not_checked",
        )

    place_id = presence.get("gbp_place_id")
    rank = next((i + 1 for i, p in enumerate(places) if p.get("id") == place_id), None)
    top_three = [p.get("displayName", {}).get("text") for p in places[:3]]
    competitor_reviews = [
        {"name": p.get("displayName", {}).get("text"), "reviews": p.get("userRatingCount") or 0}
        for p in places[:3]
        if p.get("id") != presence.get("gbp_place_id")
    ]
    print(f"[google] map-pack search {query!r}: rank {rank or 'not in top ' + str(len(places))}; top 3: {top_three}")

    if rank is not None and rank <= 3:
        score, standing = 10, f"ranked #{rank} - in the top 3 map results"
    elif rank is not None and rank <= 10:
        score, standing = 5, (
            f"ranked #{rank} - you're there, but weak: outside the top 3 that get most of the calls"
        )
    elif rank is not None:
        score, standing = 1, f"ranked #{rank} - hard to find, well below the top 3"
    else:
        score, standing = 0, (
            f"didn't appear in the top {len(places)} results - virtually invisible to new customers"
        )

    brand_query, brand_rank = presence.get("brand_query"), presence.get("brand_rank")
    brand_part = (
        f"Searching your name ('{brand_query}'), you ranked #{brand_rank}, so people who already "
        "know you can find you. "
        if brand_query and brand_rank
        else ""
    )
    others = [name for name in top_three if name and _normalize_name(name) != _normalize_name(business_name)]
    summary = (
        f"{brand_part}Searching what a new customer would type ('{query}'), you {standing}."
        + (f" The top 3 were: {', '.join(top_three)}." if (rank is None or rank > 3) else "")
        + " Map results shift slightly with the searcher's location."
    )
    return CheckResult(
        check_name="Local Search Ranking",
        score=score,
        summary=summary,
        source_type="measured",
        details={"rank": rank, "competitor_reviews": competitor_reviews},
    )


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
