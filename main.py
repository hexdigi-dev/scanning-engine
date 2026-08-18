import asyncio
import time
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, HTTPException

import config
from checks.agent_checks import (
    check_ad_activity,
    check_local_ranking,
    check_nap_consistency,
    check_reputation,
    check_social_presence,
)
from checks.ai_visibility import check_ai_visibility
from checks.api_checks import check_google_presence, check_website_health
from calculation import calculate_leak_estimate
from models import CheckResult, ScanRequest, ScanResponse
from synthesis import synthesize_report

app = FastAPI()

WEBSITE_RESOLVE_TIMEOUT = 5.0


async def _skipped_check(check_name: str, summary: str) -> CheckResult:
    return CheckResult(check_name=check_name, score=0, summary=summary, source_type="measured")


@app.get("/")
def read_root():
    return {"status": "ok"}


@app.post("/scan", response_model=ScanResponse)
async def scan(request: ScanRequest) -> ScanResponse:
    start = time.monotonic()
    print(f"[scan] starting scan for '{request.business_name}'")

    try:
        async with httpx.AsyncClient(timeout=WEBSITE_RESOLVE_TIMEOUT, follow_redirects=True) as client:
            await client.head(request.website_url)
    except httpx.HTTPError as exc:
        elapsed = time.monotonic() - start
        print(f"[scan] rejected '{request.business_name}' - website unreachable ({elapsed:.2f}s)")
        raise HTTPException(
            status_code=400,
            detail=f"Website URL does not resolve: {request.website_url} ({exc})",
        )

    # check_ai_visibility, check_local_ranking, check_reputation, and
    # check_nap_consistency all need location/address/phone, which we derive
    # from the Google Places lookup rather than collecting on the intake
    # form - so none of them can start until check_google_presence resolves.
    # Everything with no such dependency (website health, presence itself,
    # social presence, ad activity) still runs together in the first stage.
    website_checks, (presence_checks, raw_presence_data), social_check, ad_activity_check = await asyncio.gather(
        check_website_health(request.website_url),
        check_google_presence(request.business_name, request.website_url),
        check_social_presence(request.business_name, request.website_url),
        check_ad_activity(request.business_name),
    )

    location = raw_presence_data.get("location")
    address = raw_presence_data.get("address")
    phone = raw_presence_data.get("phone")

    location_skip_summary = (
        "Could not determine the business's location - no matching Google Business "
        "Profile was found, so this check was skipped."
    )
    nap_skip_summary = (
        "Could not determine the business's reference address/phone - no matching "
        "Google Business Profile was found, so this check was skipped."
    )

    ai_visibility_coro = (
        check_ai_visibility(request.business_name, request.business_type, location)
        if location
        else None
    )
    local_ranking_coro = (
        check_local_ranking(request.business_name, request.business_type, location)
        if location
        else None
    )
    reputation_coro = check_reputation(request.business_name, location) if location else None
    nap_consistency_coro = (
        check_nap_consistency(request.business_name, address, phone) if address and phone else None
    )

    (
        ai_visibility_check,
        local_ranking_check,
        reputation_check,
        nap_consistency_check,
    ) = await asyncio.gather(
        ai_visibility_coro
        or _skipped_check("AI Search Visibility", location_skip_summary),
        local_ranking_coro
        or _skipped_check("Local Search Ranking", location_skip_summary),
        reputation_coro
        or _skipped_check("Online Reputation Scan", location_skip_summary),
        nap_consistency_coro
        or _skipped_check("Consistency, Everywhere", nap_skip_summary),
    )

    all_checks = website_checks + presence_checks + [
        social_check,
        ad_activity_check,
        ai_visibility_check,
        local_ranking_check,
        reputation_check,
        nap_consistency_check,
    ]

    leak_estimate = calculate_leak_estimate(request, raw_presence_data)

    synthesis_result = await synthesize_report(request.business_name, all_checks, leak_estimate)
    if synthesis_result.review_note:
        print(f"[scan] REVIEW NEEDED for '{request.business_name}': {synthesis_result.review_note}")

    leak_estimate = leak_estimate.model_copy(update={"headline_explanation": synthesis_result.leak_narrative})

    response = ScanResponse(
        business_name=request.business_name,
        scanned_at=datetime.now(timezone.utc),
        checks=synthesis_result.checks,
        leak_estimate=leak_estimate,
    )

    elapsed = time.monotonic() - start
    print(f"[scan] finished scan for '{request.business_name}' in {elapsed:.2f}s")

    return response
