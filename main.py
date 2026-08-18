import asyncio
import time
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, HTTPException

import config
from checks.agent_checks import check_local_ranking, check_reputation, check_social_presence
from checks.ai_visibility import check_ai_visibility
from checks.api_checks import check_google_presence, check_website_health
from models import CheckResult, LeakEstimate, ScanRequest, ScanResponse

app = FastAPI()

WEBSITE_RESOLVE_TIMEOUT = 5.0


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

    # check_ai_visibility, check_local_ranking, and check_reputation all need
    # a location, which we derive from the Google Places lookup rather than
    # collecting on the intake form - so none of them can start until
    # check_google_presence resolves. Everything with no such dependency
    # (website health, presence itself, social presence) still runs together
    # in the first stage.
    website_checks, (presence_checks, raw_presence_data), social_check = await asyncio.gather(
        check_website_health(request.website_url),
        check_google_presence(request.business_name, request.website_url),
        check_social_presence(request.business_name, request.website_url),
    )

    location = raw_presence_data.get("location")
    if location:
        ai_visibility_check, local_ranking_check, reputation_check = await asyncio.gather(
            check_ai_visibility(request.business_name, request.business_type, location),
            check_local_ranking(request.business_name, request.business_type, location),
            check_reputation(request.business_name, location),
        )
    else:
        skip_summary = (
            "Could not determine the business's location - no matching Google Business "
            "Profile was found, so this check was skipped."
        )
        ai_visibility_check = CheckResult(
            check_name="AI Search Visibility", score=0, summary=skip_summary, source_type="measured"
        )
        local_ranking_check = CheckResult(
            check_name="Local Search Ranking", score=0, summary=skip_summary, source_type="measured"
        )
        reputation_check = CheckResult(
            check_name="Online Reputation Scan", score=0, summary=skip_summary, source_type="measured"
        )

    leak_estimate = LeakEstimate(
        headline_leak_monthly=0.0,
        headline_explanation="Not yet calculated - pending calculation.py wiring.",
        supporting_leaks=[],
        dormant_lead_value=0.0,
        dormant_lead_explanation="Not yet calculated - pending calculation.py wiring.",
        flagged_for_review=False,
    )

    response = ScanResponse(
        business_name=request.business_name,
        scanned_at=datetime.now(timezone.utc),
        checks=website_checks
        + presence_checks
        + [social_check, ai_visibility_check, local_ranking_check, reputation_check],
        leak_estimate=leak_estimate,
    )

    elapsed = time.monotonic() - start
    print(f"[scan] finished scan for '{request.business_name}' in {elapsed:.2f}s")

    return response
