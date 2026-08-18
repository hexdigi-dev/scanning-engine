import asyncio
import time
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, HTTPException

import config
from checks.api_checks import check_google_presence, check_website_health
from models import LeakEstimate, ScanRequest, ScanResponse

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

    website_checks, (presence_checks, _raw_presence_data) = await asyncio.gather(
        check_website_health(request.website_url),
        check_google_presence(request.business_name, request.website_url),
    )

    leak_estimate = LeakEstimate(
        headline_leak_monthly=0.0,
        headline_explanation="Not yet calculated - pending calculation.py implementation.",
        supporting_leaks=[],
        dormant_lead_value=0.0,
        flagged_for_review=False,
    )

    response = ScanResponse(
        business_name=request.business_name,
        scanned_at=datetime.now(timezone.utc),
        checks=website_checks + presence_checks,
        leak_estimate=leak_estimate,
    )

    elapsed = time.monotonic() - start
    print(f"[scan] finished scan for '{request.business_name}' in {elapsed:.2f}s")

    return response
