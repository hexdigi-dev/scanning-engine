import asyncio
import secrets
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.responses import HTMLResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

import config
import pages
import storage
from checks.agent_checks import (
    check_ad_activity,
    check_local_ranking,
    check_nap_consistency,
    check_reputation,
    check_social_presence,
)
from checks.ai_visibility import check_ai_visibility
from checks.api_checks import check_google_presence, check_website_health
from checks.website_conversion import assess_calls_to_action, check_homepage_signals
from calculation import calculate_leak_estimate
from models import CheckResult, ScanRequest, ScanResponse, form_full_address, form_location
from synthesis import synthesize_report

@asynccontextmanager
async def lifespan(app: FastAPI):
    await storage.init()
    yield
    await storage.close()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

WEBSITE_RESOLVE_TIMEOUT = 5.0

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def _skipped_check(check_name: str, summary: str) -> CheckResult:
    return CheckResult(check_name=check_name, score=0, summary=summary, source_type="not_checked")


def _score_if_checked(check: CheckResult):
    """None for checks that couldn't run, so no leak is estimated from them."""
    return None if check.source_type == "not_checked" else check.score


def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    if not api_key or not secrets.compare_digest(api_key, config.SCAN_API_KEY):
        raise HTTPException(status_code=401, detail="Missing or invalid API key")
    return api_key


@app.get("/health")
def health():
    return {"status": "ok"}


def _public_base_url(http_request: Request) -> str:
    if config.PUBLIC_BASE_URL:
        return config.PUBLIC_BASE_URL.rstrip("/")
    if config.RAILWAY_PUBLIC_DOMAIN:
        return f"https://{config.RAILWAY_PUBLIC_DOMAIN}"
    return str(http_request.base_url).rstrip("/")


# --- Public pages -----------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def landing_page(http_request: Request):
    return pages.render_landing(http_request)


@app.post("/start", response_class=HTMLResponse)
async def start_scan(http_request: Request):
    """Receives the landing-page form, validates it, and forwards it to the
    Make.com webhook - Make then calls /scan and sends the emails, exactly
    as before. The visitor sees a "your scan is running" page."""
    form = await http_request.form()
    values = {key: str(value).strip() for key, value in form.items()}

    # Honeypot: real people never see or fill this field; bots usually do.
    if values.pop("company_fax", ""):
        return pages.render_submitted(http_request, email="")

    website = values.get("website_url", "")
    if website and not website.lower().startswith(("http://", "https://")):
        values["website_url"] = f"https://{website}"

    try:
        scan_request = ScanRequest(**values)
    except ValidationError as exc:
        return pages.render_landing(http_request, values=values, errors=pages.form_errors(exc), status_code=422)
    # Location is optional in the API (older callers don't send it) but the
    # landing page requires a ZIP, since it's what lets the location checks
    # run. City and state are filled from the ZIP in the browser; this fills
    # any that are still blank (e.g. if the page's script didn't run). The
    # street address stays optional.
    zip_info = pages.lookup_zip(scan_request.zip_code)
    if zip_info is None:
        return pages.render_landing(
            http_request,
            values=values,
            errors={"zip_code": "Enter a valid 5-digit US ZIP code."},
            status_code=422,
        )
    scan_request = scan_request.model_copy(
        update={
            "zip_code": scan_request.zip_code.strip()[:5],
            "city": scan_request.city.strip() or zip_info["city"],
            "state": scan_request.state.strip() or zip_info["state"],
        }
    )

    if not config.MAKE_WEBHOOK_URL:
        print("[start] MAKE_WEBHOOK_URL is not set - cannot forward form submission")
        return pages.render_landing(
            http_request,
            values=values,
            errors={"_form": "Scans can't be started right now. Please try again later."},
            status_code=503,
        )

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(config.MAKE_WEBHOOK_URL, json=scan_request.model_dump())
        response.raise_for_status()
    except httpx.HTTPError as exc:
        print(f"[start] failed to forward '{scan_request.business_name}' to Make: {exc}")
        return pages.render_landing(
            http_request,
            values=values,
            errors={"_form": "We couldn't start your scan. Please try again in a minute."},
            status_code=502,
        )

    print(f"[start] forwarded scan request for '{scan_request.business_name}' to Make")
    return pages.render_submitted(http_request, email=scan_request.email)


@app.get("/zip/{zip_code}")
async def zip_lookup(zip_code: str):
    """Used by the landing page to fill in city and state from the ZIP."""
    info = pages.lookup_zip(zip_code)
    if info is None:
        raise HTTPException(status_code=404, detail="Unknown ZIP code")
    return info


@app.get("/report/{report_id}", response_class=HTMLResponse)
async def report_page(report_id: str, http_request: Request):
    data = await storage.get_report(report_id)
    if data is None:
        return pages.render_not_found(http_request)
    return pages.render_report(http_request, ScanResponse.model_validate(data))


# --- Scan API (called by Make) ----------------------------------------------

@app.post("/scan", response_model=ScanResponse, dependencies=[Depends(verify_api_key)])
async def scan(request: ScanRequest, http_request: Request) -> ScanResponse:
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
    # check_nap_consistency all need location/address/phone. They come from
    # the matched Google listing when there is one; otherwise the city typed
    # on the form is used as the location (address-based consistency still
    # needs a matched listing). Everything with no such dependency runs
    # together in the first stage.
    (
        (website_checks, raw_website_data),
        (presence_checks, raw_presence_data),
        social_check,
        ad_activity_check,
        homepage_signals,
    ) = await asyncio.gather(
        check_website_health(request.website_url),
        check_google_presence(
            request.business_name,
            request.website_url,
            request.phone,
            form_location(request),
            request.street_address.strip() or None,
            request.zip_code.strip() or None,
        ),
        check_social_presence(request.business_name, request.website_url),
        check_ad_activity(request.business_name),
        check_homepage_signals(request.website_url),
    )

    # Prefer what the matched Google listing says; fall back to the form.
    location = raw_presence_data.get("location") or form_location(request)
    address = raw_presence_data.get("address") or form_full_address(request)
    phone = raw_presence_data.get("phone") or (request.phone.strip() or None)

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
        cta_assessment,
    ) = await asyncio.gather(
        ai_visibility_coro
        or _skipped_check("AI Search Visibility", location_skip_summary),
        local_ranking_coro
        or _skipped_check("Local Search Ranking", location_skip_summary),
        reputation_coro
        or _skipped_check("Online Reputation Scan", location_skip_summary),
        nap_consistency_coro
        or _skipped_check("Consistency, Everywhere", nap_skip_summary),
        assess_calls_to_action(
            request.business_name,
            request.business_type,
            raw_website_data.get("screenshot"),
            homepage_signals,
        ),
    )

    all_checks = website_checks + presence_checks + [
        social_check,
        ad_activity_check,
        ai_visibility_check,
        local_ranking_check,
        reputation_check,
        nap_consistency_check,
    ]

    leak_estimate = calculate_leak_estimate(
        request,
        raw_presence_data,
        ai_visibility_score=_score_if_checked(ai_visibility_check),
        local_ranking_score=_score_if_checked(local_ranking_check),
        reputation_score=_score_if_checked(reputation_check),
        lcp_seconds=raw_website_data.get("lcp_seconds"),
        homepage_signals=homepage_signals,
        cta_assessment=cta_assessment,
        consistency_score=_score_if_checked(nap_consistency_check),
        social_score=_score_if_checked(social_check),
    )

    synthesis_result = await synthesize_report(request.business_name, all_checks, leak_estimate)
    if synthesis_result.review_note:
        print(f"[scan] REVIEW NEEDED for '{request.business_name}': {synthesis_result.review_note}")

    leak_estimate = leak_estimate.model_copy(
        update={
            "headline_explanation": synthesis_result.headline_explanation,
            "supporting_leaks": synthesis_result.supporting_leaks,
            "dormant_lead_explanation": synthesis_result.dormant_lead_explanation,
        }
    )

    response = ScanResponse(
        business_name=request.business_name,
        scanned_at=datetime.now(timezone.utc),
        checks=synthesis_result.checks,
        leak_estimate=leak_estimate,
        website_details={
            "lcp_seconds": raw_website_data.get("lcp_seconds"),
            "homepage": homepage_signals,
            "assessment": {k: v for k, v in (cta_assessment or {}).items() if k != "notes"},
        },
    )

    # Save the finished scan so it can be viewed at /report/<id>. A storage
    # failure never fails the scan itself - the emails still go out, just
    # without a report link.
    report_id = storage.new_report_id()
    try:
        await storage.save_report(report_id, request.business_name, response.model_dump(mode="json"))
        response = response.model_copy(
            update={"report_id": report_id, "report_url": f"{_public_base_url(http_request)}/report/{report_id}"}
        )
    except Exception as exc:
        print(f"[scan] could not save report for '{request.business_name}': {type(exc).__name__}: {exc}")

    elapsed = time.monotonic() - start
    print(f"[scan] finished scan for '{request.business_name}' in {elapsed:.2f}s")

    return response
