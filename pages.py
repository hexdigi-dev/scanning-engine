"""Renders the public HTML pages: the landing page with the scan form, the
"scan is running" page, and the finished report page."""
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Dict, Optional

import zipcodes
from fastapi import Request
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

import config
from models import ScanResponse

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def _money(value: float) -> str:
    # Round halves up ($1,312.50 -> $1,313) to match how the AI-written text
    # rounds; Python's default rounding would show $1,312.
    whole = Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return f"${whole:,}"


def _jobs(jobs: float, capped: bool = False) -> str:
    """Whole jobs read per month; half jobs read per two months:
    1 -> 'about 1 job/mo', 3 -> 'about 3 jobs/mo',
    0.5 -> 'about 1 job every 2 months', 1.5 -> 'about 3 jobs every 2 months'."""
    if jobs <= 0:
        return ""
    if capped and jobs >= 4:
        return "4+ jobs/mo"
    if float(jobs).is_integer():
        count = int(jobs)
        return f"about {count} job{'s' if count != 1 else ''}/mo"
    count = int(jobs * 2)
    return f"about {count} job{'s' if count != 1 else ''} every 2 months"


templates.env.filters["money"] = _money
templates.env.filters["jobs"] = _jobs

# The dropdown choices must match the Literal values in models.ScanRequest
# exactly (including the en dashes), or the form won't validate.
FORM_OPTIONS = {
    "monthly_leads": ["Under 10", "10–25", "25–50", "50–100", "100+"],
    "response_time": ["Under 5 minutes", "Within an hour", "Same day", "Next day or longer"],
    "close_rate": ["Under 10%", "10–25%", "25–50%", "50%+"],
    "dormant_leads": ["None that I know of", "A handful", "25–100", "100–500", "500+"],
    "state": [
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL", "IN",
        "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH",
        "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT",
        "VT", "VA", "WA", "WV", "WI", "WY",
    ],
}

FIELD_LABELS = {
    "contact_name": "Your name",
    "email": "Email",
    "phone": "Phone",
    "business_name": "Business name",
    "street_address": "Street address",
    "city": "City",
    "state": "State",
    "zip_code": "ZIP code",
    "website_url": "Website",
    "business_type": "What your business does",
    "monthly_leads": "New leads per month",
    "avg_job_value": "Average job value",
    "response_time": "How fast you usually respond to a new lead",
    "close_rate": "Share of leads that become customers",
    "dormant_leads": "Old leads that never became customers",
}

# The service that fixes each section's leak, with its price. Edit names
# and prices here - the report reads them from these.
FOUNDATION_FIX = {
    "name": "Foundation Fix",
    "price": "$497",
    "upgrade": "$997 with AI search (GEO/AEO) registration",
    "detail": "Completes and aligns your Google Business Profile, directory listings, and social profiles.",
}
REPUTATION_FIX = {
    "name": "Reputation Builder",
    "price": "$497 + $100/mo",
    "detail": "Replies to every unanswered review and sends every new client a review request.",
}
WEBSITE_FIX = {
    "name": "New Website",
    "price": "$997 + $150/mo",
    "detail": "Fast, mobile-first, with clear calls-to-action, tap-to-call, and a contact form. "
    "Options: online booking or ordering, and a \"Text Us\" line.",
}
SPEED_TO_LEAD_FIX = {
    "name": "Speed-to-Lead Bot",
    "price": "$997 setup + $150/mo per channel",
    "detail": "Responds to every new lead in under 60 seconds, by text, voice, and/or online form.",
}
# Kept so reports saved with the old per-leak labels still show a fix.
LEAK_FIXES = {"headline": SPEED_TO_LEAD_FIX}
# The dormant lead card's fix. Its price depends too much on each business's
# numbers to show on the report, so the card links to a call instead. The
# per-booked-call / per-new-client pricing below is kept for later use but
# isn't shown anywhere yet.
LEAD_REVIVAL_FIX = {"name": "Lead Revival Campaign"}
LEAD_REVIVAL_PRICING = {
    # Per new client closed: this share of their average job value.
    "per_close_share_of_job_value": 0.10,
    "per_close_minimum": 150,
    # A lead who books a call is further along than a raw lead, so booked
    # calls are assumed to close at this multiple of their reported close
    # rate, up to the cap. Per booked call = per-close price x that rate, so
    # both options are worth about the same to us on average.
    "booked_call_close_multiplier": 2.0,
    "booked_call_close_cap": 0.60,
    "per_call_minimum": 50,
}


def _round_price(value: float) -> int:
    """Nice round prices: nearest $5 under $100, $25 under $1,000, else $50."""
    step = 5 if value < 100 else 25 if value < 1000 else 50
    return int(round(value / step) * step)


def lead_revival_prices(avg_job_value: float, close_rate: float) -> Optional[dict]:
    """Returns {"per_call": ..., "per_close": ...} in whole dollars, or None
    for reports saved before job value and close rate were stored."""
    if avg_job_value <= 0 or close_rate <= 0:
        return None
    pricing = LEAD_REVIVAL_PRICING
    per_close = max(
        _round_price(avg_job_value * pricing["per_close_share_of_job_value"]),
        pricing["per_close_minimum"],
    )
    booked_call_close_rate = min(
        close_rate * pricing["booked_call_close_multiplier"], pricing["booked_call_close_cap"]
    )
    per_call = max(_round_price(per_close * booked_call_close_rate), pricing["per_call_minimum"])
    return {"per_call": per_call, "per_close": per_close}


def lookup_zip(zip_code: str) -> Optional[dict]:
    """'33913' -> {"city": "Fort Myers", "state": "FL"} from the bundled
    offline ZIP database, or None if it isn't a known 5-digit US ZIP."""
    zip5 = (zip_code or "").strip()[:5]
    if len(zip5) != 5 or not zip5.isdigit():
        return None
    try:
        matches = zipcodes.matching(zip5)
    except (ValueError, TypeError):
        return None
    if not matches:
        return None
    return {"city": matches[0]["city"].title(), "state": matches[0]["state"]}


def ad_status(check) -> str:
    """Simple ad status for the Other Potential Leaks section, from the ad
    check's 0-10 rubric: confirmed active ads (8-10), confirmed none (0),
    anything uncertain in between."""
    if check is None or getattr(check, "source_type", None) == "not_checked":
        return "Couldn't tell"
    if check.score >= 8:
        return "Running ads"
    if check.score == 0:
        return "Not running ads"
    return "Couldn't tell"


def form_errors(exc: ValidationError) -> Dict[str, str]:
    errors = {}
    for error in exc.errors():
        field = str(error["loc"][0]) if error.get("loc") else "_form"
        label = FIELD_LABELS.get(field, field)
        if field == "website_url":
            errors[field] = "Enter your full website address, like yourbusiness.com."
        elif error.get("type") == "missing" or error.get("input") in ("", None):
            errors[field] = f"{label} is required."
        else:
            errors[field] = f"Choose a valid option for {label.lower()}."
    return errors


def render_landing(
    request: Request,
    values: Optional[dict] = None,
    errors: Optional[dict] = None,
    status_code: int = 200,
):
    return templates.TemplateResponse(
        request,
        "landing.html",
        {"values": values or {}, "errors": errors or {}, "options": FORM_OPTIONS},
        status_code=status_code,
    )


def render_submitted(request: Request, email: str):
    return templates.TemplateResponse(request, "submitted.html", {"email": email})


def render_not_found(request: Request):
    return templates.TemplateResponse(request, "not_found.html", {}, status_code=404)


# Which scan checks belong to which report section.
FOUNDATION_CHECK_NAMES = [
    "Google Business Profile",
    "Google Reviews",
    "Online Reputation Scan",
    "Social Media Presence",
    "AI Search Visibility",
    "Local Search Ranking",
    "Consistency, Everywhere",
]
WEBSITE_CHECK_NAMES = ["Website Health", "Accessibility Basics"]


def _website_rows(details: dict) -> list:
    """Pass/fail rows for the website checks behind the Website Performance
    Leak. status is "pass", "fail", or "unchecked"."""
    homepage = details.get("homepage") or {}
    assessment = details.get("assessment") or {}
    rows = []

    def row(name, ok, pass_text, fail_text):
        status = "unchecked" if ok is None else ("pass" if ok else "fail")
        text = "Couldn't check" if ok is None else (pass_text if ok else fail_text)
        rows.append({"name": name, "status": status, "text": text})

    lcp = details.get("lcp_seconds")
    row("Load speed", None if lcp is None else lcp < 3.0, f"{lcp or 0:.1f}s on mobile", f"{lcp or 0:.1f}s on mobile (aim for under 2.5s)")
    row("Mobile-friendly layout", homepage.get("responsive"), "Adapts to phones", "Not built for phones")
    row(
        "Clear call-to-action on first screen",
        assessment.get("clear_cta_on_first_screen"),
        "Visible without scrolling",
        "Not visible without scrolling",
    )
    expected = assessment.get("expected_main_action") or "the main action"
    row(
        "Right main action for your business",
        assessment.get("main_action_fits_business"),
        f"Offers {expected}",
        f"{expected[:1].upper() + expected[1:]} missing or buried",
    )
    goals = assessment.get("distinct_goals")
    row(
        "Focused page (not too many competing goals)",
        None if goals is None else goals < 3,
        f"{goals} main goal{'s' if goals != 1 else ''}" if goals is not None else "",
        f"{goals} competing goals",
    )
    total, broken = homepage.get("total_ctas"), homepage.get("broken_ctas")
    row(
        "Working call-to-action links",
        None if not total else broken == 0,
        f"All {total} work",
        f"{broken} of {total} don't work",
    )
    row("Tap-to-call", homepage.get("tap_to_call"), "Found", "Missing")
    row("\"Text Us\" option", homepage.get("text_us"), "Found", "Missing")
    booking = None if not homepage else bool(homepage.get("online_booking") or homepage.get("online_ordering"))
    row("Online booking or ordering", booking, "Found", "Missing")
    row("Contact form", homepage.get("contact_form"), "Found", "Missing")
    return rows


def _leak_card(
    leak_label: str, title: str, value: float, jobs: float, explanation: str, issues=None, fix=None, capped=False
):
    return {
        "capped": capped,
        "featured": False,
        "label": leak_label,
        "title": title,
        "value": value,
        "jobs": jobs,
        "explanation": explanation,
        "issues": issues or [],
        "fix": fix,
    }


def render_report(request: Request, report: ScanResponse):
    estimate = report.leak_estimate
    checks_by_name = {check.check_name: check for check in report.checks}
    leaks_by_label = {leak.label: leak for leak in estimate.supporting_leaks}

    foundation = leaks_by_label.pop("Online Foundation Leak", None)
    website = leaks_by_label.pop("Website Performance Leak", None) or leaks_by_label.pop(
        "Website Conversion Leak", None
    )

    foundation_card = (
        _leak_card(
            "Online Foundation Leak",
            "Online foundation leaks found",
            foundation.monthly_value,
            foundation.jobs_per_month,
            foundation.explanation,
            foundation.issues,
            FOUNDATION_FIX,
            capped=True,
        )
        if foundation
        else None
    )
    needs_reputation = foundation is not None and any(
        word in issue.lower() for issue in foundation.issues for word in ("review", "rating", "reputation")
    )
    website_card = (
        _leak_card(
            "Website Performance Leak",
            "Website performance leaks found",
            website.monthly_value,
            website.jobs_per_month,
            website.explanation,
            website.issues,
            WEBSITE_FIX,
            capped=True,
        )
        if website
        else None
    )

    other_cards = []
    if estimate.headline_leak_monthly > 0:
        other_cards.append(
            _leak_card(
                "Slow lead response",
                "Slow lead response",
                estimate.headline_leak_monthly,
                estimate.headline_jobs_per_month,
                estimate.headline_explanation,
                fix=SPEED_TO_LEAD_FIX,
            )
        )
    # Leaks from reports saved before the new sections existed.
    for leak in leaks_by_label.values():
        other_cards.append(
            _leak_card(leak.label, leak.label, leak.monthly_value, leak.jobs_per_month, leak.explanation, leak.issues)
        )

    for card in other_cards:
        # Slow lead response is an "other opportunity", shown with the
        # brighter styling like dormant leads.
        card["featured"] = True

    # Hero: foundation + website leaks (measured online), plus a second line
    # for other opportunities (slow response + dormant leads).
    leak_total = estimate.foundation_website_leaks_monthly or sum(
        card["value"] for card in (foundation_card, website_card) if card
    )
    # Reports saved before this split didn't store the separate figures.
    if not estimate.foundation_website_leaks_monthly and not estimate.other_opportunities_monthly:
        leak_total += sum(card["value"] for card in other_cards if card["label"] != "Slow lead response")
    opportunities_total = estimate.other_opportunities_monthly or (
        estimate.headline_leak_monthly + estimate.dormant_lead_value
    )

    return templates.TemplateResponse(
        request,
        "report.html",
        {
            "report": report,
            "estimate": estimate,
            "leak_total": leak_total,
            "opportunities_total": opportunities_total,
            "checks_run": report.checks_run or 20,
            "foundation_checks": [checks_by_name[name] for name in FOUNDATION_CHECK_NAMES if name in checks_by_name],
            "foundation_card": foundation_card,
            "reputation_fix": REPUTATION_FIX if needs_reputation else None,
            "website_checks": [checks_by_name[name] for name in WEBSITE_CHECK_NAMES if name in checks_by_name],
            "website_rows": _website_rows(report.website_details or {}),
            "website_card": website_card,
            "other_cards": other_cards,
            "ad_status": ad_status(checks_by_name.get("Visible Ad Activity")),
            "ad_check": checks_by_name.get("Visible Ad Activity"),
            "scan_date": report.scanned_at.strftime("%B %-d, %Y"),
            "booking_url": config.BOOKING_URL,
            "lead_revival_fix": LEAD_REVIVAL_FIX,
        },
    )
