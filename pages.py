"""Renders the public HTML pages: the landing page with the scan form, the
"scan is running" page, and the finished report page."""
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Dict, Optional

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


def _jobs(jobs: float) -> str:
    """0.5 -> 'about 1 job every 2 months', 1 -> 'about 1 job/mo', 3 -> 'about 3 jobs/mo'."""
    if jobs <= 0:
        return ""
    if jobs < 1:
        return "about 1 job every 2 months"
    count = int(jobs)
    return f"about {count} job{'s' if count != 1 else ''}/mo"


templates.env.filters["money"] = _money
templates.env.filters["jobs"] = _jobs

# The dropdown choices must match the Literal values in models.ScanRequest
# exactly (including the en dashes), or the form won't validate.
FORM_OPTIONS = {
    "monthly_leads": ["Under 10", "10–25", "25–50", "50–100", "100+"],
    "response_time": ["Under 5 minutes", "Within an hour", "Same day", "Next day or longer"],
    "close_rate": ["Under 10%", "10–25%", "25–50%", "50%+"],
    "dormant_leads": ["None that I know of", "A handful", "25–100", "100–500", "500+"],
}

FIELD_LABELS = {
    "contact_name": "Your name",
    "email": "Email",
    "phone": "Phone",
    "business_name": "Business name",
    "website_url": "Website",
    "business_type": "What your business does",
    "monthly_leads": "New leads per month",
    "avg_job_value": "Average job value",
    "response_time": "How fast you usually respond to a new lead",
    "close_rate": "Share of leads that become customers",
    "dormant_leads": "Old leads that never became customers",
}

# The service that fixes each leak, with its price. Edit names and prices
# here - the report reads them from this table. "headline" is the slow lead
# response leak; the others match the supporting leak labels.
LEAK_FIXES = {
    "headline": {"name": "Speed-to-Lead Bot", "price": "$997 setup + $150/mo"},
    "Website Speed Leak": {"name": "New Website with Online Booking", "price": "$997 + $150/mo"},
    "Visibility Gap": {
        "name": "Google & Social Profile Consistency Fix",
        "price": "$497",
        "upgrade": "$997 with AI search (GEO/AEO) registration",
    },
    "Reputation Gap": {"name": "Review Bot", "price": "$497 + $100/mo"},
}
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


def render_report(request: Request, report: ScanResponse):
    estimate = report.leak_estimate
    leaks = []
    if estimate.headline_leak_monthly > 0:
        leaks.append(
            {
                "title": "Slow lead response",
                "value": estimate.headline_leak_monthly,
                "jobs": estimate.headline_jobs_per_month,
                "explanation": estimate.headline_explanation,
                "fix": LEAK_FIXES["headline"],
            }
        )
    for leak in estimate.supporting_leaks:
        leaks.append(
            {
                "title": leak.label,
                "value": leak.monthly_value,
                "jobs": leak.jobs_per_month,
                "explanation": leak.explanation,
                "fix": LEAK_FIXES.get(leak.label),
            }
        )
    leaks.sort(key=lambda item: item["value"], reverse=True)

    return templates.TemplateResponse(
        request,
        "report.html",
        {
            "report": report,
            "estimate": estimate,
            "leaks": leaks,
            # Reports saved before the total existed fall back to summing.
            "total_leak": estimate.total_leak_monthly or sum(item["value"] for item in leaks),
            "scan_date": report.scanned_at.strftime("%B %-d, %Y"),
            "booking_url": config.BOOKING_URL,
            "lead_revival_fix": LEAD_REVIVAL_FIX,
        },
    )
