"""Renders the public HTML pages: the landing page with the scan form, the
"scan is running" page, and the finished report page."""
from pathlib import Path
from typing import Dict, Optional

from fastapi import Request
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

import config
from models import ScanResponse

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def _money(value: float) -> str:
    return f"${value:,.0f}"


templates.env.filters["money"] = _money

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

# Known fixes for each leak. Leaks without a matching service simply don't
# show a fix line.
LEAK_FIXES = {
    "headline": "Speed-to-Lead Bot",
    "Reputation Gap": "Review Response Service",
}


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
    leaks = [
        {
            "title": "Slow lead response",
            "value": estimate.headline_leak_monthly,
            "explanation": estimate.headline_explanation,
            "fix": LEAK_FIXES["headline"],
            "is_headline": True,
        }
    ]
    for leak in estimate.supporting_leaks:
        leaks.append(
            {
                "title": leak.label,
                "value": leak.monthly_value,
                "explanation": leak.explanation,
                "fix": LEAK_FIXES.get(leak.label),
                "is_headline": False,
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
            "scan_date": report.scanned_at.strftime("%B %-d, %Y"),
            "booking_url": config.BOOKING_URL,
        },
    )
