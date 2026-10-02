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
    "after_hours": ["Goes to voicemail", "Answering service", "AI receptionist", "Someone always answers", "Not sure"],
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
    "after_hours": "When someone calls after hours, what happens?",
}

# The service that fixes each section's leak, with its price. Edit names
# and prices here - the report reads them from these.
FOUNDATION_FIX = {
    "name": "Foundation Fix",
    "starting": True,
    # Standard price, shown crossed out, and the price when they start on the
    # follow-up call booked from this report.
    "was": "$1,497",
    "price": "$997",
    "monthly": "+ $100/mo",
    "upgrade": "when you start on your follow-up call",
    "detail": "Completes and aligns your Google Business Profile, directory listings, review-site "
    "profiles, and social profiles.",
}
# Offered next to the Foundation Fix when AI assistants don't recommend them.
AI_SEARCH_ADDON = {
    "name": "AI Search Registration (AEO/GEO) add-on",
    "starting": True,
    "price": "$497 + $100/mo",
    "detail": "Sets up your business information so AI assistants like ChatGPT, Gemini, and Claude "
    "can find and recommend you.",
}
REPUTATION_FIX = {
    "name": "Reputation Builder",
    "starting": True,
    "price": "$497 + $100/mo",
    "detail": "Replies to every unanswered review and sends every new client a review request.",
}
WEBSITE_FIX = {
    "name": "New Website",
    "starting": True,
    "was": "$1,497",
    "price": "$997",
    "monthly": "+ $150/mo",
    "upgrade": "when you start on your follow-up call",
    "detail": "A fast, mobile-first site with one clear call-to-action, tap-to-call, a contact form, "
    "a booking form, and local SEO setup included. Add-ons available: online ordering, a \"Text Us\" "
    "line, extra pages, and more.",
}
SPEED_TO_LEAD_FIX = {
    "name": "Speed-to-Lead Bot",
    "starting": True,
    "price": "$497 + $200/mo",
    "detail": "Replies to every new lead (form submissions, missed calls, and lead-site requests) by "
    "text and email in under 60 seconds, any time of day, with your booking link and automatic follow-ups.",
}
OUT_OF_HOURS_FIX = {
    "name": "Out-of-Hours AI Bot",
    "starting": True,
    "price": "$497 + $200/mo",
    "detail": "An after-hours answering service: answers your calls while you're closed and books "
    "appointments or schedules call-backs.",
}
# Offered in the Paid advertising section, whatever the rating.
MARKETING_AUDIT_FIX = {
    "label": "Next step",
    "name": "Pheonix Marketing Audit",
    "price": "Baby Pheonix: $997",
    "note": "up to 2 ad channels. Full Pheonix (3 or more channels): $1,997",
    "detail": "A review of your paid advertising: where you're spending, what's being tracked, what "
    "competitors are doing, and a clear plan to get more jobs from every ad dollar.",
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
    """Headline for the Paid advertising section. New scans carry a status
    from the findings-based rating; older saved reports fall back to the old
    0-10 rubric."""
    if check is None or getattr(check, "source_type", None) == "not_checked":
        return "Not checked"
    if (getattr(check, "details", None) or {}).get("status"):
        return check.details["status"]
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


# Mobile load time (seconds) -> score out of 10. Google rates 2.5s or less
# as good; the steps line up with the website leak's load-time tiers.
LOAD_SPEED_SCORES = ((2.5, 10), (3.0, 8), (4.0, 6), (5.0, 4), (6.0, 2), (7.0, 1))


def load_speed_score(lcp: Optional[float]) -> Optional[int]:
    if lcp is None:
        return None
    return next((score for limit, score in LOAD_SPEED_SCORES if lcp <= limit), 0)


def _ad_rows(check) -> list:
    """The three paid advertising findings as report rows, same style as the
    website rows. The section score itself still comes from rate_paid_ads."""
    if check is None or check.source_type == "not_checked":
        return []
    details = check.details or {}
    running = details.get("ads_running", "unknown")
    competitors = details.get("competitors_advertising", "unknown")
    tags = details.get("ad_tags") or []
    rows = []

    def row(name, ok, pass_text, fail_text):
        status = "unchecked" if ok is None else ("pass" if ok else "fail")
        text = "Couldn't check" if ok is None else (pass_text if ok else fail_text)
        rows.append({"name": name, "status": status, "text": text, "score": None})

    row("Paid ads running", {"yes": True, "no": False}.get(running), "Ads found for your business", "No current ads found")
    row(
        "Ad tags for tracking and retargeting",
        bool(tags) if details.get("tags_checked", True) else None,
        "Installed: " + ", ".join(tags) if tags else "Installed",
        "None found on your website",
    )
    # Competitor ads: a pass only when competitors advertise and so do they;
    # with no competitor ads it's shown as neutral information, not a pass.
    if competitors == "no":
        rows.append(
            {
                "name": "Keeping up with competitor ads",
                "status": "info",
                "label": "None found",
                "text": "No competitors found advertising in your search",
                "score": None,
            }
        )
    else:
        row(
            "Keeping up with competitor ads",
            None if competitors != "yes" else running == "yes",
            "Competitors are advertising and so are you",
            "Competitors are paying to appear above you",
        )
    return rows


def overall_score(scores: list) -> Optional[int]:
    """Average of the checks that ran, on a 0-10 scale, rounded down. Checks
    we couldn't run are left out rather than counted as 0. None if nothing ran."""
    scores = [score for score in scores if score is not None]
    if not scores:
        return None
    return int(sum(scores) / len(scores))


def _website_rows(details: dict) -> list:
    """Pass/fail rows for the website checks behind the Website Performance
    Leak. status is "pass", "fail", or "unchecked"."""
    homepage = details.get("homepage") or {}
    assessment = details.get("assessment") or {}
    rows = []

    def row(name, ok, pass_text, fail_text, score=None):
        status = "unchecked" if ok is None else ("pass" if ok else "fail")
        text = "Couldn't check" if ok is None else (pass_text if ok else fail_text)
        rows.append({"name": name, "status": status, "text": text, "score": score})

    lcp = details.get("lcp_seconds")
    speed_score = load_speed_score(lcp)
    row(
        "Load speed",
        None if lcp is None else speed_score == 10,
        f"{lcp or 0:.1f}s on mobile",
        f"{lcp or 0:.1f}s on mobile (aim for 2.5s or less)",
        score=speed_score,
    )
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
        "bordered": False,
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
    reviews_details = (checks_by_name.get("Google Reviews").details or {}) if checks_by_name.get("Google Reviews") else {}
    needs_reputation = foundation is not None and (
        any(word in issue.lower() for issue in foundation.issues for word in ("review", "rating", "reputation"))
        # Fewer reviews than the top map competitors, or no new review lately.
        or bool(reviews_details.get("behind_competitors"))
        or bool(reviews_details.get("stale"))
    )
    needs_ai_search = foundation is not None and any("ai assistant" in issue.lower() for issue in foundation.issues)
    foundation_extra_fixes = [
        fix for fix, needed in ((REPUTATION_FIX, needs_reputation), (AI_SEARCH_ADDON, needs_ai_search)) if needed
    ]
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
    if estimate.after_hours_leak_monthly > 0:
        other_cards.append(
            _leak_card(
                "After-hours coverage",
                f"After-hours coverage: {estimate.after_hours_status}",
                estimate.after_hours_leak_monthly,
                estimate.after_hours_jobs_per_month,
                estimate.after_hours_explanation,
                estimate.after_hours_gaps,
                fix=OUT_OF_HOURS_FIX,
            )
        )
    # Leaks from reports saved before the new sections existed.
    for leak in leaks_by_label.values():
        other_cards.append(
            _leak_card(leak.label, leak.label, leak.monthly_value, leak.jobs_per_month, leak.explanation, leak.issues)
        )

    for card in other_cards:
        # Slow lead response: same layout as the leak cards above, with the
        # dormant-leads orange border to mark it as an opportunity.
        card["bordered"] = True

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

    foundation_checks = [checks_by_name[name] for name in FOUNDATION_CHECK_NAMES if name in checks_by_name]
    website_checks = [checks_by_name[name] for name in WEBSITE_CHECK_NAMES if name in checks_by_name]
    website_rows = _website_rows(report.website_details or {})

    return templates.TemplateResponse(
        request,
        "report.html",
        {
            "report": report,
            "estimate": estimate,
            "leak_total": leak_total,
            "opportunities_total": opportunities_total,
            "checks_run": report.checks_run or 23,
            "foundation_checks": foundation_checks,
            "foundation_card": foundation_card,
            "foundation_extra_fixes": foundation_extra_fixes,
            "website_checks": website_checks,
            "website_rows": website_rows,
            "foundation_score": overall_score(
                [c.score for c in foundation_checks if c.source_type != "not_checked"]
            ),
            # Pass = 10, needs work = 0, plus the Website Health and
            # Accessibility scores as they are.
            "website_score": overall_score(
                [
                    row["score"] if row.get("score") is not None else (10 if row["status"] == "pass" else 0)
                    for row in website_rows
                    if row["status"] != "unchecked"
                ]
                + [c.score for c in website_checks if c.source_type != "not_checked"]
            ),
            "website_card": website_card,
            "other_cards": other_cards,
            "ad_status": ad_status(checks_by_name.get("Visible Ad Activity")),
            "ad_rating": (
                (checks_by_name["Visible Ad Activity"].details or {}).get("rating", "Need more information")
                if "Visible Ad Activity" in checks_by_name
                else "Need more information"
            ),
            "marketing_audit_fix": MARKETING_AUDIT_FIX,
            "ad_check": checks_by_name.get("Visible Ad Activity"),
            "ad_rows": _ad_rows(checks_by_name.get("Visible Ad Activity")),
            "scan_date": report.scanned_at.strftime("%B %-d, %Y"),
            "booking_url": config.BOOKING_URL,
            "lead_revival_fix": LEAD_REVIVAL_FIX,
        },
    )
