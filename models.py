from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, HttpUrl, TypeAdapter, field_validator

_url_adapter = TypeAdapter(HttpUrl)


class ScanRequest(BaseModel):
    contact_name: str
    phone: str
    email: str
    business_name: str
    website_url: str
    business_type: str
    # Business location from the form. Optional in the API so requests from
    # before these fields existed still work. City/state are the location for
    # the location-based checks when no Google listing matches; the street
    # address (optional - many service-area businesses don't publish one)
    # and ZIP also help match the Google listing and feed the consistency check.
    street_address: str = ""
    city: str = ""
    state: str = ""
    zip_code: str = ""
    monthly_leads: Literal["Under 10", "10–25", "25–50", "50–100", "100+"]
    avg_job_value: str
    response_time: Literal[
        "Under 5 minutes", "Within an hour", "Same day", "Next day or longer"
    ]
    close_rate: Literal["Under 10%", "10–25%", "25–50%", "50%+"]
    dormant_leads: Literal[
        "None that I know of", "A handful", "25–100", "100–500", "500+"
    ]

    @field_validator("website_url")
    @classmethod
    def validate_website_url(cls, v: str) -> str:
        _url_adapter.validate_python(v)
        return v


def form_location(request: "ScanRequest") -> Optional[str]:
    """'Fort Myers, FL' from the form, or None if city/state weren't given."""
    city, state = request.city.strip(), request.state.strip()
    return f"{city}, {state}" if city and state else None


def form_full_address(request: "ScanRequest") -> Optional[str]:
    """'12960 Commerce Lakes Dr, Fort Myers, FL 33913', or None without a street."""
    street = request.street_address.strip()
    location = form_location(request)
    if not street or not location:
        return None
    return f"{street}, {location} {request.zip_code.strip()}".strip()


class CheckResult(BaseModel):
    check_name: str
    score: int
    summary: str
    # "not_checked": the check couldn't run (e.g. no location), so its score
    # means nothing and no leak is estimated from it.
    source_type: Literal["measured", "estimated", "not_checked"]

    @field_validator("score")
    @classmethod
    def validate_score(cls, v: int) -> int:
        if not 0 <= v <= 10:
            raise ValueError("score must be between 0 and 10")
        return v


class SupportingLeak(BaseModel):
    label: str
    monthly_value: float
    explanation: str
    # Whole jobs per month the value represents (0.5 = one job every two
    # months). Defaults keep reports saved before this field existed loadable.
    jobs_per_month: float = 0.0
    # Plain-language problems behind this leak (used by the website leak to
    # list each issue found on the report).
    issues: List[str] = []


class LeakEstimate(BaseModel):
    headline_leak_monthly: float
    headline_explanation: str
    supporting_leaks: List[SupportingLeak]
    dormant_lead_value: float
    dormant_lead_explanation: str
    flagged_for_review: bool
    headline_jobs_per_month: float = 0.0
    dormant_jobs_per_month: float = 0.0
    # Headline leak plus every supporting leak, each already rounded down to
    # whole jobs. Excludes the dormant lead value, which is upside rather
    # than a leak.
    total_leak_monthly: float = 0.0
    # Foundation + website leaks only (what the scan measured online), and
    # the "other potential opportunities": slow lead response + dormant leads.
    foundation_website_leaks_monthly: float = 0.0
    other_opportunities_monthly: float = 0.0
    avg_job_value: float = 0.0
    # The self-reported close rate (range midpoint), used to price the
    # performance-based Lead Revival options.
    close_rate: float = 0.0


class ScanResponse(BaseModel):
    business_name: str
    scanned_at: datetime
    checks: List[CheckResult]
    leak_estimate: LeakEstimate
    report_id: Optional[str] = None
    report_url: Optional[str] = None
    # Website check details behind the Website Performance section (load
    # time, homepage signals, and the call-to-action assessment).
    website_details: dict = {}
    # How many checks this scan ran (for the email: "We ran N checks").
    checks_run: int = 0
    # Ready-to-use dollar text for the email ("$10,000"), so Make needs no
    # number formatting.
    leaks_display: str = ""
    opportunities_display: str = ""
