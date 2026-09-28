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


class CheckResult(BaseModel):
    check_name: str
    score: int
    summary: str
    source_type: Literal["measured", "estimated"]

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
