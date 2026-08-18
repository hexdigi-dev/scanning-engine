from datetime import datetime
from typing import List, Literal

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


class LeakEstimate(BaseModel):
    headline_leak_monthly: float
    headline_explanation: str
    supporting_leaks: List[SupportingLeak]
    dormant_lead_value: float
    flagged_for_review: bool


class ScanResponse(BaseModel):
    business_name: str
    scanned_at: datetime
    checks: List[CheckResult]
    leak_estimate: LeakEstimate
