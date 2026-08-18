import re

from models import LeakEstimate, ScanRequest

# Midpoints for open-ended self-reported ranges. Ranges use the reported
# range's midpoint; open-ended ends ("Under 10", "100+", "50%+") use a
# conservative point estimate rather than the range's true midpoint, since
# that's unbounded.
MONTHLY_LEADS_MIDPOINTS = {
    "Under 10": 5.0,
    "10–25": 17.5,
    "25–50": 37.5,
    "50–100": 75.0,
    "100+": 100.0,
}

CLOSE_RATE_MIDPOINTS = {
    "Under 10%": 0.05,
    "10–25%": 0.175,
    "25–50%": 0.375,
    "50%+": 0.5,
}

DORMANT_LEADS_MIDPOINTS = {
    "None that I know of": 0.0,
    "A handful": 5.0,
    "25–100": 62.5,
    "100–500": 300.0,
    "500+": 500.0,
}

# Tiered close-rate multiplier: how much of the baseline (best-case) customer
# volume is actually captured at each response-time tier. "Under 5 minutes"
# is the baseline (1.0x, no leak); everything slower loses some share of
# leads to competitors or lead fatigue before the business responds.
RESPONSE_TIME_MULTIPLIERS = {
    "Under 5 minutes": 1.0,
    "Within an hour": 0.6,
    "Same day": 0.35,
    "Next day or longer": 0.15,
}

# Stated assumption, not measured: conservative estimate of what share of
# dormant leads could realistically be won back via a reactivation campaign.
DORMANT_REACTIVATION_RATE = 0.10

# Rough plausibility ceiling multiplier for the sanity check.
PLAUSIBILITY_CEILING_MULTIPLIER = 3.0


def _parse_currency(value: str) -> float:
    match = re.search(r"[\d,]+\.?\d*", value)
    if not match:
        raise ValueError(f"Could not parse a numeric value from avg_job_value: {value!r}")
    return float(match.group(0).replace(",", ""))


def calculate_leak_estimate(scan_request: ScanRequest, review_data: dict) -> LeakEstimate:
    """Pure deterministic math - no AI calls. All inputs here are either
    self-reported answers or stated assumptions, so every figure this
    function produces is an ESTIMATE, never a measured value.

    review_data is accepted for signature stability with the eventual
    review-response-gap calculation, but is intentionally unused for now -
    the Places API doesn't expose owner-response data, so that piece is
    deferred to a later step that gathers it a different way.
    """
    monthly_leads_numeric = MONTHLY_LEADS_MIDPOINTS[scan_request.monthly_leads]
    close_rate_numeric = CLOSE_RATE_MIDPOINTS[scan_request.close_rate]
    avg_job_value_numeric = _parse_currency(scan_request.avg_job_value)
    response_time_multiplier = RESPONSE_TIME_MULTIPLIERS[scan_request.response_time]

    # --- 1. Response-time leak (headline number) ---
    baseline_customers = monthly_leads_numeric * close_rate_numeric
    actual_customers = baseline_customers * response_time_multiplier
    lost_customers = baseline_customers - actual_customers
    response_time_leak = lost_customers * avg_job_value_numeric

    headline_leak_monthly = round(response_time_leak, 2)
    headline_explanation = (
        "ESTIMATE, not measured data - based on your self-reported monthly leads "
        f"({scan_request.monthly_leads} → ~{monthly_leads_numeric:g}/mo), close rate "
        f"({scan_request.close_rate} → ~{close_rate_numeric:.0%}), and average job value "
        f"({scan_request.avg_job_value} → ~${avg_job_value_numeric:,.2f}). At your self-reported "
        f"response time ({scan_request.response_time}), we assume you capture "
        f"{response_time_multiplier:.0%} of the ~{baseline_customers:.1f} customers/mo you'd close "
        "if you responded in under 5 minutes, losing the rest to competitors or lead fatigue."
    )

    # --- 2. Dormant lead value (upside, not a leak) ---
    dormant_leads_numeric = DORMANT_LEADS_MIDPOINTS[scan_request.dormant_leads]
    dormant_lead_value = round(
        dormant_leads_numeric * DORMANT_REACTIVATION_RATE * close_rate_numeric * avg_job_value_numeric,
        2,
    )
    dormant_lead_explanation = (
        "ESTIMATE, not measured data - assumes a stated 10% reactivation rate applied to your "
        f"self-reported dormant lead count ({scan_request.dormant_leads} → ~{dormant_leads_numeric:g}) "
        f"and close rate ({scan_request.close_rate} → ~{close_rate_numeric:.0%}), at your self-reported "
        f"average job value ({scan_request.avg_job_value} → ~${avg_job_value_numeric:,.2f})."
    )

    # --- 3. Sanity check ---
    implied_monthly_revenue = monthly_leads_numeric * avg_job_value_numeric * close_rate_numeric
    plausibility_ceiling = implied_monthly_revenue * PLAUSIBILITY_CEILING_MULTIPLIER
    flagged_for_review = headline_leak_monthly > plausibility_ceiling

    return LeakEstimate(
        headline_leak_monthly=headline_leak_monthly,
        headline_explanation=headline_explanation,
        supporting_leaks=[],
        dormant_lead_value=dormant_lead_value,
        dormant_lead_explanation=dormant_lead_explanation,
        flagged_for_review=flagged_for_review,
    )


if __name__ == "__main__":
    test_request = ScanRequest(
        contact_name="Test Contact",
        phone="555-000-0000",
        email="test@ericksonsdrying.com",
        business_name="Erickson's Drying Systems",
        website_url="https://www.ericksonsdrying.com",
        business_type="Water damage restoration & mold remediation",
        monthly_leads="25–50",
        avg_job_value="$4,500",
        response_time="Within an hour",
        close_rate="25–50%",
        dormant_leads="100–500",
    )
    test_review_data = {"review_count": 0, "reviews_with_owner_response": 0}

    result = calculate_leak_estimate(test_request, test_review_data)
    print(result.model_dump_json(indent=2))
