import re
from typing import Optional

from models import LeakEstimate, ScanRequest, SupportingLeak

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

# Visibility gap: only worth calculating when the two AI-agent search checks
# average out to genuinely poor visibility (below this out of 10).
VISIBILITY_GAP_SCORE_THRESHOLD = 5
# Conservative ceiling on how many additional monthly leads improved
# visibility could plausibly bring in, even at the worst-case gap.
VISIBILITY_GAP_LEAD_CEILING = 0.20

# Reputation gap: only worth calculating below this overall reputation score.
REPUTATION_GAP_SCORE_THRESHOLD = 6
# Conservative ceiling on how many monthly leads a poor reputation plausibly
# deters before they ever make contact.
REPUTATION_GAP_DETERRENCE_CEILING = 0.05

# Google's "good" Largest Contentful Paint threshold, in seconds.
LCP_BASELINE_SECONDS = 2.5
# Published conversion-rate research: roughly a 7% conversion drop per
# additional second of load time beyond the baseline, capped at 70%.
LCP_CONVERSION_PENALTY_PER_SECOND = 0.07
LCP_CONVERSION_PENALTY_CAP = 0.70


def _parse_currency(value: str) -> float:
    match = re.search(r"[\d,]+\.?\d*", value)
    if not match:
        raise ValueError(f"Could not parse a numeric value from avg_job_value: {value!r}")
    return float(match.group(0).replace(",", ""))


def calculate_visibility_gap(
    ai_visibility_score: int,
    local_ranking_score: int,
    monthly_leads: float,
    close_rate: float,
    avg_job_value: float,
) -> Optional[SupportingLeak]:
    """UPSIDE/OPPORTUNITY, not a loss - invisibility in AI search results and
    local rankings means never entering consideration in the first place,
    not losing a customer you already had. Only calculated when the two
    AI-agent search checks average out to genuinely poor visibility."""
    avg_score = (ai_visibility_score + local_ranking_score) / 2
    if avg_score >= VISIBILITY_GAP_SCORE_THRESHOLD:
        return None

    visibility_gap = (VISIBILITY_GAP_SCORE_THRESHOLD - avg_score) / VISIBILITY_GAP_SCORE_THRESHOLD
    potential_additional_leads = monthly_leads * VISIBILITY_GAP_LEAD_CEILING * visibility_gap
    value = potential_additional_leads * close_rate * avg_job_value

    explanation = (
        "UPSIDE/OPPORTUNITY, not a loss - invisibility in AI search results and local rankings "
        "means never entering consideration in the first place, not losing a customer you "
        "already had. This is a rough estimate based on two AI-agent search checks (AI Search "
        f"Visibility: {ai_visibility_score}/10, Local Search Ranking: {local_ranking_score}/10), "
        "not measured traffic data, using a conservative 20% ceiling on how many additional "
        "monthly leads improved visibility could plausibly bring."
    )
    return SupportingLeak(label="Visibility Gap", monthly_value=round(value, 2), explanation=explanation)


def calculate_reputation_gap(
    reputation_score: int, monthly_leads: float, close_rate: float, avg_job_value: float
) -> Optional[SupportingLeak]:
    """Only calculated when the overall reputation score is genuinely poor."""
    if reputation_score >= REPUTATION_GAP_SCORE_THRESHOLD:
        return None

    severity = (REPUTATION_GAP_SCORE_THRESHOLD - reputation_score) / REPUTATION_GAP_SCORE_THRESHOLD
    estimated_deterred_customers = monthly_leads * REPUTATION_GAP_DETERRENCE_CEILING * severity
    value = estimated_deterred_customers * close_rate * avg_job_value

    explanation = (
        f"Uses your overall online reputation score ({reputation_score}/10) as a proxy for how "
        "many prospects are deterred before ever contacting you - this is NOT a direct count of "
        "unanswered reviews or specific negative feedback, since we don't have that data. Treat "
        "this as a rough, conservative estimate, not a precise figure."
    )
    return SupportingLeak(label="Reputation Gap", monthly_value=round(value, 2), explanation=explanation)


def calculate_website_speed_leak(
    lcp_seconds: Optional[float], actual_customers: float, avg_job_value: float
) -> Optional[SupportingLeak]:
    """Skipped entirely if LCP couldn't be measured."""
    if lcp_seconds is None:
        return None

    excess_seconds = max(0.0, lcp_seconds - LCP_BASELINE_SECONDS)
    conversion_penalty = min(excess_seconds * LCP_CONVERSION_PENALTY_PER_SECOND, LCP_CONVERSION_PENALTY_CAP)
    value = actual_customers * conversion_penalty * avg_job_value

    explanation = (
        "Based on published conversion-rate research (roughly a 7% conversion drop per "
        "additional second of load time beyond Google's 2.5s 'good' Largest Contentful Paint "
        f"threshold), not something measured specifically for this business. Your site's LCP "
        f"measured {lcp_seconds:.2f}s."
    )
    return SupportingLeak(label="Website Speed Leak", monthly_value=round(value, 2), explanation=explanation)


def calculate_leak_estimate(
    scan_request: ScanRequest,
    review_data: dict,
    ai_visibility_score: int,
    local_ranking_score: int,
    reputation_score: int,
    lcp_seconds: Optional[float],
) -> LeakEstimate:
    """Pure deterministic math - no AI calls. All inputs here are either
    self-reported answers or stated assumptions, so every figure this
    function produces is an ESTIMATE, never a measured value.

    review_data is accepted for signature stability with the eventual
    review-response-gap calculation, but is intentionally unused for now -
    the Places API doesn't expose owner-response data, so that piece is
    deferred to a later step that gathers it a different way.

    ai_visibility_score, local_ranking_score, and reputation_score are the
    0-10 scores from the corresponding CheckResults, and lcp_seconds is the
    website's measured Largest Contentful Paint (or None if it couldn't be
    measured) - all four feed the supporting_leaks calculations below.
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
        "if you responded in under 5 minutes. Response-time research shows slower follow-up "
        "typically loses customers to faster-responding competitors or lead fatigue, though we "
        "can't confirm that's specifically what happened with your leads. This is exactly the kind "
        "of leak our Speed-to-Lead bot is built to close, by responding to every new lead in under "
        "60 seconds."
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
        f"average job value ({scan_request.avg_job_value} → ~${avg_job_value_numeric:,.2f}). This is "
        "exactly the upside our Lead Revival campaign is built to capture, by reaching back out to "
        "these dormant leads on your behalf."
    )

    # --- 3. Supporting leaks (each an independent "if only this one thing
    # were fixed" estimate - never summed into a combined total anywhere in
    # what's returned or displayed) ---
    supporting_leaks = []
    visibility_gap = calculate_visibility_gap(
        ai_visibility_score, local_ranking_score, monthly_leads_numeric, close_rate_numeric, avg_job_value_numeric
    )
    if visibility_gap is not None:
        supporting_leaks.append(visibility_gap)

    reputation_gap = calculate_reputation_gap(
        reputation_score, monthly_leads_numeric, close_rate_numeric, avg_job_value_numeric
    )
    if reputation_gap is not None:
        supporting_leaks.append(reputation_gap)

    website_speed_leak = calculate_website_speed_leak(lcp_seconds, actual_customers, avg_job_value_numeric)
    if website_speed_leak is not None:
        supporting_leaks.append(website_speed_leak)

    # --- 4. Sanity check ---
    # Flags both if the headline number alone looks implausible, and if the
    # headline plus every supporting leak combined would - catching the case
    # where each individual leak looks reasonable but the combination
    # doesn't. This combined figure is used only for this boolean flag and
    # is not stored or displayed anywhere as a total.
    implied_monthly_revenue = monthly_leads_numeric * avg_job_value_numeric * close_rate_numeric
    plausibility_ceiling = implied_monthly_revenue * PLAUSIBILITY_CEILING_MULTIPLIER
    combined_leak_total = headline_leak_monthly + sum(leak.monthly_value for leak in supporting_leaks)
    flagged_for_review = (
        headline_leak_monthly > plausibility_ceiling or combined_leak_total > plausibility_ceiling
    )

    return LeakEstimate(
        headline_leak_monthly=headline_leak_monthly,
        headline_explanation=headline_explanation,
        supporting_leaks=supporting_leaks,
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
    test_review_data = {"review_count": 22, "reviews_with_owner_response": 0}

    # Illustrative scores consistent with prior real /scan runs against this
    # business - AI Search Visibility 0/10, Local Search Ranking 1/10,
    # Online Reputation Scan 4/10, and a plausible LCP for a middling
    # performance score.
    result = calculate_leak_estimate(
        test_request,
        test_review_data,
        ai_visibility_score=0,
        local_ranking_score=1,
        reputation_score=4,
        lcp_seconds=4.2,
    )
    print(result.model_dump_json(indent=2))
