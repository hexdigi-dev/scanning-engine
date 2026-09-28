import math
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

# Share of the business a company already wins that it's estimated to lose
# by responding at each speed, compared with responding in under 5 minutes.
# Stated assumptions, kept deliberately moderate so owners find them
# believable; the reported close rate already reflects their real response
# time, so this is applied on top of their actual customers.
RESPONSE_TIME_LOSS_SHARE = {
    "Under 5 minutes": 0.0,
    "Within an hour": 0.15,
    "Same day": 0.25,
    "Next day or longer": 0.40,
}

# Stated assumption, not measured: share of the dormant-lead pool a revival
# campaign turns into jobs each month. Monthly rather than one-time because a
# business that keeps marketing keeps adding unconverted leads to the pool.
DORMANT_MONTHLY_REVIVAL_RATE = 0.04
# Revived jobs per month are capped at this share of current monthly
# customers, so a very large old-lead list can't produce an unrealistic figure.
DORMANT_REVIVAL_CAP_SHARE = 0.50

# Rough plausibility ceiling multiplier for the sanity check. The combined
# headline + supporting-leaks total is structurally bounded at ~1.2x
# implied_monthly_revenue by the underlying formulas (headline_leak_monthly
# alone maxes out at 0.40x, at the slowest response-time tier), so 1.0x sits
# comfortably above what real businesses produce while still being
# reachable by genuinely extreme combinations - unlike the old 3.0x, which
# the combined total could never mathematically reach.
PLAUSIBILITY_CEILING_MULTIPLIER = 1.0

# Visibility gap: only worth calculating when the two AI-agent search checks
# average out to genuinely poor visibility (below this out of 10).
VISIBILITY_GAP_SCORE_THRESHOLD = 5
# Conservative ceiling on how many additional monthly leads improved
# visibility could plausibly bring in, even at the worst-case gap.
VISIBILITY_GAP_LEAD_CEILING = 0.10

# Reputation gap: only worth calculating below this overall reputation score.
REPUTATION_GAP_SCORE_THRESHOLD = 6
# Conservative ceiling on how many monthly leads a poor reputation plausibly
# deters before they ever make contact.
REPUTATION_GAP_DETERRENCE_CEILING = 0.05

# Google's "good" Largest Contentful Paint threshold, in seconds.
LCP_BASELINE_SECONDS = 2.5
# Published conversion-rate research: roughly a 7% conversion drop per
# additional second of load time beyond the baseline, capped at 20%.
LCP_CONVERSION_PENALTY_PER_SECOND = 0.07
LCP_CONVERSION_PENALTY_CAP = 0.20
# Stated assumption: only about half of a local service business's leads
# come through its website (the rest are calls, referrals, repeat customers),
# so a slow site only affects that share.
WEBSITE_LEAD_SHARE = 0.50


def _parse_currency(value: str) -> float:
    match = re.search(r"[\d,]+\.?\d*", value)
    if not match:
        raise ValueError(f"Could not parse a numeric value from avg_job_value: {value!r}")
    return float(match.group(0).replace(",", ""))


def _whole_jobs(raw_jobs: float) -> float:
    """Every estimate is stated as whole jobs at the business's own average
    job value, rounded DOWN, so figures read as floors the owner can sanity
    check ("about 3 jobs a month") rather than falsely precise dollars. Any
    detected leak below one job counts as one job every two months (0.5)."""
    if raw_jobs <= 0:
        return 0.0
    whole = float(math.floor(raw_jobs))
    return whole if whole >= 1 else 0.5


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
    jobs = _whole_jobs(potential_additional_leads * close_rate)
    if jobs == 0:
        return None
    value = jobs * avg_job_value

    explanation = (
        "UPSIDE/OPPORTUNITY, not a loss - invisibility in AI search results and local rankings "
        "means never entering consideration in the first place, not losing a customer you "
        "already had. This is a rough estimate based on two AI-agent search checks (AI Search "
        f"Visibility: {ai_visibility_score}/10, Local Search Ranking: {local_ranking_score}/10), "
        "not measured traffic data. It assumes improved visibility would bring in at least 10% "
        "more monthly leads, so present this figure as a floor: 'at least $X a month'."
    )
    return SupportingLeak(
        label="Visibility Gap", monthly_value=round(value, 2), explanation=explanation, jobs_per_month=jobs
    )


def calculate_reputation_gap(
    reputation_score: int, monthly_leads: float, close_rate: float, avg_job_value: float
) -> Optional[SupportingLeak]:
    """Only calculated when the overall reputation score is genuinely poor."""
    if reputation_score >= REPUTATION_GAP_SCORE_THRESHOLD:
        return None

    severity = (REPUTATION_GAP_SCORE_THRESHOLD - reputation_score) / REPUTATION_GAP_SCORE_THRESHOLD
    estimated_deterred_customers = monthly_leads * REPUTATION_GAP_DETERRENCE_CEILING * severity
    jobs = _whole_jobs(estimated_deterred_customers * close_rate)
    if jobs == 0:
        return None
    value = jobs * avg_job_value

    explanation = (
        f"Uses your overall online reputation score ({reputation_score}/10) as a proxy for how "
        "many prospects are deterred before ever contacting you - this is NOT a direct count of "
        "unanswered reviews or specific negative feedback, since we don't have that data. Treat "
        "this as a rough, conservative estimate, not a precise figure."
    )
    return SupportingLeak(
        label="Reputation Gap", monthly_value=round(value, 2), explanation=explanation, jobs_per_month=jobs
    )


def calculate_website_speed_leak(
    lcp_seconds: Optional[float], current_customers: float, avg_job_value: float
) -> Optional[SupportingLeak]:
    """Skipped entirely if LCP couldn't be measured."""
    if lcp_seconds is None:
        return None

    excess_seconds = max(0.0, lcp_seconds - LCP_BASELINE_SECONDS)
    conversion_penalty = min(excess_seconds * LCP_CONVERSION_PENALTY_PER_SECOND, LCP_CONVERSION_PENALTY_CAP)
    jobs = _whole_jobs(current_customers * WEBSITE_LEAD_SHARE * conversion_penalty)
    if jobs == 0:
        return None
    value = jobs * avg_job_value

    explanation = (
        "Based on published conversion-rate research (roughly a 7% conversion drop per "
        "additional second of load time beyond Google's 2.5s 'good' Largest Contentful Paint "
        "threshold, capped at 20%), applied only to the roughly half of leads assumed to come "
        "through the website - not something measured specifically for this business. Your "
        f"site's LCP measured {lcp_seconds:.2f}s."
    )
    return SupportingLeak(
        label="Website Speed Leak", monthly_value=round(value, 2), explanation=explanation, jobs_per_month=jobs
    )


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
    response_time_loss_share = RESPONSE_TIME_LOSS_SHARE[scan_request.response_time]

    # --- 1. Response-time leak (headline number) ---
    current_customers = monthly_leads_numeric * close_rate_numeric
    lost_customers = current_customers * response_time_loss_share
    headline_jobs = _whole_jobs(lost_customers)
    headline_leak_monthly = round(headline_jobs * avg_job_value_numeric, 2)
    headline_explanation = (
        "ESTIMATE, not measured data - based on your self-reported monthly leads "
        f"({scan_request.monthly_leads} → ~{monthly_leads_numeric:g}/mo), close rate "
        f"({scan_request.close_rate} → ~{close_rate_numeric:.0%}), and average job value "
        f"({scan_request.avg_job_value} → ~${avg_job_value_numeric:,.2f}). At your self-reported "
        f"response time ({scan_request.response_time}), we assume you lose about "
        f"{response_time_loss_share:.0%} of the business you'd win by responding in under 5 "
        f"minutes, on top of your ~{current_customers:.1f} current customers/mo - rounded down, "
        f"about {headline_jobs:g} job(s) a month. Response-time "
        "research shows slower follow-up "
        "typically loses customers to faster-responding competitors or lead fatigue, though we "
        "can't confirm that's specifically what happened with your leads. This is exactly the kind "
        "of leak our Speed-to-Lead bot is built to close, by responding to every new lead in under "
        "60 seconds."
    )

    # --- 2. Dormant lead value (upside, not a leak) ---
    dormant_leads_numeric = DORMANT_LEADS_MIDPOINTS[scan_request.dormant_leads]
    dormant_jobs = _whole_jobs(
        min(
            dormant_leads_numeric * DORMANT_MONTHLY_REVIVAL_RATE,
            current_customers * DORMANT_REVIVAL_CAP_SHARE,
        )
    )
    dormant_lead_value = round(dormant_jobs * avg_job_value_numeric, 2)
    dormant_lead_explanation = (
        "ESTIMATE, not measured data - assumes about 4% of your self-reported dormant leads "
        f"({scan_request.dormant_leads} → ~{dormant_leads_numeric:g}) can be revived into jobs each "
        "month, rounded down to whole jobs at your self-reported average job value "
        f"({scan_request.avg_job_value} → ~${avg_job_value_numeric:,.2f}) - about "
        f"{dormant_jobs:g} job(s) a month. Ongoing marketing keeps adding unconverted leads to this "
        "pool, so this is a monthly figure. This is exactly the upside our Lead Revival campaign is "
        "built to capture, by reaching back out to these dormant leads on your behalf."
    )

    # --- 3. Supporting leaks (each rounded down to whole jobs, so their sum
    # is shown as a conservative "at least" total alongside the headline) ---
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

    website_speed_leak = calculate_website_speed_leak(lcp_seconds, current_customers, avg_job_value_numeric)
    if website_speed_leak is not None:
        supporting_leaks.append(website_speed_leak)

    # --- 4. Sanity check ---
    # Flags both if the headline number alone looks implausible, and if the
    # headline plus every supporting leak combined would - catching the case
    # where each individual leak looks reasonable but the combination
    # doesn't.
    implied_monthly_revenue = monthly_leads_numeric * avg_job_value_numeric * close_rate_numeric
    plausibility_ceiling = implied_monthly_revenue * PLAUSIBILITY_CEILING_MULTIPLIER
    combined_leak_total = round(
        headline_leak_monthly + sum(leak.monthly_value for leak in supporting_leaks), 2
    )
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
        headline_jobs_per_month=headline_jobs,
        dormant_jobs_per_month=dormant_jobs,
        total_leak_monthly=combined_leak_total,
        avg_job_value=avg_job_value_numeric,
        close_rate=close_rate_numeric,
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
