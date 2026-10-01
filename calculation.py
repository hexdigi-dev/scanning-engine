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
    "Next day or longer": 0.50,
}

# Stated assumption, not measured: share of the dormant-lead pool a revival
# campaign turns into jobs each month. Monthly rather than one-time because a
# business that keeps marketing keeps adding unconverted leads to the pool.
DORMANT_MONTHLY_REVIVAL_RATE = 0.08
# Revived jobs per month are capped at this share of current monthly
# customers, so a very large old-lead list can't produce an unrealistic figure.
DORMANT_REVIVAL_CAP_SHARE = 1.00

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
# --- Website Conversion Leak ---
# An optimized home-service website turns about 10% of visitors into leads
# (top contractor sites convert 8-12%; the median is 2-4%). Each problem
# found keeps only (1 - loss) of the conversions, and the losses multiply.
OPTIMIZED_CONVERSION_RATE = 0.10
MIN_ESTIMATED_CONVERSION_RATE = 0.005
# Load time (Portent lead-gen research, rounded): (seconds at or above, loss)
LOAD_TIME_LOSSES = [(7.0, 0.50), (6.0, 0.40), (5.0, 0.30), (4.0, 0.20), (3.0, 0.10)]
NOT_RESPONSIVE_LOSS = 0.35
NO_CLEAR_CTA_LOSS = 0.35
MAIN_ACTION_WRONG_LOSS = 0.35
# Distinct competing goals on the page (all contact channels count as one).
COMPETING_GOAL_LOSSES = {3: 0.20, 4: 0.35, 5: 0.50}
NO_TAP_TO_CALL_LOSS = 0.35
NO_TEXT_US_LOSS = 0.35
NO_BOOKING_LOSS_BOOKING_BUSINESS = 0.50  # appointment- or order-based businesses
NO_BOOKING_LOSS_OTHER = 0.25
NO_CONTACT_FORM_LOSS = 0.25
# Broken calls-to-action: loss = broken / total (each assumed equally used).
# Combined website loss = three-quarters of the sum of the individual losses,
# capped at 95%, so the total climbs gently as issues pile up. It's applied
# to the leads the website currently brings in, and the result is capped at
# WEBSITE_JOBS_CAP jobs a month (shown as "4+").
WEBSITE_COMBINED_LOSS_WEIGHT = 0.75
WEBSITE_COMBINED_LOSS_CAP = 0.95
WEBSITE_JOBS_CAP = 4.0

# --- Online Foundation Leak ---
# Share of additional leads each foundation gap could be keeping away:
#  - Google: complete profiles make customers 70% more likely to visit and
#    50% more likely to consider purchasing.
#  - BrightLocal 2026: 68% only use businesses with 4+ stars, 31% require
#    4.5+; 47% won't use a business with fewer than 20 reviews.
#  - The Google map 3-pack gets ~42% of clicks on local searches.
#  - Whitespark 2026: citations ~6-7% and social ~4% of map ranking weight.
FOUNDATION_LOSSES = {
    "gbp_incomplete": 0.50,
    "rating_below_4": 0.50,
    "rating_below_4_5": 0.25,
    "few_reviews": 0.40,
    # Map ranking for "[Google category] in [City]", in four levels:
    # top 3 = no leak; 4-10 = on the map list but weak; 11-20 = hard to find;
    # not in the top 20 (or no profile) = virtually invisible.
    "map_weak": 0.15,
    "map_hard_to_find": 0.30,
    "map_invisible": 0.40,
    "weak_off_google": 0.20,
    "not_in_ai": 0.15,
    "inconsistent_nap": 0.10,
    "no_social": 0.10,
}
GBP_MIN_PHOTOS = 3
MIN_TRUSTED_REVIEWS = 20
FOUNDATION_COMBINED_LOSS_WEIGHT = 0.50
FOUNDATION_COMBINED_CAP = 0.95
FOUNDATION_JOBS_CAP = 4.0
# Stated assumption: only about half of a local service business's leads
# come through its website (the rest are calls, referrals, repeat customers),
# so a slow site only affects that share.
WEBSITE_LEAD_SHARE = 0.50


def _parse_currency(value: str) -> float:
    match = re.search(r"[\d,]+\.?\d*", value)
    if not match:
        raise ValueError(f"Could not parse a numeric value from avg_job_value: {value!r}")
    return float(match.group(0).replace(",", ""))


JOB_ROUNDING_TOLERANCE = 0.10


def _whole_jobs(raw_jobs: float) -> float:
    """Every estimate is stated in jobs at the business's own average job
    value, rounded DOWN to the nearest half job (0.5 = one job every two
    months), so figures read as floors the owner can sanity check rather
    than falsely precise dollars. A value within JOB_ROUNDING_TOLERANCE of
    the next half job rounds up to it (0.98 -> 1, 1.43 -> 1.5). Any detected
    leak below half a job still counts as half a job."""
    if raw_jobs <= 0:
        return 0.0
    halves = math.floor((raw_jobs + JOB_ROUNDING_TOLERANCE) * 2) / 2
    return max(halves, 0.5)


def calculate_visibility_gap(
    ai_visibility_score: Optional[int],
    local_ranking_score: Optional[int],
    monthly_leads: float,
    close_rate: float,
    avg_job_value: float,
) -> Optional[SupportingLeak]:
    """UPSIDE/OPPORTUNITY, not a loss - invisibility in AI search results and
    local rankings means never entering consideration in the first place,
    not losing a customer you already had. Only calculated when the two
    AI-agent search checks average out to genuinely poor visibility. Checks
    that couldn't run (None) are left out; if neither ran, there's no
    estimate."""
    scores = [score for score in (ai_visibility_score, local_ranking_score) if score is not None]
    if not scores:
        return None
    avg_score = sum(scores) / len(scores)
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
    reputation_score: Optional[int], monthly_leads: float, close_rate: float, avg_job_value: float
) -> Optional[SupportingLeak]:
    """Only calculated when the reputation scan actually ran (not None) and
    its score is genuinely poor."""
    if reputation_score is None or reputation_score >= REPUTATION_GAP_SCORE_THRESHOLD:
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


def calculate_foundation_leak(
    presence: dict,
    ai_visibility_score: Optional[int],
    local_ranking_score: Optional[int],
    reputation_score: Optional[int],
    consistency_score: Optional[int],
    social_score: Optional[int],
    monthly_leads: float,
    close_rate: float,
    avg_job_value: float,
) -> Optional[SupportingLeak]:
    """How many more leads a complete online foundation (Google profile,
    reviews, rankings, reputation, listings, social) could be bringing in.
    Each issue found has an "up to" share of leads; they combine at
    FOUNDATION_COMBINED_LOSS_WEIGHT of the sum, capped, and apply to ALL
    current leads, since the foundation drives how many people find them at
    all. Checks that couldn't run (None) add nothing."""
    presence = presence or {}
    issues = []  # (plain description, share)
    gbp_found = presence.get("gbp_found")

    if gbp_found is False:
        issues.append(("No Google Business Profile found", FOUNDATION_LOSSES["gbp_incomplete"]))
        issues.append(("No Google reviews", FOUNDATION_LOSSES["few_reviews"]))
    elif gbp_found:
        missing = []
        if presence.get("gbp_has_hours") is False:
            missing.append("hours")
        if (presence.get("gbp_photo_count") or 0) < GBP_MIN_PHOTOS:
            missing.append("photos")
        if presence.get("gbp_has_category") is False:
            missing.append("a business category")
        if presence.get("gbp_has_website") is False:
            missing.append("a website link")
        if missing:
            issues.append(
                (f"Google Business Profile incomplete (missing {', '.join(missing)})", FOUNDATION_LOSSES["gbp_incomplete"])
            )
        rating = presence.get("rating")
        if rating is not None and rating < 4.0:
            issues.append((f"Google rating {rating} (below 4.0)", FOUNDATION_LOSSES["rating_below_4"]))
        elif rating is not None and rating < 4.5:
            issues.append((f"Google rating {rating} (below 4.5)", FOUNDATION_LOSSES["rating_below_4_5"]))
        review_count = presence.get("review_count") or 0
        if review_count < MIN_TRUSTED_REVIEWS:
            issues.append((f"Only {review_count} Google reviews (fewer than 20)", FOUNDATION_LOSSES["few_reviews"]))

    if local_ranking_score is not None and local_ranking_score < 9:
        if local_ranking_score >= 4:
            issues.append(("On Google's map for your main search, but weak - outside the top 3", FOUNDATION_LOSSES["map_weak"]))
        elif local_ranking_score >= 1:
            issues.append(("Hard to find on Google's map for your main search (ranked 11-20)", FOUNDATION_LOSSES["map_hard_to_find"]))
        else:
            issues.append(("Virtually invisible on Google's map for your main search", FOUNDATION_LOSSES["map_invisible"]))
    if reputation_score is not None and reputation_score < 6:
        issues.append(("Weak reputation on review sites outside Google", FOUNDATION_LOSSES["weak_off_google"]))
    if ai_visibility_score is not None and ai_visibility_score < 5:
        issues.append(("Not recommended by AI assistants", FOUNDATION_LOSSES["not_in_ai"]))
    if consistency_score is not None and consistency_score < 7:
        issues.append(("Name, address, or phone inconsistent across directories", FOUNDATION_LOSSES["inconsistent_nap"]))
    if social_score is not None and social_score < 4:
        issues.append(("No active social media presence", FOUNDATION_LOSSES["no_social"]))

    if not issues:
        return None

    combined = min(sum(share for _, share in issues) * FOUNDATION_COMBINED_LOSS_WEIGHT, FOUNDATION_COMBINED_CAP)
    jobs = min(_whole_jobs(monthly_leads * combined * close_rate), FOUNDATION_JOBS_CAP)
    if jobs == 0:
        return None
    value = jobs * avg_job_value

    issue_lines = [f"{description} (up to {share:.0%} more leads on its own)" for description, share in issues]
    explanation = (
        "ESTIMATE, not measured data - gaps we found in the online foundation: "
        + "; ".join(issue_lines)
        + f". Together, fixing these could be bringing in up to {combined:.0%} more leads than your "
        f"~{monthly_leads:g} a month today. The percentages come from published consumer and "
        "local-search research (Google, BrightLocal, Whitespark) and our stated estimates. Present "
        "the dollar figure as 'up to'"
        + (", and the job count as '4 or more'." if jobs >= FOUNDATION_JOBS_CAP else ".")
    )
    return SupportingLeak(
        label="Online Foundation Leak",
        monthly_value=round(value, 2),
        explanation=explanation,
        jobs_per_month=jobs,
        issues=[description for description, _ in issues],
    )


def calculate_website_conversion_leak(
    lcp_seconds: Optional[float],
    homepage: Optional[dict],
    assessment: Optional[dict],
    monthly_leads: float,
    close_rate: float,
    avg_job_value: float,
) -> Optional[SupportingLeak]:
    """Estimates how far the site falls short of an optimized site that turns
    OPTIMIZED_CONVERSION_RATE of visitors into leads. Each problem found keeps
    only (1 - loss) of the conversions; the losses multiply. The shortfall is
    applied to the share of leads assumed to come through the website, capped
    at doubling them. Anything that couldn't be checked adds no loss."""
    homepage = homepage or {}
    assessment = assessment or {}
    issues = []  # (plain description, loss)

    if lcp_seconds is not None:
        for threshold, loss in LOAD_TIME_LOSSES:
            if lcp_seconds >= threshold:
                issues.append((f"Slow to load: {lcp_seconds:.1f} seconds on mobile", loss))
                break

    if homepage.get("responsive") is False:
        issues.append(("Not built for phones (no mobile-responsive layout)", NOT_RESPONSIVE_LOSS))

    if assessment.get("clear_cta_on_first_screen") is False:
        issues.append(("No clear call-to-action on the first screen", NO_CLEAR_CTA_LOSS))

    booking_business = bool(assessment.get("booking_or_ordering_business"))
    main_action_wrong = assessment.get("main_action_fits_business") is False
    expected = (assessment.get("expected_main_action") or "").lower()
    if main_action_wrong:
        issues.append(
            (f"Main action for this business ({expected or 'unclear'}) is missing or buried", MAIN_ACTION_WRONG_LOSS)
        )

    goals = assessment.get("distinct_goals")
    if isinstance(goals, int) and goals >= 3:
        loss = COMPETING_GOAL_LOSSES[min(goals, 5)]
        issues.append((f"{goals} competing goals on the page", loss))

    total_ctas, broken_ctas = homepage.get("total_ctas") or 0, homepage.get("broken_ctas") or 0
    if total_ctas and broken_ctas:
        issues.append(
            (f"{broken_ctas} of {total_ctas} call-to-action links don't work", min(broken_ctas / total_ctas, 1.0))
        )

    # When the "main action wrong" factor already covers a missing channel,
    # that channel isn't counted a second time.
    def covered(*words):
        return main_action_wrong and any(word in expected for word in words)

    if homepage.get("tap_to_call") is False and not covered("call"):
        issues.append(("No tap-to-call button", NO_TAP_TO_CALL_LOSS))
    if homepage.get("text_us") is False and not covered("text"):
        issues.append(("No \"Text Us\" option", NO_TEXT_US_LOSS))
    if (
        homepage
        and not homepage.get("online_booking")
        and not homepage.get("online_ordering")
        and not covered("book", "order", "schedule", "reserve")
    ):
        issues.append(
            (
                "No online booking or ordering",
                NO_BOOKING_LOSS_BOOKING_BUSINESS if booking_business else NO_BOOKING_LOSS_OTHER,
            )
        )
    if homepage.get("contact_form") is False and not covered("form", "quote", "contact"):
        issues.append(("No contact form", NO_CONTACT_FORM_LOSS))

    if not issues:
        return None

    combined_loss = min(
        sum(loss for _, loss in issues) * WEBSITE_COMBINED_LOSS_WEIGHT, WEBSITE_COMBINED_LOSS_CAP
    )
    website_leads = monthly_leads * WEBSITE_LEAD_SHARE
    jobs = min(_whole_jobs(website_leads * combined_loss * close_rate), WEBSITE_JOBS_CAP)
    if jobs == 0:
        return None
    value = jobs * avg_job_value

    issue_lines = [f"{description} (up to {loss:.0%} fewer leads on its own)" for description, loss in issues]
    explanation = (
        "ESTIMATE, not measured data - problems we found on the website: "
        + "; ".join(issue_lines)
        + f". Together, these could be costing up to {combined_loss:.0%} of the leads the website "
        f"is currently bringing in (the roughly half of your ~{monthly_leads:g} monthly leads "
        "assumed to come through the website). The load-time figures follow published research; "
        "the other percentages are our stated estimates. Present the dollar figure as 'up to'"
        + (", and the job count as '4 or more'." if jobs >= WEBSITE_JOBS_CAP else ".")
    )
    return SupportingLeak(
        label="Website Performance Leak",
        monthly_value=round(value, 2),
        explanation=explanation,
        jobs_per_month=jobs,
        issues=[description for description, _ in issues],
    )


def calculate_leak_estimate(
    scan_request: ScanRequest,
    review_data: dict,
    ai_visibility_score: Optional[int],
    local_ranking_score: Optional[int],
    reputation_score: Optional[int],
    lcp_seconds: Optional[float],
    homepage_signals: Optional[dict] = None,
    cta_assessment: Optional[dict] = None,
    consistency_score: Optional[int] = None,
    social_score: Optional[int] = None,
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
        "ESTIMATE, not measured data - assumes about 8% of your self-reported dormant leads "
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
    # The old separate Visibility Gap and Reputation Gap estimates are folded
    # into the Online Foundation Leak (their functions remain for reference).
    foundation_leak = calculate_foundation_leak(
        review_data,
        ai_visibility_score,
        local_ranking_score,
        reputation_score,
        consistency_score,
        social_score,
        monthly_leads_numeric,
        close_rate_numeric,
        avg_job_value_numeric,
    )
    if foundation_leak is not None:
        supporting_leaks.append(foundation_leak)

    website_leak = calculate_website_conversion_leak(
        lcp_seconds,
        homepage_signals,
        cta_assessment,
        monthly_leads_numeric,
        close_rate_numeric,
        avg_job_value_numeric,
    )
    if website_leak is not None:
        supporting_leaks.append(website_leak)

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
