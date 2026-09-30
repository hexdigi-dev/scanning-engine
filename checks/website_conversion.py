"""Signals for the Website Conversion Leak.

Two parts:
1. check_homepage_signals: reads the homepage code for a mobile (responsive)
   layout, tap-to-call, a "Text Us" option, a contact form, online booking or
   ordering, and calls-to-action whose links are broken.
2. assess_calls_to_action: shows Claude the mobile screenshot Google's speed
   test already takes, plus the business type, and asks whether a clear
   call-to-action is visible on the first screen, whether the main action fits
   the business, how many competing goals the page pushes, and whether the
   business is booking/ordering-based.

Anything that can't be checked is returned as None and adds no loss.
"""
import asyncio
import json
import re
from html.parser import HTMLParser
from typing import List, Optional
from urllib.parse import urljoin, urlparse

import httpx

from config import ANTHROPIC_API_KEY

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-sonnet-5"
PAGE_TIMEOUT = 15.0
LINK_CHECK_TIMEOUT = 8.0
MAX_LINKS_TO_CHECK = 8
ASSESS_TIMEOUT = 45.0
USER_AGENT = "Mozilla/5.0 (compatible; FoundationScan/1.0)"

BOOKING_MARKERS = (
    "calendly.com", "acuityscheduling", "housecallpro", "servicetitan", "jobber",
    "setmore", "squareup.com/appointments", "booksy", "schedulicity", "vagaro",
    "book online", "book now", "schedule online", "schedule service",
    "schedule an appointment", "request an appointment", "book an appointment",
)
ORDERING_MARKERS = (
    "doordash.com", "ubereats.com", "grubhub.com", "toasttab.com", "chownow.com",
    "order.online", "slicelife.com", "square.site", "clover.com/online-ordering",
    "order online", "order now", "start your order", "online ordering",
)
TEXT_US_MARKERS = (
    "podium.com", "birdeye.com", "leadconnectorhq", "msgsndr", "textus", "hatchapp",
    "text us", "text now", "send us a text",
)
CONTACT_FORM_MARKERS = ("<form", "wpforms", "gform_", "contact-form", "hsforms", "jotform", "typeform")
# Links worth following to find a contact form or booking tool that isn't
# on the homepage itself (e.g. a separate Contact page).
SECONDARY_PAGE_WORDS = ("contact", "quote", "estimate", "schedule", "book", "appointment", "request")
MAX_SECONDARY_PAGES = 2
# Homepages smaller than this are usually a bot-block or redirect page.
MIN_REAL_PAGE_BYTES = 3000
BLOCKED_PAGE_MARKERS = ("captcha", "cf-browser-verification", "access denied", "are you a robot")
VIEWPORT_TAG = re.compile(r"<meta\b[^>]*\bviewport\b[^>]*>", re.I)

CTA_WORDS = (
    "call", "text", "book", "schedule", "order", "quote", "estimate", "contact",
    "request", "get started", "appointment", "reserve",
)


class _LinkCollector(HTMLParser):
    """Collects (href, visible text) for every link on the page."""

    def __init__(self):
        super().__init__()
        self.links: List[tuple] = []
        self._href: Optional[str] = None
        self._text: List[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self._href = dict(attrs).get("href") or ""
            self._text = []

    def handle_data(self, data):
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._href is not None:
            self.links.append((self._href.strip(), " ".join("".join(self._text).split())))
            self._href = None


def _is_cta(href: str, text: str) -> bool:
    lowered_href, lowered_text = href.lower(), text.lower()
    if lowered_href.startswith(("tel:", "sms:")):
        return True
    if any(marker in lowered_href for marker in BOOKING_MARKERS + ORDERING_MARKERS):
        return True
    return any(word in lowered_text for word in CTA_WORDS) and len(lowered_text) <= 40


async def _link_works(client: httpx.AsyncClient, url: str) -> bool:
    try:
        response = await client.get(url, headers={"User-Agent": USER_AGENT})
        return response.status_code < 400
    except Exception:
        return False


async def check_homepage_signals(website_url: str) -> dict:
    """Returns a dict of booleans/counts (see keys below), or {} if the
    homepage couldn't be fetched - in which case nothing is counted against
    the business. Keys: responsive, tap_to_call, text_us, contact_form,
    online_booking, online_ordering, cta_labels, total_ctas, broken_ctas."""
    try:
        async with httpx.AsyncClient(timeout=PAGE_TIMEOUT, follow_redirects=True) as client:
            response = await client.get(website_url, headers={"User-Agent": USER_AGENT})
        if response.status_code != 200:
            return {}
        raw_html = response.text
        base_url = str(response.url)
    except Exception as exc:
        print(f"[website] could not fetch {website_url}: {type(exc).__name__}: {exc}")
        return {}

    html = raw_html.lower()
    if len(html) < MIN_REAL_PAGE_BYTES or any(marker in html[:5000] for marker in BLOCKED_PAGE_MARKERS):
        print(f"[website] {website_url} looks blocked or empty ({len(html)} bytes) - website signals not checked")
        return {}
    collector = _LinkCollector()
    try:
        collector.feed(raw_html)
    except Exception:
        pass

    ctas = []
    seen = set()
    for href, text in collector.links:
        if not href or not _is_cta(href, text):
            continue
        key = href.lower()
        if key in seen:
            continue  # the same button repeated down the page counts once
        seen.add(key)
        ctas.append((href, text))

    # A phone link is broken if it doesn't hold a full number; a web link is
    # broken if it errors when opened.
    broken = 0
    web_ctas = []
    for href, _ in ctas:
        lowered = href.lower()
        if lowered.startswith(("tel:", "sms:")):
            digits = re.sub(r"\D", "", href)
            if len(digits) < 10:
                broken += 1
        elif lowered.startswith(("http://", "https://", "/")):
            web_ctas.append(urljoin(base_url, href))
    web_ctas = web_ctas[:MAX_LINKS_TO_CHECK]
    if web_ctas:
        async with httpx.AsyncClient(timeout=LINK_CHECK_TIMEOUT, follow_redirects=True) as client:
            results = await asyncio.gather(*(_link_works(client, url) for url in web_ctas))
        broken += sum(1 for ok in results if not ok)
    checked_ctas = sum(1 for href, _ in ctas if href.lower().startswith(("tel:", "sms:"))) + len(web_ctas)

    # Contact forms and booking tools often live on a separate page (Contact,
    # Get a Quote, Book), so read up to MAX_SECONDARY_PAGES of those too.
    site_host = urlparse(base_url).hostname
    secondary_urls = []
    for href, text in collector.links:
        target = urljoin(base_url, href)
        if urlparse(target).hostname != site_host or target in secondary_urls:
            continue
        label = f"{href} {text}".lower()
        if any(word in label for word in SECONDARY_PAGE_WORDS):
            secondary_urls.append(target)
        if len(secondary_urls) >= MAX_SECONDARY_PAGES:
            break
    extra_html = ""
    if secondary_urls:
        async with httpx.AsyncClient(timeout=LINK_CHECK_TIMEOUT, follow_redirects=True) as client:
            pages = await asyncio.gather(
                *(client.get(url, headers={"User-Agent": USER_AGENT}) for url in secondary_urls),
                return_exceptions=True,
            )
        extra_html = " ".join(
            page.text.lower() for page in pages if isinstance(page, httpx.Response) and page.status_code == 200
        )
    all_html = html + " " + extra_html

    viewport_tags = VIEWPORT_TAG.findall(raw_html)
    responsive = any("device-width" in tag.lower() or "initial-scale" in tag.lower() for tag in viewport_tags)

    return {
        "responsive": responsive,
        "tap_to_call": "href=\"tel:" in html or "href='tel:" in html or "href=tel:" in html,
        "text_us": "href=\"sms:" in all_html or "href='sms:" in all_html or any(m in all_html for m in TEXT_US_MARKERS),
        "contact_form": any(marker in all_html for marker in CONTACT_FORM_MARKERS),
        "online_booking": any(marker in all_html for marker in BOOKING_MARKERS),
        "online_ordering": any(marker in all_html for marker in ORDERING_MARKERS),
        "cta_labels": [text or href for href, text in ctas][:15],
        "total_ctas": checked_ctas,
        "broken_ctas": broken,
    }


ASSESS_TOOL = {
    "name": "submit_assessment",
    "description": "Submit the call-to-action assessment of this business's homepage.",
    "input_schema": {
        "type": "object",
        "properties": {
            "booking_or_ordering_business": {
                "type": "boolean",
                "description": "True if customers normally book an appointment or place an order "
                "(salon, cleaning, dental, restaurant, maintenance visits); false for urgent or "
                "talk-it-through trades (emergency plumbing, restoration, contractors, B2B equipment).",
            },
            "expected_main_action": {
                "type": "string",
                "description": "The single action this kind of business most needs visitors to take, "
                "e.g. 'tap to call', 'book online', 'order online', 'request a quote'.",
            },
            "clear_cta_on_first_screen": {
                "type": ["boolean", "null"],
                "description": "Is an obvious, tappable call-to-action visible in the screenshot "
                "without scrolling? null if there is no screenshot.",
            },
            "main_action_fits_business": {
                "type": ["boolean", "null"],
                "description": "Is expected_main_action offered prominently (on the first screen or "
                "in the list of calls-to-action found)? false if it's missing or buried.",
            },
            "distinct_goals": {
                "type": "integer",
                "description": "How many DIFFERENT goals the page asks visitors to pursue. Call, text, "
                "book, order, quote and contact all count together as ONE 'contact us' goal; other "
                "goals include newsletter signup, shopping merchandise, job applications, following "
                "on social media, downloading something.",
            },
            "notes": {"type": "string", "description": "One or two plain sentences on what you saw."},
        },
        "required": [
            "booking_or_ordering_business",
            "expected_main_action",
            "clear_cta_on_first_screen",
            "main_action_fits_business",
            "distinct_goals",
            "notes",
        ],
    },
}


async def assess_calls_to_action(
    business_name: str,
    business_type: str,
    screenshot_data_uri: Optional[str],
    homepage_signals: dict,
) -> dict:
    """Returns the submit_assessment fields, or {} if the assessment failed
    (nothing is then counted against the business)."""
    content = []
    if screenshot_data_uri and screenshot_data_uri.startswith("data:image/"):
        media_type, _, data = screenshot_data_uri[5:].partition(";base64,")
        if data:
            content.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}})
    content.append(
        {
            "type": "text",
            "text": (
                f"Business: {business_name}\nWhat they do: {business_type}\n"
                f"Calls-to-action found in the homepage code: {json.dumps(homepage_signals.get('cta_labels', []))}\n"
                f"Online booking found: {homepage_signals.get('online_booking')}; "
                f"online ordering found: {homepage_signals.get('online_ordering')}; "
                f"tap-to-call found: {homepage_signals.get('tap_to_call')}\n\n"
                + (
                    "The image is the first screen of their homepage on a phone. "
                    if len(content) > 1
                    else "No screenshot is available, so answer null for clear_cta_on_first_screen. "
                )
                + "Assess it as a customer looking for this kind of business would, and submit "
                "your assessment with the submit_assessment tool."
            ),
        }
    )
    try:
        async with httpx.AsyncClient(timeout=ASSESS_TIMEOUT) as client:
            response = await client.post(
                ANTHROPIC_URL,
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": ANTHROPIC_MODEL,
                    "max_tokens": 800,
                    "tools": [ASSESS_TOOL],
                    "tool_choice": {"type": "tool", "name": ASSESS_TOOL["name"]},
                    "messages": [{"role": "user", "content": content}],
                },
            )
        if response.status_code != 200:
            print(f"[website] CTA assessment failed: HTTP {response.status_code}: {response.text[:300]}")
            return {}
        for block in response.json().get("content", []):
            if block.get("type") == "tool_use" and isinstance(block.get("input"), dict):
                return block["input"]
    except Exception as exc:
        print(f"[website] CTA assessment failed: {type(exc).__name__}: {exc}")
    return {}
