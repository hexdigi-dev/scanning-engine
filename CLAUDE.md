# Pheonix Digital — Free Foundation Scan engine

Briefing for Claude Code. Read this before changing anything.

## Who this is for

Owner: Vincent Dyer, Pheonix Digital (AI-first marketing agency for American local service
businesses — plumbers, HVAC, trades). Goal: $1M/month, with automation maximized everywhere.
Vincent approves copy, prices, new campaigns and fixes to major issues; Claude does the rest.

How Vincent wants to work:
- Fully honest, straightforward advice. No flattery; say "good idea" only when it is one.
- Don't guess. When an estimate or guess is unavoidable, give the source(s) and a % confidence.
- Always look for what can be automated and recommend how.
- Terminal commands: each on its own separate copyable line/block.
- Downloaded files land on his Desktop, not Downloads.
- Keep changes staged while iterating; deploy once things are ironed out (ask if unsure).

## What the engine does

A prospect fills in the landing page form; the engine scans their business online, estimates
what each gap costs them per month, builds a branded report, and Make emails it to them.

Pipeline:
1. Landing page `/` (Railway, `templates/landing.html`) → posts the form to the Make webhook.
2. Make scenario **"Foundation Scan"** (id 5995516, team "My Team" 2744935, zone us2.make.com):
   Webhooks 2 → HTTP 4 (POST /scan, X-API-Key header, timeout 300s) → Iterator 7 → Text
   aggregator 9 (all checks) → Iterator 10 → Text aggregator 11 (leaks) → Gmail 6 (internal, to
   hexdigi@gmail.com) → filter "Ok to send to client" (`hold_client_email = false`) → Gmail 12
   (client email).
   - New form fields must be added to HTTP 4's JSON body as `"field": "{{2.field}}"`.
3. `POST /scan` (`main.py`) runs all checks, calculates leaks, polishes text with Claude, saves
   the report to Postgres, returns JSON (incl. `report_url`, `checks_run`, `leaks_display`,
   `opportunities_display`, `internal_alerts`, `hold_client_email`).
4. Report page `/report/<id>` (`pages.py` + `templates/report.html`, styles in `static/brand.css`).

## Stack and deploy

- FastAPI, Python 3.9 (keep code 3.9-compatible: `Optional[...]`, no `X | Y` types, no match).
- GitHub: hexdigi-dev/scanning-engine. Local: /Users/vincentdyer/Desktop/scanning-engine.
- Railway: scanning-engine-production.up.railway.app, with Postgres.
- Deploy: commit + push to `main`, then in Railway: Settings → "Check for updates" (auto-deploy
  is unreliable). Verify with Deploy Logs.
- Env vars (Railway → Variables): ANTHROPIC_API_KEY (Console credits, prepaid — auto-reload on),
  GOOGLE_API_KEY (Google Cloud project "Pheonix Scanning Engine": PageSpeed Insights, Places API,
  Places API (New)), GOOGLE_AI_API_KEY (Gemini, "Default Gemini Project", prepaid balance),
  OPENAI_API_KEY, XAI_API_KEY, SCAN_API_KEY, DATABASE_URL, optional JINA_API_KEY, BOOKING_URL
  (not set yet — the report has no "Book a call" button until it is).
- Before pushing: `python3 -m py_compile` every changed file, and import `main` and `pages`
  and load both templates to catch errors.

## Useful log lines (Railway → Deploy Logs, search the word)

`[google]` profile + map-pack searches · `[speed]` the 3 speed tests · `[website]` extra pages,
ad tags, chat widgets, contact page, Claude's notes · `[ai_visibility]` model used per assistant
· `[agent]` failed AI checks · `[synthesis]` report polishing failures · `[scan] alerts`.

## Checks (23 total; `checks_run` is computed, keep the report list in sync)

Online foundation (7): Google Business Profile, Google Reviews, Online Reputation (other review
sites), Social Media, AI Search Visibility, Local Search Ranking, Consistency (NAP).
Website (13): load speed (median of 3 PageSpeed runs, scored /10), mobile layout, clear CTA on
first screen, right main action, focused page, working CTA links, tap-to-call, Text Us, online
booking/ordering, chat widget, contact form, Website Health (PageSpeed perf), Accessibility.
Paid advertising (3 findings in one check): ads running, ad tags (tracking/retargeting, incl.
inside Google Tag Manager), competitor ads.

Rules that must hold:
- Google profile lookup searches exactly "[Business Name] [City]" via Places API (New) with
  service-area businesses included; confirm by website, phone or address.
- Map ranking searches "[Google category] in [City]": top 3 = 10, 4–10 = 5 ("there but weak"),
  11–20 = 1 ("hard to find"), not in top 20 = 0 ("virtually invisible"). Report shows both
  searches and names the top 3 competitors.
- AI visibility asks Claude, ChatGPT, Grok and Gemini, each with live web search, newest models
  first with fallbacks.
- A check that couldn't run is "Not checked": excluded from scores and leak totals, never shown
  as 0/10, never shows raw errors to the client.
- Google's API returns "most relevant" reviews, not newest — don't infer review recency from it.

## Estimates (underpromise: when in doubt, estimate lower)

- All leaks are in jobs at the client's average job value, rounded down to half jobs (up if
  within 0.1), shown as "up to". Any detected leak counts as at least half a job.
- Foundation and Website leaks = the leak total in the hero. Slow lead response, after-hours
  coverage and dormant leads = "other potential opportunities" (second figure, not in the total).
- After-hours: 15% of leads assumed to arrive after hours (sources range ~10–30%), only the extra
  loss beyond their stated response time (no double counting).
- Overall section scores = average of checks that ran, rounded down. Website rows: pass = 10,
  needs work = 0, load speed uses its /10 score. Paid advertising has a rating (Active / Needs
  work / Need more information), not a score.
- Every dollar estimate and overall score uses the brand gradient (yellow-orange → orange-red).

## Pricing shown on the report (all "Starting at"; prices end in 7)

- Foundation Fix: ~~$1,497~~ $997 + $100/mo when started on the follow-up call.
- New Website: ~~$1,497~~ $997 + $150/mo when started on the follow-up call (one CTA, contact
  form, booking form, local SEO included; anything more is an add-on).
- AI Search Registration (AEO/GEO) add-on: $497 + $100/mo. SEO: $497 + $100/mo, included with sites.
- Reputation Builder (= Review Bot): $497 + $100/mo.
- Speed-to-Lead Bot (texts + emails every new lead with a booking link): $497 + $200/mo.
- Out-of-Hours AI Bot (after-hours answering, books appointments/call-backs): $497 + $200/mo.
- Pheonix Marketing Audit: Baby Pheonix $997 (≤2 ad channels), Full Pheonix $1,997 (3+).
- Lead Revival: "Book a call for pricing" (performance-based).
Never imply bad reviews can be removed.

## Open items

- Set BOOKING_URL (Calendly) in Railway — highest value.
- Emails land in spam: plan is sending from reports@pheonix.digital (Postmark + SPF/DKIM/DMARC)
  and serving reports from a pheonix.digital subdomain.
- Make the GitHub repo private.
- Add SerpApi for measured competitor ads / Local Services Ads once scanning at volume.
- Upgrade both Google Cloud free trials to paid before they expire.
