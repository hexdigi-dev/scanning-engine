import os

from dotenv import load_dotenv

load_dotenv()

REQUIRED_ENV_VARS = ["GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "SCAN_API_KEY"]

missing = [name for name in REQUIRED_ENV_VARS if not os.getenv(name)]
if missing:
    raise EnvironmentError(
        f"Missing required environment variable(s): {', '.join(missing)}"
    )

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
SCAN_API_KEY = os.getenv("SCAN_API_KEY")

# Optional - the app still runs without these, with reduced features.
# DATABASE_URL: Postgres connection string (Railway provides this when you add
#   a Postgres database). Without it, reports are kept in memory only and are
#   lost on every redeploy - fine for local testing, not for production.
DATABASE_URL = os.getenv("DATABASE_URL")
# MAKE_WEBHOOK_URL: the Make.com webhook the landing-page form forwards to.
MAKE_WEBHOOK_URL = os.getenv("MAKE_WEBHOOK_URL")
# BOOKING_URL: where the report page's call-to-action button points.
BOOKING_URL = os.getenv("BOOKING_URL")
# PUBLIC_BASE_URL: overrides the domain used to build report links. If unset,
# Railway's own RAILWAY_PUBLIC_DOMAIN is used.
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL")
RAILWAY_PUBLIC_DOMAIN = os.getenv("RAILWAY_PUBLIC_DOMAIN")
