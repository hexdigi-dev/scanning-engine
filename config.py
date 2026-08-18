import os

from dotenv import load_dotenv

load_dotenv()

REQUIRED_ENV_VARS = ["GOOGLE_API_KEY", "ANTHROPIC_API_KEY"]

missing = [name for name in REQUIRED_ENV_VARS if not os.getenv(name)]
if missing:
    raise EnvironmentError(
        f"Missing required environment variable(s): {', '.join(missing)}"
    )

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
