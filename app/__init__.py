"""Apple Store Review Analysis API."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from a ``.env`` file at the project root, if one
# exists. Done at package import time so it runs *before* any module that
# reads env vars at import time (e.g. ``app.llm``, which captures the model
# names from ``os.environ`` at the top of the file).
#
# Precedence: real non-empty shell exports win over ``.env`` entries. But we
# treat **empty-string** ``ANTHROPIC_*`` env vars as "not set" first — those
# almost always come from accidental unset-variable expansion in a shell,
# not from intentional configuration, and they would otherwise silently
# block the .env file from populating the same key.
for _k in [k for k in os.environ if k.startswith("ANTHROPIC_") and not os.environ[k]]:
    del os.environ[_k]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# Logging must be configured before any module imports ``logging.getLogger``
# so all subsequent getLogger calls inherit the handlers set up here.
from .logging_config import setup_logging  # noqa: E402

setup_logging()

__version__ = "0.3.0"
