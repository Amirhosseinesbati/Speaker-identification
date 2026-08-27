"""Interactively store the Telegram token on a campaign worker.

The token is read with terminal echo disabled and is never printed.
"""

from __future__ import annotations

import getpass
import os
from pathlib import Path


SECRET_PATH = Path("/root/.iaaa_campaign.env")


def main() -> None:
    token = getpass.getpass("Telegram bot token: ").strip()
    if not token or ":" not in token:
        raise SystemExit("Token format is invalid; nothing was written.")
    old_umask = os.umask(0o077)
    try:
        SECRET_PATH.write_text(f"TELEGRAM_BOT_TOKEN={token}\n", encoding="utf-8")
        SECRET_PATH.chmod(0o600)
    finally:
        os.umask(old_umask)
    print(f"Secret stored at {SECRET_PATH} with mode 0600.")


if __name__ == "__main__":
    main()
