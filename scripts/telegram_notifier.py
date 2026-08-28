"""Small allowlisted Telegram notifier for the Vast.ai campaign worker."""

from __future__ import annotations

import argparse
import json
import os
import stat
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


SECRET_FILE = Path("/root/.iaaa_campaign.env")
CHAT_FILE = Path("/root/.iaaa_telegram_chat_id")


def _load_token() -> str:
    mode = stat.S_IMODE(SECRET_FILE.stat().st_mode)
    if mode & 0o077:
        raise RuntimeError(f"secret file permissions are too broad: {mode:o}")
    for line in SECRET_FILE.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key == "TELEGRAM_BOT_TOKEN":
            token = value.strip()
            if token and ":" in token:
                return token
    raise RuntimeError("TELEGRAM_BOT_TOKEN is missing or malformed")


def _api(token: str, method: str, payload: dict | None = None) -> dict:
    data = None if payload is None else urllib.parse.urlencode(payload).encode()
    request = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/{method}", data=data
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        # Never propagate urllib's exception text: it may contain the request
        # URL and therefore the bot token embedded in that URL.
        raise RuntimeError(
            f"Telegram {method} transport failed ({type(exc).__name__})"
        ) from None
    if not result.get("ok"):
        raise RuntimeError(f"Telegram {method} failed")
    return result


def _discover_private_chat(token: str) -> int:
    updates = _api(token, "getUpdates", {"timeout": 0, "limit": 100})["result"]
    candidates = []
    for update in updates:
        message = update.get("message") or update.get("edited_message") or {}
        chat = message.get("chat") or {}
        if chat.get("type") == "private" and isinstance(chat.get("id"), int):
            candidates.append((update.get("update_id", -1), chat["id"]))
    if not candidates:
        raise RuntimeError("no private Telegram chat found; send /start again")
    chat_id = max(candidates)[1]
    old_umask = os.umask(0o077)
    try:
        CHAT_FILE.write_text(str(chat_id), encoding="ascii")
        CHAT_FILE.chmod(0o600)
    finally:
        os.umask(old_umask)
    return chat_id


def send(message: str, discover: bool = False) -> int:
    """Send one message and return the verified Telegram ``message_id``.

    A successful HTTP response alone is not enough for campaign observability:
    the marker written by the heartbeat needs the server-assigned message id so
    retries can be distinguished from duplicate notifications.
    """
    token = _load_token()
    if discover or not CHAT_FILE.exists():
        chat_id = _discover_private_chat(token)
    else:
        chat_id = int(CHAT_FILE.read_text(encoding="ascii").strip())
    response = _api(token, "sendMessage", {"chat_id": chat_id, "text": message})
    message_id = (response.get("result") or {}).get("message_id")
    if not isinstance(message_id, int):
        raise RuntimeError("Telegram sendMessage response has no integer message_id")
    print(json.dumps({"status": "delivered", "message_id": message_id}))
    return message_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("message")
    parser.add_argument("--discover", action="store_true")
    args = parser.parse_args()
    send(args.message, discover=args.discover)


if __name__ == "__main__":
    main()
