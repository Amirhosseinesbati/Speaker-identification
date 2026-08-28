import json

import pytest

from scripts import telegram_notifier


def test_send_returns_verified_message_id(monkeypatch, tmp_path, capsys):
    chat_file = tmp_path / "chat_id"
    chat_file.write_text("12345", encoding="ascii")
    monkeypatch.setattr(telegram_notifier, "CHAT_FILE", chat_file)
    monkeypatch.setattr(telegram_notifier, "_load_token", lambda: "token:value")

    observed = {}

    def fake_api(token, method, payload=None):
        observed.update(token=token, method=method, payload=payload)
        return {"ok": True, "result": {"message_id": 46}}

    monkeypatch.setattr(telegram_notifier, "_api", fake_api)

    assert telegram_notifier.send("گزارش علمی") == 46
    assert observed == {
        "token": "token:value",
        "method": "sendMessage",
        "payload": {"chat_id": 12345, "text": "گزارش علمی"},
    }
    assert json.loads(capsys.readouterr().out) == {
        "status": "delivered",
        "message_id": 46,
    }


def test_send_rejects_success_without_message_id(monkeypatch, tmp_path):
    chat_file = tmp_path / "chat_id"
    chat_file.write_text("12345", encoding="ascii")
    monkeypatch.setattr(telegram_notifier, "CHAT_FILE", chat_file)
    monkeypatch.setattr(telegram_notifier, "_load_token", lambda: "token:value")
    monkeypatch.setattr(
        telegram_notifier,
        "_api",
        lambda token, method, payload=None: {"ok": True, "result": {}},
    )

    with pytest.raises(RuntimeError, match="message_id"):
        telegram_notifier.send("گزارش علمی")
