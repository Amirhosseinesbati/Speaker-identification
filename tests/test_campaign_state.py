from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from src.campaign_state import CampaignStateError, CampaignStore


def _store(tmp_path):
    return CampaignStore(tmp_path / "state.json", tmp_path / "events.jsonl")


def _initialize(store, **overrides):
    params = {
        "campaign_id": "test-campaign",
        "instance_id": "123",
        "hourly_rate_usd": 0.20,
        "max_campaign_cost_usd": 10.0,
        "max_run_hours": 4.0,
        "waiting_keepalive_hours": 6.0,
        "git_commit": "abc123",
    }
    params.update(overrides)
    return store.initialize(**params)


def test_state_transitions_are_atomic_and_evented(tmp_path):
    store = _store(tmp_path)
    state = _initialize(store)
    assert state["status"] == "PREFLIGHT"

    state = store.transition("READY", "preflight passed")
    assert state["revision"] == 1
    state = store.start_run({"profile": "p0", "git_commit": "abc123"})
    assert state["status"] == "RUNNING_EXPERIMENT"
    state = store.finish_run(
        success=True, exit_code=0, reason="run completed", artifacts=[],
    )
    assert state["status"] == "ANALYZING"
    assert len(state["completed_runs"]) == 1

    events = [json.loads(line) for line in store.event_path.read_text().splitlines()]
    assert [event["event_type"] for event in events] == [
        "INITIALIZED", "TRANSITION", "TRANSITION", "TRANSITION",
    ]


def test_invalid_transition_is_rejected(tmp_path):
    store = _store(tmp_path)
    _initialize(store)
    with pytest.raises(CampaignStateError, match="invalid campaign transition"):
        store.transition("WAITING_FOR_LEADERBOARD", "not ready")


def test_budget_guard_reserves_run_cost(tmp_path):
    store = _store(tmp_path)
    _initialize(
        store,
        hourly_rate_usd=1.0,
        max_campaign_cost_usd=2.0,
        billing_started_at=(
            datetime.now(timezone.utc) - timedelta(hours=1.5)
        ).isoformat(),
    )
    with pytest.raises(CampaignStateError, match="budget guard"):
        store.assert_budget(reserve_hours=1.0)


def test_leaderboard_result_requires_waiting_state(tmp_path):
    store = _store(tmp_path)
    _initialize(store)
    with pytest.raises(CampaignStateError, match="only while waiting"):
        store.record_leaderboard(0.973)
