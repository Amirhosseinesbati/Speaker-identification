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


def test_max_run_hours_policy_update_is_atomic_evented_and_idempotent(tmp_path):
    store = _store(tmp_path)
    initial = _initialize(store)
    assert initial["policy"]["max_run_hours"] == 4.0

    updated = store.set_max_run_hours(15.0, "locked treatment requires 15 hours")
    assert updated["policy"]["max_run_hours"] == 15.0
    assert updated["revision"] == 1
    events = [json.loads(line) for line in store.event_path.read_text().splitlines()]
    assert events[-1]["event_type"] == "POLICY_UPDATED"
    assert events[-1]["metadata"] == {
        "field": "max_run_hours",
        "previous": 4.0,
        "updated": 15.0,
    }

    repeated = store.set_max_run_hours(15.0, "safe retry")
    assert repeated["revision"] == 1
    assert len(store.event_path.read_text().splitlines()) == len(events)


def test_max_run_hours_policy_update_rejects_active_or_invalid_change(tmp_path):
    store = _store(tmp_path)
    _initialize(store)
    with pytest.raises(CampaignStateError, match="positive"):
        store.set_max_run_hours(0.0, "invalid")
    with pytest.raises(CampaignStateError, match="reason"):
        store.set_max_run_hours(5.0, "")

    store.transition("READY", "preflight passed")
    store.start_run({"profile": "p0", "git_commit": "abc123"})
    with pytest.raises(CampaignStateError, match="no experiment is active"):
        store.set_max_run_hours(15.0, "must not mutate an active run")


def test_campaign_cost_policy_update_is_atomic_evented_and_idempotent(tmp_path):
    store = _store(tmp_path)
    initial = _initialize(store)
    assert initial["policy"]["max_campaign_cost_usd"] == 10.0

    updated = store.set_max_campaign_cost(50.0, "user raised campaign budget")
    assert updated["policy"]["max_campaign_cost_usd"] == 50.0
    assert updated["revision"] == 1
    events = [json.loads(line) for line in store.event_path.read_text().splitlines()]
    assert events[-1]["event_type"] == "POLICY_UPDATED"
    assert events[-1]["metadata"] == {
        "field": "max_campaign_cost_usd",
        "previous": 10.0,
        "updated": 50.0,
    }

    repeated = store.set_max_campaign_cost(50.0, "safe retry")
    assert repeated["revision"] == 1
    assert len(store.event_path.read_text().splitlines()) == len(events)


def test_campaign_cost_policy_rejects_active_invalid_or_spent_change(tmp_path):
    store = _store(tmp_path)
    _initialize(store)
    with pytest.raises(CampaignStateError, match="positive"):
        store.set_max_campaign_cost(0.0, "invalid")
    with pytest.raises(CampaignStateError, match="reason"):
        store.set_max_campaign_cost(20.0, "")

    state = store.load()
    state["billing_started_at_utc"] = (
        datetime.now(timezone.utc) - timedelta(hours=60.0)
    ).isoformat()
    store._persist(state)
    with pytest.raises(CampaignStateError, match="already estimated spend"):
        store.set_max_campaign_cost(5.0, "cannot undercut spend")

    state = store.load()
    state["billing_started_at_utc"] = datetime.now(timezone.utc).isoformat()
    store._persist(state)
    store.transition("READY", "preflight passed")
    store.start_run({"profile": "p0", "git_commit": "abc123"})
    with pytest.raises(CampaignStateError, match="no experiment is active"):
        store.set_max_campaign_cost(50.0, "must not mutate active run")
