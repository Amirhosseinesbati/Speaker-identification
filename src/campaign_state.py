"""Durable state machine and budget accounting for remote GPU campaigns.

Runtime state is intentionally kept under ``data/experiments`` (gitignored),
while every scientific input remains a committed config/profile.  Each state
transition is written atomically and also appended to an event log so a worker
restart never turns an interrupted run into an apparently successful one.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


SCHEMA_VERSION = 1

ALLOWED_TRANSITIONS = {
    "CREATED": {"BOOTSTRAPPING", "FAILED"},
    "BOOTSTRAPPING": {"PREFLIGHT", "FAILED"},
    "PREFLIGHT": {"READY", "CAMPAIGN_BLOCKED", "FAILED"},
    "READY": {"RUNNING_EXPERIMENT", "PAUSED", "FAILED"},
    "RUNNING_EXPERIMENT": {
        "ANALYZING", "CAMPAIGN_BLOCKED", "STOPPED_FOR_BUDGET", "FAILED",
    },
    "ANALYZING": {
        "READY", "RUNNING_EXPERIMENT", "WAITING_FOR_LEADERBOARD",
        "CAMPAIGN_COMPLETE", "CAMPAIGN_BLOCKED", "FAILED",
    },
    "WAITING_FOR_LEADERBOARD": {
        "ANALYZING", "PAUSED", "STOPPED_FOR_BUDGET", "FAILED",
    },
    "CAMPAIGN_BLOCKED": {"PREFLIGHT", "READY", "FAILED"},
    "PAUSED": {"READY", "STOPPED_FOR_BUDGET", "FAILED"},
    "STOPPED_FOR_BUDGET": set(),
    "CAMPAIGN_COMPLETE": set(),
    "FAILED": set(),
}


class CampaignStateError(RuntimeError):
    """Raised when campaign state or a requested transition is invalid."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class CampaignStore:
    """Atomic campaign state plus an append-only JSONL event ledger."""

    def __init__(self, state_path: Path, event_path: Optional[Path] = None):
        self.state_path = Path(state_path)
        self.event_path = Path(event_path or self.state_path.with_suffix(".events.jsonl"))

    def exists(self) -> bool:
        return self.state_path.is_file()

    def load(self) -> dict[str, Any]:
        if not self.exists():
            raise CampaignStateError(f"campaign state does not exist: {self.state_path}")
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        if state.get("schema_version") != SCHEMA_VERSION:
            raise CampaignStateError(
                f"unsupported campaign schema: {state.get('schema_version')}"
            )
        if state.get("status") not in ALLOWED_TRANSITIONS:
            raise CampaignStateError(f"unknown campaign status: {state.get('status')}")
        return state

    def initialize(
        self,
        *,
        campaign_id: str,
        instance_id: str,
        hourly_rate_usd: float,
        max_campaign_cost_usd: float,
        max_run_hours: float,
        waiting_keepalive_hours: float,
        billing_started_at: Optional[str] = None,
        git_commit: str = "",
    ) -> dict[str, Any]:
        if self.exists():
            raise CampaignStateError(
                f"refusing to overwrite existing campaign state: {self.state_path}"
            )
        if hourly_rate_usd <= 0 or max_campaign_cost_usd <= 0:
            raise CampaignStateError("hourly rate and campaign budget must be positive")
        if max_run_hours <= 0 or waiting_keepalive_hours <= 0:
            raise CampaignStateError("run and waiting time limits must be positive")

        now = utc_now()
        billing_start = billing_started_at or now
        parse_utc(billing_start)
        state: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "revision": 0,
            "campaign_id": campaign_id,
            "instance_id": str(instance_id),
            "status": "PREFLIGHT",
            "created_at_utc": now,
            "updated_at_utc": now,
            "billing_started_at_utc": billing_start,
            "git_commit": git_commit,
            "policy": {
                "hourly_rate_usd": float(hourly_rate_usd),
                "max_campaign_cost_usd": float(max_campaign_cost_usd),
                "max_run_hours": float(max_run_hours),
                "waiting_keepalive_hours": float(waiting_keepalive_hours),
            },
            "current_run": None,
            "completed_runs": [],
            "leaderboard_results": [],
            "waiting_since_utc": None,
            "last_reason": "campaign state initialized after worker bootstrap",
        }
        self._persist(state)
        self._append_event(state, "INITIALIZED", state["last_reason"], {})
        return state

    def estimate_cost_usd(
        self, state: Optional[dict[str, Any]] = None, now: Optional[datetime] = None,
    ) -> float:
        state = state or self.load()
        now = now or datetime.now(timezone.utc)
        started = parse_utc(state["billing_started_at_utc"])
        elapsed_hours = max(0.0, (now - started).total_seconds() / 3600.0)
        return elapsed_hours * float(state["policy"]["hourly_rate_usd"])

    def budget_snapshot(self, state: Optional[dict[str, Any]] = None) -> dict[str, float]:
        state = state or self.load()
        spent = self.estimate_cost_usd(state)
        maximum = float(state["policy"]["max_campaign_cost_usd"])
        return {
            "estimated_cost_usd": round(spent, 6),
            "max_campaign_cost_usd": maximum,
            "remaining_usd": round(max(0.0, maximum - spent), 6),
            "used_fraction": spent / maximum,
        }

    def assert_budget(self, reserve_hours: float = 0.0) -> dict[str, float]:
        state = self.load()
        snapshot = self.budget_snapshot(state)
        projected = snapshot["estimated_cost_usd"] + (
            max(0.0, reserve_hours) * float(state["policy"]["hourly_rate_usd"])
        )
        if projected > snapshot["max_campaign_cost_usd"]:
            raise CampaignStateError(
                "campaign budget guard rejected the operation: "
                f"projected=${projected:.3f}, "
                f"limit=${snapshot['max_campaign_cost_usd']:.3f}"
            )
        return snapshot

    def set_max_run_hours(self, hours: float, reason: str) -> dict[str, Any]:
        """Atomically revise the per-run ceiling while no experiment is active.

        Runtime policy is durable operational state rather than a scientific
        profile input.  Changes are nevertheless evented because an otherwise
        identical profile can be truncated at a different horizon when this
        guard is lower than its requested timeout.
        """
        state = self.load()
        allowed = {
            "PREFLIGHT", "READY", "ANALYZING", "CAMPAIGN_BLOCKED", "PAUSED",
        }
        if state["status"] not in allowed or state.get("current_run"):
            raise CampaignStateError(
                "max_run_hours can be changed only while no experiment is active"
            )
        value = float(hours)
        if not value > 0.0:
            raise CampaignStateError("max_run_hours must be positive")
        if not str(reason).strip():
            raise CampaignStateError("policy change reason must be non-empty")

        previous = float(state["policy"]["max_run_hours"])
        if value == previous:
            return state
        state["revision"] = int(state["revision"]) + 1
        state["updated_at_utc"] = utc_now()
        state["last_reason"] = str(reason).strip()
        state["policy"]["max_run_hours"] = value
        self._persist(state)
        self._append_event(state, "POLICY_UPDATED", state["last_reason"], {
            "field": "max_run_hours",
            "previous": previous,
            "updated": value,
        })
        return state

    def set_max_campaign_cost(
        self, cost_usd: float, reason: str,
    ) -> dict[str, Any]:
        """Atomically revise the total budget while no experiment is active.

        The update is deliberately separate from scientific profiles: it
        changes only the supervisor's operational spending guard.  Every
        effective change is written to the append-only event ledger so a
        larger budget cannot be introduced by an unaudited JSON edit.
        """
        state = self.load()
        allowed = {
            "PREFLIGHT", "READY", "ANALYZING", "CAMPAIGN_BLOCKED", "PAUSED",
        }
        if state["status"] not in allowed or state.get("current_run"):
            raise CampaignStateError(
                "max_campaign_cost can be changed only while no experiment "
                "is active"
            )
        value = float(cost_usd)
        if not value > 0.0:
            raise CampaignStateError("max_campaign_cost must be positive")
        if not str(reason).strip():
            raise CampaignStateError("policy change reason must be non-empty")

        spent = self.estimate_cost_usd(state)
        if value < spent:
            raise CampaignStateError(
                "max_campaign_cost cannot be lower than already estimated "
                f"spend: requested=${value:.3f}, spent=${spent:.3f}"
            )
        previous = float(state["policy"]["max_campaign_cost_usd"])
        if value == previous:
            return state
        state["revision"] = int(state["revision"]) + 1
        state["updated_at_utc"] = utc_now()
        state["last_reason"] = str(reason).strip()
        state["policy"]["max_campaign_cost_usd"] = value
        self._persist(state)
        self._append_event(state, "POLICY_UPDATED", state["last_reason"], {
            "field": "max_campaign_cost_usd",
            "previous": previous,
            "updated": value,
        })
        return state

    def transition(
        self,
        target: str,
        reason: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        state = self.load()
        source = state["status"]
        target = target.upper()
        if target not in ALLOWED_TRANSITIONS:
            raise CampaignStateError(f"unknown target status: {target}")
        if target not in ALLOWED_TRANSITIONS[source]:
            raise CampaignStateError(f"invalid campaign transition: {source} -> {target}")
        if target in {"RUNNING_EXPERIMENT", "WAITING_FOR_LEADERBOARD"}:
            self.assert_budget()

        now = utc_now()
        state["revision"] = int(state["revision"]) + 1
        state["status"] = target
        state["updated_at_utc"] = now
        state["last_reason"] = reason
        if target == "WAITING_FOR_LEADERBOARD":
            state["waiting_since_utc"] = now
        elif source == "WAITING_FOR_LEADERBOARD":
            state["waiting_since_utc"] = None
        self._persist(state)
        self._append_event(state, "TRANSITION", reason, {
            "from": source, "to": target, **(metadata or {}),
        })
        return state

    def start_run(self, run: dict[str, Any]) -> dict[str, Any]:
        state = self.transition(
            "RUNNING_EXPERIMENT",
            f"starting experiment {run['profile']}",
            {"profile": run["profile"], "git_commit": run.get("git_commit", "")},
        )
        run = {**run, "started_at_utc": utc_now(), "status": "running"}
        state["current_run"] = run
        self._persist(state)
        return state

    def finish_run(
        self,
        *,
        success: bool,
        exit_code: int,
        reason: str,
        artifacts: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        state = self.load()
        if state["status"] != "RUNNING_EXPERIMENT" or not state.get("current_run"):
            raise CampaignStateError("there is no active experiment to finish")
        run = dict(state["current_run"])
        run.update({
            "finished_at_utc": utc_now(),
            "status": "complete" if success else "failed",
            "exit_code": int(exit_code),
            "artifacts": artifacts or [],
        })
        state["completed_runs"].append(run)
        state["current_run"] = None
        self._persist(state)
        target = "ANALYZING" if success else "CAMPAIGN_BLOCKED"
        return self.transition(target, reason, {
            "profile": run["profile"], "exit_code": int(exit_code),
        })

    def record_leaderboard(self, score: float, note: str = "") -> dict[str, Any]:
        state = self.load()
        if state["status"] != "WAITING_FOR_LEADERBOARD":
            raise CampaignStateError(
                "leaderboard results are accepted only while waiting for leaderboard"
            )
        state["leaderboard_results"].append({
            "score": float(score), "note": note, "received_at_utc": utc_now(),
        })
        self._persist(state)
        return self.transition(
            "ANALYZING", f"leaderboard score received: {score:.6f}",
            {"score": float(score), "note": note},
        )

    def _persist(self, state: dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(state, indent=2, ensure_ascii=False) + "\n"
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{self.state_path.name}.", dir=str(self.state_path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.state_path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    def _append_event(
        self,
        state: dict[str, Any],
        event_type: str,
        reason: str,
        metadata: dict[str, Any],
    ) -> None:
        self.event_path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "timestamp_utc": utc_now(),
            "campaign_id": state["campaign_id"],
            "revision": state["revision"],
            "event_type": event_type,
            "status": state["status"],
            "reason": reason,
            "metadata": metadata,
        }
        with self.event_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
