from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREREG = (
    ROOT
    / "configs"
    / "analyses"
    / "no-proto-es21-f0-locked-lme20-prereg.json"
)


def test_no_proto_lme20_prereg_is_locked_and_non_tuning() -> None:
    contract = json.loads(PREREG.read_text(encoding="utf-8"))

    assert contract["status"] == "preregistered_before_lme20_output"
    assert contract["source_run"]["fold"] == 0
    assert contract["source_run"]["raw_primary_gate_unchanged"][
        "macro_f1_min"
    ] == 0.96
    assert contract["locked_backend"] == {
        "variant": "logmeanexp_b20",
        "beta": 20.0,
        "alpha": 0.15,
        "kappa": 16.0,
        "tau": 0.0,
        "lambda_unknown": 0.75,
        "search_dimensions": 0,
        "target_fold_used_for_tuning": False,
        "leaderboard_used_for_selection": False,
    }
    assert contract["observed_before_backend_audit"][
        "locked_lme20_output_seen"
    ] is False
    assert contract["expansion_gate"]["all_checks_required"] is True
    assert contract["expansion_gate"]["single_fold_submission_authorized"] is False
    assert contract["runtime"]["second_training_run_allowed_during_audit"] is False
