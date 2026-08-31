from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import scripts.evaluate_locked_lme20_single_fold as audit


def test_locked_lme20_contract_has_no_search_dimensions() -> None:
    assert audit.LOCKED_VARIANT == "logmeanexp_b20"
    assert audit.LOCKED_PARAMETERS == {
        "alpha": 0.15,
        "kappa": 16.0,
        "tau": 0.0,
        "lambda_unknown": 0.75,
    }


def test_builder_computes_only_beta20_logmeanexp(monkeypatch) -> None:
    captured = {}

    monkeypatch.setattr(
        audit,
        "group_indices",
        lambda artifact: [np.array([0, 1]), np.array([2])],
    )

    def fake_evidence(*, fold, oof, scores):
        captured["fold"] = fold
        captured["oof"] = oof
        captured["scores"] = scores
        return "evidence"

    monkeypatch.setattr(audit, "score_matrix_to_evidence", fake_evidence)
    artifact = {
        "train_embeddings": np.array(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32
        )
    }
    oof = {"embeddings": np.array([[1.0, 0.0]], dtype=np.float32)}

    evidence, diagnostics = audit.build_locked_lme20_evidence(
        fold=2, artifact=artifact, oof=oof
    )

    expected_first = float(np.log((np.exp(20.0) + 1.0) / 2.0) / 20.0)
    assert evidence == "evidence"
    assert captured["fold"] == 2
    np.testing.assert_allclose(
        captured["scores"], [[expected_first, -1.0]], rtol=0, atol=1e-6
    )
    assert diagnostics["logmeanexp_beta"] == 20.0
    assert diagnostics["enrollment_group_size_min"] == 1
    assert diagnostics["enrollment_group_size_max"] == 2


def test_evaluation_uses_exact_locked_policy(monkeypatch) -> None:
    evidence = SimpleNamespace(
        labels=np.array([1, 2, 0, 0], dtype=np.int64),
        files=np.array(["a.wav", "b.wav", "c.wav", "d.wav"]),
        baseline_predictions=np.array([1, 0, 1, 0], dtype=np.int64),
    )

    def fake_evidence(*, fold, artifact, oof):
        assert fold == 0
        assert artifact == {"artifact": "sentinel"}
        assert oof == {"oof": "sentinel"}
        return evidence, {"score_mean": 0.1}

    def fake_predict(received, parameters):
        assert received is evidence
        assert parameters == audit.LOCKED_PARAMETERS
        return np.array([1, 2, 0, 1], dtype=np.int64)

    monkeypatch.setattr(audit, "build_locked_lme20_evidence", fake_evidence)
    monkeypatch.setattr(audit, "predict", fake_predict)
    result = audit.evaluate_locked_lme20(
        fold=0,
        artifact={"artifact": "sentinel"},
        oof={"oof": "sentinel"},
    )

    assert result["parameters"] == audit.LOCKED_PARAMETERS
    assert result["rescued_errors"] == 2
    assert result["introduced_errors"] == 1
    assert result["rescued_files"] == ["b.wav", "c.wav"]
    assert result["introduced_files"] == ["d.wav"]
    assert result["rescue_rate"] == 1.0
