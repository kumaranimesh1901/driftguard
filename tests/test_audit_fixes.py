"""Regression tests for the audit fixes and the features added during the audit.

Covers: null-floor calibration, detector agreement, low-impact drift messaging,
partial-write compensation, CORS, reuse signature, ADWIN streams, drift-type
separation, and the champion-challenger retraining workflow.
"""

import numpy as np
import pytest
from scipy.stats import ks_2samp

from app.core.config import settings
from app.database.models import Model, MonitoringJob, RetrainingEvent
from app.services.drift.adwin import adwin_stream
from app.services.drift.calibration import numerical_noise_floors
from app.services.drift.psi import psi
from app.services.monitoring import store
from app.services.monitoring.analysis import compute_feature_drift
from app.services.recommendations.risk import risk_assessment
from tests.test_banking_platform import env  # noqa: F401  (pytest fixture)

API = "/api/v1/banking"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def test_noise_floor_false_alarm_rate_is_near_nominal() -> None:
    """Under no drift, ~5% of fresh batches should exceed the 95% floor.

    The previous n-vs-n null design gave ~0% (floors too high, D biased down).
    """
    rng = np.random.default_rng(3)
    draw = lambda k: rng.lognormal(np.log(35_000), 0.4, k)
    ref = draw(2500)
    floors = numerical_noise_floors(ref, 1000, b=200)
    trials = 200
    ks_rate = np.mean(
        [ks_2samp(ref, draw(1000)).statistic > floors["ks"] for _ in range(trials)]
    )
    psi_rate = np.mean([psi(ref, draw(1000)) > floors["psi"] for _ in range(trials)])
    assert 0.01 <= ks_rate <= 0.12
    assert 0.01 <= psi_rate <= 0.12


def test_detector_agreement_reports_disagreement() -> None:
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 3000)
    stable = compute_feature_drift(
        __import__("pandas").Series(ref), __import__("pandas").Series(rng.normal(0, 1, 1000)),
        "age", "numerical", {"psi": 0.05, "ks": 0.05, "jsd": 0.02},
    )
    shifted = compute_feature_drift(
        __import__("pandas").Series(ref), __import__("pandas").Series(rng.normal(1.5, 1, 1000)),
        "age", "numerical", {"psi": 0.05, "ks": 0.05, "jsd": 0.02},
    )
    # Only KS crosses a tiny floor: the three detectors disagree.
    partial = compute_feature_drift(
        __import__("pandas").Series(ref), __import__("pandas").Series(rng.normal(0.15, 1, 1000)),
        "age", "numerical", {"psi": 10.0, "ks": 0.01, "jsd": 10.0},
    )
    assert stable["detector_agreement"]["status"] == "agree-stable"
    assert shifted["detector_agreement"]["status"] == "agree-drift"
    assert partial["detector_agreement"]["status"] == "disagree"
    assert partial["detector_agreement"]["exceeding_noise_floor"] == ["ks"]


def _feature(label: str, severity: str, dis: float) -> dict:
    return {"label": label, "severity": severity, "dis": dis, "d": 0.9, "fi": 0.01,
            "s": 0.01, "psi": 0.6, "ks_stat": 0.4, "jsd": 0.2, "normalised": {}}


def test_low_risk_batch_with_strong_low_impact_drift_is_not_called_stable() -> None:
    out = risk_assessment("low", "unknown", [_feature("FD Balance", "high", 0.001)], None)
    assert out["risk_category"] == "LOW RISK"
    assert out["low_impact_drift_features"] == ["FD Balance"]
    assert not any("remain relatively stable" in a for a in out["recommended_actions"])
    assert "data quality" in out["recommendation"]
    clean = risk_assessment("low", "unknown", [_feature("Age", "low", 0.0)], None)
    assert any("remain relatively stable" in a for a in clean["recommended_actions"])


def test_adwin_detects_shift_but_not_stationary_stream() -> None:
    rng = np.random.default_rng(1)
    ref = rng.beta(2, 8, 1500)
    stable = adwin_stream(ref, rng.beta(2, 8, 1500), "prediction")
    shifted = adwin_stream(ref, rng.beta(5, 5, 1500), "prediction")
    assert stable["available"] and stable["detected"] is False
    assert shifted["detected"] is True and shifted["first_detection_index"] >= 0
    assert shifted["n_detections_reference"] == 0


# ---------------------------------------------------------------------------
# Persistence / API
# ---------------------------------------------------------------------------


def _seed(client, admin):
    r = client.post(f"{API}/simulation/generate-baseline", headers=admin,
                    json={"n": 2500, "seed": 7})
    assert r.status_code == 201, r.text
    return r.json()


def _batch(client, hdr, batch_id, scenario, n=1000, labels=False, concept=False, seed=1):
    r = client.post(f"{API}/simulation/generate-batch", headers=hdr, json={
        "scenario": scenario, "batch_id": batch_id, "batch_date": "2026-03-31", "n": n,
        "with_labels": labels, "concept_shift": concept, "seed": seed})
    assert r.status_code == 201, r.text


def test_failed_mongo_event_write_leaves_no_orphan_report(env, monkeypatch) -> None:  # noqa: F811
    client, tokens, session_factory = env
    _seed(client, tokens["admin"])
    _batch(client, tokens["analyst"], "Orphan_Test", "C")

    def boom(*_a, **_k):
        raise RuntimeError("simulated drift_events failure")

    coll = store.db().drift_events
    with monkeypatch.context() as patch:
        patch.setattr(type(coll), "insert_many", boom)
        r = client.post(f"{API}/analysis/run", headers=tokens["analyst"],
                        json={"batch_ids": ["Orphan_Test"]})
    assert r.status_code == 500
    assert store.db().root_cause_reports.count_documents({}) == 0
    with session_factory() as s:
        assert [j.status for j in s.query(MonitoringJob).all()] == ["failed"]


def test_cors_does_not_allow_arbitrary_origins(env) -> None:  # noqa: F811
    client, _, _ = env
    r = client.get("/health", headers={"Origin": "https://evil.example"})
    assert r.headers.get("access-control-allow-origin") != "https://evil.example"
    ok = client.get("/health", headers={"Origin": "http://localhost:8501"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:8501"


def test_reuse_signature_changes_with_result_settings(env, monkeypatch) -> None:  # noqa: F811
    client, tokens, _ = env
    _seed(client, tokens["admin"])
    _batch(client, tokens["analyst"], "Sig_Test", "B")
    run = lambda: client.post(f"{API}/analysis/run", headers=tokens["analyst"],
                              json={"batch_ids": ["Sig_Test"]}).json()["reports"][0]
    first = run()
    assert run()["reused"] is True
    monkeypatch.setattr(settings, "batch_dis_high", 0.9)
    assert run()["reused"] is False and first["reused"] is False


def test_report_separates_drift_types_and_stores_adwin_events(env) -> None:  # noqa: F811
    client, tokens, _ = env
    _seed(client, tokens["admin"])
    _batch(client, tokens["analyst"], "Unlabelled_C", "C", n=1500)
    rep = client.post(f"{API}/analysis/run", headers=tokens["analyst"],
                      json={"batch_ids": ["Unlabelled_C"]}).json()["reports"][0]
    types = rep["drift_types"]
    assert types["data_drift"] and types["prediction_drift"]
    # Without labels, degradation / concept drift are unknown, never "False".
    assert types["confirmed_degradation"] is None
    assert types["concept_drift_suspected"] is None
    assert rep["model_health"]["streams"]["prediction"]["detected"] is True
    docs = list(store.db().stream_events.find({"job_id": rep["job_id"]}))
    assert [d["stream"] for d in docs] == ["prediction"]
    assert all("detector_agreement" in f for f in rep["feature_details"])


# ---------------------------------------------------------------------------
# Champion-challenger retraining
# ---------------------------------------------------------------------------


def test_retraining_refuses_without_enough_labels(env) -> None:  # noqa: F811
    client, tokens, _ = env
    _seed(client, tokens["admin"])
    _batch(client, tokens["analyst"], "No_Labels", "C")
    r = client.post(f"{API}/models/retrain", headers=tokens["admin"],
                    json={"batch_ids": ["No_Labels"]})
    assert r.status_code == 409 and "Unlabelled" in r.json()["detail"]
    # Analysts and reviewers may not retrain.
    assert client.post(f"{API}/models/retrain", headers=tokens["analyst"],
                       json={"batch_ids": ["No_Labels"]}).status_code == 403


def test_champion_challenger_deploy_and_rollback(env) -> None:  # noqa: F811
    client, tokens, session_factory = env
    admin = tokens["admin"]
    _seed(client, admin)
    _batch(client, tokens["analyst"], "Concept_Labelled", "B", n=4000, labels=True,
           concept=True, seed=9)
    r = client.post(f"{API}/models/retrain", headers=admin,
                    json={"batch_ids": ["Concept_Labelled"]})
    assert r.status_code == 201, r.text
    ev = r.json()
    assert ev["data"]["unlabelled_rows_used"] == 0
    # The post-drift test rows are disjoint from training rows by construction.
    assert ev["data"]["production_train_rows"] + ev["data"]["post_drift_test_rows"] == 4000
    assert ev["acceptable"], ev["criteria"]  # concept shift: retraining helps
    event_id = ev["event_id"]
    # Only one pending challenger per champion.
    assert client.post(f"{API}/models/retrain", headers=admin,
                       json={"batch_ids": ["Concept_Labelled"]}).status_code == 409

    models = {m["version"]: m for m in client.get(f"{API}/models", headers=admin).json()}
    assert models[1]["status"] == "champion" and models[2]["status"] == "challenger"

    d = client.post(f"{API}/models/retraining/{event_id}/deploy", headers=admin)
    assert d.status_code == 200, d.text
    models = {m["version"]: m for m in client.get(f"{API}/models", headers=admin).json()}
    assert models[1]["status"] == "retired" and models[2]["status"] == "champion"

    # Analyses now use V2, and re-analysing its own training batch is flagged.
    rep = client.post(f"{API}/analysis/run", headers=tokens["analyst"],
                      json={"batch_ids": ["Concept_Labelled"]}).json()["reports"][0]
    assert rep["model_id"] == models[2]["model_id"]
    assert rep["leakage_warning"] == ["Concept_Labelled"]

    rb = client.post(f"{API}/models/retraining/{event_id}/rollback", headers=admin)
    assert rb.status_code == 200, rb.text
    with session_factory() as s:
        statuses = {m.version: m.status for m in s.query(Model).all()}
        event = s.get(RetrainingEvent, event_id)
        assert statuses == {1: "champion", 2: "retired"}
        assert event.decision == "rolled_back" and event.rollback_flag
    assert client.post(f"{API}/models/retraining/{event_id}/deploy",
                       headers=admin).status_code == 409


def test_unacceptable_challenger_cannot_be_deployed(env) -> None:  # noqa: F811
    client, tokens, _ = env
    admin = tokens["admin"]
    _seed(client, admin)
    # Stable population, same outcome relationship: retraining cannot clearly help.
    _batch(client, tokens["analyst"], "Stable_Labelled", "A", n=600, labels=True, seed=4)
    ev = client.post(f"{API}/models/retrain", headers=admin,
                     json={"batch_ids": ["Stable_Labelled"]}).json()
    if ev["acceptable"]:
        pytest.skip("challenger happened to pass on this seed")
    r = client.post(f"{API}/models/retraining/{ev['event_id']}/deploy", headers=admin)
    assert r.status_code == 409 and "never deployed" in r.json()["detail"]
    assert client.post(f"{API}/models/retraining/{ev['event_id']}/reject",
                       headers=admin).json()["decision"] == "rejected"
