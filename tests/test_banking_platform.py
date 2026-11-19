"""Banking credit-risk monitoring platform: services, statistics and API workflow.

Integration tests run against in-memory SQLite (PostgreSQL ORM schema) and
mongomock, so they never touch the developer's real databases.
"""

import io

import mongomock
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.core.rate_limit import limiter
from app.database import mongo
from app.database.models import (
    Alert,
    Dataset,
    DriftSummary,
    Feature,
    Model,
    MonitoringJob,
)
from app.database.postgres import Base, get_db
from app.main import app
from app.services.datasets import (
    prepare_banking_frame,
    validate_banking_dataset,
)
from app.services.monitoring import store as banking_store
from app.services.monitoring.analysis import (
    analyze_banking_drift,
    compute_feature_drift,
)
from app.services.reports import generate_committee_pdf
from app.services.synthetic_data import BankingDataGenerator

PASSWORD = "Password123"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _baseline(n: int = 400, seed: int = 1) -> pd.DataFrame:
    return BankingDataGenerator(seed).generate_baseline_2020(n)


def test_valid_baseline_passes_and_detects_labels() -> None:
    result = validate_banking_dataset(_baseline())
    assert result.is_valid and result.has_labels and result.labelled_rows == 400


def test_missing_required_columns_rejected() -> None:
    result = validate_banking_dataset(pd.DataFrame({"random_col": [1, 2, 3]}))
    assert not result.is_valid
    assert "credit_score" in result.missing_columns


def test_empty_dataset_rejected() -> None:
    result = validate_banking_dataset(_baseline().iloc[0:0])
    assert not result.is_valid
    assert any("empty" in e for e in result.errors)


def test_small_sample_rejected() -> None:
    result = validate_banking_dataset(_baseline(20))
    assert not result.is_valid
    assert any("at least 30" in e for e in result.errors)


def test_headers_are_normalised() -> None:
    df = _baseline().rename(
        columns={"monthly_income": "Monthly Income ", "credit_score": "CREDIT-SCORE"}
    )
    assert validate_banking_dataset(df).is_valid


def test_missing_and_infinite_values_become_warnings() -> None:
    df = _baseline()
    df.loc[:9, "monthly_income"] = np.nan
    df.loc[10:12, "fd_balance"] = np.inf
    clean, result = prepare_banking_frame(df)
    assert result.is_valid
    assert any("monthly_income" in w and "missing" in w for w in result.warnings)
    assert any("infinite" in w for w in result.warnings)
    assert clean["fd_balance"].isna().sum() == 3


def test_incompatible_types_rejected_but_sparse_junk_tolerated() -> None:
    df = _baseline().astype({"credit_score": object})
    df.loc[:4, "credit_score"] = "n/a"  # 5/400 = 1.25% -> warning
    assert validate_banking_dataset(df).is_valid
    df.loc[:99, "credit_score"] = "bad"  # 25% -> incompatible
    result = validate_banking_dataset(df)
    assert not result.is_valid
    assert any("incompatible" in e for e in result.errors)


def test_invalid_labels_rejected() -> None:
    df = _baseline().astype({"loan_default": object})
    df.loc[0, "loan_default"] = 7
    assert not validate_banking_dataset(df).is_valid


def test_synthetic_augmentation_preserves_real_columns_and_labels() -> None:
    real = _baseline()[["monthly_income", "credit_score", "loan_default"]]
    clean, result = prepare_banking_frame(real, fill_missing_with_synthetic=True)
    assert result.is_valid
    assert "monthly_income" not in result.synthetic_columns
    assert "loan_default" not in result.synthetic_columns
    assert "fd_balance" in result.synthetic_columns
    assert np.allclose(clean["monthly_income"], real["monthly_income"])
    assert (clean["loan_default"] == real["loan_default"]).all()
    assert any("SYNTHETIC" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Statistical edge cases
# ---------------------------------------------------------------------------


def test_constant_columns_do_not_crash() -> None:
    res = compute_feature_drift(
        pd.Series([5.0] * 100), pd.Series([5.0] * 100), "loan_tenure"
    )
    assert res["psi"] == pytest.approx(0, abs=1e-9) and res["severity"] == "low"
    shifted = compute_feature_drift(
        pd.Series([5.0] * 100), pd.Series([9.0] * 100), "loan_tenure"
    )
    assert shifted["psi"] > 0.25 and shifted["severity"] == "high"


def test_small_feature_sample_is_flagged_not_fabricated() -> None:
    res = compute_feature_drift(
        pd.Series(np.arange(100.0)), pd.Series(np.arange(10.0)), "age"
    )
    assert res["drift_status"] == "Insufficient data"
    assert res["psi"] is None and res["d"] == 0.0


def test_all_missing_production_feature() -> None:
    res = compute_feature_drift(
        pd.Series(np.arange(100.0)), pd.Series([np.nan] * 100), "age"
    )
    assert res["drift_status"] == "Insufficient data"


def test_categorical_different_category_sets() -> None:
    ref = pd.Series(["salaried"] * 80 + ["contract"] * 20)
    prod = pd.Series(["salaried"] * 40 + ["gig-worker"] * 60)
    res = compute_feature_drift(ref, prod, "employment_type", "categorical")
    assert res["ks_stat"] is None
    assert res["new_categories"] == ["gig-worker"]
    assert res["missing_categories"] == ["contract"]
    assert res["severity"] == "high"


def test_scenarios_rank_by_drift_and_use_config_thresholds() -> None:
    gen = BankingDataGenerator(5)
    ref, _ = prepare_banking_frame(gen.generate_baseline_2020(1500))
    from app.services.monitoring import model as credit_model

    model, metrics = credit_model.train_baseline_model(ref)
    out = {}
    for sc in "ABC":
        prod, _ = prepare_banking_frame(gen.generate_production_batch(sc, 800, sc))
        out[sc] = analyze_banking_drift(ref, prod, sc, model, metrics, floor_key="t")
    assert out["A"]["overall_dis"] < out["B"]["overall_dis"] < out["C"]["overall_dis"]
    assert out["A"]["risk_category"] == "LOW RISK"
    assert out["C"]["risk_category"] in {"HIGH RISK", "CRITICAL RISK"}
    assert (
        out["A"]["performance_state"] == "unknown"
    )  # no labels -> never claims degradation
    for f in out["C"]["feature_details"]:
        assert f["dis"] == pytest.approx(f["d"] * f["fi"] * f["s"])


def test_unlabelled_baseline_falls_back_to_drift_only() -> None:
    gen = BankingDataGenerator(8)
    ref, _ = prepare_banking_frame(
        gen.generate_baseline_2020(500).drop(columns="loan_default")
    )
    prod, _ = prepare_banking_frame(gen.generate_production_batch("x", 300, "C"))
    res = analyze_banking_drift(ref, prod, "x")
    assert res["dis_mode"] == "drift_only"
    assert not res["model_health"]["has_model"]
    assert all(f["fi"] == 1.0 and f["s"] == 1.0 for f in res["feature_details"])


def test_pdf_generation_from_analysis() -> None:
    gen = BankingDataGenerator(77)
    ref, _ = prepare_banking_frame(gen.generate_baseline_2020(400))
    prod, _ = prepare_banking_frame(gen.generate_production_batch("S", 300, "D", True))
    res = analyze_banking_drift(ref, prod, "S")
    pdf = generate_committee_pdf(res)
    assert pdf.startswith(b"%PDF") and len(pdf) > 3000


def test_values_fit_postgres_real_columns() -> None:
    """Regression: a KS p-value of 1e-300 underflowed PostgreSQL REAL."""
    assert banking_store.to_real(1e-300) == 0.0
    assert banking_store.to_real(0.123) == pytest.approx(0.123)
    assert banking_store.to_real(None) is None
    assert banking_store.to_real(float("nan")) is None


# ---------------------------------------------------------------------------
# API workflow
# ---------------------------------------------------------------------------


@pytest.fixture()
def env(monkeypatch, tmp_path):
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)

    def override():
        s = session_factory()
        try:
            yield s
        finally:
            s.close()

    monkeypatch.setattr(mongo, "client", mongomock.MongoClient())
    monkeypatch.setattr(settings, "artifact_dir", tmp_path)
    app.dependency_overrides[get_db] = override
    limiter.reset()
    with TestClient(app) as client:
        client.post(
            "/api/v1/auth/register",
            json={
                "name": "Admin",
                "email": "admin@example.com",
                "password": PASSWORD,
                "role": "administrator",
            },
        )
        tokens = {}
        login = client.post(
            "/api/v1/auth/login",
            json={"email": "admin@example.com", "password": PASSWORD},
        )
        tokens["admin"] = {"Authorization": f"Bearer {login.json()['access_token']}"}
        for role in ("analyst", "reviewer"):
            client.post(
                "/api/v1/auth/register",
                headers=tokens["admin"],
                json={
                    "name": role,
                    "email": f"{role}@example.com",
                    "password": PASSWORD,
                    "role": role,
                },
            )
            tok = client.post(
                "/api/v1/auth/login",
                json={"email": f"{role}@example.com", "password": PASSWORD},
            ).json()["access_token"]
            tokens[role] = {"Authorization": f"Bearer {tok}"}
        yield client, tokens, session_factory
    app.dependency_overrides.clear()
    limiter.reset()


def _csv(df: pd.DataFrame, name: str) -> tuple:
    return ("files", (name, io.BytesIO(df.to_csv(index=False).encode()), "text/csv"))


def _register_baseline(client, headers, n=1200, seed=11):
    df = BankingDataGenerator(seed).generate_baseline_2020(n)
    return client.post(
        "/api/v1/banking/datasets/baseline",
        headers=headers,
        files={
            "file": (
                "baseline_2020.csv",
                io.BytesIO(df.to_csv(index=False).encode()),
                "text/csv",
            )
        },
        data={"dataset_name": "2020_Historical_Baseline"},
    )


def test_endpoints_require_authentication(env) -> None:
    client, _, _ = env
    assert client.get("/api/v1/banking/overview").status_code in (401, 403)
    assert client.get("/api/v1/banking/batches").status_code in (401, 403)


def test_role_based_access(env) -> None:
    client, tokens, _ = env
    assert _register_baseline(client, tokens["analyst"]).status_code == 403
    df = BankingDataGenerator(1).generate_production_batch("x", 200, "A")
    r = client.post(
        "/api/v1/banking/batches/upload",
        headers=tokens["reviewer"],
        files=[_csv(df, "x.csv")],
    )
    assert r.status_code == 403
    assert (
        client.get("/api/v1/banking/overview", headers=tokens["reviewer"]).status_code
        == 200
    )


def test_analysis_requires_baseline(env) -> None:
    client, tokens, _ = env
    df = BankingDataGenerator(1).generate_production_batch("x", 200, "A")
    client.post(
        "/api/v1/banking/batches/upload",
        headers=tokens["analyst"],
        files=[_csv(df, "January_2026.csv")],
    )
    r = client.post(
        "/api/v1/banking/analysis/run",
        headers=tokens["analyst"],
        json={"batch_ids": ["January_2026"]},
    )
    assert r.status_code == 409 and r.json()["code"] == "baseline_required"


def test_invalid_files_rejected(env) -> None:
    client, tokens, _ = env
    h = tokens["admin"]
    bad = [
        ("notes.txt", b"a,b\n1,2\n"),
        ("empty.csv", b""),
        ("binary.csv", b"\x00\x01\x02garbage"),
    ]
    for name, content in bad:
        r = client.post(
            "/api/v1/banking/datasets/baseline",
            headers=h,
            files={"file": (name, io.BytesIO(content), "text/csv")},
        )
        assert r.status_code == 400, name
    r = client.post(
        "/api/v1/banking/datasets/baseline",
        headers=h,
        files={"file": ("cols.csv", io.BytesIO(b"a,b\n1,2\n"), "text/csv")},
    )
    assert r.status_code == 422 and "Missing required columns" in r.json()["detail"]


def test_full_banking_workflow(env) -> None:
    client, tokens, session_factory = env
    admin, analyst = tokens["admin"], tokens["analyst"]

    # 1-2. Baseline registration (validation + model training)
    r = _register_baseline(client, admin)
    assert r.status_code == 201, r.text
    base = r.json()
    assert (
        base["model"]["metrics"]["has_model"]
        and base["model"]["metrics"]["roc_auc"] > 0.5
    )
    assert _register_baseline(client, admin).status_code == 409  # identical content

    # 3-4. Multiple production batch upload with validation
    gen = BankingDataGenerator(21)
    frames = {
        "January_2026.csv": gen.generate_production_batch("Jan", 600, "A"),
        "March_2026.csv": gen.generate_production_batch("Mar", 600, "B"),
        "June_2026.csv": gen.generate_production_batch("Jun", 600, "C"),
        "September_2026.csv": gen.generate_production_batch(
            "Sep", 600, "D", with_labels=True
        ),
    }
    bad = pd.DataFrame({"customer_id": ["a"] * 40})
    files = [_csv(df, name) for name, df in frames.items()] + [
        _csv(bad, "Broken_2026.csv")
    ]
    r = client.post(
        "/api/v1/banking/batches/upload",
        headers=analyst,
        files=files,
        data={"descriptions": ["Q1", "Q1 end", "Q2", "Q3", "bad"]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["registered_count"] == 4 and body["rejected_count"] == 1
    statuses = {x["batch_id"]: x for x in body["results"]}
    assert statuses["March_2026"]["batch_date"] == "2026-03-31"
    assert statuses["Broken_2026"]["code"] == "validation_error"

    # Duplicate batch prevention, overwrite, and duplicate content
    again = client.post(
        "/api/v1/banking/batches/upload",
        headers=analyst,
        files=[_csv(frames["January_2026.csv"], "January_2026.csv")],
    ).json()
    assert again["results"][0]["code"] == "duplicate_batch"
    twin = client.post(
        "/api/v1/banking/batches/upload",
        headers=analyst,
        files=[_csv(frames["January_2026.csv"], "Jan_copy.csv")],
    ).json()
    assert twin["results"][0]["code"] == "duplicate_content"
    over = client.post(
        "/api/v1/banking/batches/upload",
        headers=analyst,
        files=[_csv(frames["January_2026.csv"], "January_2026.csv")],
        data={"overwrite": "true"},
    ).json()
    assert over["results"][0]["status"] == "registered"

    batches = client.get("/api/v1/banking/batches", headers=analyst).json()
    assert [b["batch_id"] for b in batches] == [
        "January_2026",
        "March_2026",
        "June_2026",
        "September_2026",
    ]

    # 5. Individual analysis
    ids = [b["batch_id"] for b in batches]
    r = client.post(
        "/api/v1/banking/analysis/run", headers=analyst, json={"batch_ids": ids}
    )
    assert r.status_code == 200, r.text
    reports = {x["batch_id"]: x for x in r.json()["reports"]}
    assert reports["January_2026"]["risk_category"] == "LOW RISK"
    assert (
        reports["January_2026"]["overall_dis"]
        < reports["March_2026"]["overall_dis"]
        < reports["June_2026"]["overall_dis"] + 1e-9
    )
    assert reports["June_2026"]["performance_state"] == "unknown"
    assert reports["September_2026"]["model_health"]["has_production_labels"]
    assert reports["September_2026"]["performance_state"] in {
        "stable",
        "degraded",
        "severe",
    }

    # Duplicate processing is prevented unless forced
    rerun = client.post(
        "/api/v1/banking/analysis/run",
        headers=analyst,
        json={"batch_ids": ["June_2026"]},
    ).json()["reports"][0]
    assert rerun["reused"] and rerun["job_id"] == reports["June_2026"]["job_id"]
    forced = client.post(
        "/api/v1/banking/analysis/run",
        headers=analyst,
        json={"batch_ids": ["June_2026"], "force": True},
    ).json()["reports"][0]
    assert not forced["reused"] and forced["job_id"] != reports["June_2026"]["job_id"]
    assert forced["overall_dis"] == pytest.approx(reports["June_2026"]["overall_dis"])

    # 6. Combined analysis preserves individual results
    r = client.post(
        "/api/v1/banking/analysis/run",
        headers=analyst,
        json={"batch_ids": ids, "mode": "combined"},
    )
    combined = r.json()["reports"][0]
    assert (
        combined["analysis_type"] == "combined" and combined["customer_count"] == 2400
    )
    assert (
        client.get(
            "/api/v1/banking/analysis/latest/January_2026", headers=analyst
        ).json()["job_id"]
        == reports["January_2026"]["job_id"]
    )
    single = client.post(
        "/api/v1/banking/analysis/run",
        headers=analyst,
        json={"batch_ids": ["June_2026"], "mode": "combined"},
    )
    assert single.status_code == 422

    # 7-10. Feature detail, recommendations, history, comparison, overview, model health
    jid = reports["June_2026"]["job_id"]
    feat = client.get(
        f"/api/v1/banking/analysis/{jid}/features/credit_score", headers=analyst
    ).json()
    assert (
        feat["feature"]["ks_stat"] is not None
        and feat["distribution"]["series"]["production"]
    )
    rec = client.get(
        f"/api/v1/banking/recommendations/{jid}", headers=tokens["reviewer"]
    ).json()
    assert (
        rec["risk_category"] in {"HIGH RISK", "CRITICAL RISK"}
        and rec["recommended_actions"]
    )
    assert any("labelled records" in a for a in rec["recommended_actions"])
    hist = client.get("/api/v1/banking/analysis/history", headers=analyst).json()
    assert len(hist["latest_per_batch"]) == 4
    assert hist["first_drift"]["first_moderate_or_higher"]["batch_id"] == "March_2026"
    cmp = client.get(
        "/api/v1/banking/compare",
        headers=analyst,
        params={"batch_a": "January_2026", "batch_b": "June_2026"},
    ).json()
    by_feat = {f["feature"]: f for f in cmp["direct_drift"]}
    assert (
        by_feat["credit_score"]["severity"] == "high"
        and by_feat["age"]["severity"] == "low"
    )
    ov = client.get("/api/v1/banking/overview", headers=analyst).json()
    assert ov["kpis"]["total_customers_monitored"] == 2400
    assert (
        ov["kpis"]["production_batches"] == 4 and ov["kpis"]["features_monitored"] == 12
    )
    assert ov["kpis"]["model_health"]["confirmed"]
    assert len(ov["heatmap"]["batches"]) == 4
    mh = client.get("/api/v1/banking/model-health", headers=analyst).json()
    assert len(mh["batches"]) == 4

    # 11. PDF
    pdf = client.get(f"/api/v1/banking/reports/{jid}/pdf", headers=tokens["reviewer"])
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")

    # 12. Persistence in PostgreSQL and MongoDB
    with session_factory() as s:
        assert s.scalar(select(func.count(Dataset.dataset_id))) == 1
        assert s.scalar(select(func.count(Feature.feature_id))) == 12
        assert s.scalar(select(Model.status)) == "champion"
        jobs = s.scalars(select(MonitoringJob)).all()
        assert len(jobs) == 6 and all(j.status == "completed" for j in jobs)
        assert s.scalar(select(func.count(DriftSummary.summary_id))) == 6 * 12
        assert s.scalar(select(func.count(Alert.alert_id))) >= 3
    m = mongo.database()
    assert m.root_cause_reports.count_documents({}) == 6
    assert m.drift_events.count_documents({"job_id": jid}) == 12
    logs = client.get("/api/v1/banking/audit-logs", headers=analyst).json()
    events = {e["event"] for e in logs["events"]}
    assert {
        "baseline_registered",
        "batch_uploaded",
        "batch_rejected",
        "analysis_completed",
        "report_generated",
    } <= events
    assert logs["jobs"][0]["features_summarised"] == 12


def test_database_failure_marks_job_failed(env, monkeypatch) -> None:
    client, tokens, session_factory = env
    _register_baseline(client, tokens["admin"])
    df = BankingDataGenerator(3).generate_production_batch("x", 300, "B")
    client.post(
        "/api/v1/banking/batches/upload",
        headers=tokens["analyst"],
        files=[_csv(df, "March_2026.csv")],
    )

    from pymongo.errors import ServerSelectionTimeoutError

    real_db = banking_store.db

    def flaky():
        d = real_db()
        reports = d.root_cause_reports

        class Reports:
            """Reads work; the report write fails like an unreachable server."""

            def __getattr__(self, name):
                if name == "replace_one":

                    def fail(*a, **k):
                        raise ServerSelectionTimeoutError("mongo down")

                    return fail
                return getattr(reports, name)

        class Proxy:
            def __getattr__(self, name):
                return Reports() if name == "root_cause_reports" else getattr(d, name)

        return Proxy()

    monkeypatch.setattr(banking_store, "db", flaky)
    r = client.post(
        "/api/v1/banking/analysis/run",
        headers=tokens["analyst"],
        json={"batch_ids": ["March_2026"]},
    )
    assert r.status_code == 500 and r.json()["code"] == "analysis_failed"
    with session_factory() as s:
        job = s.scalars(select(MonitoringJob)).one()
        assert job.status == "failed"
        assert s.scalar(select(func.count(DriftSummary.summary_id))) == 0


def test_seed_demo_and_synthetic_labelling(env) -> None:
    client, tokens, _ = env
    r = client.post(
        "/api/v1/banking/simulation/seed-demo", headers=tokens["admin"], json={}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["synthetic"] and len(body["analyses"]) == 5
    risk = {a["batch_id"]: a["risk_category"] for a in body["analyses"]}
    assert risk["January_2026"] == "LOW RISK"
    assert risk["March_2026"] == "MEDIUM RISK"
    assert risk["November_2026"] == "CRITICAL RISK"
    batches = client.get("/api/v1/banking/batches", headers=tokens["admin"]).json()
    assert all(b["is_synthetic"] for b in batches)
    assert client.get(
        "/api/v1/banking/datasets/baseline", headers=tokens["admin"]
    ).json()["is_synthetic"]
    # Seeding again is idempotent (reuses baseline, batches and analyses)
    again = client.post(
        "/api/v1/banking/simulation/seed-demo", headers=tokens["admin"], json={}
    ).json()
    assert again["baseline"]["reused"]
