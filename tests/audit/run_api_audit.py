"""Adversarial API and security audit script using FastAPI TestClient."""

import datetime
from fastapi.testclient import TestClient
import jwt
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.db.postgres import Base, get_db
from app.config import settings

# In-memory SQLite engine for testing
test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

def run_api_audit():
    Base.metadata.create_all(bind=test_engine)
    client = TestClient(app)
    
    print("=== 7.1 ENDPOINT EXISTENCE AUDIT (Section 18) ===")
    spec_endpoints = [
        # (method, path, spec_section, required_role)
        ("POST", "/api/v1/auth/register", "18.1 Auth", "admin/bootstrap"),
        ("POST", "/api/v1/auth/login", "18.1 Auth", "public"),
        ("GET", "/api/v1/auth/me", "18.1 Auth", "authenticated"),
        ("POST", "/api/v1/datasets", "18.2 Datasets", "analyst/admin"),
        ("GET", "/api/v1/datasets", "18.2 Datasets", "any"),
        ("GET", "/api/v1/datasets/1", "18.2 Datasets", "any"),
        ("POST", "/api/v1/models", "18.3 Models", "analyst/admin"),
        ("GET", "/api/v1/models", "18.3 Models", "any"),
        ("GET", "/api/v1/models/1", "18.3 Models", "any"),
        ("GET", "/api/v1/models/1/champion", "18.3 Models", "any"),
        ("GET", "/api/v1/models/1/reference", "18.3 Models", "any"),
        ("POST", "/api/v1/batches", "18.4 Batches", "analyst/admin"),
        ("GET", "/api/v1/batches", "18.4 Batches", "any"),
        ("GET", "/api/v1/batches/b1", "18.4 Batches", "any"),
        ("POST", "/api/v1/labels", "18.5 Labels", "analyst/admin"),
        ("POST", "/api/v1/monitoring/jobs", "18.6 Monitoring", "analyst/admin"),
        ("GET", "/api/v1/monitoring/jobs", "18.6 Monitoring", "any"),
        ("GET", "/api/v1/monitoring/jobs/1", "18.6 Monitoring", "any"),
        ("GET", "/api/v1/monitoring/jobs/1/report", "18.6 Monitoring", "any"),
        ("GET", "/api/v1/monitoring/jobs/1/events", "18.6 Monitoring", "any"),
        ("POST", "/api/v1/monitoring/jobs/1/reevaluate", "18.6 Monitoring", "analyst/admin"),
        ("GET", "/api/v1/monitoring/trend", "18.6 Monitoring", "any"),
        ("GET", "/api/v1/alerts", "18.7 Alerts", "any"),
        ("GET", "/api/v1/alerts/1", "18.7 Alerts", "any"),
        ("POST", "/api/v1/retrain", "18.8 Retraining", "analyst/admin"),
        ("GET", "/api/v1/retrain", "18.8 Retraining", "any"),
        ("GET", "/api/v1/retrain/1", "18.8 Retraining", "any"),
        ("POST", "/api/v1/retrain/1/approve", "18.8 Retraining", "administrator"),
        ("POST", "/api/v1/retrain/1/reject", "18.8 Retraining", "administrator"),
        ("POST", "/api/v1/retrain/1/rollback", "18.8 Retraining", "administrator"),
        ("GET", "/api/v1/thresholds", "18.9 Thresholds", "any"),
        ("PUT", "/api/v1/thresholds/severity_moderate", "18.9 Thresholds", "administrator"),
        ("GET", "/api/v1/thresholds/history", "18.9 Thresholds", "any"),
        ("GET", "/api/v1/simulation/scenarios", "18.10 Simulation", "any"),
        ("POST", "/api/v1/simulation/run", "18.10 Simulation", "analyst/admin"),
        ("POST", "/api/v1/experiments/run", "18.10 Experiments", "administrator"),
        ("GET", "/api/v1/experiments/1", "18.10 Experiments", "any"),
        ("GET", "/api/v1/experiments/1/results", "18.10 Experiments", "any"),
        ("GET", "/api/v1/monitoring/jobs/1/export", "18.11 Export", "any"),
    ]
    
    table_results = []
    for method, path, sec, role in spec_endpoints:
        resp = client.request(method, path)
        exists = resp.status_code != 404
        table_results.append((method, path, sec, role, "YES" if exists else "NO", resp.status_code))
        
    print(f"{'Method':6s} | {'Path':38s} | {'Role':15s} | {'Exists':6s} | {'Status'}")
    print("-" * 75)
    for m, p, sec, role, ex, sc in table_results:
        print(f"{m:6s} | {p:38s} | {role:15s} | {ex:6s} | {sc}")
        
    print("\n=== 7.2 ADVERSARIAL AUTH & RBAC AUDIT ===")
    
    # 1. Bootstrap: first user must be admin
    resp = client.post("/api/v1/auth/register", json={
        "name": "First Analyst", "email": "analyst@example.com", "password": "Password123", "role": "analyst"
    })
    print(f"Bootstrap non-admin registration: {resp.status_code} (expected 403) -> {resp.json()}")
    
    # 2. Register first admin
    resp = client.post("/api/v1/auth/register", json={
        "name": "Admin User", "email": "admin@example.com", "password": "AdminPassword123", "role": "administrator"
    })
    print(f"Bootstrap admin registration: {resp.status_code} (expected 201) -> {resp.json()}")
    admin_id = resp.json().get("user_id")
    
    # 3. Duplicate email registration
    resp = client.post("/api/v1/auth/register", json={
        "name": "Admin Dup", "email": "admin@example.com", "password": "AdminPassword123", "role": "administrator"
    })
    print(f"Duplicate email registration: {resp.status_code} (expected 409) -> {resp.json()}")
    
    # 4. Anonymous registration when users already exist
    resp = client.post("/api/v1/auth/register", json={
        "name": "Second User", "email": "user2@example.com", "password": "UserPassword123", "role": "analyst"
    })
    print(f"Anonymous registration after bootstrap: {resp.status_code} (expected 401) -> {resp.json()}")
    
    # 5. Login
    resp = client.post("/api/v1/auth/login", json={"email": "admin@example.com", "password": "AdminPassword123"})
    print(f"Login success: {resp.status_code} -> role: {resp.json().get('role')}")
    admin_token = resp.json().get("access_token")
    
    # 6. Invalid credentials (generic error)
    resp = client.post("/api/v1/auth/login", json={"email": "admin@example.com", "password": "WrongPassword999"})
    print(f"Login wrong password: {resp.status_code} (expected 401) -> {resp.json()}")
    resp = client.post("/api/v1/auth/login", json={"email": "nonexistent@example.com", "password": "AdminPassword123"})
    print(f"Login unknown email: {resp.status_code} (expected 401) -> {resp.json()}")
    
    # 7. Admin creates an Analyst
    headers_admin = {"Authorization": f"Bearer {admin_token}"}
    resp = client.post("/api/v1/auth/register", json={
        "name": "Analyst User", "email": "analyst@example.com", "password": "AnalystPassword123", "role": "analyst"
    }, headers=headers_admin)
    print(f"Admin creates analyst: {resp.status_code} (expected 201) -> {resp.json()}")
    
    # Login as analyst
    resp = client.post("/api/v1/auth/login", json={"email": "analyst@example.com", "password": "AnalystPassword123"})
    analyst_token = resp.json().get("access_token")
    headers_analyst = {"Authorization": f"Bearer {analyst_token}"}
    
    # Analyst tries to create user
    resp = client.post("/api/v1/auth/register", json={
        "name": "User 3", "email": "user3@example.com", "password": "UserPassword123", "role": "reviewer"
    }, headers=headers_analyst)
    print(f"Analyst tries to create user: {resp.status_code} (expected 403) -> {resp.json()}")

    # 8. Token tests: no token, garbage token, expired token, deleted user token
    resp = client.get("/api/v1/auth/me")
    print(f"GET /auth/me with NO token: {resp.status_code} (expected 401)")
    
    resp = client.get("/api/v1/auth/me", headers={"Authorization": "Bearer not-a-valid-token"})
    print(f"GET /auth/me with GARBAGE token: {resp.status_code} (expected 401) -> {resp.json()}")
    
    # Expired token
    expired_payload = {
        "sub": str(admin_id),
        "role": "administrator",
        "iat": datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=2),
        "exp": datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=1),
    }
    expired_token = jwt.encode(expired_payload, settings.secret_key, algorithm="HS256")
    resp = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {expired_token}"})
    print(f"GET /auth/me with EXPIRED token: {resp.status_code} (expected 401) -> {resp.json()}")
    
    # Token for deleted user (e.g. sub=99999)
    deleted_payload = {
        "sub": "99999",
        "role": "administrator",
        "iat": datetime.datetime.now(datetime.timezone.utc),
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1),
    }
    deleted_token = jwt.encode(deleted_payload, settings.secret_key, algorithm="HS256")
    resp = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {deleted_token}"})
    print(f"GET /auth/me with DELETED user token: {resp.status_code} (expected 401) -> {resp.json()}")

    print("\n=== 7.3 ADVERSARIAL VALIDATION AUDIT ===")
    # Password too short (<8)
    resp = client.post("/api/v1/auth/register", json={
        "name": "Short", "email": "short@example.com", "password": "Pass1", "role": "analyst"
    }, headers=headers_admin)
    print(f"Password < 8 chars validation: {resp.status_code} -> body: {resp.json()}")
    
    # Password with no digits
    resp = client.post("/api/v1/auth/register", json={
        "name": "NoDigits", "email": "nodigit@example.com", "password": "PasswordOnly", "role": "analyst"
    }, headers=headers_admin)
    print(f"Password without digits validation: {resp.status_code} -> body: {resp.json()}")
    
    # Extra unexpected fields (ConfigDict(extra='forbid'))
    resp = client.post("/api/v1/auth/register", json={
        "name": "Extra", "email": "extra@example.com", "password": "Password123", "role": "analyst", "is_admin": True
    }, headers=headers_admin)
    print(f"Extra payload field validation: {resp.status_code} -> body: {resp.json()}")

    print("\n=== 7.4 RATE LIMITING AUDIT ===")
    # Send 50 rapid login requests
    rate_limited = False
    for i in range(50):
        resp = client.post("/api/v1/auth/login", json={"email": "admin@example.com", "password": "WrongPassword"})
        if resp.status_code == 429:
            rate_limited = True
            print(f"Rate limiting triggered on request {i+1} with 429!")
            break
    if not rate_limited:
        print("Rate limiting: NOT ENFORCED (50 consecutive requests all returned 401, no 429 returned)")

    print("\n=== 7.5 HEALTH ENDPOINT AUDIT ===")
    resp = client.get("/health")
    print(f"GET /health: {resp.status_code} -> body: {resp.json()}")
    
if __name__ == "__main__":
    run_api_audit()
