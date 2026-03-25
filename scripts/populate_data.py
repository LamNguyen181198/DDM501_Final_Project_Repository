"""
populate_data.py — Seed the project with synthetic data + trigger pipeline

Steps:
  1. Generate 2 500-row synthetic CSV → data/raw/online_shopping_ai.csv
  2. Trigger the Airflow DAG via REST API (waits for completion)
  3. Fire 300 prediction requests at the FastAPI to seed Prometheus
"""

import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent
DATA_RAW = REPO / "data" / "raw" / "online_shopping_ai.csv"
AIRFLOW_URL = "http://localhost:8080"
API_URL = "http://localhost:8000"
DAG_ID = "satisfaction_model_training"

# ── valid category values (mirror pipeline/data_ingestion.py) ──────────────
COUNTRIES = ["CANADA", "CHINA", "INDIA"]
AGE_GROUPS = ["Gen Z", "Millennials", "Gen X", "Baby Boomers"]
SALARIES = ["Low", "Medium", "Medium High", "High"]
EDUCATIONS = [
    "Highschool Graduate",
    "University Graduate",
    "Masters' Degree",
    "Doctorate Degree",
]
GENDERS = ["Female", "Male", "Prefer not to say"]
REGIONS = ["Metropolitan", "Suburban Areas", "Rural Areas"]
YES_NO = ["YES", "NO"]
TARGETS = ["Satisfied", "Unsatisfied"]


# ── 1. Generate synthetic dataset ─────────────────────────────────────────

def generate_raw_csv(n: int = 2500, seed: int = 42) -> None:
    """Generate plausible synthetic rows using the RAW column names that
    data_ingestion.SOURCE_COLUMN_MAP expects."""
    rng = random.Random(seed)
    np.random.seed(seed)
    DATA_RAW.parent.mkdir(parents=True, exist_ok=True)

    def yn(p_yes: float = 0.6) -> str:
        return "YES" if rng.random() < p_yes else "NO"

    rows = []
    for _ in range(n):
        country = rng.choice(COUNTRIES)
        age = rng.choice(AGE_GROUPS)
        salary = rng.choice(SALARIES)
        education = rng.choice(EDUCATIONS)
        gender = rng.choice(GENDERS)
        region = rng.choice(REGIONS)

        # correlated satisfaction: higher salary + more tech tools → more satisfied
        sat_score = 0
        if salary in ("Medium High", "High"):
            sat_score += 2
        if education in ("Masters' Degree", "Doctorate Degree"):
            sat_score += 1
        ai_chat = yn(0.55)
        ai_va = yn(0.5)
        ai_vps = yn(0.45)
        ai_enrich = yn(0.6)
        if ai_chat == "YES":
            sat_score += 1
        if ai_enrich == "YES":
            sat_score += 1
        target = "Satisfied" if rng.random() < (0.3 + sat_score * 0.08) else "Unsatisfied"

        rows.append(
            {
                "country": country,
                "online_consumer": yn(0.9),
                "age": age,
                "annual_salary": salary,
                "gender": gender,
                "education": education,
                "payment_method_credit_debit": yn(0.55),
                "living_region": region,
                "online_service_preference": yn(0.7),
                "ai_endorsement": yn(0.5),
                "ai_privacy_no_trust": yn(0.35),
                "ai_enhance_experience": ai_enrich,
                "ai_satisfication": target,
                "ai_tools_used_chatbots": ai_chat,
                "ai_tools_used_virtual_assistant": ai_va,
                "ai_tools_used_voice_photo_search": ai_vps,
                "payment_method_cod": yn(0.4),
                "payment_method_ewallet": yn(0.6),
                "product_category_appliances": yn(0.35),
                "product_category_electronics": yn(0.55),
                "product_category_groceries": yn(0.7),
                "product_category_personal_care": yn(0.45),
                "product_category_clothing": yn(0.6),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(DATA_RAW, index=False)
    sat_pct = (df["ai_satisfication"] == "Satisfied").mean() * 100
    print(
        f"[1/3] ✅  Generated {len(df)} rows → {DATA_RAW}"
        f"  (satisfied: {sat_pct:.1f}%)"
    )


# ── 2. Trigger Airflow DAG ─────────────────────────────────────────────────

def _airflow_request(path: str, method: str = "GET", body: dict | None = None) -> dict:
    url = f"{AIRFLOW_URL}/api/v1{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    # Basic auth: admin / admin
    import base64
    creds = base64.b64encode(b"admin:admin").decode()
    req.add_header("Authorization", f"Basic {creds}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def trigger_dag_and_wait(max_wait: int = 600) -> None:
    """Trigger DAG, then poll until it finishes (success or failure)."""
    run_id = f"populate_data_{int(time.time())}"
    print(f"[2/3] ⏳  Triggering DAG '{DAG_ID}' (run_id={run_id}) …")

    # trigger
    _airflow_request(
        f"/dags/{DAG_ID}/dagRuns",
        method="POST",
        body={"dag_run_id": run_id, "conf": {}},
    )
    print(f"       DAG run submitted. Waiting up to {max_wait}s …")

    deadline = time.time() + max_wait
    dot_count = 0
    while time.time() < deadline:
        time.sleep(15)
        try:
            info = _airflow_request(f"/dags/{DAG_ID}/dagRuns/{run_id}")
            state = info.get("state", "unknown")
        except urllib.error.HTTPError as exc:
            print(f"  poll error: {exc}")
            continue

        dot_count += 1
        print(f"  [{dot_count:>3}] state = {state}", flush=True)

        if state == "success":
            print("[2/3] ✅  DAG run completed successfully.")
            return
        if state in ("failed", "upstream_failed"):
            print(f"[2/3] ❌  DAG run ended with state '{state}'.")
            return

    print("[2/3] ⚠️  Timed out waiting for DAG. Check Airflow UI.")


# ── 3. Generate API traffic (seed Prometheus) ──────────────────────────────

# Representative prediction records (vary country/age/education)
SAMPLE_PROFILES = [
    {
        "country": "INDIA",
        "online_consumer": "YES",
        "age_group": "Millennials",
        "annual_salary_band": "Medium High",
        "gender": "Female",
        "education": "University Graduate",
        "payment_method_card": "YES",
        "living_region": "Metropolitan",
        "online_service_preference": "YES",
        "ai_endorsement": "YES",
        "ai_privacy_no_trust": "NO",
        "ai_enhance_experience": "YES",
        "ai_tool_chatbots": "YES",
        "ai_tool_virtual_assistant": "NO",
        "ai_tool_voice_photo_search": "YES",
        "payment_method_cod": "NO",
        "payment_method_ewallet": "YES",
        "product_category_appliances": "NO",
        "product_category_electronics": "YES",
        "product_category_groceries": "YES",
        "product_category_personal_care": "YES",
        "product_category_clothing": "NO",
    },
    {
        "country": "CHINA",
        "online_consumer": "YES",
        "age_group": "Gen Z",
        "annual_salary_band": "Low",
        "gender": "Male",
        "education": "Highschool Graduate",
        "payment_method_card": "NO",
        "living_region": "Suburban Areas",
        "online_service_preference": "NO",
        "ai_endorsement": "NO",
        "ai_privacy_no_trust": "YES",
        "ai_enhance_experience": "NO",
        "ai_tool_chatbots": "NO",
        "ai_tool_virtual_assistant": "NO",
        "ai_tool_voice_photo_search": "NO",
        "payment_method_cod": "YES",
        "payment_method_ewallet": "NO",
        "product_category_appliances": "YES",
        "product_category_electronics": "NO",
        "product_category_groceries": "YES",
        "product_category_personal_care": "NO",
        "product_category_clothing": "YES",
    },
    {
        "country": "CANADA",
        "online_consumer": "YES",
        "age_group": "Gen X",
        "annual_salary_band": "High",
        "gender": "Prefer not to say",
        "education": "Doctorate Degree",
        "payment_method_card": "YES",
        "living_region": "Rural Areas",
        "online_service_preference": "YES",
        "ai_endorsement": "YES",
        "ai_privacy_no_trust": "NO",
        "ai_enhance_experience": "YES",
        "ai_tool_chatbots": "YES",
        "ai_tool_virtual_assistant": "YES",
        "ai_tool_voice_photo_search": "YES",
        "payment_method_cod": "NO",
        "payment_method_ewallet": "YES",
        "product_category_appliances": "NO",
        "product_category_electronics": "YES",
        "product_category_groceries": "NO",
        "product_category_personal_care": "YES",
        "product_category_clothing": "YES",
    },
    {
        "country": "INDIA",
        "online_consumer": "NO",
        "age_group": "Baby Boomers",
        "annual_salary_band": "Medium",
        "gender": "Female",
        "education": "Masters' Degree",
        "payment_method_card": "YES",
        "living_region": "Metropolitan",
        "online_service_preference": "YES",
        "ai_endorsement": "NO",
        "ai_privacy_no_trust": "YES",
        "ai_enhance_experience": "NO",
        "ai_tool_chatbots": "NO",
        "ai_tool_virtual_assistant": "YES",
        "ai_tool_voice_photo_search": "NO",
        "payment_method_cod": "YES",
        "payment_method_ewallet": "NO",
        "product_category_appliances": "YES",
        "product_category_electronics": "YES",
        "product_category_groceries": "NO",
        "product_category_personal_care": "NO",
        "product_category_clothing": "YES",
    },
]


def generate_api_traffic(n_requests: int = 300) -> None:
    """Fire single-prediction requests at the API to populate Prometheus."""
    print(f"[3/3] ⏳  Sending {n_requests} prediction requests …", flush=True)
    success = 0
    rng = random.Random(99)
    for i in range(n_requests):
        profile = dict(rng.choice(SAMPLE_PROFILES))
        # small random variation on country/age to spread metrics
        profile["country"] = rng.choice(COUNTRIES)
        profile["age_group"] = rng.choice(AGE_GROUPS)
        profile["annual_salary_band"] = rng.choice(SALARIES)

        payload = json.dumps(profile).encode()
        req = urllib.request.Request(
            f"{API_URL}/api/v1/predict",
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                resp.read()
            success += 1
        except Exception as exc:
            if i < 3:
                print(f"  request {i} failed: {exc}")

        # small delay to get time-series spread on Prometheus
        if i % 50 == 49:
            print(f"  … {i+1}/{n_requests} done ({success} ok)", flush=True)
            time.sleep(1)

    print(f"[3/3] ✅  Sent {n_requests} requests → {success} successful.")


# ── main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  DDM501 — Populate data (DAG + MLflow + Prometheus)")
    print("=" * 60)

    generate_raw_csv()
    trigger_dag_and_wait(max_wait=600)
    generate_api_traffic(n_requests=300)

    print("\n✅  Done. Check:")
    print(f"   • Airflow   → http://localhost:8080  (DAG runs)")
    print(f"   • MLflow    → http://localhost:5001  (experiments)")
    print(f"   • API       → http://localhost:8000/metrics  (Prometheus metrics)")
    print(f"   • Grafana   → http://localhost:3000  (dashboards)")
    print(f"   • Prometheus→ http://localhost:9090  (targets & graphs)")
