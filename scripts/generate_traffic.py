"""
generate_traffic.py — Fire 400 prediction requests at the FastAPI
to seed Prometheus metrics with realistic time-series data.
"""

import json
import random
import time
import urllib.request

API_URL = "http://localhost:8000"

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

BASE_PROFILES = [
    {
        "country": "INDIA", "online_consumer": "YES",
        "age_group": "Millennials", "annual_salary_band": "Medium High",
        "gender": "Female", "education": "University Graduate",
        "payment_method_card": "YES", "living_region": "Metropolitan",
        "online_service_preference": "YES", "ai_endorsement": "YES",
        "ai_privacy_no_trust": "NO", "ai_enhance_experience": "YES",
        "ai_tool_chatbots": "YES", "ai_tool_virtual_assistant": "NO",
        "ai_tool_voice_photo_search": "YES", "payment_method_cod": "NO",
        "payment_method_ewallet": "YES", "product_category_appliances": "NO",
        "product_category_electronics": "YES", "product_category_groceries": "YES",
        "product_category_personal_care": "YES", "product_category_clothing": "NO",
    },
    {
        "country": "CHINA", "online_consumer": "YES",
        "age_group": "Gen Z", "annual_salary_band": "Low",
        "gender": "Male", "education": "Highschool Graduate",
        "payment_method_card": "NO", "living_region": "Suburban Areas",
        "online_service_preference": "NO", "ai_endorsement": "NO",
        "ai_privacy_no_trust": "YES", "ai_enhance_experience": "NO",
        "ai_tool_chatbots": "NO", "ai_tool_virtual_assistant": "NO",
        "ai_tool_voice_photo_search": "NO", "payment_method_cod": "YES",
        "payment_method_ewallet": "NO", "product_category_appliances": "YES",
        "product_category_electronics": "NO", "product_category_groceries": "YES",
        "product_category_personal_care": "NO", "product_category_clothing": "YES",
    },
    {
        "country": "CANADA", "online_consumer": "YES",
        "age_group": "Gen X", "annual_salary_band": "High",
        "gender": "Prefer not to say", "education": "Doctorate Degree",
        "payment_method_card": "YES", "living_region": "Rural Areas",
        "online_service_preference": "YES", "ai_endorsement": "YES",
        "ai_privacy_no_trust": "NO", "ai_enhance_experience": "YES",
        "ai_tool_chatbots": "YES", "ai_tool_virtual_assistant": "YES",
        "ai_tool_voice_photo_search": "YES", "payment_method_cod": "NO",
        "payment_method_ewallet": "YES", "product_category_appliances": "NO",
        "product_category_electronics": "YES", "product_category_groceries": "NO",
        "product_category_personal_care": "YES", "product_category_clothing": "YES",
    },
    {
        "country": "INDIA", "online_consumer": "NO",
        "age_group": "Baby Boomers", "annual_salary_band": "Medium",
        "gender": "Female", "education": "Masters' Degree",
        "payment_method_card": "YES", "living_region": "Metropolitan",
        "online_service_preference": "YES", "ai_endorsement": "NO",
        "ai_privacy_no_trust": "YES", "ai_enhance_experience": "NO",
        "ai_tool_chatbots": "NO", "ai_tool_virtual_assistant": "YES",
        "ai_tool_voice_photo_search": "NO", "payment_method_cod": "YES",
        "payment_method_ewallet": "NO", "product_category_appliances": "YES",
        "product_category_electronics": "YES", "product_category_groceries": "NO",
        "product_category_personal_care": "NO", "product_category_clothing": "YES",
    },
    {
        "country": "CANADA", "online_consumer": "YES",
        "age_group": "Millennials", "annual_salary_band": "Medium High",
        "gender": "Male", "education": "University Graduate",
        "payment_method_card": "YES", "living_region": "Suburban Areas",
        "online_service_preference": "YES", "ai_endorsement": "YES",
        "ai_privacy_no_trust": "NO", "ai_enhance_experience": "YES",
        "ai_tool_chatbots": "YES", "ai_tool_virtual_assistant": "YES",
        "ai_tool_voice_photo_search": "NO", "payment_method_cod": "NO",
        "payment_method_ewallet": "YES", "product_category_appliances": "YES",
        "product_category_electronics": "YES", "product_category_groceries": "NO",
        "product_category_personal_care": "YES", "product_category_clothing": "NO",
    },
]


def send_requests(n: int = 400) -> None:
    rng = random.Random(2025)
    success = 0
    print(f"Sending {n} prediction requests to {API_URL} …", flush=True)

    for i in range(n):
        profile = dict(rng.choice(BASE_PROFILES))
        profile["country"] = rng.choice(COUNTRIES)
        profile["age_group"] = rng.choice(AGE_GROUPS)
        profile["annual_salary_band"] = rng.choice(SALARIES)
        profile["gender"] = rng.choice(GENDERS)
        profile["education"] = rng.choice(EDUCATIONS)
        profile["living_region"] = rng.choice(REGIONS)

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
        except Exception as e:
            if i < 5:
                print(f"  [!] req {i}: {e}")

        # small gaps for time-series spread
        if i % 100 == 99:
            print(f"  {i + 1}/{n} sent ({success} ok)", flush=True)
            time.sleep(2)

    print(f"Done: {success}/{n} succeeded.")


if __name__ == "__main__":
    send_requests(400)
