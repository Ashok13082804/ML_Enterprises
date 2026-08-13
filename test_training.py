"""Quick E2E test for MLVerse X API: generate-dataset, train, status, predict"""
import requests
import time
import sys

BASE = "http://localhost:8000/api/v1"

# Login
print("1. Logging in...")
resp = requests.post(f"{BASE}/auth/login", json={"email": "testuser2@mlverse.ai", "password": "Test@123456"})
if resp.status_code != 200:
    print(f"  Login failed: {resp.text}")
    sys.exit(1)
token = resp.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}
print(f"  ✅ Logged in. Token: {token[:20]}...")

# Generate Dataset
print("\n2. Generating AI dataset for house-price-prediction (80 rows)...")
resp = requests.post(f"{BASE}/modules/house-price-prediction/generate-dataset", 
                     headers=headers, params={"num_rows": 80})
if resp.status_code != 200:
    print(f"  Error: {resp.text}")
    sys.exit(1)
data = resp.json()
print(f"  ✅ Generated {data['num_rows']} rows")
print(f"  Columns: {data['feature_columns']} + [{data['target_column']}]")
print(f"  Sample row: {data['records'][0]}")
csv_content = data["csv_content"]
target_col = data["target_column"]

# Upload CSV + Train
print("\n3. Starting training...")
csv_bytes = csv_content.encode("utf-8")
files = {"file": ("house_price_data.csv", csv_bytes, "text/csv")}
form_data = {
    "experiment_name": "E2E Test Training",
    "target_column": target_col,
    "algorithm": "random_forest",
}
resp = requests.post(f"{BASE}/modules/house-price-prediction/train", 
                     headers=headers, files=files, data=form_data)
if resp.status_code != 200:
    print(f"  Training start failed: {resp.status_code} {resp.text[:400]}")
    sys.exit(1)
train_resp = resp.json()
exp_id = train_resp["experiment_id"]
status = train_resp["status"]
print(f"  ✅ Training response: experiment_id={exp_id}, status={status}")

# Poll status if not immediately complete
if status != "completed":
    print("  Polling training status...")
    for i in range(30):  # up to 30 seconds
        time.sleep(1)
        resp = requests.get(f"{BASE}/modules/house-price-prediction/experiments/{exp_id}/status", headers=headers)
        s = resp.json().get("status")
        print(f"  [{i+1}s] Status: {s}")
        if s == "completed":
            status = "completed"
            break
        elif s == "failed":
            print(f"  ❌ Training failed: {resp.json()}")
            sys.exit(1)

if status == "completed":
    # Get final status with metrics
    resp = requests.get(f"{BASE}/modules/house-price-prediction/experiments/{exp_id}/status", headers=headers)
    result = resp.json()
    print(f"\n4. ✅ Training complete!")
    print(f"   Metrics: {result.get('metrics')}")
    print(f"   Feature Importance: {str(result.get('feature_importance'))[:100]}")
    print(f"   CV Scores: {result.get('training_history')}")
    
    # Test Prediction
    print("\n5. Testing prediction...")
    pred_resp = requests.post(f"{BASE}/modules/house-price-prediction/predict", 
                              headers=headers, json={
                                  "experiment_id": exp_id,
                                  "input_data": {
                                      "area_sqft": 2500,
                                      "bedrooms": 3,
                                      "bathrooms": 2,
                                      "location": "Urban",
                                      "year_built": 2010,
                                      "garage": 1
                                  }
                              })
    if pred_resp.status_code == 200:
        p = pred_resp.json()
        print(f"   ✅ Prediction: {p.get('prediction')} | Confidence: {p.get('confidence')}")
    else:
        print(f"   ❌ Prediction failed: {pred_resp.text[:200]}")
else:
    print("  ⚠️  Training did not complete in time")

print("\n✅ All API tests done!")
