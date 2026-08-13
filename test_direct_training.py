"""Direct test of run_training_direct to catch the real error"""
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'backend'))
os.chdir(os.getcwd())  # ensure CWD is project root

import logging
logging.basicConfig(level=logging.DEBUG)

from app.core.config import settings
from app.workers.training_tasks import _get_sync_db_url, run_training_direct
from app.core.storage import upload_file
import io, requests

# Get the sync DB URL
sync_url = _get_sync_db_url()
print(f"\n=== SYNC DB URL: {sync_url} ===\n")

# Generate a mini CSV dataset directly (no API call needed)
csv_content = """area_sqft,bedrooms,bathrooms,location,year_built,garage,price
1200,2,1,Urban,2000,0,250000
1800,3,2,Suburbs,2010,1,380000
2500,4,2,Downtown,2015,2,520000
1000,1,1,Urban,1995,0,200000
3000,4,3,Suburbs,2020,2,650000
2200,3,2,Metropolitan,2012,1,450000
1600,2,2,Urban,2005,1,310000
2800,4,3,Downtown,2018,2,600000
1400,3,1,Suburbs,2008,0,290000
2000,3,2,Urban,2014,1,410000
1100,2,1,Urban,1998,0,220000
2600,4,2,Suburbs,2016,2,540000
1900,3,2,Downtown,2011,1,400000
2300,3,2,Metropolitan,2013,1,470000
1700,2,2,Urban,2007,1,340000
"""

# Upload to local storage (MinIO fallback)
file_bytes = csv_content.encode("utf-8")
object_key = "test/house_price_direct_test.csv"
try:
    upload_file(settings.MINIO_BUCKET_DATASETS, object_key, io.BytesIO(file_bytes), len(file_bytes))
    print("✅ File uploaded to storage")
except Exception as e:
    print(f"❌ Upload failed: {e}")
    sys.exit(1)

# Get experiment_id = 1 (first one that succeeded training)
# Actually we need to create one first via the DB
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models.models import Experiment, JobStatus, ModuleCategory, User
from sqlalchemy import select as sa_select

sync_engine = create_engine(sync_url, connect_args={"check_same_thread": False})
Session = sessionmaker(bind=sync_engine)
session = Session()

# Find the most recent user
users = session.execute(sa_select(User)).scalars().all()
print(f"Users in DB: {[(u.id, u.email) for u in users]}")
if not users:
    print("No users found! Run the test_training.py first to create a user.")
    session.close()
    sys.exit(1)

user = users[-1]

# Create a new experiment
exp = Experiment(
    owner_id=user.id,
    module_id="house-price-prediction",
    module_category=ModuleCategory.BEGINNER_ML,
    name="Direct Test Experiment",
    status=JobStatus.PENDING,
    target_column="price",
    algorithm="random_forest",
    hyperparameters={},
    config={"test_size": 0.2, "file_type": "csv", "object_key": object_key},
)
session.add(exp)
session.commit()
exp_id = exp.id
print(f"✅ Created experiment id={exp_id}")
session.close()
sync_engine.dispose()

# Now run training directly
print(f"\nRunning training directly for experiment_id={exp_id}...")
try:
    run_training_direct(
        experiment_id=exp_id,
        module_id="house-price-prediction",
        object_key=object_key,
        file_type="csv",
        target_column="price",
        algorithm="random_forest",
        hyperparameters={},
        test_size=0.2,
    )
    print("✅ run_training_direct completed without raising an exception")
except Exception as e:
    print(f"❌ run_training_direct raised: {e}")
    import traceback; traceback.print_exc()

# Verify DB updated
sync_engine2 = create_engine(sync_url, connect_args={"check_same_thread": False})
Session2 = sessionmaker(bind=sync_engine2)
session2 = Session2()
exp_check = session2.query(Experiment).filter(Experiment.id == exp_id).first()
print(f"\n=== DB Experiment State ===")
print(f"  Status: {exp_check.status}")
print(f"  Metrics: {exp_check.metrics}")
print(f"  Feature Importance keys: {list(exp_check.feature_importance.keys()) if exp_check.feature_importance else None}")
print(f"  Model Key: {exp_check.model_minio_key}")
session2.close()
sync_engine2.dispose()
