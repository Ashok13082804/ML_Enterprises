"""
MLVerse X — Celery App & Training Task
"""
from celery import Celery
import logging
import time
import io
import uuid
import socket
import os
from datetime import datetime, timezone

from app.core.config import settings

logger = logging.getLogger(__name__)


def _get_sync_db_url() -> str:
    """Return correct sync DB URL — falls back to SQLite if PostgreSQL is not reachable."""
    db_url = settings.DATABASE_URL_SYNC
    if "localhost:5432" in db_url or "postgres:5432" in db_url:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            result = sock.connect_ex(("localhost", 5432))
            sock.close()
            if result != 0:
                # Use CWD-relative path — same as how async engine creates `sqlite+aiosqlite:///./mlverse.db`
                db_path = os.path.abspath("mlverse.db")
                logger.info(f"PostgreSQL not reachable. Using SQLite fallback: {db_path}")
                return f"sqlite:///{db_path}"
        except Exception:
            db_path = os.path.abspath("mlverse.db")
            return f"sqlite:///{db_path}"
    return db_url

celery_app = Celery(
    "mlverse",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_soft_time_limit=3600,  # 1 hour
    task_time_limit=7200,       # 2 hours hard limit
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)


@celery_app.task(bind=True, name="mlverse.train_model")
def train_model_task(
    self,
    experiment_id: int,
    module_id: str,
    object_key: str,
    file_type: str,
    target_column: str,
    algorithm: str,
    hyperparameters: dict,
    test_size: float = 0.2,
):
    """
    Async Celery task for ML model training.
    Updates experiment status and results in the database.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.models.models import Experiment, JobStatus
    from app.ml.module_registry import get_module
    from app.ml.pipeline_engine import MLPipelineEngine
    from app.core.storage import download_file, upload_file

    # Sync DB session for Celery worker
    sync_db_url = _get_sync_db_url()
    connect_args = {"check_same_thread": False} if "sqlite" in sync_db_url else {}
    sync_engine = create_engine(sync_db_url, pool_pre_ping=True, connect_args=connect_args)
    Session = sessionmaker(bind=sync_engine)
    session = Session()

    try:
        # Update status to running
        exp = session.query(Experiment).filter(Experiment.id == experiment_id).first()
        if not exp:
            logger.error(f"Experiment {experiment_id} not found")
            return

        exp.status = JobStatus.RUNNING
        exp.started_at = datetime.utcnow()  # naive UTC — SQLite strips timezone info on roundtrip
        session.commit()

        # Update Celery task state
        self.update_state(state="PROGRESS", meta={"stage": "loading_data", "percent": 10})

        # Load file from MinIO
        file_bytes = download_file(settings.MINIO_BUCKET_DATASETS, object_key)

        self.update_state(state="PROGRESS", meta={"stage": "preprocessing", "percent": 25})

        # Get module config
        module = get_module(module_id)
        if not module:
            raise ValueError(f"Module {module_id} not found")

        # Train
        engine = MLPipelineEngine(module)
        self.update_state(state="PROGRESS", meta={"stage": "training", "percent": 50})

        results = engine.train(
            file_bytes=file_bytes,
            file_type=file_type,
            target_column=target_column,
            algorithm=algorithm,
            hyperparameters=hyperparameters,
            test_size=test_size,
        )

        self.update_state(state="PROGRESS", meta={"stage": "evaluating", "percent": 80})

        # Save model to MinIO
        model_bytes = engine.save_model()
        model_key = f"models/{exp.owner_id}/{module_id}/{experiment_id}/{algorithm}.pkl"
        upload_file(
            settings.MINIO_BUCKET_MODELS,
            model_key,
            io.BytesIO(model_bytes),
            len(model_bytes),
            content_type="application/octet-stream",
        )

        self.update_state(state="PROGRESS", meta={"stage": "saving", "percent": 95})

        # Update experiment with results
        completed_at = datetime.utcnow()  # naive UTC — consistent with SQLite roundtrip
        started_at = exp.started_at
        # Make both naive for subtraction (SQLite may strip timezone on reload)
        if hasattr(started_at, 'tzinfo') and started_at.tzinfo is not None:
            started_at = started_at.replace(tzinfo=None)
        exp.status = JobStatus.COMPLETED
        exp.completed_at = completed_at
        exp.duration_seconds = (completed_at - started_at).total_seconds()
        exp.metrics = results["metrics"]
        exp.feature_importance = results.get("feature_importance")
        exp.confusion_matrix = results.get("confusion_matrix")
        exp.training_history = results.get("cv_scores")
        exp.predictions_preview = {"samples": results.get("sample_predictions", [])}
        exp.model_minio_key = model_key
        exp.model_version = f"v1.{experiment_id}"
        session.commit()

        logger.info(f"Training completed for experiment {experiment_id}: {results['metrics']}")
        return {"status": "completed", "experiment_id": experiment_id, "metrics": results["metrics"]}

    except Exception as e:
        logger.error(f"Training failed for experiment {experiment_id}: {e}", exc_info=True)
        if exp:
            exp.status = JobStatus.FAILED
            exp.training_history = {"error": str(e)}
            session.commit()
        raise self.retry(exc=e, countdown=0, max_retries=0)
    finally:
        session.close()
        sync_engine.dispose()


def run_training_direct(
    experiment_id: int,
    module_id: str,
    object_key: str,
    file_type: str,
    target_column: str,
    algorithm: str,
    hyperparameters: dict,
    test_size: float = 0.2,
):
    """
    Direct background training function (used when Celery/Redis is offline).
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.models.models import Experiment, JobStatus
    from app.ml.module_registry import get_module
    from app.ml.pipeline_engine import MLPipelineEngine
    from app.core.storage import download_file, upload_file

    sync_db_url = _get_sync_db_url()
    connect_args = {"check_same_thread": False} if "sqlite" in sync_db_url else {}
    sync_engine = create_engine(sync_db_url, pool_pre_ping=True, connect_args=connect_args)
    Session = sessionmaker(bind=sync_engine)
    session = Session()

    try:
        exp = session.query(Experiment).filter(Experiment.id == experiment_id).first()
        if not exp:
            logger.error(f"Direct training: Experiment {experiment_id} not found")
            return

        exp.status = JobStatus.RUNNING
        exp.started_at = datetime.utcnow()  # naive UTC — SQLite strips timezone info on roundtrip
        session.commit()

        file_bytes = download_file(settings.MINIO_BUCKET_DATASETS, object_key)
        module = get_module(module_id)
        if not module:
            raise ValueError(f"Module {module_id} not found")

        engine = MLPipelineEngine(module)
        results = engine.train(
            file_bytes=file_bytes,
            file_type=file_type,
            target_column=target_column,
            algorithm=algorithm,
            hyperparameters=hyperparameters,
            test_size=test_size,
        )

        model_bytes = engine.save_model()
        model_key = f"models/{exp.owner_id}/{module_id}/{experiment_id}/{algorithm}.pkl"
        upload_file(
            settings.MINIO_BUCKET_MODELS,
            model_key,
            io.BytesIO(model_bytes),
            len(model_bytes),
            content_type="application/octet-stream",
        )

        completed_at = datetime.utcnow()  # naive UTC — consistent with SQLite roundtrip
        started_at = exp.started_at
        # Make both naive for subtraction (SQLite may strip timezone on reload)
        if hasattr(started_at, 'tzinfo') and started_at.tzinfo is not None:
            started_at = started_at.replace(tzinfo=None)
        exp.status = JobStatus.COMPLETED
        exp.completed_at = completed_at
        exp.duration_seconds = (completed_at - started_at).total_seconds()
        exp.metrics = results["metrics"]
        exp.feature_importance = results.get("feature_importance")
        exp.confusion_matrix = results.get("confusion_matrix")
        exp.training_history = results.get("cv_scores")
        exp.predictions_preview = {"samples": results.get("sample_predictions", [])}
        exp.model_minio_key = model_key
        exp.model_version = f"v1.{experiment_id}"
        session.commit()

        logger.info(f"Direct training completed for experiment {experiment_id}: {results['metrics']}")

    except Exception as e:
        logger.error(f"Direct training failed for experiment {experiment_id}: {e}", exc_info=True)
        if 'exp' in locals() and exp:
            exp.status = JobStatus.FAILED
            exp.training_history = {"error": str(e)}
            session.commit()
    finally:
        session.close()
        sync_engine.dispose()

