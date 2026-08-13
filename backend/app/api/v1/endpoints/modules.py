"""
MLVerse X — Universal Module API
Handles training, prediction, evaluation, history, and reports for all 100 ML modules.
"""
import io
import uuid
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks, WebSocket
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from pydantic import BaseModel
import json

from app.core.database import get_db
from app.core.storage import upload_file, download_file
from app.core.config import settings
from app.models.models import User, Experiment, Prediction, Dataset, JobStatus, ModuleCategory
from app.api.v1.endpoints.auth import get_current_active_user
from app.ml.module_registry import MODULE_REGISTRY, get_module, get_all_modules, get_modules_by_category
from app.ml.pipeline_engine import MLPipelineEngine
from app.workers.training_tasks import train_model_task

router = APIRouter()


# ─── Schemas ───────────────────────────────────────────────────────────────────
class TrainRequest(BaseModel):
    module_id: str
    experiment_name: str
    dataset_id: Optional[int] = None
    target_column: str
    feature_columns: Optional[List[str]] = None
    algorithm: Optional[str] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    test_size: float = 0.2


class PredictRequest(BaseModel):
    experiment_id: int
    input_data: Dict[str, Any]


class BatchPredictRequest(BaseModel):
    experiment_id: int
    inputs: List[Dict[str, Any]]


# ─── List All Modules ──────────────────────────────────────────────────────────
@router.get("/")
async def list_modules(category: Optional[str] = None):
    """List all 100 ML modules, optionally filtered by category."""
    if category:
        try:
            cat = ModuleCategory(category)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid category: {category}")
        modules = get_modules_by_category(cat)
    else:
        modules = get_all_modules()

    return {
        "total": len(modules),
        "modules": [
            {
                "id": m.id,
                "name": m.name,
                "category": m.category.value,
                "task_type": m.task_type.value,
                "description": m.description,
                "icon": m.icon,
                "tags": m.tags,
                "input_type": m.input_type,
                "default_algorithms": m.default_algorithms,
                "color": m.color,
                "supports_batch_predict": m.supports_batch_predict,
                "supports_realtime": m.supports_realtime,
            }
            for m in modules
        ],
    }


# ─── Get Module Details ────────────────────────────────────────────────────────
@router.get("/{module_id}")
async def get_module_detail(module_id: str):
    """Get details of a specific module."""
    module = get_module(module_id)
    if not module:
        raise HTTPException(status_code=404, detail=f"Module '{module_id}' not found")
    return {
        "id": module.id,
        "name": module.name,
        "category": module.category.value,
        "task_type": module.task_type.value,
        "description": module.description,
        "icon": module.icon,
        "tags": module.tags,
        "input_type": module.input_type,
        "default_algorithms": module.default_algorithms,
        "feature_hints": module.feature_hints,
        "target_description": module.target_description,
        "color": module.color,
    }


# ─── Upload Dataset + Train (single-step) ─────────────────────────────────────
@router.post("/{module_id}/train")
async def train_module(
    module_id: str,
    background_tasks: BackgroundTasks,
    experiment_name: str = Form(...),
    target_column: str = Form(...),
    algorithm: Optional[str] = Form(None),
    hyperparameters: str = Form("{}"),
    test_size: float = Form(0.2),
    file: Optional[UploadFile] = File(None),
    dataset_id: Optional[int] = Form(None),
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Train a model for the given module. Upload file or use existing dataset."""
    module = get_module(module_id)
    if not module:
        raise HTTPException(status_code=404, detail=f"Module '{module_id}' not found")

    # Get file bytes
    file_bytes = None
    file_type = None
    if file:
        file_bytes = await file.read()
        file_type = file.filename.split(".")[-1] if file.filename else "csv"
        # Upload dataset to MinIO
        object_key = f"{user.id}/{uuid.uuid4()}/{file.filename}"
        upload_file(
            settings.MINIO_BUCKET_DATASETS,
            object_key,
            io.BytesIO(file_bytes),
            len(file_bytes),
            content_type=file.content_type or "application/octet-stream",
        )
    elif dataset_id:
        result = await db.execute(
            select(Dataset).where(Dataset.id == dataset_id, Dataset.owner_id == user.id)
        )
        dataset = result.scalar_one_or_none()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        file_bytes = download_file(settings.MINIO_BUCKET_DATASETS, dataset.minio_object_key)
        file_type = dataset.file_type
        object_key = dataset.minio_object_key
    else:
        raise HTTPException(status_code=400, detail="Provide a file or dataset_id")

    # Parse hyperparameters
    try:
        hparams = json.loads(hyperparameters)
    except json.JSONDecodeError:
        hparams = {}

    # Create experiment record
    exp = Experiment(
        owner_id=user.id,
        module_id=module_id,
        module_category=ModuleCategory(module.category.value),
        name=experiment_name,
        status=JobStatus.PENDING,
        target_column=target_column,
        algorithm=algorithm or module.default_algorithms[0],
        hyperparameters=hparams,
        config={
            "test_size": test_size,
            "file_type": file_type,
            "object_key": object_key,
        },
    )
    db.add(exp)
    await db.flush()
    experiment_id = exp.id

    # Dispatch async Celery task or run direct training thread if Redis is offline
    celery_task_id = None
    try:
        task = train_model_task.delay(
            experiment_id=experiment_id,
            module_id=module_id,
            object_key=object_key,
            file_type=file_type,
            target_column=target_column,
            algorithm=algorithm or module.default_algorithms[0],
            hyperparameters=hparams,
            test_size=test_size,
        )
        celery_task_id = task.id
        exp.celery_task_id = task.id
        exp.status = JobStatus.RUNNING
        await db.flush()
    except Exception as e:
        import logging, asyncio
        logging.getLogger(__name__).warning(f"Celery/Redis unavailable ({e}). Executing instant direct training thread.")
        from app.workers.training_tasks import run_training_direct
        # CRITICAL: commit the experiment row BEFORE the thread runs so the sync session can see it
        await db.commit()
        await asyncio.to_thread(
            run_training_direct,
            experiment_id,
            module_id,
            object_key,
            file_type,
            target_column,
            algorithm or module.default_algorithms[0],
            hparams,
            test_size,
        )

    return {
        "experiment_id": experiment_id,
        "celery_task_id": celery_task_id,
        "status": "completed" if celery_task_id is None else "running",
        "message": "Training completed." if celery_task_id is None else "Training started.",
    }


# ─── Training Status ───────────────────────────────────────────────────────────
@router.get("/{module_id}/experiments/{experiment_id}/status")
async def get_training_status(
    module_id: str,
    experiment_id: int,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Experiment).where(
            Experiment.id == experiment_id,
            Experiment.owner_id == user.id,
        )
    )
    exp = result.scalar_one_or_none()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")

    return {
        "experiment_id": exp.id,
        "status": exp.status.value,
        "module_id": exp.module_id,
        "algorithm": exp.algorithm,
        "started_at": exp.started_at.isoformat() if exp.started_at else None,
        "completed_at": exp.completed_at.isoformat() if exp.completed_at else None,
        "duration_seconds": exp.duration_seconds,
        "metrics": exp.metrics,
        "feature_importance": exp.feature_importance,
        "confusion_matrix": exp.confusion_matrix,
        "training_history": exp.training_history,
        "predictions_preview": exp.predictions_preview,
    }


# ─── Predict ───────────────────────────────────────────────────────────────────
@router.post("/{module_id}/predict")
async def predict(
    module_id: str,
    body: PredictRequest,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Single-row prediction using a trained model. Missing features are auto-imputed."""
    import numpy as np
    module = get_module(module_id)
    if not module:
        raise HTTPException(status_code=404, detail=f"Module '{module_id}' not found")

    result = await db.execute(
        select(Experiment).where(
            Experiment.id == body.experiment_id,
            Experiment.owner_id == user.id,
            Experiment.status == JobStatus.COMPLETED,
        )
    )
    exp = result.scalar_one_or_none()
    if not exp:
        raise HTTPException(status_code=404, detail="Trained experiment not found")

    # Load model from MinIO / local disk
    model_bytes = download_file(settings.MINIO_BUCKET_MODELS, exp.model_minio_key)
    engine = MLPipelineEngine.load_model(model_bytes, module)
    prediction_result = engine.predict(body.input_data)

    p_val = prediction_result.get("prediction")
    conf = prediction_result.get("confidence")
    # Handle numpy types
    if isinstance(p_val, (np.integer,)):
        p_val = int(p_val)
    elif isinstance(p_val, (np.floating,)):
        p_val = float(p_val)
    if isinstance(conf, (np.floating, np.integer)):
        conf = float(conf)

    is_num = isinstance(p_val, (int, float)) and not isinstance(p_val, bool)
    numericals = {
        "total_rows": 1,
        "type": "regression" if is_num else "classification",
        "mean_prediction": float(p_val) if is_num else None,
        "min_prediction": float(p_val) if is_num else None,
        "max_prediction": float(p_val) if is_num else None,
        "std_prediction": 0.0 if is_num else None,
        "primary_class": str(p_val) if not is_num else None,
        "primary_class_percentage": 100.0 if not is_num else None,
        "class_counts": {str(p_val): 1} if not is_num else None,
        "avg_confidence": conf,
        "high_confidence_count": 1 if (conf and conf >= 0.8) else 0,
    }
    charts = {
        "feature_importance": exp.feature_importance,
    }
    predictions_sample = [{
        "id": 1,
        "inputs": body.input_data,
        "prediction": p_val,
        "confidence": conf,
    }]
    output_data_full = {
        "prediction": p_val,
        "confidence": conf,
        "numericals": numericals,
        "charts": charts,
        "predictions_sample": predictions_sample,
    }

    # Store prediction record
    pred = Prediction(
        experiment_id=exp.id,
        owner_id=user.id,
        input_data=body.input_data,
        output_data=output_data_full,
        confidence=conf,
        explanation=prediction_result.get("explanation"),
        is_batch=False,
        batch_size=1,
    )
    db.add(pred)
    await db.flush()

    return {
        "prediction_id": pred.id,
        "module_id": module_id,
        "experiment_id": exp.id,
        "prediction": p_val,
        "confidence": conf,
        "numericals": numericals,
        "charts": charts,
        "predictions_sample": predictions_sample,
        "explanation": prediction_result.get("explanation"),
    }



# ─── Batch Predict ─────────────────────────────────────────────────────────────
@router.post("/{module_id}/predict/batch")
async def batch_predict(
    module_id: str,
    body: BatchPredictRequest,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Batch prediction using a trained model."""
    module = get_module(module_id)
    if not module:
        raise HTTPException(status_code=404, detail=f"Module '{module_id}' not found")

    result = await db.execute(
        select(Experiment).where(
            Experiment.id == body.experiment_id,
            Experiment.owner_id == user.id,
            Experiment.status == JobStatus.COMPLETED,
        )
    )
    exp = result.scalar_one_or_none()
    if not exp:
        raise HTTPException(status_code=404, detail="Trained experiment not found")

    import pandas as pd
    model_bytes = download_file(settings.MINIO_BUCKET_MODELS, exp.model_minio_key)
    engine = MLPipelineEngine.load_model(model_bytes, module)
    df = pd.DataFrame(body.inputs)
    results = engine.predict_batch(df)

    # Store as batch prediction
    pred = Prediction(
        experiment_id=exp.id,
        owner_id=user.id,
        input_data={"batch": body.inputs[:5]},  # store sample
        output_data={"predictions": results},
        is_batch=True,
        batch_size=len(body.inputs),
    )
    db.add(pred)

    return {
        "module_id": module_id,
        "experiment_id": exp.id,
        "total": len(results),
        "predictions": results,
    }


# ─── Batch CSV Predict ─────────────────────────────────────────────────────────
@router.post("/{module_id}/predict/csv")
async def predict_csv(
    module_id: str,
    file: UploadFile = File(...),
    experiment_id: int = Form(...),
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Batch prediction from an uploaded CSV file containing unknown prediction data."""
    import pandas as pd
    import numpy as np

    module = get_module(module_id)
    if not module:
        raise HTTPException(status_code=404, detail=f"Module '{module_id}' not found")

    result = await db.execute(
        select(Experiment).where(
            Experiment.id == experiment_id,
            Experiment.owner_id == user.id,
            Experiment.status == JobStatus.COMPLETED,
        )
    )
    exp = result.scalar_one_or_none()
    if not exp:
        raise HTTPException(status_code=404, detail="Trained experiment not found or not completed")

    # Read CSV file
    file_bytes = await file.read()
    try:
        if file.filename and (file.filename.endswith(".xlsx") or file.filename.endswith(".xls")):
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse dataset file: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV file is empty")

    # Load model from MinIO
    model_bytes = download_file(settings.MINIO_BUCKET_MODELS, exp.model_minio_key)
    engine = MLPipelineEngine.load_model(model_bytes, module)

    # Run predictions
    results = engine.predict_batch(df)

    # Combine input features with predicted target & confidence
    combined_predictions = []
    preds_list = []
    confs_list = []

    for i, row_data in enumerate(df.to_dict(orient="records")):
        p = results[i]["prediction"]
        c = results[i].get("confidence")
        preds_list.append(p)
        if c is not None:
            confs_list.append(c)

        # Clean NaN/Inf for JSON serialization
        clean_row = {k: (None if pd.isna(v) else v) for k, v in row_data.items()}
        combined_predictions.append({
            "id": i + 1,
            "inputs": clean_row,
            "prediction": p,
            "confidence": c,
        })

    # Compute Numericals
    is_numeric_pred = isinstance(preds_list[0], (int, float, np.number)) and not isinstance(preds_list[0], bool)
    
    if is_numeric_pred:
        numeric_arr = np.array(preds_list, dtype=float)
        numericals = {
            "total_rows": len(preds_list),
            "type": "regression",
            "mean_prediction": float(np.mean(numeric_arr)),
            "min_prediction": float(np.min(numeric_arr)),
            "max_prediction": float(np.max(numeric_arr)),
            "std_prediction": float(np.std(numeric_arr)),
            "avg_confidence": float(np.mean(confs_list)) if confs_list else None,
        }
    else:
        from collections import Counter
        counts = Counter(str(p) for p in preds_list)
        most_common_class, top_count = counts.most_common(1)[0]
        numericals = {
            "total_rows": len(preds_list),
            "type": "classification",
            "primary_class": most_common_class,
            "primary_class_percentage": float((top_count / len(preds_list)) * 100),
            "class_counts": dict(counts),
            "avg_confidence": float(np.mean(confs_list)) if confs_list else None,
            "high_confidence_count": sum(1 for c in confs_list if c >= 0.8) if confs_list else len(preds_list),
        }

    # Compute Charts
    if is_numeric_pred:
        hist, bin_edges = np.histogram(preds_list, bins=min(10, max(3, len(preds_list))))
        bin_labels = [f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}" for i in range(len(hist))]
        distribution_chart = {
            "type": "histogram",
            "labels": bin_labels,
            "values": [int(x) for x in hist],
        }
    else:
        distribution_chart = {
            "type": "pie",
            "labels": list(numericals["class_counts"].keys()),
            "values": list(numericals["class_counts"].values()),
        }

    confidence_chart = None
    if confs_list:
        high = sum(1 for c in confs_list if c >= 0.85)
        medium = sum(1 for c in confs_list if 0.70 <= c < 0.85)
        low = sum(1 for c in confs_list if c < 0.70)
        confidence_chart = {
            "labels": ["High (≥85%)", "Medium (70-85%)", "Low (<70%)"],
            "values": [high, medium, low],
        }

    charts = {
        "distribution": distribution_chart,
        "confidence": confidence_chart,
        "feature_importance": exp.feature_importance,
    }

    # Compute SHAP explanation summary
    explanation_summary = engine._explain_prediction(df.head(10))

    # Store prediction record
    pred = Prediction(
        experiment_id=exp.id,
        owner_id=user.id,
        input_data={"filename": file.filename, "sample_inputs": combined_predictions[:5]},
        output_data={
            "numericals": numericals,
            "charts": charts,
            "predictions_sample": combined_predictions[:100],
            "total_rows": len(preds_list),
        },
        confidence=numericals.get("avg_confidence"),
        explanation=explanation_summary,
        is_batch=True,
        batch_size=len(preds_list),
    )
    db.add(pred)
    await db.flush()

    return {
        "prediction_id": pred.id,
        "filename": file.filename,
        "module_id": module_id,
        "experiment_id": exp.id,
        "numericals": numericals,
        "charts": charts,
        "predictions": combined_predictions,
        "explanation": explanation_summary,
        "created_at": pred.created_at.isoformat() if hasattr(pred, "created_at") and pred.created_at else None,
    }


# ─── Prediction History ────────────────────────────────────────────────────────
@router.get("/{module_id}/predictions")
async def get_predictions_history(
    module_id: str,
    experiment_id: Optional[int] = None,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Fetch prediction runs history for a module/experiment."""
    query = (
        select(Prediction)
        .join(Experiment)
        .where(
            Experiment.module_id == module_id,
            Prediction.owner_id == user.id,
        )
    )
    if experiment_id:
        query = query.where(Prediction.experiment_id == experiment_id)
    
    query = query.order_by(Prediction.created_at.desc()).limit(50)
    res = await db.execute(query)
    predictions = res.scalars().all()

    return {
        "predictions": [
            {
                "id": p.id,
                "experiment_id": p.experiment_id,
                "is_batch": p.is_batch,
                "batch_size": p.batch_size or 1,
                "confidence": p.confidence,
                "input_data": p.input_data,
                "output_data": p.output_data,
                "explanation": p.explanation,
                "created_at": p.created_at.isoformat() if p.created_at else None,
            }
            for p in predictions
        ]
    }


# ─── Experiment History ────────────────────────────────────────────────────────
@router.get("/{module_id}/experiments")
async def get_experiments(
    module_id: str,
    page: int = 1,
    limit: int = 20,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    offset = (page - 1) * limit
    result = await db.execute(
        select(Experiment)
        .where(Experiment.module_id == module_id, Experiment.owner_id == user.id)
        .order_by(Experiment.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    exps = result.scalars().all()
    return {
        "page": page,
        "limit": limit,
        "experiments": [
            {
                "id": e.id,
                "name": e.name,
                "status": e.status.value,
                "algorithm": e.algorithm,
                "metrics": e.metrics,
                "created_at": e.created_at.isoformat(),
                "duration_seconds": e.duration_seconds,
            }
            for e in exps
        ],
    }



# ─── SHAP/LIME Explanation ─────────────────────────────────────────────────────
class ExplainRequest(BaseModel):
    experiment_id: int
    input_data: Dict[str, Any]


@router.post("/{module_id}/explain")
async def explain_prediction(
    module_id: str,
    body: ExplainRequest,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Get SHAP explanation for a prediction input."""
    module = get_module(module_id)
    if not module:
        raise HTTPException(status_code=404, detail="Module not found")

    result = await db.execute(
        select(Experiment).where(
            Experiment.id == body.experiment_id,
            Experiment.owner_id == user.id,
            Experiment.status == JobStatus.COMPLETED,
        )
    )
    exp = result.scalar_one_or_none()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")

    model_bytes = download_file(settings.MINIO_BUCKET_MODELS, exp.model_minio_key)
    engine = MLPipelineEngine.load_model(model_bytes, module)

    import pandas as pd
    df = pd.DataFrame([body.input_data])
    explanation = engine._explain_prediction(df)

    return {
        "experiment_id": body.experiment_id,
        "module_id": module_id,
        "input_data": body.input_data,
        "explanation": explanation,
        "feature_importance": exp.feature_importance,
    }


# ─── AI Synthetic Dataset Generator ────────────────────────────────────────────
class GenerateDatasetRequest(BaseModel):
    num_rows: int = 100


@router.post("/{module_id}/generate-dataset")
async def generate_synthetic_dataset(
    module_id: str,
    body: GenerateDatasetRequest = GenerateDatasetRequest(),
    user: User = Depends(get_current_active_user),
):
    """Generate realistic, problem-specific synthetic training data matching the module's exact feature columns and target."""
    import numpy as np
    import pandas as pd
    import random

    module = get_module(module_id)
    if not module:
        raise HTTPException(status_code=404, detail=f"Module '{module_id}' not found")

    n = max(10, min(1000, body.num_rows))
    feature_hints = module.feature_hints or ["feature_1", "feature_2", "feature_3"]

    # Determine target column name
    target_desc = (module.target_description or "").lower()
    if "price" in target_desc or "cost" in target_desc or "value" in target_desc:
        target_col = "price"
    elif "label" in target_desc or "class" in target_desc:
        target_col = "label"
    elif "score" in target_desc:
        target_col = "score"
    elif "churn" in target_desc:
        target_col = "churn"
    elif "fraud" in target_desc:
        target_col = "is_fraud"
    elif "disease" in target_desc or "diagnosis" in target_desc:
        target_col = "diagnosis"
    else:
        target_col = "target"

    data = {}
    target_signal = np.zeros(n)

    # Domain specific categorical choices
    LOCATIONS = ["Downtown", "Suburbs", "Urban", "Beachfront", "Countryside", "Metropolitan"]
    CATEGORIES = ["Standard", "Premium", "Basic", "Deluxe", "Enterprise"]
    GENDERS = ["Male", "Female", "Other"]
    YES_NO = ["Yes", "No"]

    for feat in feature_hints:
        f_name = feat.lower()
        if "sqft" in f_name or "area" in f_name or "size" in f_name:
            vals = np.random.randint(500, 4500, size=n)
            data[feat] = vals
            target_signal += vals * 150
        elif "bedroom" in f_name or "room" in f_name:
            vals = np.random.randint(1, 6, size=n)
            data[feat] = vals
            target_signal += vals * 25000
        elif "bathroom" in f_name or "bath" in f_name:
            vals = np.random.randint(1, 5, size=n)
            data[feat] = vals
            target_signal += vals * 18000
        elif "year" in f_name or "built" in f_name:
            vals = np.random.randint(1975, 2024, size=n)
            data[feat] = vals
            target_signal += (vals - 1970) * 1200
        elif "garage" in f_name or "car" in f_name:
            vals = np.random.randint(0, 4, size=n)
            data[feat] = vals
            target_signal += vals * 12000
        elif "location" in f_name or "city" in f_name or "region" in f_name:
            loc_choices = [random.choice(LOCATIONS) for _ in range(n)]
            data[feat] = loc_choices
            target_signal += np.array([LOCATIONS.index(c) * 15000 for c in loc_choices])
        elif "age" in f_name:
            vals = np.random.randint(18, 75, size=n)
            data[feat] = vals
            target_signal += vals * 500
        elif "income" in f_name or "salary" in f_name:
            vals = np.random.randint(30000, 180000, size=n)
            data[feat] = vals
            target_signal += vals * 1.5
        elif "score" in f_name or "rating" in f_name:
            vals = np.round(np.random.uniform(1.0, 5.0, size=n), 1)
            data[feat] = vals
            target_signal += vals * 10000
        elif "category" in f_name or "type" in f_name:
            cat_choices = [random.choice(CATEGORIES) for _ in range(n)]
            data[feat] = cat_choices
            target_signal += np.array([CATEGORIES.index(c) * 5000 for c in cat_choices])
        else:
            # Fallback numeric feature
            vals = np.round(np.random.normal(loc=50.0, scale=15.0, size=n), 2)
            data[feat] = vals
            target_signal += vals * 100

    # Calculate target variable based on task type
    is_classification = module.task_type.value in ["binary_classification", "multiclass_classification", "nlp_classification"]

    if is_classification:
        median_signal = np.median(target_signal)
        prob = 1 / (1 + np.exp(-(target_signal - median_signal) / (np.std(target_signal) + 1e-5)))
        data[target_col] = (prob >= 0.5).astype(int)
    else:
        # Add realistic noise for regression
        noise = np.random.normal(0, np.std(target_signal) * 0.08, size=n)
        raw_target = target_signal + noise
        if "price" in target_col or "cost" in target_col:
            data[target_col] = np.round(np.clip(raw_target + 50000, 100000, 2000000), -2)
        else:
            data[target_col] = np.round(raw_target, 2)

    df = pd.DataFrame(data)

    # Convert to CSV string & list of records
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()

    records = df.to_dict(orient="records")

    return {
        "module_id": module_id,
        "target_column": target_col,
        "feature_columns": feature_hints,
        "num_rows": len(records),
        "records": records,
        "csv_content": csv_content,
        "filename": f"ai_generated_{module_id}_dataset.csv",
    }

