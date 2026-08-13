"""
MLVerse X — Analytics endpoints
"""
import psutil
from datetime import datetime, timezone
from fastapi import APIRouter, Depends
from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.models import User, Experiment, Prediction, Dataset, JobStatus
from app.api.v1.endpoints.auth import get_current_active_user, require_admin

router = APIRouter()


@router.get("/overview")
async def get_overview(
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Dashboard overview stats for the current user."""
    # Experiments count
    total_exp = await db.execute(
        select(func.count(Experiment.id)).where(Experiment.owner_id == user.id)
    )

    # Datasets count
    total_ds = await db.execute(
        select(func.count(Dataset.id)).where(Dataset.owner_id == user.id)
    )

    # Predictions count
    total_preds = await db.execute(
        select(func.count(Prediction.id)).where(Prediction.owner_id == user.id)
    )

    # Running jobs
    running_jobs = await db.execute(
        select(func.count(Experiment.id)).where(
            Experiment.owner_id == user.id,
            Experiment.status == JobStatus.RUNNING,
        )
    )

    # Recent experiments (last 10)
    recent = await db.execute(
        select(Experiment)
        .where(Experiment.owner_id == user.id)
        .order_by(Experiment.created_at.desc())
        .limit(10)
    )
    recent_experiments = [
        {
            "id": e.id,
            "name": e.name,
            "module_id": e.module_id,
            "status": e.status.value,
            "metrics": e.metrics,
            "algorithm": e.algorithm,
            "created_at": e.created_at.isoformat(),
        }
        for e in recent.scalars().all()
    ]

    return {
        "total_experiments": total_exp.scalar() or 0,
        "total_datasets": total_ds.scalar() or 0,
        "total_predictions": total_preds.scalar() or 0,
        "running_jobs": running_jobs.scalar() or 0,
        "recent_experiments": recent_experiments,
    }


@router.get("/system")
async def get_system_metrics(user: User = Depends(get_current_active_user)):
    """Real-time system resource metrics."""
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    cpu = psutil.cpu_percent(interval=0.5)

    return {
        "cpu_percent": cpu,
        "memory_percent": memory.percent,
        "memory_used_gb": round(memory.used / 1e9, 2),
        "memory_total_gb": round(memory.total / 1e9, 2),
        "disk_percent": disk.percent,
        "disk_used_gb": round(disk.used / 1e9, 2),
        "disk_total_gb": round(disk.total / 1e9, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/recent-experiments")
async def get_recent_experiments(
    limit: int = 20,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Experiment)
        .where(Experiment.owner_id == user.id)
        .order_by(Experiment.created_at.desc())
        .limit(limit)
    )
    experiments = result.scalars().all()
    return {
        "experiments": [
            {
                "id": e.id,
                "name": e.name,
                "module_id": e.module_id,
                "status": e.status.value,
                "metrics": e.metrics,
                "algorithm": e.algorithm,
                "duration_seconds": e.duration_seconds,
                "created_at": e.created_at.isoformat(),
            }
            for e in experiments
        ]
    }


@router.get("/admin/platform-stats")
async def get_admin_stats(
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Platform-wide stats for admins."""
    total_users = await db.execute(select(func.count(User.id)))
    total_exp = await db.execute(select(func.count(Experiment.id)))
    total_preds = await db.execute(select(func.count(Prediction.id)))
    total_ds = await db.execute(select(func.count(Dataset.id)))

    return {
        "total_users": total_users.scalar() or 0,
        "total_experiments": total_exp.scalar() or 0,
        "total_predictions": total_preds.scalar() or 0,
        "total_datasets": total_ds.scalar() or 0,
    }
