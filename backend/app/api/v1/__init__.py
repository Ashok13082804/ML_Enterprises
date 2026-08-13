"""
MLVerse X — Master API Router (v1)
All endpoints registered here.
"""
from fastapi import APIRouter

from app.api.v1.endpoints import auth, modules, analytics, rag, assistant
from app.api.v1.endpoints.stubs import (
    router_users as users_router,
    router_datasets as datasets_router,
    router_agents as agents_router,
    router_automl as automl_router,
    router_reports as reports_router,
    router_admin as admin_router,
    router_models as models_router,
    router_notifications as notifications_router,
    router_api_keys as api_keys_router,
    router_websocket as websocket_router,
)

api_router = APIRouter()

# ── Auth ──────────────────────────────────────────────────────────────────────
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])

# ── Users ─────────────────────────────────────────────────────────────────────
api_router.include_router(users_router, prefix="/users", tags=["Users"])

# ── Datasets ──────────────────────────────────────────────────────────────────
api_router.include_router(datasets_router, prefix="/datasets", tags=["Datasets"])

# ── ML Modules (covers all 100) ───────────────────────────────────────────────
api_router.include_router(modules.router, prefix="/modules", tags=["ML Modules"])

# ── AI Assistant ──────────────────────────────────────────────────────────────
api_router.include_router(assistant.router, prefix="/assistant", tags=["AI Assistant"])

# ── RAG ───────────────────────────────────────────────────────────────────────
api_router.include_router(rag.router, prefix="/rag", tags=["RAG System"])

# ── Agents ────────────────────────────────────────────────────────────────────
api_router.include_router(agents_router, prefix="/agents", tags=["Multi-Agent AI"])

# ── AutoML ────────────────────────────────────────────────────────────────────
api_router.include_router(automl_router, prefix="/automl", tags=["AutoML"])

# ── Reports ───────────────────────────────────────────────────────────────────
api_router.include_router(reports_router, prefix="/reports", tags=["Reports"])

# ── Admin ─────────────────────────────────────────────────────────────────────
api_router.include_router(admin_router, prefix="/admin", tags=["Admin"])

# ── Analytics ─────────────────────────────────────────────────────────────────
api_router.include_router(analytics.router, prefix="/analytics", tags=["Analytics"])

# ── Notifications ─────────────────────────────────────────────────────────────
api_router.include_router(notifications_router, prefix="/notifications", tags=["Notifications"])

# ── API Keys ──────────────────────────────────────────────────────────────────
api_router.include_router(api_keys_router, prefix="/api-keys", tags=["API Keys"])

# ── Model Registry ────────────────────────────────────────────────────────────
api_router.include_router(models_router, prefix="/model-registry", tags=["Model Registry"])

# ── WebSocket ─────────────────────────────────────────────────────────────────
api_router.include_router(websocket_router, prefix="/ws", tags=["WebSocket"])
