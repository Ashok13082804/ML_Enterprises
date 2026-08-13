"""
MLVerse X — FastAPI Application Entry Point
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import logging

from app.core.config import settings
from app.core.database import engine, Base
from app.core.redis import init_redis_pool
from app.api.v1 import api_router
from app.middleware.rate_limit import RateLimitMiddleware
from app.middleware.audit_log import AuditLogMiddleware
from app.middleware.security import SecurityHeadersMiddleware

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # ── Startup ──────────────────────────────────────────────────
    logger.info("Starting MLVerse X...")
    
    # Initialize Redis
    await init_redis_pool()
    logger.info("Redis connected")
    
    # Create DB tables (alembic handles migrations, this is a safety net)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database ready")
    
    # Initialize MinIO buckets
    from app.core.storage import init_storage
    await init_storage()
    logger.info("Storage ready")
    
    logger.info(f"MLVerse X started — {settings.APP_NAME} v{settings.APP_VERSION}")
    
    yield
    
    # ── Shutdown ─────────────────────────────────────────────────
    logger.info("Shutting down MLVerse X...")
    await engine.dispose()


def create_application() -> FastAPI:
    application = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="The Ultimate Offline AI, ML, DL, NLP & Computer Vision Platform",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── Middleware ────────────────────────────────────────────────
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.add_middleware(GZipMiddleware, minimum_size=1000)
    application.add_middleware(SecurityHeadersMiddleware)
    application.add_middleware(RateLimitMiddleware)
    application.add_middleware(AuditLogMiddleware)

    # ── Routes ───────────────────────────────────────────────────
    application.include_router(api_router, prefix="/api/v1")

    # Mount static storage directory for local disk fallback
    from pathlib import Path
    storage_path = Path(__file__).resolve().parents[2] / "storage"
    storage_path.mkdir(parents=True, exist_ok=True)
    application.mount("/storage", StaticFiles(directory=str(storage_path)), name="storage")

    # ── Health Check ─────────────────────────────────────────────
    @application.get("/health", tags=["health"])
    async def health_check():
        return JSONResponse(
            content={
                "status": "healthy",
                "app": settings.APP_NAME,
                "version": settings.APP_VERSION,
                "environment": settings.APP_ENV,
            }
        )

    @application.get("/", tags=["root"])
    async def root():
        return JSONResponse(
            content={
                "message": f"Welcome to {settings.APP_NAME}",
                "docs": "/docs",
                "health": "/health",
            }
        )

    # ── Global Exception Handler (ensures CORS headers on 500s) ──
    from fastapi import Request as _Request
    from fastapi.responses import JSONResponse as _JSONResponse

    @application.exception_handler(Exception)
    async def global_exception_handler(request: _Request, exc: Exception):
        origin = request.headers.get("origin", "")
        headers = {}
        if origin in settings.origins_list:
            headers["Access-Control-Allow-Origin"] = origin
            headers["Access-Control-Allow-Credentials"] = "true"
        logger.exception(f"Unhandled exception on {request.method} {request.url}: {exc}")
        return _JSONResponse(
            status_code=500,
            content={"detail": "Internal server error. Please try again."},
            headers=headers,
        )

    return application


app = create_application()
