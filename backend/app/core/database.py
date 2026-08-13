"""
MLVerse X — Async Database Engine and Session with Automatic SQLite Fallback
"""
import logging
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import MetaData
from app.core.config import settings

logger = logging.getLogger(__name__)

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=NAMING_CONVENTION)


class Base(DeclarativeBase):
    metadata = metadata


def _create_engine():
    db_url = settings.DATABASE_URL
    # If using localhost Postgres or default docker host when running outside docker, fall back to SQLite
    if "postgres:5432" in db_url or "localhost:5432" in db_url:
        # Check if local postgres is reachable or use SQLite
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            result = sock.connect_ex(("localhost", 5432))
            sock.close()
            if result != 0:
                logger.info("PostgreSQL not running locally. Falling back to local SQLite database (mlverse.db).")
                db_url = "sqlite+aiosqlite:///./mlverse.db"
        except Exception:
            db_url = "sqlite+aiosqlite:///./mlverse.db"

    if db_url.startswith("sqlite"):
        return create_async_engine(db_url, echo=settings.APP_DEBUG, connect_args={"check_same_thread": False})
    else:
        return create_async_engine(db_url, echo=settings.APP_DEBUG, pool_pre_ping=True, pool_size=10, max_overflow=20)


engine = _create_engine()

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def init_db():
    """Create all database tables on startup."""
    from app.models import models  # Ensure all models are imported before metadata create_all
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables verified/created successfully.")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency: provide a database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
