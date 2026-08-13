"""
MLVerse X — Application Configuration
"""
from typing import List, Optional, Any, Union
from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl, field_validator
import secrets


class Settings(BaseSettings):
    # ── App ──────────────────────────────────────────────────────
    APP_NAME: str = "MLVerse X"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = "development"
    APP_DEBUG: bool = True
    ALLOWED_ORIGINS: Union[List[str], str] = ["http://localhost:3000", "http://localhost:8000"]
    FRONTEND_URL: str = "http://localhost:3000"

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_origins(cls, v):
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("[") and v.endswith("]"):
                import json
                try:
                    return json.loads(v)
                except Exception:
                    pass
            return [i.strip() for i in v.split(",") if i.strip()]
        return v

    @property
    def origins_list(self) -> List[str]:
        if isinstance(self.ALLOWED_ORIGINS, list):
            return self.ALLOWED_ORIGINS
        return [self.ALLOWED_ORIGINS]



    # ── JWT ──────────────────────────────────────────────────────
    JWT_SECRET_KEY: str = secrets.token_urlsafe(32)
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    @property
    def SECRET_KEY(self) -> str:
        """Alias for JWT_SECRET_KEY (used by security.py)."""
        return self.JWT_SECRET_KEY

    @property
    def ALGORITHM(self) -> str:
        """Alias for JWT_ALGORITHM (used by security.py)."""
        return self.JWT_ALGORITHM

    # ── Database ─────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://mlverse_user:mlverse_password_change_in_prod@localhost:5432/mlverse"
    DATABASE_URL_SYNC: str = "postgresql://mlverse_user:mlverse_password_change_in_prod@localhost:5432/mlverse"

    # ── Redis ────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"

    # ── MinIO ────────────────────────────────────────────────────
    MINIO_HOST: str = "localhost"
    MINIO_PORT: int = 9000
    MINIO_ROOT_USER: str = "mlverse_minio_user"
    MINIO_ROOT_PASSWORD: str = "mlverse_minio_password_change_in_prod"
    MINIO_SECURE: bool = False
    MINIO_BUCKET_DATASETS: str = "datasets"
    MINIO_BUCKET_MODELS: str = "models"
    MINIO_BUCKET_REPORTS: str = "reports"
    MINIO_BUCKET_AVATARS: str = "avatars"

    # ── MLflow ───────────────────────────────────────────────────
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"

    # ── Ollama ───────────────────────────────────────────────────
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_DEFAULT_MODEL: str = "llama3.2"
    OLLAMA_TIMEOUT: int = 120

    # ── ChromaDB ─────────────────────────────────────────────────
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8001

    # ── Vector DB ────────────────────────────────────────────────
    VECTOR_DB_BACKEND: str = "chromadb"

    # ── Embeddings ───────────────────────────────────────────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DEVICE: str = "cpu"

    # ── Email ────────────────────────────────────────────────────
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_USERNAME: str = ""  # alias for SMTP_USER
    SMTP_PASSWORD: str = ""
    SMTP_FROM_NAME: str = "MLVerse X"
    SMTP_FROM_EMAIL: str = "noreply@mlverse.ai"
    SMTP_TLS: bool = True

    # ── OAuth (optional) ─────────────────────────────────────────
    GOOGLE_CLIENT_ID: Optional[str] = None
    GOOGLE_CLIENT_SECRET: Optional[str] = None
    GITHUB_CLIENT_ID: Optional[str] = None
    GITHUB_CLIENT_SECRET: Optional[str] = None

    # ── Rate Limiting ────────────────────────────────────────────
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_PER_HOUR: int = 1000

    # ── File Upload ──────────────────────────────────────────────
    MAX_UPLOAD_SIZE_MB: int = 500
    ALLOWED_UPLOAD_EXTENSIONS: str = "csv,xlsx,xls,json,parquet,txt,md,pdf,docx,pptx,zip,png,jpg,jpeg,mp4,mp3,wav"

    # ── Admin ────────────────────────────────────────────────────
    FIRST_SUPERUSER_EMAIL: str = "admin@mlverse.ai"
    FIRST_SUPERUSER_PASSWORD: str = "Admin@123!"

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"




settings = Settings()
