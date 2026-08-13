"""
MLVerse X — All SQLAlchemy Models
"""
from datetime import datetime, timezone
from typing import Optional, List
from sqlalchemy import (
    String, Text, Boolean, Integer, Float, DateTime, JSON,
    ForeignKey, Enum as SAEnum, Index, BigInteger
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
import enum

from app.core.database import Base


def utcnow():
    return datetime.now(timezone.utc)


# ─── Enums ─────────────────────────────────────────────────────────────────────
class UserRole(str, enum.Enum):
    SUPER_ADMIN = "super_admin"
    ADMIN = "admin"
    DEVELOPER = "developer"
    RESEARCHER = "researcher"
    STUDENT = "student"
    ORGANIZATION = "organization"


class UserStatus(str, enum.Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING_VERIFICATION = "pending_verification"


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModuleCategory(str, enum.Enum):
    BEGINNER_ML = "beginner_ml"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"
    TIME_SERIES = "time_series"
    FINANCE = "finance"
    HEALTHCARE = "healthcare"
    INDUSTRIAL = "industrial"


class DatasetStatus(str, enum.Enum):
    UPLOADING = "uploading"
    VALIDATING = "validating"
    READY = "ready"
    CLEANING = "cleaning"
    ERROR = "error"


# ─── User ──────────────────────────────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    full_name: Mapped[Optional[str]] = mapped_column(String(255))
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(SAEnum(UserRole), default=UserRole.STUDENT)
    status: Mapped[UserStatus] = mapped_column(SAEnum(UserStatus), default=UserStatus.PENDING_VERIFICATION)
    avatar_url: Mapped[Optional[str]] = mapped_column(String(500))
    bio: Mapped[Optional[str]] = mapped_column(Text)
    organization: Mapped[Optional[str]] = mapped_column(String(255))

    # Auth flags
    is_email_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    is_2fa_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    totp_secret: Mapped[Optional[str]] = mapped_column(String(64))
    otp_code: Mapped[Optional[str]] = mapped_column(String(10))
    otp_expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # OAuth
    google_id: Mapped[Optional[str]] = mapped_column(String(255), unique=True)
    github_id: Mapped[Optional[str]] = mapped_column(String(255), unique=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    last_login_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Relationships
    datasets: Mapped[List["Dataset"]] = relationship("Dataset", back_populates="owner", cascade="all, delete-orphan")
    experiments: Mapped[List["Experiment"]] = relationship("Experiment", back_populates="owner", cascade="all, delete-orphan")
    api_keys: Mapped[List["APIKey"]] = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    refresh_tokens: Mapped[List["RefreshToken"]] = relationship("RefreshToken", back_populates="user", cascade="all, delete-orphan")
    notifications: Mapped[List["Notification"]] = relationship("Notification", back_populates="user", cascade="all, delete-orphan")
    audit_logs: Mapped[List["AuditLog"]] = relationship("AuditLog", back_populates="user")


# ─── Refresh Token ─────────────────────────────────────────────────────────────
class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    token_hash: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)
    device_info: Mapped[Optional[str]] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    user: Mapped["User"] = relationship("User", back_populates="refresh_tokens")


# ─── API Key ───────────────────────────────────────────────────────────────────
class APIKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(100))
    key_hash: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    key_prefix: Mapped[str] = mapped_column(String(20))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    rate_limit: Mapped[int] = mapped_column(Integer, default=1000)
    usage_count: Mapped[int] = mapped_column(Integer, default=0)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    user: Mapped["User"] = relationship("User", back_populates="api_keys")


# ─── Dataset ───────────────────────────────────────────────────────────────────
class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    owner_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[Optional[str]] = mapped_column(Text)
    file_type: Mapped[str] = mapped_column(String(50))
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    minio_object_key: Mapped[str] = mapped_column(String(500))
    status: Mapped[DatasetStatus] = mapped_column(SAEnum(DatasetStatus), default=DatasetStatus.UPLOADING)

    # Stats (populated after validation)
    num_rows: Mapped[Optional[int]] = mapped_column(Integer)
    num_columns: Mapped[Optional[int]] = mapped_column(Integer)
    column_names: Mapped[Optional[dict]] = mapped_column(JSON)
    column_types: Mapped[Optional[dict]] = mapped_column(JSON)
    statistics: Mapped[Optional[dict]] = mapped_column(JSON)
    missing_values: Mapped[Optional[dict]] = mapped_column(JSON)
    sample_data: Mapped[Optional[dict]] = mapped_column(JSON)

    is_public: Mapped[bool] = mapped_column(Boolean, default=False)
    tags: Mapped[Optional[List[str]]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    owner: Mapped["User"] = relationship("User", back_populates="datasets")
    experiments: Mapped[List["Experiment"]] = relationship("Experiment", back_populates="dataset")


# ─── Experiment ────────────────────────────────────────────────────────────────
class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    owner_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    dataset_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("datasets.id", ondelete="SET NULL"))
    module_id: Mapped[str] = mapped_column(String(100), index=True)
    module_category: Mapped[ModuleCategory] = mapped_column(SAEnum(ModuleCategory))
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[Optional[str]] = mapped_column(Text)
    status: Mapped[JobStatus] = mapped_column(SAEnum(JobStatus), default=JobStatus.PENDING)
    celery_task_id: Mapped[Optional[str]] = mapped_column(String(255))
    mlflow_run_id: Mapped[Optional[str]] = mapped_column(String(255))

    # Config
    config: Mapped[Optional[dict]] = mapped_column(JSON)
    target_column: Mapped[Optional[str]] = mapped_column(String(255))
    feature_columns: Mapped[Optional[List[str]]] = mapped_column(JSON)
    algorithm: Mapped[Optional[str]] = mapped_column(String(100))
    hyperparameters: Mapped[Optional[dict]] = mapped_column(JSON)

    # Results
    metrics: Mapped[Optional[dict]] = mapped_column(JSON)
    feature_importance: Mapped[Optional[dict]] = mapped_column(JSON)
    confusion_matrix: Mapped[Optional[dict]] = mapped_column(JSON)
    training_history: Mapped[Optional[dict]] = mapped_column(JSON)
    predictions_preview: Mapped[Optional[dict]] = mapped_column(JSON)
    shap_values: Mapped[Optional[dict]] = mapped_column(JSON)

    # Saved model
    model_minio_key: Mapped[Optional[str]] = mapped_column(String(500))
    model_version: Mapped[Optional[str]] = mapped_column(String(50))

    # Timing
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    owner: Mapped["User"] = relationship("User", back_populates="experiments")
    dataset: Mapped[Optional["Dataset"]] = relationship("Dataset", back_populates="experiments")
    predictions: Mapped[List["Prediction"]] = relationship("Prediction", back_populates="experiment", cascade="all, delete-orphan")
    reports: Mapped[List["Report"]] = relationship("Report", back_populates="experiment", cascade="all, delete-orphan")


# ─── Prediction ────────────────────────────────────────────────────────────────
class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(Integer, ForeignKey("experiments.id", ondelete="CASCADE"))
    owner_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    input_data: Mapped[dict] = mapped_column(JSON)
    output_data: Mapped[dict] = mapped_column(JSON)
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    explanation: Mapped[Optional[dict]] = mapped_column(JSON)
    is_batch: Mapped[bool] = mapped_column(Boolean, default=False)
    batch_size: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    experiment: Mapped["Experiment"] = relationship("Experiment", back_populates="predictions")


# ─── Report ────────────────────────────────────────────────────────────────────
class Report(Base):
    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(Integer, ForeignKey("experiments.id", ondelete="CASCADE"))
    owner_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(255))
    format: Mapped[str] = mapped_column(String(20))  # pdf, docx, pptx, xlsx, html, md
    minio_object_key: Mapped[str] = mapped_column(String(500))
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    experiment: Mapped["Experiment"] = relationship("Experiment", back_populates="reports")


# ─── RAG Document ──────────────────────────────────────────────────────────────
class RAGDocument(Base):
    __tablename__ = "rag_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    owner_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(255))
    file_type: Mapped[str] = mapped_column(String(50))
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    minio_object_key: Mapped[str] = mapped_column(String(500))
    chroma_collection_id: Mapped[Optional[str]] = mapped_column(String(255))
    num_chunks: Mapped[Optional[int]] = mapped_column(Integer)
    is_indexed: Mapped[bool] = mapped_column(Boolean, default=False)
    indexing_error: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


# ─── Chat Session ──────────────────────────────────────────────────────────────
class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    owner_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    title: Mapped[str] = mapped_column(String(255), default="New Chat")
    model: Mapped[str] = mapped_column(String(100))
    mode: Mapped[str] = mapped_column(String(50), default="assistant")  # assistant, rag, agent
    rag_document_ids: Mapped[Optional[List[int]]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    messages: Mapped[List["ChatMessage"]] = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(Integer, ForeignKey("chat_sessions.id", ondelete="CASCADE"))
    role: Mapped[str] = mapped_column(String(20))  # user, assistant, system
    content: Mapped[str] = mapped_column(Text)
    citations: Mapped[Optional[List[dict]]] = mapped_column(JSON)
    tokens_used: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    session: Mapped["ChatSession"] = relationship("ChatSession", back_populates="messages")


# ─── Agent Task ────────────────────────────────────────────────────────────────
class AgentTask(Base):
    __tablename__ = "agent_tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    owner_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    agent_type: Mapped[str] = mapped_column(String(100))
    task_description: Mapped[str] = mapped_column(Text)
    status: Mapped[JobStatus] = mapped_column(SAEnum(JobStatus), default=JobStatus.PENDING)
    input_data: Mapped[Optional[dict]] = mapped_column(JSON)
    output_data: Mapped[Optional[dict]] = mapped_column(JSON)
    steps: Mapped[Optional[List[dict]]] = mapped_column(JSON)
    celery_task_id: Mapped[Optional[str]] = mapped_column(String(255))
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


# ─── Notification ──────────────────────────────────────────────────────────────
class Notification(Base):
    __tablename__ = "notifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    title: Mapped[str] = mapped_column(String(255))
    message: Mapped[str] = mapped_column(Text)
    type: Mapped[str] = mapped_column(String(50), default="info")  # info, success, warning, error
    is_read: Mapped[bool] = mapped_column(Boolean, default=False)
    action_url: Mapped[Optional[str]] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    user: Mapped["User"] = relationship("User", back_populates="notifications")


# ─── Audit Log ─────────────────────────────────────────────────────────────────
class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id", ondelete="SET NULL"))
    action: Mapped[str] = mapped_column(String(100), index=True)
    resource_type: Mapped[Optional[str]] = mapped_column(String(100))
    resource_id: Mapped[Optional[str]] = mapped_column(String(255))
    details: Mapped[Optional[dict]] = mapped_column(JSON)
    ip_address: Mapped[Optional[str]] = mapped_column(String(50))
    user_agent: Mapped[Optional[str]] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)

    user: Mapped[Optional["User"]] = relationship("User", back_populates="audit_logs")


# ─── Indexes ───────────────────────────────────────────────────────────────────
Index("ix_experiments_module_id_owner", Experiment.module_id, Experiment.owner_id)
Index("ix_predictions_experiment_id_created", Prediction.experiment_id, Prediction.created_at)
Index("ix_audit_logs_user_id_created", AuditLog.user_id, AuditLog.created_at)
