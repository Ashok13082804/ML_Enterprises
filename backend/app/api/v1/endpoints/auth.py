"""
MLVerse X — Authentication Endpoints
Handles: register, login, refresh, logout, email verify, OTP, 2FA, OAuth, password reset
"""
from datetime import datetime, timezone, timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status, Request
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, EmailStr, validator
import hashlib
import secrets

from app.core.database import get_db
from app.core.config import settings
from app.core.security import (
    hash_password, verify_password,
    create_access_token, create_refresh_token, decode_token,
    create_email_verification_token, create_password_reset_token,
    generate_totp_secret, get_totp_uri, verify_totp, generate_qr_code_base64,
    generate_otp,
)
from app.core.redis import RedisCache
from app.models.models import User, RefreshToken, UserRole, UserStatus
from app.services.email_service import send_verification_email, send_password_reset_email, send_otp_email
from sqlalchemy import select

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")
cache = RedisCache("auth")


# ─── Schemas ───────────────────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    email: EmailStr
    username: str
    full_name: Optional[str] = None
    password: str
    role: Optional[UserRole] = UserRole.STUDENT

    @validator("password")
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v

    @validator("username")
    def username_valid(cls, v):
        if not v.isalnum() or len(v) < 3:
            raise ValueError("Username must be alphanumeric and at least 3 characters")
        return v.lower()


class LoginRequest(BaseModel):
    email: EmailStr
    password: str
    remember_me: bool = False


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user_id: int
    role: str


class RefreshRequest(BaseModel):
    refresh_token: str


class EmailVerifyRequest(BaseModel):
    token: str


class OTPVerifyRequest(BaseModel):
    email: EmailStr
    otp: str


class TOTPSetupResponse(BaseModel):
    secret: str
    uri: str
    qr_code: str


class TOTPVerifyRequest(BaseModel):
    code: str


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str

    @validator("new_password")
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


# ─── Dependency: Current User ───────────────────────────────────────────────────
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    try:
        payload = decode_token(token)
        user_id = int(payload["sub"])
        if payload.get("type") != "access":
            raise HTTPException(status_code=401, detail="Invalid token type")
    except (ValueError, KeyError):
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    if user.status == UserStatus.SUSPENDED:
        raise HTTPException(status_code=403, detail="Account suspended")
    return user


async def get_current_active_user(user: User = Depends(get_current_user)) -> User:
    if user.status == UserStatus.PENDING_VERIFICATION:
        raise HTTPException(status_code=403, detail="Please verify your email first")
    return user


async def require_admin(user: User = Depends(get_current_active_user)) -> User:
    if user.role not in (UserRole.ADMIN, UserRole.SUPER_ADMIN):
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


# ─── Register ──────────────────────────────────────────────────────────────────
@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(
    body: RegisterRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    # Check existing email
    result = await db.execute(select(User).where(User.email == body.email))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Email already registered")

    # Check existing username
    result = await db.execute(select(User).where(User.username == body.username))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Username already taken")

    user = User(
        email=body.email,
        username=body.username,
        full_name=body.full_name,
        hashed_password=hash_password(body.password),
        role=body.role,
        status=UserStatus.PENDING_VERIFICATION,
    )
    db.add(user)
    await db.flush()

    # In dev mode, auto-verify so users can log in immediately without email setup
    dev_mode = settings.APP_ENV == "development"
    if dev_mode:
        user.is_email_verified = True
        user.status = UserStatus.ACTIVE
    else:
        # Send verification email in background
        token = create_email_verification_token(body.email)
        background_tasks.add_task(send_verification_email, body.email, token)

    return {
        "message": "Registration successful." + (" Please verify your email." if not dev_mode else " You can log in now."),
        "user_id": user.id,
        "email": user.email,
        "email_verification_required": not dev_mode,
    }


# ─── Login ─────────────────────────────────────────────────────────────────────
@router.post("/login", response_model=TokenResponse)
async def login(
    body: LoginRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()

    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    if user.status == UserStatus.SUSPENDED:
        raise HTTPException(status_code=403, detail="Account suspended")

    # 2FA check
    if user.is_2fa_enabled:
        # Issue temporary token, require TOTP verification
        otp = generate_otp()
        await cache.set(f"2fa_pending:{user.id}", otp, expire=300)
        raise HTTPException(
            status_code=202,
            detail={"message": "2FA required", "user_id": user.id, "requires_2fa": True}
        )

    # Update last login
    user.last_login_at = datetime.now(timezone.utc)

    # Create tokens
    extra = {"role": user.role.value, "username": user.username}
    access_token = create_access_token(user.id, extra)
    refresh_token_str = create_refresh_token(user.id)
    refresh_hash = hashlib.sha256(refresh_token_str.encode()).hexdigest()

    # Store refresh token
    rt = RefreshToken(
        user_id=user.id,
        token_hash=refresh_hash,
        expires_at=datetime.now(timezone.utc) + timedelta(days=7 if not body.remember_me else 30),
        device_info=request.headers.get("user-agent", "")[:500],
    )
    db.add(rt)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token_str,
        user_id=user.id,
        role=user.role.value,
    )


# ─── Verify 2FA ────────────────────────────────────────────────────────────────
@router.post("/login/2fa")
async def login_2fa(
    user_id: int,
    code: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not user.totp_secret or not verify_totp(user.totp_secret, code):
        raise HTTPException(status_code=401, detail="Invalid 2FA code")

    user.last_login_at = datetime.now(timezone.utc)
    extra = {"role": user.role.value, "username": user.username}
    access_token = create_access_token(user.id, extra)
    refresh_token_str = create_refresh_token(user.id)
    refresh_hash = hashlib.sha256(refresh_token_str.encode()).hexdigest()

    rt = RefreshToken(
        user_id=user.id,
        token_hash=refresh_hash,
        expires_at=datetime.now(timezone.utc) + timedelta(days=7),
        device_info=request.headers.get("user-agent", "")[:500],
    )
    db.add(rt)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token_str,
        user_id=user.id,
        role=user.role.value,
    )


# ─── Refresh Token ─────────────────────────────────────────────────────────────
@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    body: RefreshRequest,
    db: AsyncSession = Depends(get_db),
):
    try:
        payload = decode_token(body.refresh_token)
        if payload.get("type") != "refresh":
            raise ValueError("Wrong token type")
        user_id = int(payload["sub"])
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    refresh_hash = hashlib.sha256(body.refresh_token.encode()).hexdigest()
    result = await db.execute(
        select(RefreshToken).where(
            RefreshToken.token_hash == refresh_hash,
            RefreshToken.user_id == user_id,
            RefreshToken.revoked == False,
        )
    )
    rt = result.scalar_one_or_none()
    if not rt or rt.expires_at.replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
        raise HTTPException(status_code=401, detail="Refresh token expired or revoked")

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Rotate tokens
    rt.revoked = True
    extra = {"role": user.role.value, "username": user.username}
    new_access = create_access_token(user.id, extra)
    new_refresh = create_refresh_token(user.id)
    new_hash = hashlib.sha256(new_refresh.encode()).hexdigest()

    new_rt = RefreshToken(
        user_id=user.id,
        token_hash=new_hash,
        expires_at=rt.expires_at,
    )
    db.add(new_rt)

    return TokenResponse(
        access_token=new_access,
        refresh_token=new_refresh,
        user_id=user.id,
        role=user.role.value,
    )


# ─── Logout ────────────────────────────────────────────────────────────────────
@router.post("/logout")
async def logout(
    body: RefreshRequest,
    db: AsyncSession = Depends(get_db),
):
    refresh_hash = hashlib.sha256(body.refresh_token.encode()).hexdigest()
    result = await db.execute(
        select(RefreshToken).where(RefreshToken.token_hash == refresh_hash)
    )
    rt = result.scalar_one_or_none()
    if rt:
        rt.revoked = True
    return {"message": "Logged out successfully"}


# ─── Email Verification ────────────────────────────────────────────────────────
@router.post("/verify-email")
async def verify_email(body: EmailVerifyRequest, db: AsyncSession = Depends(get_db)):
    try:
        payload = decode_token(body.token)
        if payload.get("type") != "email_verification":
            raise ValueError
        email = payload["sub"]
    except (ValueError, KeyError):
        raise HTTPException(status_code=400, detail="Invalid or expired verification token")

    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.is_email_verified = True
    user.status = UserStatus.ACTIVE
    return {"message": "Email verified successfully"}


# ─── Resend Verification ───────────────────────────────────────────────────────
@router.post("/resend-verification")
async def resend_verification(
    email: EmailStr,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.is_email_verified:
        return {"message": "Email already verified"}
    token = create_email_verification_token(email)
    background_tasks.add_task(send_verification_email, email, token)
    return {"message": "Verification email sent"}


# ─── Forgot Password ───────────────────────────────────────────────────────────
@router.post("/forgot-password")
async def forgot_password(
    body: ForgotPasswordRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()
    if user:
        token = create_password_reset_token(body.email)
        background_tasks.add_task(send_password_reset_email, body.email, token)
    # Always return same response to prevent email enumeration
    return {"message": "If this email exists, a reset link has been sent"}


# ─── Reset Password ────────────────────────────────────────────────────────────
@router.post("/reset-password")
async def reset_password(body: ResetPasswordRequest, db: AsyncSession = Depends(get_db)):
    try:
        payload = decode_token(body.token)
        if payload.get("type") != "password_reset":
            raise ValueError
        email = payload["sub"]
    except (ValueError, KeyError):
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")

    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.hashed_password = hash_password(body.new_password)
    # Revoke all refresh tokens for security
    rts = await db.execute(select(RefreshToken).where(RefreshToken.user_id == user.id))
    for rt in rts.scalars():
        rt.revoked = True

    return {"message": "Password reset successfully"}


# ─── 2FA Setup ─────────────────────────────────────────────────────────────────
@router.post("/2fa/setup", response_model=TOTPSetupResponse)
async def setup_2fa(user: User = Depends(get_current_active_user), db: AsyncSession = Depends(get_db)):
    secret = generate_totp_secret()
    uri = get_totp_uri(secret, user.email)
    qr = generate_qr_code_base64(uri)
    # Store secret temporarily in redis until verified
    cache_key = f"totp_setup:{user.id}"
    await cache.set(cache_key, secret, expire=600)
    return TOTPSetupResponse(secret=secret, uri=uri, qr_code=qr)


# ─── 2FA Enable ────────────────────────────────────────────────────────────────
@router.post("/2fa/enable")
async def enable_2fa(
    body: TOTPVerifyRequest,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    cache_key = f"totp_setup:{user.id}"
    secret = await cache.get(cache_key)
    if not secret:
        raise HTTPException(status_code=400, detail="2FA setup session expired, restart setup")

    if not verify_totp(secret, body.code):
        raise HTTPException(status_code=400, detail="Invalid TOTP code")

    user.totp_secret = secret
    user.is_2fa_enabled = True
    await cache.delete(cache_key)
    return {"message": "2FA enabled successfully"}


# ─── 2FA Disable ───────────────────────────────────────────────────────────────
@router.post("/2fa/disable")
async def disable_2fa(
    body: TOTPVerifyRequest,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    if not user.totp_secret or not verify_totp(user.totp_secret, body.code):
        raise HTTPException(status_code=400, detail="Invalid TOTP code")
    user.totp_secret = None
    user.is_2fa_enabled = False
    return {"message": "2FA disabled"}


# ─── Me ────────────────────────────────────────────────────────────────────────
@router.get("/me")
async def get_me(user: User = Depends(get_current_user)):
    """Returns the current user's profile. Works even if email is not yet verified."""
    return {
        "id": user.id,
        "email": user.email,
        "username": user.username,
        "full_name": user.full_name,
        "role": user.role.value,
        "status": user.status.value,
        "avatar_url": user.avatar_url,
        "is_email_verified": user.is_email_verified,
        "is_2fa_enabled": user.is_2fa_enabled,
        "created_at": user.created_at.isoformat(),
        "email_verification_required": not user.is_email_verified,
    }
