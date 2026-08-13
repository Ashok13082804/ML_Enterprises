"""Email service for MLVerse X (uses SMTP or Mailhog in development)"""
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from app.core.config import settings

logger = logging.getLogger(__name__)


def _send_email(to_email: str, subject: str, html_body: str):
    """Send email via SMTP. Falls back to log if SMTP not configured."""
    if not settings.SMTP_HOST:
        logger.info(f"[EMAIL STUB] To: {to_email} | Subject: {subject}")
        return

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = settings.SMTP_FROM_EMAIL
        msg["To"] = to_email
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
            if settings.SMTP_TLS:
                server.starttls()
            if settings.SMTP_USERNAME:
                server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
            server.sendmail(settings.SMTP_FROM_EMAIL, to_email, msg.as_string())
    except Exception as e:
        logger.error(f"Failed to send email to {to_email}: {e}")


def send_verification_email(email: str, token: str):
    url = f"{settings.FRONTEND_URL}/auth/verify-email?token={token}"
    html = f"""
    <div style="font-family: Inter, sans-serif; max-width: 480px; margin: 0 auto; padding: 32px; background: #020817; color: #fff; border-radius: 16px;">
      <div style="text-align: center; margin-bottom: 32px;">
        <div style="display: inline-block; width: 56px; height: 56px; background: linear-gradient(135deg, #6366f1, #06b6d4); border-radius: 14px; line-height: 56px; font-size: 24px; font-weight: 900; color: white;">M</div>
        <h1 style="margin: 16px 0 0; font-size: 24px; font-weight: 900;">MLVerse X</h1>
      </div>
      <h2 style="font-size: 20px; font-weight: 700; margin-bottom: 8px;">Verify your email</h2>
      <p style="color: rgba(255,255,255,0.6); margin-bottom: 24px;">Click the button below to verify your email address.</p>
      <a href="{url}" style="display: block; text-align: center; padding: 14px 24px; background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white; text-decoration: none; border-radius: 12px; font-weight: 600; font-size: 15px;">
        Verify Email Address
      </a>
      <p style="margin-top: 24px; color: rgba(255,255,255,0.3); font-size: 12px;">This link expires in 24 hours. If you didn't register, ignore this email.</p>
    </div>
    """
    _send_email(email, "Verify your MLVerse X account", html)


def send_password_reset_email(email: str, token: str):
    url = f"{settings.FRONTEND_URL}/auth/reset-password?token={token}"
    html = f"""
    <div style="font-family: Inter, sans-serif; max-width: 480px; margin: 0 auto; padding: 32px; background: #020817; color: #fff; border-radius: 16px;">
      <h2 style="font-size: 20px; font-weight: 700; margin-bottom: 8px;">Reset your password</h2>
      <p style="color: rgba(255,255,255,0.6); margin-bottom: 24px;">You requested a password reset for your MLVerse X account.</p>
      <a href="{url}" style="display: block; text-align: center; padding: 14px 24px; background: linear-gradient(135deg, #ef4444, #f59e0b); color: white; text-decoration: none; border-radius: 12px; font-weight: 600;">
        Reset Password
      </a>
      <p style="margin-top: 24px; color: rgba(255,255,255,0.3); font-size: 12px;">This link expires in 1 hour. If you didn't request this, ignore this email.</p>
    </div>
    """
    _send_email(email, "Reset your MLVerse X password", html)


def send_otp_email(email: str, otp: str):
    html = f"""
    <div style="font-family: Inter, sans-serif; max-width: 480px; margin: 0 auto; padding: 32px; background: #020817; color: #fff; border-radius: 16px;">
      <h2 style="font-size: 20px; font-weight: 700; margin-bottom: 8px;">Your OTP Code</h2>
      <div style="font-size: 40px; font-weight: 900; letter-spacing: 8px; text-align: center; color: #6366f1; margin: 24px 0;">{otp}</div>
      <p style="color: rgba(255,255,255,0.3); font-size: 12px; text-align: center;">Expires in 5 minutes</p>
    </div>
    """
    _send_email(email, "Your MLVerse X OTP code", html)
