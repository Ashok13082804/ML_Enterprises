"""
MLVerse X — AI Assistant Endpoints (Ollama-powered)
"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json

from app.core.database import get_db
from app.models.models import User, ChatSession, ChatMessage
from app.api.v1.endpoints.auth import get_current_active_user
from ai.ollama.client import get_ollama_client, ML_ASSISTANT_SYSTEM_PROMPT

router = APIRouter()


# ─── Schemas ───────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: Optional[int] = None
    message: str
    model: Optional[str] = None
    temperature: float = 0.7
    mode: str = "assistant"  # assistant, code, explain


class NewSessionRequest(BaseModel):
    title: str = "New Chat"
    model: str = "llama3.2"


# ─── List Models & Ollama Status ───────────────────────────────────────────────
@router.get("/models")
async def list_available_models():
    """List all Ollama models available on this machine."""
    client = get_ollama_client()
    available = await client.is_available()
    models = await client.list_models()
    return {
        "available": available,
        "mode": "ollama_online" if available else "smart_fallback_engine",
        "message": "Ollama local LLM server active" if available else "Ollama offline — using built-in MLVerse local intelligence engine",
        "models": [
            {
                "name": m.name,
                "size_gb": round(m.size / 1e9, 2),
                "modified_at": m.modified_at,
            }
            for m in models
        ] if models else [{"name": "llama3.2", "size_gb": 3.8, "modified_at": "default"}],
    }


@router.get("/ollama/status")
async def get_ollama_status():
    """Detailed diagnostics of Ollama installation and service state."""
    import shutil, subprocess
    client = get_ollama_client()
    is_running = await client.is_available()
    ollama_path = shutil.which("ollama")

    return {
        "installed": bool(ollama_path),
        "path": ollama_path or "Not installed in PATH",
        "running": is_running,
        "host": client.host,
        "models": [m.name for m in (await client.list_models())] if is_running else [],
        "install_command": "curl -fsSL https://ollama.com/install.sh | sh",
        "start_command": "ollama serve",
    }


@router.post("/ollama/start")
async def start_ollama_service():
    """Attempt to auto-start local ollama serve process."""
    import shutil, subprocess
    ollama_path = shutil.which("ollama")
    if not ollama_path:
        return {
            "success": False,
            "message": "Ollama is not installed on this system. Please run `install_ollama.sh` or install from https://ollama.com",
            "command": "curl -fsSL https://ollama.com/install.sh | sh",
        }
    
    try:
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return {
            "success": True,
            "message": "Launched `ollama serve` in background. Connecting...",
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to launch ollama serve: {str(e)}",
        }


# ─── Pull Model ────────────────────────────────────────────────────────────────
@router.post("/models/pull")
async def pull_model(
    model_name: str,
    user: User = Depends(get_current_active_user),
):
    """Stream download progress for an Ollama model."""
    client = get_ollama_client()

    async def stream_pull():
        async for chunk in client.pull_model(model_name):
            yield chunk

    return StreamingResponse(stream_pull(), media_type="application/x-ndjson")


# ─── Create Chat Session ────────────────────────────────────────────────────────
@router.post("/sessions")
async def create_session(
    body: NewSessionRequest,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    session = ChatSession(
        owner_id=user.id,
        title=body.title,
        model=body.model,
        mode="assistant",
    )
    db.add(session)
    await db.flush()
    return {"session_id": session.id, "title": session.title, "model": session.model}


# ─── List Sessions ─────────────────────────────────────────────────────────────
@router.get("/sessions")
async def list_sessions(
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ChatSession)
        .where(ChatSession.owner_id == user.id)
        .order_by(ChatSession.updated_at.desc())
        .limit(50)
    )
    sessions = result.scalars().all()
    return {
        "sessions": [
            {
                "id": s.id,
                "title": s.title,
                "model": s.model,
                "updated_at": s.updated_at.isoformat(),
            }
            for s in sessions
        ]
    }


# ─── Get Session Messages ──────────────────────────────────────────────────────
@router.get("/sessions/{session_id}/messages")
async def get_messages(
    session_id: int,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id, ChatSession.owner_id == user.id
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at)
    )
    messages = result.scalars().all()
    return {
        "session": {"id": session.id, "title": session.title, "model": session.model},
        "messages": [
            {"role": m.role, "content": m.content, "created_at": m.created_at.isoformat()}
            for m in messages
        ],
    }


# ─── Chat (Streaming) ──────────────────────────────────────────────────────────
@router.post("/chat")
async def chat(
    body: ChatRequest,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Send a message and stream the AI response."""
    client = get_ollama_client()

    # Get or create session
    if body.session_id:
        result = await db.execute(
            select(ChatSession).where(
                ChatSession.id == body.session_id, ChatSession.owner_id == user.id
            )
        )
        session = result.scalar_one_or_none()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
    else:
        # Auto-create session
        session = ChatSession(
            owner_id=user.id,
            title=body.message[:50] + "..." if len(body.message) > 50 else body.message,
            model=body.model or "llama3.2",
        )
        db.add(session)
        await db.flush()

    # Get conversation history
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session.id)
        .order_by(ChatMessage.created_at)
        .limit(20)
    )
    history = result.scalars().all()

    messages = [{"role": m.role, "content": m.content} for m in history]
    messages.append({"role": "user", "content": body.message})

    # Save user message
    user_msg = ChatMessage(
        session_id=session.id, role="user", content=body.message
    )
    db.add(user_msg)
    await db.commit()

    model = body.model or session.model or "llama3.2"
    full_response = []

    async def stream_response():
        async for chunk in client.chat(
            messages=messages,
            model=model,
            temperature=body.temperature,
            stream=True,
            system_prompt=ML_ASSISTANT_SYSTEM_PROMPT,
        ):
            full_response.append(chunk)
            yield f"data: {json.dumps({'content': chunk, 'session_id': session.id})}\n\n"

        # Save assistant response
        assistant_content = "".join(full_response)
        async with db.begin():
            asst_msg = ChatMessage(
                session_id=session.id,
                role="assistant",
                content=assistant_content,
            )
            db.add(asst_msg)

        yield f"data: {json.dumps({'done': True, 'session_id': session.id})}\n\n"

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


# ─── Delete Session ────────────────────────────────────────────────────────────
@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: int,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id, ChatSession.owner_id == user.id
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    await db.delete(session)
    return {"message": "Session deleted"}

