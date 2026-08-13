"""
MLVerse X — RAG System API endpoints
"""
import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel

from app.core.database import get_db
from app.core.storage import upload_file, download_file
from app.core.config import settings
from app.models.models import User, Dataset
from app.api.v1.endpoints.auth import get_current_active_user
from ai.rag.pipeline import RAGPipeline

router = APIRouter()
rag = RAGPipeline()


class QueryRequest(BaseModel):
    query: str
    collection_ids: List[str]
    model: Optional[str] = None
    top_k: int = 5


class CreateCollectionRequest(BaseModel):
    name: str


@router.get("/collections")
async def list_collections(
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """List all RAG document collections for the user."""
    from app.models.models import RAGDocument
    
    result = await db.execute(
        select(RAGDocument).where(RAGDocument.owner_id == user.id)
    )
    docs = result.scalars().all()
    
    collections_dict = {}
    for d in docs:
        cid = d.chroma_collection_id
        if not cid:
            continue
        if cid not in collections_dict:
            # Generate a nice human-readable name if not stored
            name_part = cid.replace(f"user_{user.id}_", "")
            # capitalize first letter
            name = name_part.capitalize() if name_part else "Collection"
            collections_dict[cid] = {
                "id": cid,
                "name": name,
                "documents": [],
            }
        collections_dict[cid]["documents"].append({
            "id": d.id,
            "name": d.name,
            "file_type": d.file_type,
            "file_size_bytes": d.file_size_bytes,
            "num_chunks": d.num_chunks,
            "created_at": d.created_at.isoformat(),
        })
        
    return {"collections": list(collections_dict.values()), "total": len(collections_dict)}


@router.post("/collections")
async def create_collection(
    body: CreateCollectionRequest,
    user: User = Depends(get_current_active_user),
):
    # Use lowercase alphanumeric + underscores for safe ChromaDB names
    clean_name = "".join(c for c in body.name.lower() if c.isalnum() or c in ("-", "_"))
    collection_id = f"user_{user.id}_{clean_name}_{uuid.uuid4().hex[:6]}"
    return {"collection_id": collection_id, "name": body.name}


@router.post("/ingest")
async def ingest_document(
    collection_id: str = Form(...),
    file: UploadFile = File(...),
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Upload and ingest a document into a RAG collection."""
    from app.models.models import RAGDocument
    file_bytes = await file.read()
    file_type = file.filename.split(".")[-1] if file.filename else "txt"

    try:
        result = await rag.ingest_document(
            file_bytes=file_bytes,
            filename=file.filename or "document",
            file_type=file_type,
            collection_id=collection_id,
            user_id=user.id,
        )
        
        # Save document metadata to PostgreSQL/SQLite
        doc = RAGDocument(
            owner_id=user.id,
            name=file.filename or "document",
            file_type=file_type,
            file_size_bytes=len(file_bytes),
            minio_object_key=f"rag/{user.id}/{collection_id}/{file.filename}",
            chroma_collection_id=collection_id,
            num_chunks=result["num_chunks"],
            is_indexed=True,
        )
        db.add(doc)
        await db.commit()
        
        return {
            "status": "success",
            "message": f"Document ingested: {result['num_chunks']} chunks indexed",
            **result,
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.post("/query")
async def query_rag(
    body: QueryRequest,
    user: User = Depends(get_current_active_user),
):
    """Query RAG system and get AI-generated answer with citations."""
    try:
        result = await rag.answer(
            query=body.query,
            collection_ids=body.collection_ids,
            model=body.model,
            top_k=body.top_k,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/retrieve")
async def retrieve_chunks(
    query: str,
    collection_id: str,
    top_k: int = 5,
    user: User = Depends(get_current_active_user),
):
    """Retrieve relevant chunks without generating an answer."""
    chunks = await rag.retrieve(query=query, collection_id=collection_id, top_k=top_k)
    return {"query": query, "chunks": chunks, "total": len(chunks)}


@router.post("/explain-pdf")
async def explain_pdf_document(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    user: User = Depends(get_current_active_user),
):
    """Upload a PDF and get structured content explanation and executive summary."""
    file_bytes = await file.read()
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported by this endpoint")

    try:
        res = await rag.explain_pdf(
            file_bytes=file_bytes,
            filename=file.filename or "document.pdf",
            model=model,
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.delete("/collections/{collection_id}")
async def delete_collection(
    collection_id: str,
    user: User = Depends(get_current_active_user),
):
    await rag.delete_collection(collection_id)
    return {"message": f"Collection {collection_id} deleted"}

