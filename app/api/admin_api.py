"""Admin API: knowledge CRUD, categories, test search, ElevenLabs sync."""
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from sqlalchemy import select, func
from datetime import datetime
from pathlib import Path
import logging
import uuid

from ..models.database import async_session, KnowledgeDocument

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin"])

KNOWLEDGE_DIR = Path(__file__).parent.parent.parent / "knowledge"


# =============================================================================
# Knowledge Base CRUD
# =============================================================================


class KnowledgeDocumentCreate(BaseModel):
    title: str
    content: str
    category: str
    filePath: Optional[str] = None
    workspace: Optional[str] = None
    tags: Optional[List[str]] = None


class KnowledgeDocumentUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    category: Optional[str] = None
    filePath: Optional[str] = None
    workspace: Optional[str] = None
    tags: Optional[List[str]] = None


@router.get("/knowledge")
async def list_knowledge_documents():
    """List all knowledge base documents."""
    async with async_session() as db:
        result = await db.execute(
            select(KnowledgeDocument).order_by(KnowledgeDocument.updated_at.desc())
        )
        documents = result.scalars().all()
        return {
            "documents": [doc.to_dict() for doc in documents],
            "count": len(documents),
        }


@router.get("/knowledge/{doc_id}")
async def get_knowledge_document(doc_id: str):
    """Get a specific knowledge document."""
    async with async_session() as db:
        result = await db.execute(
            select(KnowledgeDocument).where(KnowledgeDocument.id == doc_id)
        )
        doc = result.scalar_one_or_none()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        return doc.to_dict()


@router.post("/knowledge")
async def create_knowledge_document(data: KnowledgeDocumentCreate):
    """Create a new knowledge document."""
    async with async_session() as db:
        doc = KnowledgeDocument(
            id=str(uuid.uuid4()),
            title=data.title,
            content=data.content,
            category=data.category,
            file_path=data.filePath,
            workspace=data.workspace,
            tags=data.tags or [],
        )
        db.add(doc)
        await db.commit()
        await db.refresh(doc)
        logger.info(f"Created knowledge document: {doc.title}")
        return doc.to_dict()


@router.put("/knowledge/{doc_id}")
async def update_knowledge_document(doc_id: str, data: KnowledgeDocumentUpdate):
    """Update a knowledge document."""
    async with async_session() as db:
        result = await db.execute(
            select(KnowledgeDocument).where(KnowledgeDocument.id == doc_id)
        )
        doc = result.scalar_one_or_none()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        if data.title is not None:
            doc.title = data.title
        if data.content is not None:
            doc.content = data.content
        if data.category is not None:
            doc.category = data.category
        if data.filePath is not None:
            doc.file_path = data.filePath
        if data.workspace is not None:
            doc.workspace = data.workspace
        if data.tags is not None:
            doc.tags = data.tags

        doc.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(doc)
        logger.info(f"Updated knowledge document: {doc.title}")
        return doc.to_dict()


@router.delete("/knowledge/{doc_id}")
async def delete_knowledge_document(doc_id: str):
    """Delete a knowledge document."""
    async with async_session() as db:
        result = await db.execute(
            select(KnowledgeDocument).where(KnowledgeDocument.id == doc_id)
        )
        doc = result.scalar_one_or_none()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        await db.delete(doc)
        await db.commit()
        logger.info(f"Deleted knowledge document: {doc.title}")
        return {"status": "deleted", "id": doc_id}


# =============================================================================
# Categories
# =============================================================================


@router.get("/categories")
async def list_categories():
    """List all unique knowledge document categories."""
    async with async_session() as db:
        result = await db.execute(
            select(KnowledgeDocument.category, func.count(KnowledgeDocument.id))
            .group_by(KnowledgeDocument.category)
            .order_by(KnowledgeDocument.category)
        )
        categories = [{"name": name, "count": count} for name, count in result.all()]
        return {"categories": categories}


# =============================================================================
# Import from Files
# =============================================================================


@router.post("/knowledge/import")
async def import_knowledge_from_files():
    """Import knowledge documents from markdown files."""
    if not KNOWLEDGE_DIR.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Knowledge directory not found: {KNOWLEDGE_DIR}"
        )

    md_files = list(KNOWLEDGE_DIR.glob("**/*.md"))
    logger.info(f"Found {len(md_files)} markdown files to import")

    imported = 0
    updated = 0
    skipped = 0

    async with async_session() as db:
        for md_file in md_files:
            if md_file.name == "README.md" or md_file.name.endswith("~.md"):
                skipped += 1
                continue

            relative_path = md_file.relative_to(KNOWLEDGE_DIR)
            if len(relative_path.parts) > 1:
                category = relative_path.parts[0].replace("-", " ").replace("_", " ")
            else:
                category = "general"

            title = md_file.stem.replace("-", " ").replace("_", " ").title()

            try:
                content = md_file.read_text(encoding="utf-8")
            except Exception as e:
                logger.error(f"Error reading {md_file}: {e}")
                skipped += 1
                continue

            file_path_str = str(relative_path)
            result = await db.execute(
                select(KnowledgeDocument).where(
                    KnowledgeDocument.file_path == file_path_str
                )
            )
            existing = result.scalar_one_or_none()

            if existing:
                existing.title = title
                existing.content = content
                existing.category = category
                existing.updated_at = datetime.utcnow()
                updated += 1
            else:
                doc = KnowledgeDocument(
                    id=str(uuid.uuid4()),
                    title=title,
                    content=content,
                    category=category,
                    file_path=file_path_str,
                )
                db.add(doc)
                imported += 1

        await db.commit()

    logger.info(f"Import complete: {imported} imported, {updated} updated, {skipped} skipped")
    return {
        "status": "success",
        "imported": imported,
        "updated": updated,
        "skipped": skipped,
        "total": imported + updated,
    }


# =============================================================================
# Test Search
# =============================================================================


class TestKnowledgeRequest(BaseModel):
    query: str


@router.post("/test-knowledge")
async def test_knowledge_search(request: TestKnowledgeRequest):
    """Test knowledge base search and return matching documents with scores."""
    query = request.query
    if not query or len(query.strip()) < 3:
        return {"results": [], "query": query}

    search_text = query.lower()

    async with async_session() as db:
        result = await db.execute(select(KnowledgeDocument))
        documents = result.scalars().all()

        if not documents:
            return {"results": [], "query": query}

        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "need", "to", "of",
            "in", "for", "on", "with", "at", "by", "from", "as", "into", "through",
            "during", "before", "after", "above", "below", "between", "under",
            "i", "me", "my", "we", "our", "you", "your", "he", "him", "she", "her",
            "it", "its", "they", "them", "their", "this", "that", "these", "those",
            "what", "when", "where", "why", "how",
        }

        words = [w for w in search_text.split() if len(w) > 2 and w not in stop_words]

        results = []
        for doc in documents:
            content = doc.content.lower()
            title = doc.title.lower()

            score = 0
            matched_words = []
            for word in words:
                if word in title:
                    score += 3
                    matched_words.append(word)
                if word in content:
                    score += content.count(word)
                    if word not in matched_words:
                        matched_words.append(word)

            if score > 0:
                snippet = _extract_snippet(doc.content, matched_words)
                results.append({
                    "id": doc.id,
                    "title": doc.title,
                    "category": doc.category,
                    "score": score,
                    "matchedWords": matched_words,
                    "snippet": snippet,
                })

        results.sort(key=lambda x: x["score"], reverse=True)

        return {
            "results": results[:10],
            "query": query,
            "totalMatches": len(results),
        }


def _extract_snippet(content: str, keywords: List[str], max_length: int = 300) -> str:
    """Extract a relevant snippet from content based on keywords."""
    content_lower = content.lower()

    best_pos = len(content)
    for keyword in keywords:
        pos = content_lower.find(keyword)
        if 0 <= pos < best_pos:
            best_pos = pos

    if best_pos == len(content):
        return content[:max_length] + "..." if len(content) > max_length else content

    start = max(0, best_pos - 50)
    end = min(len(content), best_pos + max_length - 50)
    snippet = content[start:end]

    if start > 0:
        snippet = "..." + snippet
    if end < len(content):
        snippet = snippet + "..."

    return snippet


# =============================================================================
# ElevenLabs Sync
# =============================================================================


@router.post("/elevenlabs/sync")
async def sync_elevenlabs_knowledge():
    """Sync knowledge base to ElevenLabs voice agent."""
    from ..integrations.elevenlabs_sync import sync_knowledge_to_elevenlabs
    result = await sync_knowledge_to_elevenlabs()
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.get("/elevenlabs/status")
async def elevenlabs_sync_status():
    """Get ElevenLabs sync status."""
    from ..integrations.elevenlabs_sync import get_elevenlabs_sync_status
    return await get_elevenlabs_sync_status()


@router.get("/elevenlabs/verify")
async def elevenlabs_verify_documents():
    """Verify documents exist in ElevenLabs workspace."""
    from ..integrations.elevenlabs_sync import verify_elevenlabs_documents
    result = await verify_elevenlabs_documents()
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result
