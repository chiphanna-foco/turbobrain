"""Chat API: question answering with KB context and feedback collection."""
from __future__ import annotations

import uuid
import logging
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
from sqlalchemy import select

from anthropic import AsyncAnthropic
from ..models.database import async_session, KnowledgeDocument, InstantAnswer, ChatFeedback
from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/api/chat", tags=["chat"])

STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "to", "of",
    "in", "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "i", "me", "my", "we", "our", "you", "your", "he", "him", "she", "her",
    "it", "its", "they", "them", "their", "this", "that", "these", "those",
    "what", "when", "where", "why", "how", "about", "just", "like", "get",
    "not", "but", "all", "also", "some", "any", "out",
}


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None


class FeedbackRequest(BaseModel):
    message_id: str
    feedback: str   # "positive" | "negative"
    notes: Optional[str] = None


async def _search_kb(query: str):
    """Keyword-search instant answers and KB docs. Returns (ia_list, doc_list)."""
    words = [w for w in query.lower().split() if len(w) > 2 and w not in STOP_WORDS]
    if not words:
        return [], []

    # Instant answers
    ia_matches = []
    async with async_session() as db:
        result = await db.execute(select(InstantAnswer))
        for ia in result.scalars().all():
            score = 0
            key_text = ia.key.lower().replace("_", " ")
            answer_text = (ia.answer or "").lower()
            for w in words:
                if w in key_text:
                    score += 5
                if w in answer_text:
                    score += answer_text.count(w)
            if score > 0:
                ia_matches.append((ia, score))
    ia_matches.sort(key=lambda x: x[1], reverse=True)

    # KB documents
    doc_matches = []
    async with async_session() as db:
        result = await db.execute(select(KnowledgeDocument))
        for doc in result.scalars().all():
            score = 0
            title_lower = doc.title.lower()
            content_lower = doc.content.lower()
            for w in words:
                if w in title_lower:
                    score += 3
                if w in content_lower:
                    score += content_lower.count(w)
            if score > 0:
                doc_matches.append((doc, score))
    doc_matches.sort(key=lambda x: x[1], reverse=True)

    return [m[0] for m in ia_matches[:3]], [m[0] for m in doc_matches[:5]]


@router.post("")
async def chat(body: ChatRequest):
    """Answer a question using KB context + Claude."""
    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    ia_list, doc_list = await _search_kb(message)

    # Build context string
    context_parts = []
    if ia_list:
        context_parts.append("### Quick Reference")
        for ia in ia_list:
            context_parts.append(f"**{ia.key.replace('_', ' ').title()}**: {ia.answer}")
            if ia.talking_points:
                for pt in ia.talking_points:
                    context_parts.append(f"  • {pt}")

    sources = []
    if doc_list:
        context_parts.append("\n### Knowledge Base Articles")
        for doc in doc_list:
            context_parts.append(f"\n**{doc.title}**")
            context_parts.append(doc.content[:2000])
            sources.append({"title": doc.title, "source_url": doc.source_url})

    context = "\n".join(context_parts) if context_parts else "No relevant articles found in the knowledge base."

    # Generate answer with Claude
    answer = "I couldn't find relevant information to answer that question. Try rephrasing or check the knowledge base directly."

    if settings.anthropic_api_key:
        try:
            client = AsyncAnthropic(api_key=settings.anthropic_api_key)
            response = await client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=800,
                system=(
                    "You are a helpful internal knowledge base assistant. "
                    "Answer the question using ONLY the provided context. "
                    "Be concise and direct. If the context doesn't contain the answer, say so honestly — do not make up information. "
                    "Format your response in plain text (no markdown)."
                ),
                messages=[{
                    "role": "user",
                    "content": f"Context:\n{context}\n\n---\nQuestion: {message}",
                }],
            )
            answer = response.content[0].text
        except Exception as e:
            logger.error(f"Claude error: {e}")
            # Fall back to top KB snippet
            if doc_list:
                answer = doc_list[0].content[:600]
            elif ia_list:
                answer = ia_list[0].answer
    else:
        # No Claude key — return best snippet
        if ia_list:
            answer = ia_list[0].answer
        elif doc_list:
            answer = doc_list[0].content[:600]

    # Persist for feedback tracking
    message_id = str(uuid.uuid4())
    async with async_session() as db:
        db.add(ChatFeedback(
            id=str(uuid.uuid4()),
            message_id=message_id,
            question=message,
            answer=answer,
            sources=sources,
        ))
        await db.commit()

    return {"message_id": message_id, "answer": answer, "sources": sources}


@router.post("/feedback")
async def submit_feedback(body: FeedbackRequest):
    """Record thumbs up / down and optional notes for a chat response."""
    if body.feedback not in ("positive", "negative"):
        raise HTTPException(status_code=400, detail="feedback must be 'positive' or 'negative'")

    async with async_session() as db:
        result = await db.execute(
            select(ChatFeedback).where(ChatFeedback.message_id == body.message_id)
        )
        record = result.scalar_one_or_none()
        if not record:
            raise HTTPException(status_code=404, detail="Message not found")
        record.feedback = body.feedback
        record.notes = body.notes or None
        await db.commit()

    return {"ok": True}


@router.get("/feedback")
async def list_feedback(limit: int = Query(default=200)):
    """List all chat feedback entries (newest first)."""
    async with async_session() as db:
        result = await db.execute(
            select(ChatFeedback).order_by(ChatFeedback.created_at.desc()).limit(limit)
        )
        records = result.scalars().all()
    return {"feedback": [r.to_dict() for r in records]}
