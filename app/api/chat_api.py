"""Chat API: multi-turn question answering with KB context, correction rules, and feedback."""
from __future__ import annotations

import uuid
import logging
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
from sqlalchemy import select, desc

from anthropic import AsyncAnthropic
from ..models.database import (
    async_session, KnowledgeDocument, InstantAnswer,
    ChatFeedback, ConversationMessage, CorrectionRule,
)
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


def _extract_best_snippet(content: str, keywords: list[str], max_chars: int = 1200) -> str:
    """Return the most relevant portion of a document using a sliding window approach."""
    if not keywords or len(content) <= max_chars:
        return content[:max_chars]

    window_size = 400
    step = 150
    content_lower = content.lower()

    windows: list[tuple[int, int, int]] = []  # (score, start, end)
    pos = 0
    while pos < len(content):
        end = min(pos + window_size, len(content))
        window_text = content_lower[pos:end]
        score = sum(window_text.count(kw) for kw in keywords)
        if score > 0:
            windows.append((score, pos, end))
        pos += step

    if not windows:
        return content[:max_chars]

    windows.sort(key=lambda x: x[0], reverse=True)

    # Pick top 2 non-overlapping windows
    selected: list[tuple[int, int]] = []
    for score, start, end in windows:
        overlaps = any(
            not (end <= s or start >= e) for s, e in selected
        )
        if not overlaps:
            selected.append((start, end))
        if len(selected) == 2:
            break

    selected.sort()
    parts = [content[s:e].strip() for s, e in selected]
    result = " ... ".join(parts)
    return result[:max_chars]


async def _rewrite_query_if_short(message: str, client: AsyncAnthropic) -> str:
    """Expand short queries via Haiku to improve keyword search recall."""
    words = [w for w in message.lower().split() if w not in STOP_WORDS and len(w) > 2]
    if len(words) >= 4:
        return message

    try:
        resp = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=80,
            system="Rewrite the user's short question as a more specific search query (1 sentence, no fluff). Return only the rewritten query.",
            messages=[{"role": "user", "content": message}],
        )
        return resp.content[0].text.strip()
    except Exception:
        return message


async def _search_kb(query: str):
    """Weighted keyword search across instant answers and KB docs. Returns (ia_list, doc_list, keywords)."""
    words = [w for w in query.lower().split() if len(w) > 2 and w not in STOP_WORDS]
    if not words:
        return [], [], words

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
                    score += 10
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
                    score += 5
                if w in content_lower:
                    score += content_lower.count(w)
            if score > 0:
                doc_matches.append((doc, score))
    doc_matches.sort(key=lambda x: x[1], reverse=True)

    # Adaptive result count based on top score
    top_score = doc_matches[0][1] if doc_matches else 0
    if top_score >= 20:
        n_docs, n_ias = 2, 2
    elif top_score >= 10:
        n_docs, n_ias = 3, 3
    else:
        n_docs, n_ias = 5, 3

    return (
        [m[0] for m in ia_matches[:n_ias]],
        [m[0] for m in doc_matches[:n_docs]],
        words,
    )


async def _load_conversation_history(conversation_id: str, limit: int = 8) -> list[dict]:
    """Return last N turns of a conversation in chronological order."""
    async with async_session() as db:
        result = await db.execute(
            select(ConversationMessage)
            .where(ConversationMessage.conversation_id == conversation_id)
            .order_by(desc(ConversationMessage.created_at))
            .limit(limit)
        )
        messages = result.scalars().all()
    return [{"role": m.role, "content": m.content} for m in reversed(messages)]


async def _load_correction_rules() -> list[str]:
    """Return enabled correction rule texts."""
    async with async_session() as db:
        result = await db.execute(
            select(CorrectionRule).where(CorrectionRule.enabled == True)
        )
        return [r.rule_text for r in result.scalars().all()]


def _build_system_prompt(correction_rules: list[str]) -> str:
    base = (
        "You are TurboBrain, an internal knowledge base assistant for a support team.\n"
        "1. Answer using ONLY the context provided. Do not invent facts.\n"
        "2. If the context doesn't contain a reliable answer, say exactly: "
        "\"I don't have a reliable answer for that in the knowledge base.\"\n"
        "3. Reference which article your answer comes from when possible.\n"
        "4. Be concise. Bullets are fine for steps or lists.\n"
        "5. This is a multi-turn conversation — use conversation history for context.\n"
        "6. Format your response in plain text (no markdown)."
    )
    if correction_rules:
        rules_block = "\n\n[Organization-Specific Rules — follow these over general guidelines]:\n"
        rules_block += "\n".join(f"- {r}" for r in correction_rules)
        return base + rules_block
    return base


@router.post("")
async def chat(body: ChatRequest):
    """Answer a question using KB context + Claude Sonnet with conversation history."""
    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    conversation_id = body.conversation_id or str(uuid.uuid4())

    client = AsyncAnthropic(api_key=settings.anthropic_api_key) if settings.anthropic_api_key else None

    # Optionally expand short queries
    search_query = message
    if client:
        search_query = await _rewrite_query_if_short(message, client)
        if search_query != message:
            logger.info(f"Query rewritten: '{message}' → '{search_query}'")

    ia_list, doc_list, query_words = await _search_kb(search_query)

    # Build context string using density-aware snippet extraction
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
            snippet = _extract_best_snippet(doc.content, query_words)
            context_parts.append(f"\n**{doc.title}**")
            context_parts.append(snippet)
            sources.append({"title": doc.title, "source_url": doc.source_url})

    context = "\n".join(context_parts) if context_parts else "No relevant articles found in the knowledge base."

    # Load conversation history and correction rules
    history = await _load_conversation_history(conversation_id)
    correction_rules = await _load_correction_rules()
    system_prompt = _build_system_prompt(correction_rules)

    answer = "I don't have a reliable answer for that in the knowledge base."

    if client:
        try:
            # Build messages array: history + current user turn
            claude_messages = history + [
                {"role": "user", "content": f"Context:\n{context}\n\n---\nQuestion: {message}"}
            ]
            response = await client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1000,
                system=system_prompt,
                messages=claude_messages,
            )
            answer = response.content[0].text
        except Exception as e:
            logger.error(f"Claude error: {e}")
            if doc_list:
                answer = _extract_best_snippet(doc_list[0].content, query_words, max_chars=600)
            elif ia_list:
                answer = ia_list[0].answer
    else:
        if ia_list:
            answer = ia_list[0].answer
        elif doc_list:
            answer = _extract_best_snippet(doc_list[0].content, query_words, max_chars=600)

    # Persist conversation turns
    async with async_session() as db:
        db.add(ConversationMessage(
            id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            role="user",
            content=message,
        ))
        db.add(ConversationMessage(
            id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            role="assistant",
            content=answer,
        ))
        await db.commit()

    # Persist for feedback tracking
    message_id = str(uuid.uuid4())
    async with async_session() as db:
        db.add(ChatFeedback(
            id=str(uuid.uuid4()),
            message_id=message_id,
            conversation_id=conversation_id,
            question=message,
            answer=answer,
            sources=sources,
        ))
        await db.commit()

    return {"message_id": message_id, "conversation_id": conversation_id, "answer": answer, "sources": sources}


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
