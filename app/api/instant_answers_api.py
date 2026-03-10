"""Instant answers CRUD, suggested answers management, and unanswered questions."""
from __future__ import annotations
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List
from sqlalchemy import select, func, delete
from datetime import datetime
import logging
import uuid

from ..models.database import (
    async_session,
    InstantAnswer,
    SuggestedInstantAnswer,
    UnansweredQuestion,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["instant-answers"])


# =============================================================================
# Unanswered / Flagged Questions
# =============================================================================


@router.get("/unanswered")
async def list_unanswered(
    limit: int = 50,
    unresolved_only: bool = True,
):
    """List unanswered/flagged questions."""
    async with async_session() as db:
        query = select(UnansweredQuestion).order_by(
            UnansweredQuestion.last_asked.desc()
        ).limit(limit)
        if unresolved_only:
            query = query.where(UnansweredQuestion.resolved == False)
        result = await db.execute(query)
        questions = result.scalars().all()
        return {"questions": [q.to_dict() for q in questions]}


@router.post("/unanswered/{question}/resolve")
async def resolve_unanswered(question: str):
    """Mark an unanswered question as resolved."""
    async with async_session() as db:
        result = await db.execute(
            select(UnansweredQuestion).where(
                UnansweredQuestion.question == question
            )
        )
        q = result.scalar_one_or_none()
        if not q:
            raise HTTPException(status_code=404, detail="Question not found")
        q.resolved = True
        await db.commit()
        return {"status": "resolved"}


@router.delete("/unanswered/resolved")
async def clear_resolved_questions():
    """Delete all resolved unanswered questions."""
    async with async_session() as db:
        await db.execute(
            delete(UnansweredQuestion).where(UnansweredQuestion.resolved == True)
        )
        await db.commit()
        return {"status": "cleared"}


# =============================================================================
# Instant Answers CRUD
# =============================================================================


class InstantAnswerCreate(BaseModel):
    key: str
    answer: str
    talking_points: Optional[List[str]] = None
    suggested_response: Optional[str] = None
    confidence: Optional[str] = "medium"
    source_topic: Optional[str] = None


@router.get("/instant-answers")
async def list_instant_answers():
    """List all instant answers."""
    async with async_session() as db:
        result = await db.execute(
            select(InstantAnswer).order_by(InstantAnswer.key)
        )
        answers = result.scalars().all()
        return {"answers": [a.to_dict() for a in answers]}


@router.get("/instant-answers/{key:path}")
async def get_instant_answer(key: str):
    """Get a single instant answer by key."""
    async with async_session() as db:
        result = await db.execute(
            select(InstantAnswer).where(InstantAnswer.key == key)
        )
        answer = result.scalar_one_or_none()
        if not answer:
            raise HTTPException(status_code=404, detail="Answer not found")
        return answer.to_dict()


@router.post("/instant-answers")
async def create_instant_answer(data: InstantAnswerCreate):
    """Create a new instant answer."""
    async with async_session() as db:
        result = await db.execute(
            select(InstantAnswer).where(InstantAnswer.key == data.key)
        )
        if result.scalar_one_or_none():
            raise HTTPException(status_code=409, detail="Key already exists")

        answer = InstantAnswer(
            key=data.key,
            answer=data.answer,
            talking_points=data.talking_points or [],
            suggested_response=data.suggested_response,
            confidence=data.confidence or "medium",
            source_topic=data.source_topic,
        )
        db.add(answer)
        await db.commit()
        await db.refresh(answer)
        return answer.to_dict()


@router.put("/instant-answers/{key:path}")
async def update_instant_answer(key: str, data: InstantAnswerCreate):
    """Update an existing instant answer."""
    async with async_session() as db:
        result = await db.execute(
            select(InstantAnswer).where(InstantAnswer.key == key)
        )
        answer = result.scalar_one_or_none()
        if not answer:
            raise HTTPException(status_code=404, detail="Answer not found")

        answer.answer = data.answer
        answer.talking_points = data.talking_points or []
        answer.suggested_response = data.suggested_response
        answer.confidence = data.confidence or "medium"
        answer.source_topic = data.source_topic
        answer.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(answer)
        return answer.to_dict()


@router.delete("/instant-answers/{key:path}")
async def delete_instant_answer(key: str):
    """Delete an instant answer."""
    async with async_session() as db:
        result = await db.execute(
            select(InstantAnswer).where(InstantAnswer.key == key)
        )
        answer = result.scalar_one_or_none()
        if not answer:
            raise HTTPException(status_code=404, detail="Answer not found")
        await db.delete(answer)
        await db.commit()
        return {"status": "deleted", "key": key}


# =============================================================================
# Suggested Instant Answers
# =============================================================================


@router.get("/suggested-answers")
async def list_suggested_answers(
    status: str = Query(default="pending", description="Filter by status"),
    limit: int = 50,
):
    """List suggested instant answers."""
    async with async_session() as db:
        query = select(SuggestedInstantAnswer).order_by(
            SuggestedInstantAnswer.created_at.desc()
        ).limit(limit)
        if status != "all":
            query = query.where(SuggestedInstantAnswer.status == status)
        result = await db.execute(query)
        suggestions = result.scalars().all()
        return {"suggestions": [s.to_dict() for s in suggestions]}


@router.get("/suggested-answers/count")
async def suggested_answers_count():
    """Count pending suggested answers (for badge)."""
    async with async_session() as db:
        count = (await db.execute(
            select(func.count(SuggestedInstantAnswer.id)).where(
                SuggestedInstantAnswer.status == "pending"
            )
        )).scalar() or 0
        return {"count": count}


@router.post("/suggested-answers/{suggestion_id}/approve")
async def approve_suggested_answer(suggestion_id: str):
    """Approve a suggested answer — creates a real InstantAnswer."""
    async with async_session() as db:
        result = await db.execute(
            select(SuggestedInstantAnswer).where(SuggestedInstantAnswer.id == suggestion_id)
        )
        suggestion = result.scalar_one_or_none()
        if not suggestion:
            raise HTTPException(status_code=404, detail="Suggestion not found")

        # Check if key already exists
        existing = (await db.execute(
            select(InstantAnswer).where(InstantAnswer.key == suggestion.key)
        )).scalar_one_or_none()
        if existing:
            raise HTTPException(status_code=409, detail=f"Key '{suggestion.key}' already exists")

        # Create real instant answer
        answer = InstantAnswer(
            key=suggestion.key,
            answer=suggestion.answer,
            talking_points=suggestion.talking_points or [],
            suggested_response=suggestion.suggested_response,
            confidence=suggestion.confidence or "medium",
            source_topic=suggestion.source_topic,
        )
        db.add(answer)

        # Mark suggestion as approved
        suggestion.status = "approved"
        suggestion.reviewed_at = datetime.utcnow()
        await db.commit()

        return {"status": "approved", "key": suggestion.key}


@router.put("/suggested-answers/{suggestion_id}")
async def update_suggested_answer(suggestion_id: str, data: InstantAnswerCreate):
    """Edit a suggested answer before approving."""
    async with async_session() as db:
        result = await db.execute(
            select(SuggestedInstantAnswer).where(SuggestedInstantAnswer.id == suggestion_id)
        )
        suggestion = result.scalar_one_or_none()
        if not suggestion:
            raise HTTPException(status_code=404, detail="Suggestion not found")

        suggestion.key = data.key
        suggestion.answer = data.answer
        suggestion.talking_points = data.talking_points or []
        suggestion.suggested_response = data.suggested_response
        suggestion.confidence = data.confidence or "medium"
        suggestion.source_topic = data.source_topic
        await db.commit()
        return suggestion.to_dict()


@router.post("/suggested-answers/{suggestion_id}/dismiss")
async def dismiss_suggested_answer(suggestion_id: str):
    """Dismiss a suggested answer."""
    async with async_session() as db:
        result = await db.execute(
            select(SuggestedInstantAnswer).where(SuggestedInstantAnswer.id == suggestion_id)
        )
        suggestion = result.scalar_one_or_none()
        if not suggestion:
            raise HTTPException(status_code=404, detail="Suggestion not found")

        suggestion.status = "dismissed"
        suggestion.reviewed_at = datetime.utcnow()
        await db.commit()
        return {"status": "dismissed"}


@router.delete("/suggested-answers/dismissed")
async def clear_dismissed_suggestions():
    """Delete all dismissed suggestions."""
    async with async_session() as db:
        await db.execute(
            delete(SuggestedInstantAnswer).where(SuggestedInstantAnswer.status == "dismissed")
        )
        await db.commit()
        return {"status": "cleared"}
