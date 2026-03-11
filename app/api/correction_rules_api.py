"""CRUD API for admin-managed correction rules injected into chat system prompt."""
from __future__ import annotations

import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from sqlalchemy import select

from ..models.database import async_session, CorrectionRule

router = APIRouter(prefix="/api/correction-rules", tags=["correction-rules"])


class RuleCreate(BaseModel):
    rule_text: str


class RuleUpdate(BaseModel):
    rule_text: Optional[str] = None
    enabled: Optional[bool] = None


@router.get("")
async def list_rules():
    """List all correction rules (newest first)."""
    async with async_session() as db:
        result = await db.execute(
            select(CorrectionRule).order_by(CorrectionRule.created_at.desc())
        )
        rules = result.scalars().all()
    return {"rules": [r.to_dict() for r in rules]}


@router.post("")
async def create_rule(body: RuleCreate):
    """Create a new correction rule."""
    rule = CorrectionRule(
        id=str(uuid.uuid4()),
        rule_text=body.rule_text.strip(),
    )
    async with async_session() as db:
        db.add(rule)
        await db.commit()
        await db.refresh(rule)
    return rule.to_dict()


@router.patch("/{rule_id}")
async def update_rule(rule_id: str, body: RuleUpdate):
    """Update rule text and/or enabled state."""
    async with async_session() as db:
        result = await db.execute(
            select(CorrectionRule).where(CorrectionRule.id == rule_id)
        )
        rule = result.scalar_one_or_none()
        if not rule:
            raise HTTPException(status_code=404, detail="Rule not found")
        if body.rule_text is not None:
            rule.rule_text = body.rule_text.strip()
        if body.enabled is not None:
            rule.enabled = body.enabled
        await db.commit()
        await db.refresh(rule)
    return rule.to_dict()


@router.delete("/{rule_id}")
async def delete_rule(rule_id: str):
    """Delete a correction rule."""
    async with async_session() as db:
        result = await db.execute(
            select(CorrectionRule).where(CorrectionRule.id == rule_id)
        )
        rule = result.scalar_one_or_none()
        if not rule:
            raise HTTPException(status_code=404, detail="Rule not found")
        await db.delete(rule)
        await db.commit()
    return {"ok": True}
