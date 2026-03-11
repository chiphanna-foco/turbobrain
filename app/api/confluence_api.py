"""Confluence Space management endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from sqlalchemy import select
from datetime import datetime
import logging
import uuid

from ..models.database import async_session, ConfluenceSpace, KnowledgeDocument

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["confluence"])


class ConfluenceSpaceCreate(BaseModel):
    name: str
    domain: str          # e.g. "mycompany.atlassian.net"
    email: str
    apiToken: str
    spaceKey: str
    workspace: Optional[str] = None


@router.get("/confluence/spaces")
async def list_confluence_spaces():
    """List all Confluence spaces."""
    async with async_session() as db:
        result = await db.execute(
            select(ConfluenceSpace).order_by(ConfluenceSpace.created_at.desc())
        )
        spaces = result.scalars().all()
        return {"spaces": [s.to_dict() for s in spaces]}


@router.post("/confluence/spaces")
async def add_confluence_space(data: ConfluenceSpaceCreate):
    """Add a new Confluence space and trigger initial sync."""
    async with async_session() as db:
        existing = (await db.execute(
            select(ConfluenceSpace).where(
                (ConfluenceSpace.domain == data.domain) &
                (ConfluenceSpace.space_key == data.spaceKey)
            )
        )).scalar_one_or_none()
        if existing:
            raise HTTPException(status_code=409, detail="Confluence space already configured")

        space = ConfluenceSpace(
            id=str(uuid.uuid4()),
            name=data.name,
            domain=data.domain,
            email=data.email,
            api_token=data.apiToken,
            space_key=data.spaceKey,
            workspace=data.workspace,
        )
        db.add(space)
        await db.commit()
        await db.refresh(space)

    from ..services.confluence_sync import sync_space
    sync_result = await sync_space(space)
    if "error" in sync_result:
        logger.warning(f"Initial sync warning for Confluence space {data.name}: {sync_result['error']}")

    return {**space.to_dict(), "syncResult": sync_result}


@router.delete("/confluence/spaces/{space_id}")
async def delete_confluence_space(space_id: str):
    """Remove a Confluence space and all its synced articles."""
    async with async_session() as db:
        result = await db.execute(
            select(ConfluenceSpace).where(ConfluenceSpace.id == space_id)
        )
        space = result.scalar_one_or_none()
        if not space:
            raise HTTPException(status_code=404, detail="Space not found")

        # Delete all KB docs whose file_path starts with confluence:{space_id}:
        kd_result = await db.execute(select(KnowledgeDocument))
        all_docs = kd_result.scalars().all()
        prefix = f"confluence:{space_id}:"
        for doc in all_docs:
            if doc.file_path and doc.file_path.startswith(prefix):
                await db.delete(doc)

        await db.delete(space)
        await db.commit()
        return {"status": "deleted", "id": space_id}


@router.post("/confluence/spaces/{space_id}/sync")
async def sync_confluence_space(space_id: str):
    """Trigger sync for a single Confluence space."""
    async with async_session() as db:
        result = await db.execute(
            select(ConfluenceSpace).where(ConfluenceSpace.id == space_id)
        )
        space = result.scalar_one_or_none()
        if not space:
            raise HTTPException(status_code=404, detail="Space not found")

    from ..services.confluence_sync import sync_space
    result = await sync_space(space)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.post("/confluence/sync-all")
async def sync_all_confluence_spaces():
    """Trigger sync for all enabled Confluence spaces."""
    from ..services.confluence_sync import sync_all_confluence
    return await sync_all_confluence()
