"""API endpoints for Intercom workspace management and sync."""
from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from ..models.database import async_session, IntercomWorkspace, KnowledgeDocument
from ..services.intercom_sync import sync_workspace, sync_all_intercom

router = APIRouter()


class WorkspaceCreate(BaseModel):
    name: str
    access_token: str


@router.get("/api/intercom/workspaces")
async def list_workspaces():
    async with async_session() as db:
        result = await db.execute(
            select(IntercomWorkspace).order_by(IntercomWorkspace.created_at)
        )
        workspaces = result.scalars().all()
    return {"workspaces": [ws.to_dict() for ws in workspaces]}


@router.post("/api/intercom/workspaces")
async def add_workspace(body: WorkspaceCreate):
    ws = IntercomWorkspace(
        id=str(uuid.uuid4()),
        name=body.name,
        access_token=body.access_token,
    )
    async with async_session() as db:
        db.add(ws)
        await db.commit()
        await db.refresh(ws)

    try:
        sync_result = await sync_workspace(ws)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Workspace saved but initial sync failed: {e}"
        )

    # Re-fetch updated workspace (sync updated article_count etc.)
    async with async_session() as db:
        result = await db.execute(
            select(IntercomWorkspace).where(IntercomWorkspace.id == ws.id)
        )
        ws_updated = result.scalar_one()
        return {**ws_updated.to_dict(), "sync": sync_result}


@router.delete("/api/intercom/workspaces/{workspace_id}")
async def delete_workspace(workspace_id: str):
    async with async_session() as db:
        result = await db.execute(
            select(IntercomWorkspace).where(IntercomWorkspace.id == workspace_id)
        )
        ws = result.scalar_one_or_none()
        if not ws:
            raise HTTPException(status_code=404, detail="Workspace not found")
        workspace_name = ws.name
        await db.delete(ws)
        await db.commit()

    # Also remove synced articles for this workspace
    async with async_session() as db:
        result = await db.execute(
            select(KnowledgeDocument).where(
                KnowledgeDocument.file_path.like(f"intercom:{workspace_name}:%")
            )
        )
        docs = result.scalars().all()
        for doc in docs:
            await db.delete(doc)
        await db.commit()

    return {"ok": True, "articles_removed": len(docs)}


@router.post("/api/intercom/workspaces/{workspace_id}/sync")
async def sync_one(workspace_id: str):
    async with async_session() as db:
        result = await db.execute(
            select(IntercomWorkspace).where(IntercomWorkspace.id == workspace_id)
        )
        ws = result.scalar_one_or_none()
        if not ws:
            raise HTTPException(status_code=404, detail="Workspace not found")

    try:
        return await sync_workspace(ws)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/intercom/sync-all")
async def sync_all():
    return await sync_all_intercom()
