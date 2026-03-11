"""Google Docs and Drive folder management endpoints."""
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from sqlalchemy import select
from datetime import datetime
import logging
import uuid

from ..models.database import (
    async_session,
    GoogleDocSource,
    GoogleDriveFolder,
    KnowledgeDocument,
)
from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/api", tags=["google-docs"])


# =============================================================================
# Google Docs Sources
# =============================================================================


class GoogleDocCreate(BaseModel):
    googleDocId: str
    title: str
    category: str = "general"
    workspace: Optional[str] = None


class GoogleDocUpdate(BaseModel):
    title: Optional[str] = None
    category: Optional[str] = None
    workspace: Optional[str] = None
    enabled: Optional[bool] = None


@router.get("/google-docs")
async def list_google_docs():
    """List all Google Doc sources."""
    async with async_session() as db:
        result = await db.execute(
            select(GoogleDocSource).order_by(GoogleDocSource.created_at.desc())
        )
        sources = result.scalars().all()
        return {"sources": [s.to_dict() for s in sources]}


@router.post("/google-docs")
async def add_google_doc(data: GoogleDocCreate):
    """Add a new Google Doc source."""
    async with async_session() as db:
        existing = (await db.execute(
            select(GoogleDocSource).where(
                GoogleDocSource.google_doc_id == data.googleDocId
            )
        )).scalar_one_or_none()
        if existing:
            raise HTTPException(status_code=409, detail="Google Doc already configured")

        source = GoogleDocSource(
            id=str(uuid.uuid4()),
            google_doc_id=data.googleDocId,
            title=data.title,
            category=data.category,
            workspace=data.workspace,
        )
        db.add(source)
        await db.commit()
        await db.refresh(source)
        return source.to_dict()


@router.patch("/google-docs/{source_id}")
async def update_google_doc(source_id: str, data: GoogleDocUpdate):
    """Update a Google Doc source."""
    async with async_session() as db:
        result = await db.execute(
            select(GoogleDocSource).where(GoogleDocSource.id == source_id)
        )
        source = result.scalar_one_or_none()
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        if data.title is not None:
            source.title = data.title
        if data.category is not None:
            source.category = data.category
        if data.workspace is not None:
            source.workspace = data.workspace
        if data.enabled is not None:
            source.enabled = data.enabled
        source.updated_at = datetime.utcnow()
        await db.commit()
        return source.to_dict()


@router.delete("/google-docs/{source_id}")
async def delete_google_doc(source_id: str):
    """Remove a Google Doc source."""
    async with async_session() as db:
        result = await db.execute(
            select(GoogleDocSource).where(GoogleDocSource.id == source_id)
        )
        source = result.scalar_one_or_none()
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        # Also delete the associated KnowledgeDocument
        kd_result = await db.execute(
            select(KnowledgeDocument).where(
                KnowledgeDocument.file_path == f"gdoc:{source.google_doc_id}"
            )
        )
        kd = kd_result.scalar_one_or_none()
        if kd:
            await db.delete(kd)

        await db.delete(source)
        await db.commit()
        return {"status": "deleted", "id": source_id}


@router.post("/google-docs/{source_id}/sync")
async def sync_google_doc(source_id: str):
    """Trigger sync for a single Google Doc."""
    from ..services.google_docs_sync import sync_single_doc
    result = await sync_single_doc(source_id)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.post("/google-docs/sync-all")
async def sync_all_docs():
    """Trigger sync for all enabled Google Docs."""
    from ..services.google_docs_sync import sync_all_google_docs
    return await sync_all_google_docs()


# =============================================================================
# Google Drive Folders
# =============================================================================


class DriveFolderCreate(BaseModel):
    folderId: str
    title: str
    category: str = "general"
    workspace: Optional[str] = None


class DriveFolderUpdate(BaseModel):
    title: Optional[str] = None
    category: Optional[str] = None
    workspace: Optional[str] = None
    enabled: Optional[bool] = None


@router.get("/drive-folders")
async def list_drive_folders():
    """List all Google Drive folder sources."""
    async with async_session() as db:
        result = await db.execute(
            select(GoogleDriveFolder).order_by(GoogleDriveFolder.created_at.desc())
        )
        folders = result.scalars().all()
        return {"folders": [f.to_dict() for f in folders]}


@router.post("/drive-folders")
async def add_drive_folder(data: DriveFolderCreate):
    """Add a new Google Drive folder source."""
    async with async_session() as db:
        existing = (await db.execute(
            select(GoogleDriveFolder).where(
                GoogleDriveFolder.folder_id == data.folderId
            )
        )).scalar_one_or_none()
        if existing:
            raise HTTPException(status_code=409, detail="Folder already configured")

        folder = GoogleDriveFolder(
            id=str(uuid.uuid4()),
            folder_id=data.folderId,
            title=data.title,
            category=data.category,
            workspace=data.workspace,
        )
        db.add(folder)
        await db.commit()
        await db.refresh(folder)
        return folder.to_dict()


@router.patch("/drive-folders/{folder_record_id}")
async def update_drive_folder(folder_record_id: str, data: DriveFolderUpdate):
    """Update a Drive folder source."""
    async with async_session() as db:
        result = await db.execute(
            select(GoogleDriveFolder).where(GoogleDriveFolder.id == folder_record_id)
        )
        folder = result.scalar_one_or_none()
        if not folder:
            raise HTTPException(status_code=404, detail="Folder not found")

        if data.title is not None:
            folder.title = data.title
        if data.category is not None:
            folder.category = data.category
        if data.workspace is not None:
            folder.workspace = data.workspace
        if data.enabled is not None:
            folder.enabled = data.enabled
        folder.updated_at = datetime.utcnow()
        await db.commit()
        return folder.to_dict()


@router.delete("/drive-folders/{folder_record_id}")
async def delete_drive_folder(folder_record_id: str):
    """Remove a Drive folder source."""
    async with async_session() as db:
        result = await db.execute(
            select(GoogleDriveFolder).where(GoogleDriveFolder.id == folder_record_id)
        )
        folder = result.scalar_one_or_none()
        if not folder:
            raise HTTPException(status_code=404, detail="Folder not found")

        await db.delete(folder)
        await db.commit()
        return {"status": "deleted", "id": folder_record_id}


@router.post("/drive-folders/{folder_record_id}/sync")
async def sync_drive_folder(folder_record_id: str):
    """Trigger sync for a single Drive folder."""
    from ..services.google_docs_sync import sync_folder
    result = await sync_folder(folder_record_id)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.post("/drive-folders/sync-all")
async def sync_all_drive_folders():
    """Trigger sync for all enabled Drive folders."""
    from ..services.google_docs_sync import sync_all_folders
    return await sync_all_folders()


# =============================================================================
# Sources Status
# =============================================================================


@router.get("/sources/status")
async def get_sources_status():
    """Get status of all knowledge sources."""
    from ..models.database import async_session as get_session
    from sqlalchemy import func

    async with get_session() as db:
        # Knowledge documents count
        kb_count = (await db.execute(
            select(func.count(KnowledgeDocument.id))
        )).scalar() or 0

        # Google Docs sources
        gdoc_count = (await db.execute(
            select(func.count(GoogleDocSource.id))
        )).scalar() or 0
        gdoc_enabled = (await db.execute(
            select(func.count(GoogleDocSource.id)).where(GoogleDocSource.enabled == True)
        )).scalar() or 0

        # Drive folders
        folder_count = (await db.execute(
            select(func.count(GoogleDriveFolder.id))
        )).scalar() or 0
        folder_enabled = (await db.execute(
            select(func.count(GoogleDriveFolder.id)).where(GoogleDriveFolder.enabled == True)
        )).scalar() or 0

    return {
        "knowledge_documents": kb_count,
        "google_docs": {
            "status": "active" if gdoc_enabled > 0 else "not_configured",
            "total": gdoc_count,
            "enabled": gdoc_enabled,
        },
        "drive_folders": {
            "status": "active" if folder_enabled > 0 else "not_configured",
            "total": folder_count,
            "enabled": folder_enabled,
        },
        "elevenlabs": {
            "status": "active" if settings.elevenlabs_api_key else "not_configured",
        },
    }
