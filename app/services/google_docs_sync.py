"""Google Docs knowledge base sync: fetches Google Docs via public export URLs."""
from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import uuid
from datetime import datetime

import httpx
from sqlalchemy import select

from ..models.database import async_session, GoogleDocSource, GoogleDriveFolder, KnowledgeDocument
from ..config import get_settings

logger = logging.getLogger(__name__)

EXPORT_URL = "https://docs.google.com/document/d/{doc_id}/export?format=txt"
DRIVE_PDF_URL = "https://drive.google.com/uc?id={file_id}&export=download"
DRIVE_FILES_URL = "https://www.googleapis.com/drive/v3/files"
REFRESH_INTERVAL_SECONDS = 24 * 60 * 60  # 24 hours


async def fetch_google_doc(doc_id: str) -> str:
    """Fetch plain text content of a Google Doc via its public export URL."""
    url = EXPORT_URL.format(doc_id=doc_id)
    async with httpx.AsyncClient(follow_redirects=True, timeout=60) as client:
        resp = await client.get(url)
        resp.raise_for_status()
    return resp.text


async def sync_single_doc(source_id: str) -> dict:
    """Sync a single Google Doc source by its database ID."""
    async with async_session() as db:
        result = await db.execute(
            select(GoogleDocSource).where(GoogleDocSource.id == source_id)
        )
        source = result.scalar_one_or_none()
        if not source:
            return {"error": "Source not found"}

    return await _sync_source(source)


async def _sync_source(source: GoogleDocSource) -> dict:
    """Fetch a Google Doc and upsert its content into KnowledgeDocument."""
    try:
        content = await fetch_google_doc(source.google_doc_id)
        content_hash = hashlib.md5(content.encode()).hexdigest()

        # Check if content changed
        if content_hash == source.content_hash:
            # Update last_synced_at even if unchanged
            async with async_session() as db:
                result = await db.execute(
                    select(GoogleDocSource).where(GoogleDocSource.id == source.id)
                )
                src = result.scalar_one()
                src.last_synced_at = datetime.utcnow()
                src.last_sync_status = "success"
                src.last_error = None
                await db.commit()
            return {"status": "unchanged", "source_id": source.id}

        # Content changed — upsert KnowledgeDocument
        file_path_key = f"gdoc:{source.google_doc_id}"
        changed_doc_id = None
        doc_url = f"https://docs.google.com/document/d/{source.google_doc_id}"

        async with async_session() as db:
            # Upsert KnowledgeDocument
            result = await db.execute(
                select(KnowledgeDocument).where(
                    KnowledgeDocument.file_path == file_path_key
                )
            )
            existing = result.scalar_one_or_none()

            if existing:
                existing.title = source.title
                existing.content = content
                existing.category = source.category
                existing.workspace = source.workspace
                existing.source_url = doc_url
                existing.updated_at = datetime.utcnow()
                changed_doc_id = existing.id
            else:
                import uuid
                doc = KnowledgeDocument(
                    id=str(uuid.uuid4()),
                    title=source.title,
                    content=content,
                    category=source.category,
                    file_path=file_path_key,
                    workspace=source.workspace,
                    source_url=doc_url,
                )
                db.add(doc)
                changed_doc_id = doc.id

            # Update source sync status
            result2 = await db.execute(
                select(GoogleDocSource).where(GoogleDocSource.id == source.id)
            )
            src = result2.scalar_one()
            src.content_hash = content_hash
            src.last_synced_at = datetime.utcnow()
            src.last_sync_status = "success"
            src.last_error = None

            await db.commit()

        logger.info(f"Synced Google Doc '{source.title}' (doc_id={source.google_doc_id})")
        return {"status": "updated", "source_id": source.id, "changed_doc_id": changed_doc_id}

    except Exception as e:
        logger.error(f"Failed to sync Google Doc '{source.title}': {e}")
        try:
            async with async_session() as db:
                result = await db.execute(
                    select(GoogleDocSource).where(GoogleDocSource.id == source.id)
                )
                src = result.scalar_one()
                src.last_synced_at = datetime.utcnow()
                src.last_sync_status = "error"
                src.last_error = str(e)[:500]
                await db.commit()
        except Exception:
            pass
        return {"status": "error", "source_id": source.id, "error": str(e)}


async def sync_all_google_docs() -> dict:
    """Sync all enabled Google Doc sources."""
    async with async_session() as db:
        result = await db.execute(
            select(GoogleDocSource).where(GoogleDocSource.enabled == True)
        )
        sources = result.scalars().all()

    if not sources:
        return {"synced": 0, "skipped": 0, "errors": 0, "changed_doc_ids": []}

    synced = 0
    skipped = 0
    errors = 0
    changed_doc_ids = []

    for source in sources:
        result = await _sync_source(source)
        if result.get("status") == "updated":
            synced += 1
            if result.get("changed_doc_id"):
                changed_doc_ids.append(result["changed_doc_id"])
        elif result.get("status") == "unchanged":
            skipped += 1
        else:
            errors += 1

    logger.info(f"Google Docs sync: {synced} updated, {skipped} unchanged, {errors} errors")
    return {
        "synced": synced,
        "skipped": skipped,
        "errors": errors,
        "changed_doc_ids": changed_doc_ids,
    }


async def fetch_drive_pdf_text(file_id: str) -> str:
    """Download a PDF from Google Drive and extract text with pdfplumber."""
    url = DRIVE_PDF_URL.format(file_id=file_id)

    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()

        # Google shows a virus-scan confirmation page for larger files;
        # look for a confirm token and retry if present
        if b"%PDF" not in resp.content[:1024] and b"confirm=" in resp.content:
            import re as _re
            m = _re.search(rb'confirm=([0-9A-Za-z_-]+)', resp.content)
            if m:
                confirm_url = f"{url}&confirm={m.group(1).decode()}"
                resp = await client.get(confirm_url)
                resp.raise_for_status()

    import pdfplumber

    text_parts = []
    with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

    return "\n\n".join(text_parts)


async def _sync_pdf(file_id: str, name: str, category: str, workspace: str = None) -> dict:
    """Download a PDF from Drive, extract text, and upsert into KnowledgeDocument."""
    try:
        text = await fetch_drive_pdf_text(file_id)
        if not text.strip():
            logger.warning(f"PDF '{name}' ({file_id}) has no extractable text")
            return {"status": "skipped", "file_id": file_id}

        content_hash = hashlib.md5(text.encode()).hexdigest()
        file_path_key = f"gdrive_pdf:{file_id}"
        changed_doc_id = None

        async with async_session() as db:
            result = await db.execute(
                select(KnowledgeDocument).where(
                    KnowledgeDocument.file_path == file_path_key
                )
            )
            existing = result.scalar_one_or_none()

            if existing:
                old_hash = hashlib.md5(existing.content.encode()).hexdigest()
                if old_hash == content_hash:
                    return {"status": "unchanged", "file_id": file_id}
                existing.title = name
                existing.content = text
                existing.category = category
                existing.workspace = workspace
                existing.updated_at = datetime.utcnow()
                changed_doc_id = existing.id
            else:
                doc = KnowledgeDocument(
                    id=str(uuid.uuid4()),
                    title=name,
                    content=text,
                    category=category,
                    file_path=file_path_key,
                    workspace=workspace,
                )
                db.add(doc)
                changed_doc_id = doc.id

            await db.commit()

        logger.info(f"Synced PDF '{name}' (file_id={file_id})")
        return {"status": "updated", "file_id": file_id, "changed_doc_id": changed_doc_id}

    except Exception as e:
        logger.error(f"Failed to sync PDF '{name}': {e}")
        return {"status": "error", "file_id": file_id, "error": str(e)}


async def list_folder_files(folder_id: str, _depth: int = 0) -> list[dict]:
    """List all Google Docs and PDFs in a Drive folder, recursing into subfolders.

    Returns a list of dicts with 'id', 'name', and 'type' ('doc' or 'pdf').
    Requires GOOGLE_DRIVE_API env var to be set.
    """
    if _depth > 5:
        logger.warning(f"Skipping folder {folder_id}: max recursion depth reached")
        return []

    settings = get_settings()
    if not settings.google_drive_api:
        raise ValueError("GOOGLE_DRIVE_API is not configured")

    files = []
    page_token = None

    mime_filter = (
        f"'{folder_id}' in parents and trashed=false and ("
        "mimeType='application/vnd.google-apps.document' or "
        "mimeType='application/pdf' or "
        "mimeType='application/vnd.google-apps.folder'"
        ")"
    )

    async with httpx.AsyncClient(timeout=30) as client:
        while True:
            params = {
                "q": mime_filter,
                "key": settings.google_drive_api,
                "fields": "nextPageToken,files(id,name,mimeType)",
                "pageSize": 100,
            }
            if page_token:
                params["pageToken"] = page_token

            resp = await client.get(DRIVE_FILES_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

            for f in data.get("files", []):
                mime = f["mimeType"]
                if mime == "application/vnd.google-apps.folder":
                    subfolder_files = await list_folder_files(f["id"], _depth + 1)
                    files.extend(subfolder_files)
                elif mime == "application/vnd.google-apps.document":
                    files.append({"id": f["id"], "name": f["name"], "type": "doc"})
                elif mime == "application/pdf":
                    files.append({"id": f["id"], "name": f["name"], "type": "pdf"})

            page_token = data.get("nextPageToken")
            if not page_token:
                break

    return files


async def sync_folder(folder_record_id: str) -> dict:
    """Sync a single Google Drive folder: discover docs/PDFs and sync each one.

    Recursively discovers Google Docs and PDFs in the folder via Drive API,
    auto-creates GoogleDocSource entries for docs, syncs PDFs directly,
    and stores all content in KnowledgeDocument.
    """
    async with async_session() as db:
        result = await db.execute(
            select(GoogleDriveFolder).where(GoogleDriveFolder.id == folder_record_id)
        )
        folder = result.scalar_one_or_none()
        if not folder:
            return {"error": "Folder not found"}

    try:
        discovered = await list_folder_files(folder.folder_id)
        docs = [f for f in discovered if f["type"] == "doc"]
        pdfs = [f for f in discovered if f["type"] == "pdf"]

        new_sources = 0
        synced = 0
        errors = 0
        changed_doc_ids = []

        # --- Handle Google Docs (via GoogleDocSource) ---
        async with async_session() as db:
            for doc_info in docs:
                existing = (await db.execute(
                    select(GoogleDocSource).where(
                        GoogleDocSource.google_doc_id == doc_info["id"]
                    )
                )).scalar_one_or_none()

                if not existing:
                    source = GoogleDocSource(
                        id=str(uuid.uuid4()),
                        google_doc_id=doc_info["id"],
                        title=doc_info["name"],
                        category=folder.category,
                        workspace=folder.workspace,
                    )
                    db.add(source)
                    new_sources += 1

            await db.commit()

        async with async_session() as db:
            for doc_info in docs:
                src_result = await db.execute(
                    select(GoogleDocSource).where(
                        GoogleDocSource.google_doc_id == doc_info["id"]
                    )
                )
                source = src_result.scalar_one_or_none()
                if source and source.enabled:
                    sync_result = await _sync_source(source)
                    if sync_result.get("status") == "updated":
                        synced += 1
                        if sync_result.get("changed_doc_id"):
                            changed_doc_ids.append(sync_result["changed_doc_id"])
                    elif sync_result.get("status") == "error":
                        errors += 1

        # --- Handle PDFs (direct download + text extraction) ---
        for pdf_info in pdfs:
            sync_result = await _sync_pdf(pdf_info["id"], pdf_info["name"], folder.category, folder.workspace)
            if sync_result.get("status") == "updated":
                synced += 1
                if sync_result.get("changed_doc_id"):
                    changed_doc_ids.append(sync_result["changed_doc_id"])
            elif sync_result.get("status") == "error":
                errors += 1

        # Update folder record
        async with async_session() as db:
            result = await db.execute(
                select(GoogleDriveFolder).where(GoogleDriveFolder.id == folder_record_id)
            )
            f = result.scalar_one()
            f.last_synced_at = datetime.utcnow()
            f.last_sync_status = "success"
            f.last_error = None
            f.docs_found = len(discovered)
            await db.commit()

        logger.info(
            f"Folder sync '{folder.title}': {len(docs)} docs + {len(pdfs)} PDFs found, "
            f"{new_sources} new sources, {synced} updated, {errors} errors"
        )
        return {
            "status": "success",
            "folder_id": folder_record_id,
            "docs_found": len(discovered),
            "new_sources": new_sources,
            "synced": synced,
            "errors": errors,
            "changed_doc_ids": changed_doc_ids,
        }

    except Exception as e:
        logger.error(f"Failed to sync folder '{folder.title}': {e}")
        try:
            async with async_session() as db:
                result = await db.execute(
                    select(GoogleDriveFolder).where(GoogleDriveFolder.id == folder_record_id)
                )
                f = result.scalar_one()
                f.last_synced_at = datetime.utcnow()
                f.last_sync_status = "error"
                f.last_error = str(e)[:500]
                await db.commit()
        except Exception:
            pass
        return {"status": "error", "folder_id": folder_record_id, "error": str(e)}


async def sync_all_folders() -> dict:
    """Sync all enabled Google Drive folders."""
    settings = get_settings()
    if not settings.google_drive_api:
        return {"synced": 0, "errors": 0, "message": "GOOGLE_DRIVE_API not configured"}

    async with async_session() as db:
        result = await db.execute(
            select(GoogleDriveFolder).where(GoogleDriveFolder.enabled == True)
        )
        folders = result.scalars().all()

    if not folders:
        return {"synced": 0, "errors": 0, "changed_doc_ids": []}

    total_synced = 0
    total_errors = 0
    all_changed_doc_ids = []

    for folder in folders:
        result = await sync_folder(folder.id)
        if result.get("status") == "success":
            total_synced += result.get("synced", 0)
            total_errors += result.get("errors", 0)
            all_changed_doc_ids.extend(result.get("changed_doc_ids", []))
        else:
            total_errors += 1

    return {
        "synced": total_synced,
        "errors": total_errors,
        "changed_doc_ids": all_changed_doc_ids,
    }


async def google_docs_refresh_loop() -> None:
    """Background task: sync Google Docs and Drive folders periodically."""
    while True:
        await asyncio.sleep(REFRESH_INTERVAL_SECONDS)
        logger.info("Running periodic Google Docs sync...")
        result = await sync_all_google_docs()

        # Also sync Drive folders
        folder_result = await sync_all_folders()
        all_changed = result.get("changed_doc_ids", []) + folder_result.get("changed_doc_ids", [])

        # Trigger instant answer generation for changed docs
        if all_changed:
            try:
                from ..analysis.suggestion_generator import generate_suggestions_for_documents
                logger.info(f"Queueing suggestion generation for {len(all_changed)} updated docs")
                asyncio.create_task(generate_suggestions_for_documents(all_changed))
            except Exception as e:
                logger.error(f"Failed to queue suggestion generation: {e}")
