"""Confluence Cloud page sync: fetches published pages and upserts into KnowledgeDocument."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import re
import uuid
from datetime import datetime

import httpx
from sqlalchemy import select

from ..models.database import async_session, ConfluenceSpace, KnowledgeDocument

logger = logging.getLogger(__name__)

CONFLUENCE_CONTENT_URL = "https://{domain}/wiki/rest/api/content"
REFRESH_INTERVAL_SECONDS = 24 * 60 * 60  # 24 hours


def _basic_auth(email: str, api_token: str) -> str:
    credentials = base64.b64encode(f"{email}:{api_token}".encode()).decode()
    return f"Basic {credentials}"


def _strip_html(html: str) -> str:
    """Remove HTML tags and decode common entities."""
    text = re.sub(r"<[^>]+>", " ", html)
    for entity, char in [
        ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
        ("&nbsp;", " "), ("&quot;", '"'), ("&#39;", "'"),
    ]:
        text = text.replace(entity, char)
    return re.sub(r"\s+", " ", text).strip()


def _file_path_for(space_id: str, page_id: str) -> str:
    return f"confluence:{space_id}:{page_id}"


async def sync_space(space: ConfluenceSpace) -> dict:
    """Fetch all published pages from one Confluence space and upsert into KnowledgeDocument."""
    try:
        return await _sync_space_inner(space)
    except Exception as e:
        logger.error(f"Confluence sync failed [{space.name}]: {e}")
        try:
            async with async_session() as db:
                result = await db.execute(select(ConfluenceSpace).where(ConfluenceSpace.id == space.id))
                sp = result.scalar_one()
                sp.last_sync_status = "error"
                sp.last_error = str(e)[:500]
                sp.last_synced_at = datetime.utcnow()
                await db.commit()
        except Exception:
            pass
        return {"space": space.name, "error": str(e)}


async def _sync_space_inner(space: ConfluenceSpace) -> dict:
    """Inner sync logic — called by sync_space which wraps it in try/except."""
    headers = {
        "Authorization": _basic_auth(space.email, space.api_token),
        "Accept": "application/json",
    }
    base_url = CONFLUENCE_CONTENT_URL.format(domain=space.domain)

    pages: list[dict] = []
    start = 0
    limit = 50

    async with httpx.AsyncClient(timeout=60) as client:
        while True:
            resp = await client.get(
                base_url,
                headers=headers,
                params={
                    "spaceKey": space.space_key,
                    "type": "page",
                    "status": "current",
                    "expand": "body.view",
                    "start": start,
                    "limit": limit,
                },
            )
            if not resp.is_success:
                raise ValueError(f"Confluence API returned {resp.status_code}: {resp.text[:200]}")
            data = resp.json()
            results = data.get("results", [])
            pages.extend(results)

            size = data.get("size", 0)
            total_size = data.get("totalSize", size)
            if start + size >= total_size or size == 0:
                break
            start += size

    synced = 0
    updated = 0

    async with async_session() as db:
        for page in pages:
            page_id = str(page.get("id", ""))
            title = page.get("title") or "Untitled"
            body_html = page.get("body", {}).get("view", {}).get("value", "") or ""
            content = _strip_html(body_html)
            if not content.strip():
                continue

            file_path = _file_path_for(space.id, page_id)
            new_hash = hashlib.md5(content.encode()).hexdigest()

            result = await db.execute(
                select(KnowledgeDocument).where(KnowledgeDocument.file_path == file_path)
            )
            existing = result.scalar_one_or_none()

            if existing:
                old_hash = hashlib.md5(existing.content.encode()).hexdigest()
                if old_hash != new_hash:
                    existing.title = title
                    existing.content = content
                    existing.workspace = space.workspace
                    existing.updated_at = datetime.utcnow()
                    updated += 1
            else:
                db.add(KnowledgeDocument(
                    id=str(uuid.uuid4()),
                    title=title,
                    content=content,
                    category="general",
                    file_path=file_path,
                    workspace=space.workspace,
                ))
                synced += 1

        # Update space sync status
        sp_result = await db.execute(
            select(ConfluenceSpace).where(ConfluenceSpace.id == space.id)
        )
        sp = sp_result.scalar_one()
        sp.last_synced_at = datetime.utcnow()
        sp.last_sync_status = "success"
        sp.last_error = None
        sp.page_count = len(pages)

        await db.commit()

    logger.info(f"Confluence sync [{space.name}/{space.space_key}]: {synced} new, {updated} updated, {len(pages)} total pages")
    return {
        "space": space.name,
        "spaceKey": space.space_key,
        "synced": synced,
        "updated": updated,
        "total_pages": len(pages),
    }


async def sync_all_confluence() -> dict:
    """Sync all enabled Confluence spaces."""
    async with async_session() as db:
        result = await db.execute(
            select(ConfluenceSpace).where(ConfluenceSpace.enabled == True)
        )
        spaces = result.scalars().all()

    results = []
    for sp in spaces:
        try:
            r = await sync_space(sp)
            results.append(r)
        except Exception as e:
            logger.error(f"Confluence sync error [{sp.name}]: {e}")
            async with async_session() as db:
                r2 = await db.execute(
                    select(ConfluenceSpace).where(ConfluenceSpace.id == sp.id)
                )
                s = r2.scalar_one()
                s.last_sync_status = "error"
                s.last_error = str(e)[:500]
                s.last_synced_at = datetime.utcnow()
                await db.commit()
            results.append({"space": sp.name, "error": str(e)})

    return {"results": results}


async def confluence_refresh_loop():
    """Background loop: sync Confluence every 24 hours."""
    while True:
        await asyncio.sleep(REFRESH_INTERVAL_SECONDS)
        logger.info("Running scheduled Confluence sync...")
        try:
            await sync_all_confluence()
        except Exception as e:
            logger.error(f"Confluence refresh loop error: {e}")
