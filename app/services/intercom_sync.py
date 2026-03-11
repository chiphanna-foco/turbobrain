"""Intercom Help Center article sync: fetches published articles and upserts into KnowledgeDocument."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import uuid
from datetime import datetime

import httpx
from sqlalchemy import select

from ..models.database import async_session, IntercomWorkspace, KnowledgeDocument

logger = logging.getLogger(__name__)

INTERCOM_ARTICLES_URL = "https://api.intercom.io/articles"
REFRESH_INTERVAL_SECONDS = 24 * 60 * 60  # 24 hours


def _strip_html(html: str) -> str:
    """Remove HTML tags and decode common entities."""
    text = re.sub(r"<[^>]+>", " ", html)
    for entity, char in [
        ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
        ("&nbsp;", " "), ("&quot;", '"'), ("&#39;", "'"),
    ]:
        text = text.replace(entity, char)
    return re.sub(r"\s+", " ", text).strip()


def _file_path_for(workspace_name: str, article_id: str) -> str:
    return f"intercom:{workspace_name}:{article_id}"


async def sync_workspace(workspace: IntercomWorkspace) -> dict:
    """Fetch all published articles from one Intercom workspace and upsert into KnowledgeDocument."""
    headers = {
        "Authorization": f"Bearer {workspace.access_token}",
        "Accept": "application/json",
        "Intercom-Version": "2.10",
    }

    # Paginate through all articles
    articles: list[dict] = []
    page = 1
    async with httpx.AsyncClient(timeout=60) as client:
        while True:
            resp = await client.get(
                INTERCOM_ARTICLES_URL,
                headers=headers,
                params={"per_page": 250, "page": page},
            )
            resp.raise_for_status()
            data = resp.json()
            articles.extend(data.get("data", []))
            pages = data.get("pages", {})
            if page >= pages.get("total_pages", 1):
                break
            page += 1

    published = [a for a in articles if a.get("state") == "published"]
    synced = 0
    updated = 0

    async with async_session() as db:
        for article in published:
            article_id = str(article.get("id", ""))
            title = article.get("title") or "Untitled"
            body_html = article.get("body") or ""
            content = _strip_html(body_html)
            if not content.strip():
                continue

            file_path = _file_path_for(workspace.name, article_id)
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
                    existing.updated_at = datetime.utcnow()
                    updated += 1
            else:
                db.add(KnowledgeDocument(
                    id=str(uuid.uuid4()),
                    title=title,
                    content=content,
                    category="general",
                    file_path=file_path,
                ))
                synced += 1

        # Update workspace sync status
        ws_result = await db.execute(
            select(IntercomWorkspace).where(IntercomWorkspace.id == workspace.id)
        )
        ws = ws_result.scalar_one()
        ws.last_synced_at = datetime.utcnow()
        ws.last_sync_status = "success"
        ws.last_error = None
        ws.article_count = len(published)

        await db.commit()

    logger.info(f"Intercom sync [{workspace.name}]: {synced} new, {updated} updated, {len(published)} total published")
    return {
        "workspace": workspace.name,
        "synced": synced,
        "updated": updated,
        "total_published": len(published),
    }


async def sync_all_intercom() -> dict:
    """Sync all enabled Intercom workspaces."""
    async with async_session() as db:
        result = await db.execute(
            select(IntercomWorkspace).where(IntercomWorkspace.enabled == True)
        )
        workspaces = result.scalars().all()

    results = []
    for ws in workspaces:
        try:
            r = await sync_workspace(ws)
            results.append(r)
        except Exception as e:
            logger.error(f"Intercom sync error [{ws.name}]: {e}")
            async with async_session() as db:
                r2 = await db.execute(
                    select(IntercomWorkspace).where(IntercomWorkspace.id == ws.id)
                )
                w = r2.scalar_one()
                w.last_sync_status = "error"
                w.last_error = str(e)
                await db.commit()
            results.append({"workspace": ws.name, "error": str(e)})

    return {"results": results}


async def intercom_refresh_loop():
    """Background loop: sync Intercom every 24 hours."""
    while True:
        await asyncio.sleep(REFRESH_INTERVAL_SECONDS)
        logger.info("Running scheduled Intercom sync...")
        try:
            await sync_all_intercom()
        except Exception as e:
            logger.error(f"Intercom refresh loop error: {e}")
