"""Intercom sync: Help Center articles + closed conversations → KnowledgeDocument."""
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
INTERCOM_CONVERSATIONS_URL = "https://api.intercom.io/conversations"
REFRESH_INTERVAL_SECONDS = 24 * 60 * 60  # 24 hours

# Conversation parts to skip (system events, not useful content)
_SKIP_PART_TYPES = {"assignment", "open", "close", "snoozed", "unsnoozed",
                    "away_mode_assignment", "conversation_rating_changed"}


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


def _convo_file_path(workspace_name: str, convo_id: str) -> str:
    return f"intercom-convo:{workspace_name}:{convo_id}"


def _build_conversation_content(convo: dict) -> str:
    """Format a conversation as readable Q&A text for KB ingestion."""
    lines = []

    source = convo.get("source", {})
    source_body = _strip_html(source.get("body", "") or "")
    if source_body:
        author_name = (source.get("author") or {}).get("name") or "Customer"
        lines.append(f"Customer ({author_name}): {source_body}")

    parts = convo.get("conversation_parts", {}).get("conversation_parts", [])
    for part in parts:
        if part.get("part_type", "") in _SKIP_PART_TYPES:
            continue
        body = _strip_html(part.get("body", "") or "")
        if not body:
            continue
        author = part.get("author") or {}
        author_type = author.get("type", "")
        name = author.get("name") or author_type.title() or "Unknown"
        prefix = "Support" if author_type in ("admin", "bot") else "Customer"
        lines.append(f"\n{prefix} ({name}): {body}")

    return "\n".join(lines).strip()


async def sync_workspace(workspace: IntercomWorkspace) -> dict:
    """Fetch ALL articles (any state) from one Intercom workspace and upsert into KnowledgeDocument."""
    headers = {
        "Authorization": f"Bearer {workspace.access_token}",
        "Accept": "application/json",
        "Intercom-Version": "2.10",
    }

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

    # Include all articles that have body content (published + draft/internal)
    with_content = [a for a in articles if a.get("body")]
    synced = 0
    updated = 0

    async with async_session() as db:
        for article in with_content:
            article_id = str(article.get("id", ""))
            title = article.get("title") or "Untitled"
            body_html = article.get("body") or ""
            content = _strip_html(body_html)
            if not content.strip():
                continue

            state = article.get("state", "")
            category = "internal" if state != "published" else "general"
            file_path = _file_path_for(workspace.name, article_id)
            new_hash = hashlib.md5(content.encode()).hexdigest()
            # Intercom articles have a url field (public help center URL)
            source_url = article.get("url") or None

            result = await db.execute(
                select(KnowledgeDocument).where(KnowledgeDocument.file_path == file_path)
            )
            existing = result.scalar_one_or_none()

            if existing:
                old_hash = hashlib.md5(existing.content.encode()).hexdigest()
                if old_hash != new_hash or not existing.source_url:
                    existing.title = title
                    existing.content = content
                    existing.workspace = workspace.name
                    existing.category = category
                    existing.source_url = source_url
                    existing.updated_at = datetime.utcnow()
                    updated += 1
            else:
                db.add(KnowledgeDocument(
                    id=str(uuid.uuid4()),
                    title=title,
                    content=content,
                    category=category,
                    file_path=file_path,
                    workspace=workspace.name,
                    source_url=source_url,
                ))
                synced += 1

        ws_result = await db.execute(
            select(IntercomWorkspace).where(IntercomWorkspace.id == workspace.id)
        )
        ws = ws_result.scalar_one()
        ws.last_synced_at = datetime.utcnow()
        ws.last_sync_status = "success"
        ws.last_error = None
        ws.article_count = len(with_content)
        await db.commit()

    published_count = sum(1 for a in articles if a.get("state") == "published")
    draft_count = len(with_content) - published_count
    logger.info(
        f"Intercom article sync [{workspace.name}]: {synced} new, {updated} updated "
        f"({published_count} published + {draft_count} internal/draft)"
    )
    return {
        "workspace": workspace.name,
        "synced": synced,
        "updated": updated,
        "total_articles": len(with_content),
        "published": published_count,
        "internal_draft": draft_count,
    }


async def sync_conversations(workspace: IntercomWorkspace, limit: int = 500) -> dict:
    """Fetch recent closed conversations and ingest as KB documents."""
    headers = {
        "Authorization": f"Bearer {workspace.access_token}",
        "Accept": "application/json",
        "Intercom-Version": "2.10",
    }

    conversations: list[dict] = []
    page = 1
    async with httpx.AsyncClient(timeout=90) as client:
        while len(conversations) < limit:
            resp = await client.get(
                INTERCOM_CONVERSATIONS_URL,
                headers=headers,
                params={
                    "per_page": 150,
                    "page": page,
                    "sort_field": "updated_at",
                    "sort_order": "desc",
                },
            )
            if not resp.is_success:
                raise ValueError(f"Intercom API {resp.status_code}: {resp.text[:200]}")
            data = resp.json()
            batch = data.get("conversations", [])
            if not batch:
                break
            conversations.extend(batch)
            pages = data.get("pages", {})
            if page >= pages.get("total_pages", 1):
                break
            page += 1

    # Only closed (resolved) conversations
    closed = [c for c in conversations if c.get("state") == "closed"][:limit]
    synced = 0
    updated = 0

    async with async_session() as db:
        for convo in closed:
            convo_id = str(convo.get("id", ""))
            content = _build_conversation_content(convo)
            if not content or len(content) < 60:
                continue

            source = convo.get("source", {})
            first_msg = _strip_html(source.get("body", "") or "")
            title = first_msg[:90] + ("..." if len(first_msg) > 90 else "") or f"Conversation {convo_id}"

            file_path = _convo_file_path(workspace.name, convo_id)
            new_hash = hashlib.md5(content.encode()).hexdigest()
            # Build deep-link URL to conversation in Intercom inbox
            app_id = workspace.workspace_id or ""
            source_url = f"https://app.intercom.com/a/apps/{app_id}/conversations/{convo_id}" if app_id else None

            result = await db.execute(
                select(KnowledgeDocument).where(KnowledgeDocument.file_path == file_path)
            )
            existing = result.scalar_one_or_none()

            if existing:
                old_hash = hashlib.md5(existing.content.encode()).hexdigest()
                if old_hash != new_hash:
                    existing.title = title
                    existing.content = content
                    existing.source_url = source_url
                    existing.updated_at = datetime.utcnow()
                    updated += 1
            else:
                db.add(KnowledgeDocument(
                    id=str(uuid.uuid4()),
                    title=title,
                    content=content,
                    category="conversations",
                    file_path=file_path,
                    workspace=workspace.name,
                    source_url=source_url,
                ))
                synced += 1

        await db.commit()

    logger.info(
        f"Intercom conversation sync [{workspace.name}]: "
        f"{synced} new, {updated} updated of {len(closed)} closed conversations"
    )
    return {
        "workspace": workspace.name,
        "synced": synced,
        "updated": updated,
        "total_closed": len(closed),
    }


async def sync_all_intercom() -> dict:
    """Sync all enabled Intercom workspaces (articles only)."""
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
