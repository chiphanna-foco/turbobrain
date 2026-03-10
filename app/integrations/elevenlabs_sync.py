"""Sync Ziprent knowledge base to ElevenLabs voice agent."""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from typing import Optional

import httpx
from sqlalchemy import select

from ..config import get_settings
from ..models.database import (
    async_session,
    KnowledgeDocument,
    InstantAnswer,
    ElevenLabsSync,
)

logger = logging.getLogger(__name__)

ELEVENLABS_API_BASE = "https://api.elevenlabs.io/v1/convai/knowledge-base"
ELEVENLABS_AGENTS_BASE = "https://api.elevenlabs.io/v1/convai/agents"


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _format_instant_answers(answers: list[InstantAnswer]) -> str:
    """Combine all instant answers into a single structured text document."""
    lines = ["# Ziprent Quick Answers Reference\n"]
    lines.append("This document contains curated answers to common questions.\n")

    for ia in answers:
        key_label = ia.key.replace("_", " ").title()
        lines.append(f"## {key_label}\n")
        lines.append(f"{ia.answer}\n")

        if ia.talking_points:
            lines.append("Key points:")
            for point in ia.talking_points:
                lines.append(f"- {point}")
            lines.append("")

        if ia.suggested_response:
            lines.append(f"Suggested response: {ia.suggested_response}\n")

    return "\n".join(lines)


async def _elevenlabs_create_text(
    client: httpx.AsyncClient,
    api_key: str,
    name: str,
    text: str,
) -> tuple[Optional[str], Optional[str]]:
    """Upload a text document to ElevenLabs. Returns (doc_id, error_msg)."""
    resp = await client.post(
        f"{ELEVENLABS_API_BASE}/text",
        headers={"xi-api-key": api_key, "Content-Type": "application/json"},
        json={"name": name, "text": text},
        timeout=30.0,
    )
    if resp.status_code != 200:
        err = f"{resp.status_code}: {resp.text[:200]}"
        logger.error(f"ElevenLabs create failed for '{name}': {err}")
        return None, err
    data = resp.json()
    return data.get("id"), None


async def _elevenlabs_delete(
    client: httpx.AsyncClient,
    api_key: str,
    doc_id: str,
) -> bool:
    """Delete a document from ElevenLabs. Returns True on success."""
    resp = await client.delete(
        f"{ELEVENLABS_API_BASE}/{doc_id}",
        headers={"xi-api-key": api_key},
        params={"force": "true"},
        timeout=15.0,
    )
    if resp.status_code not in (200, 204):
        logger.warning(f"ElevenLabs delete failed for {doc_id}: {resp.status_code}")
        return False
    return True


async def _attach_docs_to_agent(
    client: httpx.AsyncClient,
    api_key: str,
    agent_id: str,
    doc_records: list[ElevenLabsSync],
) -> bool:
    """PATCH the ElevenLabs agent to attach all synced documents to its KB."""
    kb_entries = []
    for r in doc_records:
        kb_entries.append({
            "type": "text",
            "name": r.doc_name or r.local_doc_id,
            "id": r.elevenlabs_doc_id,
            "usage_mode": "auto",
        })

    resp = await client.patch(
        f"{ELEVENLABS_AGENTS_BASE}/{agent_id}",
        headers={"xi-api-key": api_key, "Content-Type": "application/json"},
        json={
            "conversation_config": {
                "agent": {
                    "prompt": {
                        "knowledge_base": kb_entries,
                    }
                }
            }
        },
        timeout=30.0,
    )
    if resp.status_code != 200:
        logger.error(
            f"ElevenLabs agent update failed ({resp.status_code}): {resp.text}"
        )
        return False
    return True


async def sync_knowledge_to_elevenlabs() -> dict:
    """Sync all knowledge documents and instant answers to ElevenLabs.

    Strategy:
    - For each local document, compute a content hash.
    - If a sync record exists with the same hash, skip (already up to date).
    - If a sync record exists with a different hash, delete the old ElevenLabs
      doc and re-upload.
    - If no sync record exists, upload fresh.

    Returns a summary dict with counts.
    """
    settings = get_settings()
    api_key = settings.elevenlabs_api_key
    if not api_key:
        return {"error": "ELEVENLABS_API_KEY not configured"}

    created = 0
    updated = 0
    skipped = 0
    errors = 0
    deleted_old = 0
    error_details = []

    async with httpx.AsyncClient() as client:
        async with async_session() as db:
            # Load existing sync records
            result = await db.execute(select(ElevenLabsSync))
            sync_records = {r.local_doc_id: r for r in result.scalars().all()}

            # --- Knowledge Documents ---
            result = await db.execute(select(KnowledgeDocument))
            docs = result.scalars().all()

            for doc in docs:
                content = doc.content or ""
                h = _content_hash(content)
                existing = sync_records.get(doc.id)

                if existing and existing.content_hash == h:
                    skipped += 1
                    continue

                # Delete old version if it exists
                if existing:
                    ok = await _elevenlabs_delete(client, api_key, existing.elevenlabs_doc_id)
                    if ok:
                        deleted_old += 1
                    await db.delete(existing)

                # Upload new version
                name = f"[KB] {doc.title}"
                el_id, err = await _elevenlabs_create_text(client, api_key, name, content)
                if el_id:
                    db.add(ElevenLabsSync(
                        local_doc_id=doc.id,
                        elevenlabs_doc_id=el_id,
                        doc_type="knowledge",
                        doc_name=name,
                        content_hash=h,
                    ))
                    if existing:
                        updated += 1
                    else:
                        created += 1
                else:
                    errors += 1
                    error_details.append(f"{name}: {err}")

            # --- Instant Answers (combined into one document) ---
            result = await db.execute(select(InstantAnswer))
            answers = result.scalars().all()

            if answers:
                combined_text = _format_instant_answers(answers)
                h = _content_hash(combined_text)
                ia_key = "instant_answers_combined"
                existing = sync_records.get(ia_key)

                if existing and existing.content_hash == h:
                    skipped += 1
                else:
                    if existing:
                        await _elevenlabs_delete(client, api_key, existing.elevenlabs_doc_id)
                        deleted_old += 1
                        await db.delete(existing)

                    el_id, err = await _elevenlabs_create_text(
                        client, api_key, "Ziprent Quick Answers", combined_text
                    )
                    if el_id:
                        db.add(ElevenLabsSync(
                            local_doc_id=ia_key,
                            elevenlabs_doc_id=el_id,
                            doc_type="instant_answers",
                            doc_name="Ziprent Quick Answers",
                            content_hash=h,
                        ))
                        if existing:
                            updated += 1
                        else:
                            created += 1
                    else:
                        errors += 1
                        error_details.append(f"Ziprent Quick Answers: {err}")

            await db.commit()

        # Attach all synced docs to the agent (if agent_id is configured)
        agent_id = settings.elevenlabs_agent_id
        agent_updated = False
        if agent_id:
            async with async_session() as db:
                result = await db.execute(select(ElevenLabsSync))
                all_synced = result.scalars().all()

            if all_synced:
                agent_updated = await _attach_docs_to_agent(
                    client, api_key, agent_id, all_synced
                )

    return {
        "created": created,
        "updated": updated,
        "skipped": skipped,
        "deleted_old": deleted_old,
        "errors": errors,
        "error_details": error_details[:5],  # first 5 to keep response small
        "total_synced": created + updated + skipped,
        "agent_updated": agent_updated,
    }


async def get_elevenlabs_sync_status() -> dict:
    """Get current sync status."""
    settings = get_settings()
    configured = bool(settings.elevenlabs_api_key)

    async with async_session() as db:
        result = await db.execute(select(ElevenLabsSync))
        records = result.scalars().all()

    if not records:
        return {
            "configured": configured,
            "synced": False,
            "document_count": 0,
            "last_sync": None,
            "documents": [],
        }

    last_sync = max(r.synced_at for r in records) if records else None
    knowledge_count = sum(1 for r in records if r.doc_type == "knowledge")
    ia_synced = any(r.doc_type == "instant_answers" for r in records)

    return {
        "configured": configured,
        "synced": True,
        "document_count": len(records),
        "knowledge_documents": knowledge_count,
        "instant_answers_synced": ia_synced,
        "last_sync": last_sync.isoformat() if last_sync else None,
        "documents": [
            {
                "name": r.doc_name,
                "type": r.doc_type,
                "elevenlabs_id": r.elevenlabs_doc_id,
                "synced_at": r.synced_at.isoformat() if r.synced_at else None,
            }
            for r in records
        ],
    }


async def verify_elevenlabs_documents() -> dict:
    """Verify documents exist in ElevenLabs by calling their list API."""
    settings = get_settings()
    api_key = settings.elevenlabs_api_key
    if not api_key:
        return {"error": "ELEVENLABS_API_KEY not configured"}

    async with httpx.AsyncClient() as client:
        # List all documents from ElevenLabs workspace KB
        resp = await client.get(
            ELEVENLABS_API_BASE,
            headers={"xi-api-key": api_key},
            params={
                "page_size": 100,
                "show_only_owned_documents": "true",
            },
            timeout=15.0,
        )
        if resp.status_code != 200:
            return {
                "error": f"ElevenLabs list API returned {resp.status_code}: {resp.text[:300]}"
            }

        data = resp.json()
        el_docs = data.get("documents", [])

    # Cross-reference with our sync records
    async with async_session() as db:
        result = await db.execute(select(ElevenLabsSync))
        sync_records = result.scalars().all()

    synced_ids = {r.elevenlabs_doc_id for r in sync_records}

    return {
        "elevenlabs_total": len(el_docs),
        "elevenlabs_documents": [
            {
                "id": d.get("id"),
                "name": d.get("name"),
                "type": d.get("type"),
                "in_our_sync": d.get("id") in synced_ids,
            }
            for d in el_docs
        ],
        "our_sync_records": len(sync_records),
        "has_more": data.get("has_more", False),
    }
