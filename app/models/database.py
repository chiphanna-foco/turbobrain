"""SQLAlchemy models for TurboBrain knowledge base."""
from __future__ import annotations

import json
from sqlalchemy import Column, String, Text, DateTime, Float, Boolean, JSON, create_engine, select, func
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
from pathlib import Path
import logging
import uuid

from ..config import get_settings

logger = logging.getLogger(__name__)

Base = declarative_base()
settings = get_settings()


class KnowledgeDocument(Base):
    """Stores knowledge base documents for context retrieval."""

    __tablename__ = "knowledge_documents"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    category = Column(String(100), nullable=False)
    file_path = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "category": self.category,
            "filePath": self.file_path,
            "createdAt": self.created_at.isoformat() if self.created_at else None,
            "updatedAt": self.updated_at.isoformat() if self.updated_at else None,
        }


class GoogleDocSource(Base):
    """Tracks Google Docs configured for knowledge base sync."""

    __tablename__ = "google_doc_sources"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    google_doc_id = Column(String(200), nullable=False, unique=True)
    title = Column(String(500), nullable=False)
    category = Column(String(100), nullable=False, default="general")
    enabled = Column(Boolean, default=True)
    last_synced_at = Column(DateTime, nullable=True)
    last_sync_status = Column(String(20), nullable=True)
    last_error = Column(Text, nullable=True)
    content_hash = Column(String(32), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "googleDocId": self.google_doc_id,
            "title": self.title,
            "category": self.category,
            "enabled": self.enabled,
            "lastSyncedAt": self.last_synced_at.isoformat() if self.last_synced_at else None,
            "lastSyncStatus": self.last_sync_status,
            "lastError": self.last_error,
            "createdAt": self.created_at.isoformat() if self.created_at else None,
            "updatedAt": self.updated_at.isoformat() if self.updated_at else None,
        }


class GoogleDriveFolder(Base):
    """Tracks Google Drive folders configured for knowledge base sync."""

    __tablename__ = "google_drive_folders"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    folder_id = Column(String(200), nullable=False, unique=True)
    title = Column(String(500), nullable=False)
    category = Column(String(100), nullable=False, default="general")
    enabled = Column(Boolean, default=True)
    last_synced_at = Column(DateTime, nullable=True)
    last_sync_status = Column(String(20), nullable=True)
    last_error = Column(Text, nullable=True)
    docs_found = Column(Float, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "folderId": self.folder_id,
            "title": self.title,
            "category": self.category,
            "enabled": self.enabled,
            "lastSyncedAt": self.last_synced_at.isoformat() if self.last_synced_at else None,
            "lastSyncStatus": self.last_sync_status,
            "lastError": self.last_error,
            "docsFound": int(self.docs_found) if self.docs_found else 0,
            "createdAt": self.created_at.isoformat() if self.created_at else None,
            "updatedAt": self.updated_at.isoformat() if self.updated_at else None,
        }


class InstantAnswer(Base):
    """Curated instant answers for common questions."""

    __tablename__ = "instant_answers"

    key = Column(String(100), primary_key=True)
    answer = Column(Text, nullable=False)
    talking_points = Column(JSON, default=list)
    suggested_response = Column(Text, nullable=True)
    confidence = Column(String(20), default="medium")
    source_topic = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "answer": self.answer,
            "talking_points": self.talking_points or [],
            "suggested_response": self.suggested_response,
            "confidence": self.confidence,
            "source_topic": self.source_topic,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class SuggestedInstantAnswer(Base):
    """AI-suggested instant answers extracted from knowledge base documents, pending admin review."""

    __tablename__ = "suggested_instant_answers"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    key = Column(String(100), nullable=False)
    answer = Column(Text, nullable=False)
    talking_points = Column(JSON, default=list)
    suggested_response = Column(Text, nullable=True)
    confidence = Column(String(20), default="medium")
    source_topic = Column(String(100), nullable=True)
    source_document_id = Column(String(36), nullable=True, index=True)
    source_document_title = Column(String(500), nullable=True)
    status = Column(String(20), default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)
    reviewed_at = Column(DateTime, nullable=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "key": self.key,
            "answer": self.answer,
            "talking_points": self.talking_points or [],
            "suggested_response": self.suggested_response,
            "confidence": self.confidence,
            "source_topic": self.source_topic,
            "source_document_id": self.source_document_id,
            "source_document_title": self.source_document_title,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
        }


class ElevenLabsSync(Base):
    """Tracks documents synced to ElevenLabs knowledge base."""

    __tablename__ = "elevenlabs_sync"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    local_doc_id = Column(String(100), index=True, nullable=False)
    elevenlabs_doc_id = Column(String(100), nullable=False)
    doc_type = Column(String(20), nullable=False)
    doc_name = Column(String(500), nullable=True)
    content_hash = Column(String(64), nullable=False)
    synced_at = Column(DateTime, default=datetime.utcnow)


class IntercomWorkspace(Base):
    """Tracks Intercom workspaces for Help Center article sync."""

    __tablename__ = "intercom_workspaces"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(200), nullable=False)
    access_token = Column(Text, nullable=False)
    workspace_id = Column(String(100), nullable=True)
    enabled = Column(Boolean, default=True)
    last_synced_at = Column(DateTime, nullable=True)
    last_sync_status = Column(String(20), nullable=True)
    last_error = Column(Text, nullable=True)
    article_count = Column(Float, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "workspaceId": self.workspace_id,
            "enabled": self.enabled,
            "lastSyncedAt": self.last_synced_at.isoformat() if self.last_synced_at else None,
            "lastSyncStatus": self.last_sync_status,
            "lastError": self.last_error,
            "articleCount": int(self.article_count or 0),
            "createdAt": self.created_at.isoformat() if self.created_at else None,
        }


class UnansweredQuestion(Base):
    """Tracks questions the system couldn't answer."""

    __tablename__ = "unanswered_questions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    question = Column(Text, nullable=False)
    count = Column(Float, default=1)
    last_asked = Column(DateTime, default=datetime.utcnow)
    resolved = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "count": int(self.count) if self.count else 1,
            "last_asked": self.last_asked.isoformat() if self.last_asked else None,
            "resolved": self.resolved,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# Async engine and session
_db_url = settings.effective_database_url
_is_postgres = "postgresql" in _db_url
engine = create_async_engine(_db_url, echo=settings.debug)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

logger.info(f"Database: {'PostgreSQL' if _is_postgres else 'SQLite'}")


async def init_db():
    """Initialize the database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def import_knowledge_from_files():
    """Import knowledge base markdown files into the database."""
    # Look for knowledge directory relative to project root
    app_dir = Path(__file__).parent.parent.parent  # turbobrain/
    candidates = [
        app_dir / "knowledge",
    ]

    knowledge_dir = None
    for candidate in candidates:
        if candidate.exists():
            knowledge_dir = candidate
            break

    if not knowledge_dir:
        logger.warning("Knowledge directory not found, skipping import")
        return

    md_files = list(knowledge_dir.glob("**/*.md"))
    if not md_files:
        logger.info("No markdown files found in knowledge directory")
        return

    imported = 0
    updated = 0
    skipped = 0
    changed_doc_ids = []

    async with async_session() as db:
        for md_file in md_files:
            if md_file.name == "README.md" or md_file.name.endswith("~.md"):
                skipped += 1
                continue

            relative_path = md_file.relative_to(knowledge_dir)
            category = relative_path.parts[0].replace("-", " ").replace("_", " ") if len(relative_path.parts) > 1 else "general"
            title = md_file.stem.replace("-", " ").replace("_", " ").title()
            content = md_file.read_text(encoding="utf-8")
            file_path_str = str(relative_path)

            result = await db.execute(
                select(KnowledgeDocument).where(
                    KnowledgeDocument.file_path == file_path_str
                )
            )
            existing = result.scalar_one_or_none()

            if existing:
                existing.title = title
                existing.content = content
                existing.category = category
                existing.updated_at = datetime.utcnow()
                changed_doc_ids.append(existing.id)
                updated += 1
            else:
                doc = KnowledgeDocument(
                    id=str(uuid.uuid4()),
                    title=title,
                    content=content,
                    category=category,
                    file_path=file_path_str,
                )
                db.add(doc)
                changed_doc_ids.append(doc.id)
                imported += 1

        await db.commit()

    logger.info(f"Knowledge import: {imported} new, {updated} updated, {skipped} skipped")
    return {
        "imported": imported,
        "updated": updated,
        "skipped": skipped,
        "changed_doc_ids": changed_doc_ids,
    }


async def import_instant_answers_from_file(force=False):
    """Import instant answers from instant_answers.json."""
    app_dir = Path(__file__).parent.parent.parent  # turbobrain/
    candidates = [
        app_dir / "instant_answers.json",
    ]

    json_path = None
    for candidate in candidates:
        if candidate.exists():
            json_path = candidate
            break

    if not json_path:
        logger.info("No instant_answers.json found, skipping seed")
        return

    async with async_session() as db:
        count = (await db.execute(select(func.count(InstantAnswer.key)))).scalar() or 0
        if count > 0 and not force:
            logger.info(f"Instant answers table already has {count} rows, skipping seed")
            return

        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"Failed to read instant_answers.json: {e}")
            return

        imported = 0
        updated = 0
        for key, value in data.items():
            existing = (await db.execute(select(InstantAnswer).where(InstantAnswer.key == key))).scalar_one_or_none()
            if existing:
                if force:
                    existing.answer = value.get("answer", "")
                    existing.talking_points = value.get("talking_points", [])
                    existing.suggested_response = value.get("suggested_response")
                    existing.confidence = value.get("confidence", "medium")
                    existing.source_topic = value.get("source_topic")
                    updated += 1
            else:
                answer = InstantAnswer(
                    key=key,
                    answer=value.get("answer", ""),
                    talking_points=value.get("talking_points", []),
                    suggested_response=value.get("suggested_response"),
                    confidence=value.get("confidence", "medium"),
                    source_topic=value.get("source_topic"),
                )
                db.add(answer)
                imported += 1

        await db.commit()
        logger.info(f"Instant answers: {imported} new, {updated} updated from {json_path.name}")
