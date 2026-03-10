"""Orchestrate Q&A extraction from knowledge base documents into suggested instant answers."""

import logging
import uuid
from sqlalchemy import select

from ..models.database import (
    async_session,
    KnowledgeDocument,
    InstantAnswer,
    SuggestedInstantAnswer,
)
from .qa_extractor import get_qa_extractor

logger = logging.getLogger(__name__)


async def generate_suggestions_for_documents(document_ids: list[str]) -> dict:
    """Extract Q&A pairs from the given documents and store as SuggestedInstantAnswer rows.

    Args:
        document_ids: List of KnowledgeDocument IDs to process.

    Returns:
        Stats dict: {total_suggestions, documents_processed, skipped}
    """
    if not document_ids:
        return {"total_suggestions": 0, "documents_processed": 0, "skipped": 0}

    extractor = get_qa_extractor()
    total_suggestions = 0
    documents_processed = 0
    skipped = 0

    # Load existing keys for dedup (instant answers + pending suggestions)
    async with async_session() as db:
        ia_result = await db.execute(select(InstantAnswer.key))
        existing_keys = [row[0] for row in ia_result.all()]

        sa_result = await db.execute(
            select(SuggestedInstantAnswer.key).where(
                SuggestedInstantAnswer.status == "pending"
            )
        )
        existing_keys.extend(row[0] for row in sa_result.all())

    for doc_id in document_ids:
        try:
            async with async_session() as db:
                result = await db.execute(
                    select(KnowledgeDocument).where(KnowledgeDocument.id == doc_id)
                )
                doc = result.scalar_one_or_none()

            if not doc:
                logger.warning(f"Document {doc_id} not found, skipping")
                skipped += 1
                continue

            pairs = await extractor.extract_qa_pairs(
                document_title=doc.title,
                document_content=doc.content,
                document_category=doc.category,
                existing_keys=existing_keys,
            )

            if not pairs:
                skipped += 1
                continue

            async with async_session() as db:
                for pair in pairs:
                    key = pair["key"]
                    if key in existing_keys:
                        continue

                    suggestion = SuggestedInstantAnswer(
                        id=str(uuid.uuid4()),
                        key=key,
                        answer=pair["answer"],
                        talking_points=pair.get("talking_points", []),
                        suggested_response=pair.get("suggested_response", ""),
                        confidence=pair.get("confidence", "medium"),
                        source_topic=doc.category,
                        source_document_id=doc.id,
                        source_document_title=doc.title,
                        status="pending",
                    )
                    db.add(suggestion)
                    existing_keys.append(key)
                    total_suggestions += 1

                await db.commit()

            documents_processed += 1

        except Exception as e:
            logger.error(f"Failed to generate suggestions for doc {doc_id}: {e}")
            skipped += 1

    logger.info(
        f"Suggestion generation complete: {total_suggestions} suggestions "
        f"from {documents_processed} docs, {skipped} skipped"
    )
    return {
        "total_suggestions": total_suggestions,
        "documents_processed": documents_processed,
        "skipped": skipped,
    }
