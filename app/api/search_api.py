"""Search API for querying the knowledge base."""
from __future__ import annotations
from fastapi import APIRouter, Query
from typing import List, Optional
from sqlalchemy import select
import logging

from ..models.database import async_session, KnowledgeDocument, InstantAnswer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["search"])

# Stop words to ignore in search
STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "to", "of",
    "in", "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "under",
    "i", "me", "my", "we", "our", "you", "your", "he", "him", "she", "her",
    "it", "its", "they", "them", "their", "this", "that", "these", "those",
    "what", "when", "where", "why", "how", "about", "just", "like", "get",
    "got", "not", "but", "all", "also", "been", "some", "any", "out",
}


def extract_snippet(content: str, keywords: List[str], max_length: int = 1500) -> str:
    """Extract a relevant snippet from content based on keywords."""
    content_lower = content.lower()

    best_pos = len(content)
    for keyword in keywords:
        pos = content_lower.find(keyword)
        if 0 <= pos < best_pos:
            best_pos = pos

    if best_pos == len(content):
        return content[:max_length] + "..." if len(content) > max_length else content

    start = max(0, best_pos - 100)
    end = min(len(content), best_pos + max_length - 100)
    snippet = content[start:end]

    if start > 0:
        snippet = "..." + snippet
    if end < len(content):
        snippet = snippet + "..."

    return snippet


@router.get("/search")
async def search_knowledge_base(
    query: str = Query(..., min_length=3, description="Search query"),
    max_results: int = Query(default=10, description="Maximum results to return"),
    workspace: Optional[str] = Query(default=None, description="Filter by workspace/brand"),
    tag: Optional[str] = Query(default=None, description="Filter by content tag (features, SOP, workaround, known issue, legal, general)"),
):
    """Search instant answers and knowledge documents."""
    search_text = query.lower()
    words = [w for w in search_text.split() if len(w) > 2 and w not in STOP_WORDS]

    if not words:
        return {"instant_answers": [], "knowledge_results": [], "query": query}

    # Search instant answers
    instant_answers = []
    async with async_session() as db:
        result = await db.execute(select(InstantAnswer))
        all_instant = result.scalars().all()

        ia_matches = []
        for ia in all_instant:
            score = 0
            key_text = ia.key.lower().replace("_", " ")
            answer_text = ia.answer.lower() if ia.answer else ""
            tp_text = " ".join(ia.talking_points).lower() if ia.talking_points else ""

            for word in words:
                if word in key_text:
                    score += 5
                if word in answer_text:
                    score += answer_text.count(word)
                if word in tp_text:
                    score += tp_text.count(word)

            if score > 0:
                ia_matches.append({"ia": ia, "score": score})

        ia_matches.sort(key=lambda x: x["score"], reverse=True)
        for match in ia_matches[:max_results]:
            ia = match["ia"]
            instant_answers.append({
                "key": ia.key,
                "answer": ia.answer or "",
                "talking_points": ia.talking_points or [],
                "suggested_response": ia.suggested_response or "",
                "confidence": ia.confidence or "medium",
                "score": match["score"],
            })

    # Search knowledge documents
    knowledge_results = []
    async with async_session() as db:
        query_stmt = select(KnowledgeDocument)
        if workspace:
            query_stmt = query_stmt.where(KnowledgeDocument.workspace == workspace)
        result = await db.execute(query_stmt)
        documents = result.scalars().all()
        # Client-side tag filter (JSON array contains check)
        if tag:
            documents = [d for d in documents if d.tags and tag in d.tags]

        if documents:
            matches = []
            for doc in documents:
                content = doc.content.lower()
                title = doc.title.lower()
                score = 0
                matched_words = []

                for word in words:
                    if word in title:
                        score += 3
                        matched_words.append(word)
                    if word in content:
                        score += content.count(word)
                        if word not in matched_words:
                            matched_words.append(word)

                if score > 0:
                    snippet = extract_snippet(doc.content, matched_words)
                    matches.append({
                        "doc": doc,
                        "score": score,
                        "snippet": snippet,
                        "matched_words": matched_words,
                    })

            matches.sort(key=lambda x: x["score"], reverse=True)

            for match in matches[:max_results]:
                doc = match["doc"]
                knowledge_results.append({
                    "id": doc.id,
                    "title": doc.title,
                    "category": doc.category,
                    "workspace": doc.workspace,
                    "tags": doc.tags or [],
                    "content": match["snippet"],        # relevant excerpt
                    "full_content": doc.content,        # complete document text
                    "relevance_score": min(1.0, match["score"] / 10.0),
                    "matched_words": match["matched_words"],
                })

    return {
        "instant_answers": instant_answers,
        "knowledge_results": knowledge_results,
        "query": query,
        "total_instant_matches": len(instant_answers),
        "total_knowledge_matches": len(knowledge_results),
    }
