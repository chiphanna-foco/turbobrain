"""Extract Q&A pairs from knowledge base documents using Claude Haiku."""

import json
import logging
from typing import Optional
from anthropic import AsyncAnthropic
from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class QAExtractor:
    """Extracts potential instant-answer Q&A pairs from knowledge docs."""

    def __init__(self):
        self._client = None

    def _get_client(self) -> AsyncAnthropic:
        if self._client is None:
            self._client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        return self._client

    async def extract_qa_pairs(
        self,
        document_title: str,
        document_content: str,
        document_category: str,
        existing_keys: list[str],
    ) -> list[dict]:
        """Extract Q&A pairs from a knowledge base document.

        Args:
            document_title: Title of the knowledge document
            document_content: Full markdown content
            document_category: Category (e.g., "ach billing", "showings")
            existing_keys: List of existing InstantAnswer keys to avoid duplicates

        Returns:
            List of dicts with: key, answer, talking_points, suggested_response, confidence
        """
        client = self._get_client()

        truncated_content = document_content[:6000]

        system_prompt = (
            "You are extracting FAQ-style instant answers from a TurboTenant/Ziprent "
            "customer support knowledge base document.\n\n"
            "For each distinct topic in the document, create a Q&A pair that a customer "
            "support agent could use to quickly answer a customer's question.\n\n"
            "Rules:\n"
            "- Extract 2-6 Q&A pairs per document (only the most important, actionable ones)\n"
            "- Each \"key\" should be a snake_case identifier (max 60 chars) describing the topic\n"
            "- Each \"answer\" should be a direct, concise answer (1-3 sentences)\n"
            "- \"talking_points\" should be 2-4 bullet points the agent can reference\n"
            "- \"suggested_response\" should be what the agent could say to the customer verbatim\n"
            "- \"confidence\" should be \"high\" if the document is very specific, \"medium\" if general\n"
            "- Do NOT create Q&A pairs that duplicate these existing keys: "
            + json.dumps(existing_keys[:50])
            + "\n\nRespond with ONLY a JSON array, no markdown formatting."
        )

        user_prompt = (
            f"Extract instant answer Q&A pairs from this knowledge base document.\n\n"
            f"Document Title: {document_title}\n"
            f"Category: {document_category}\n\n"
            f"Content:\n{truncated_content}\n\n"
            "Return a JSON array of objects with these fields:\n"
            '[\n  {\n    "key": "snake_case_identifier",\n'
            '    "answer": "Direct concise answer",\n'
            '    "talking_points": ["Point 1", "Point 2"],\n'
            '    "suggested_response": "What the agent should say",\n'
            '    "confidence": "high" | "medium" | "low"\n  }\n]'
        )

        try:
            response = await client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=2000,
                messages=[{"role": "user", "content": user_prompt}],
                system=system_prompt,
            )

            response_text = response.content[0].text.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            response_text = response_text.strip()

            pairs = json.loads(response_text)
            if not isinstance(pairs, list):
                pairs = [pairs]

            validated = []
            for pair in pairs:
                if not isinstance(pair, dict):
                    continue
                if not pair.get("key") or not pair.get("answer"):
                    continue
                validated.append({
                    "key": pair["key"][:100],
                    "answer": pair["answer"],
                    "talking_points": pair.get("talking_points", []),
                    "suggested_response": pair.get("suggested_response", ""),
                    "confidence": pair.get("confidence", "medium"),
                })
            return validated

        except Exception as e:
            logger.error(f"QA extraction failed for '{document_title}': {e}")
            return []


_extractor: Optional[QAExtractor] = None


def get_qa_extractor() -> QAExtractor:
    global _extractor
    if _extractor is None:
        _extractor = QAExtractor()
    return _extractor
