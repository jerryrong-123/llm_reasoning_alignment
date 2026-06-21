from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ProcessedQuery:
    original_query: str
    search_queries: List[str]
    mode: str
    metadata: Dict[str, Any]


class QueryProcessor:
    def __init__(
        self,
        mode: str = "decompose",
        lowercase: bool = False,
        strip_whitespace: bool = True,
        remove_question_mark: bool = False,
        max_search_queries: int = 4,
    ) -> None:
        self.mode = mode
        self.lowercase = lowercase
        self.strip_whitespace = strip_whitespace
        self.remove_question_mark = remove_question_mark
        self.max_search_queries = max_search_queries

        valid_modes = {"none", "rewrite", "decompose"}
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid query processing mode: {self.mode}")

    def process(self, question: str) -> ProcessedQuery:
        original_query = question
        normalized_query = self._normalize(question)

        if self.mode == "none":
            search_queries = [normalized_query]
            metadata = {"strategy": "original_query_only"}

        elif self.mode == "rewrite":
            rewritten = self._rewrite(normalized_query)
            search_queries = self._deduplicate([rewritten, normalized_query])
            metadata = {"strategy": "rule_based_rewrite"}

        elif self.mode == "decompose":
            decomposed_queries = self._decompose(normalized_query)
            search_queries = self._deduplicate(decomposed_queries + [normalized_query])
            metadata = {
                "strategy": "rule_based_decompose",
                "decomposed_query_count": len(decomposed_queries),
            }

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        search_queries = [
            query for query in search_queries
            if isinstance(query, str) and query.strip()
        ][: self.max_search_queries]

        return ProcessedQuery(
            original_query=original_query,
            search_queries=search_queries,
            mode=self.mode,
            metadata=metadata,
        )

    def _normalize(self, question: str) -> str:
        query = question.strip()
        query = re.sub(r"\s+", " ", query)

        if self.remove_question_mark:
            query = query.rstrip("?？").strip()

        if self.lowercase:
            query = query.lower()

        return query

    def _rewrite(self, query: str) -> str:
        rewritten = query
        rewritten = rewritten.replace("Which magazine was started first", "start date comparison")
        rewritten = rewritten.replace("Which came first", "start date comparison")
        rewritten = rewritten.replace("Which was founded first", "founding date comparison")
        rewritten = rewritten.replace("Which was released first", "release date comparison")
        rewritten = re.sub(r"\s+", " ", rewritten).strip()
        return rewritten

    def _decompose(self, query: str) -> List[str]:
        candidates: List[str] = []

        patterns = [
            (
                r"which\s+(.+?)\s+was\s+started\s+first\s+(.+?)\s+or\s+(.+?)\??$",
                "start date",
            ),
            (
                r"which\s+was\s+started\s+first\s+(.+?)\s+or\s+(.+?)\??$",
                "start date",
            ),
            (
                r"which\s+(.+?)\s+was\s+founded\s+first\s+(.+?)\s+or\s+(.+?)\??$",
                "founding date",
            ),
            (
                r"which\s+was\s+founded\s+first\s+(.+?)\s+or\s+(.+?)\??$",
                "founding date",
            ),
            (
                r"which\s+(.+?)\s+was\s+released\s+first\s+(.+?)\s+or\s+(.+?)\??$",
                "release date",
            ),
            (
                r"which\s+was\s+released\s+first\s+(.+?)\s+or\s+(.+?)\??$",
                "release date",
            ),
        ]

        for pattern, suffix in patterns:
            match = re.search(pattern, query, flags=re.IGNORECASE)
            if match:
                if len(match.groups()) == 3:
                    entity_a = self._clean_entity(match.group(2))
                    entity_b = self._clean_entity(match.group(3))
                else:
                    entity_a = self._clean_entity(match.group(1))
                    entity_b = self._clean_entity(match.group(2))

                if entity_a and entity_b:
                    candidates.append(f"{entity_a} {suffix}")
                    candidates.append(f"{entity_b} {suffix}")
                    candidates.append(f"{entity_a} {entity_b} comparison")
                break

        if " or " in query.lower() and not candidates:
            parts = re.split(r"\s+or\s+", query, flags=re.IGNORECASE)
            if len(parts) == 2:
                left = self._clean_entity(parts[0])
                right = self._clean_entity(parts[1])
                if left and right:
                    candidates.append(left)
                    candidates.append(right)
                    candidates.append(f"{left} {right} comparison")

        if not candidates:
            candidates.append(self._rewrite(query))

        return self._deduplicate(candidates)

    def _clean_entity(self, text: Optional[str]) -> str:
        if text is None:
            return ""

        cleaned = text.strip()
        cleaned = cleaned.strip("?？.,;:!\"'")
        cleaned = re.sub(r"^(the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _deduplicate(self, queries: List[str]) -> List[str]:
        seen = set()
        results: List[str] = []

        for query in queries:
            normalized = query.strip()
            key = normalized.lower()

            if not normalized or key in seen:
                continue

            seen.add(key)
            results.append(normalized)

        return results
