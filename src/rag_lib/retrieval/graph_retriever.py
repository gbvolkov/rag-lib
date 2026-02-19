from __future__ import annotations

import asyncio
import hashlib
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.stores import BaseStore
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, ConfigDict, Field

from rag_lib.core.logger import logger
from rag_lib.graph.domain import GraphEdge, GraphNode
from rag_lib.graph.store import BaseGraphStore


class GraphRetrieverError(Exception):
    """Base retriever exception for strict deterministic graph retrieval."""


class GraphConfigurationError(GraphRetrieverError):
    """Raised when retriever configuration is invalid for selected behavior."""


class GraphCapabilityError(GraphRetrieverError):
    """Raised when a required backend capability is missing."""


class GraphDataError(GraphRetrieverError):
    """Raised when required data format/content is invalid."""


@dataclass
class GraphQueryConfig:
    mode: Literal["local", "global", "hybrid", "mix"] = "hybrid"
    top_k_entities: int = 12
    top_k_relations: int = 24
    top_k_chunks: int = 10
    max_hops: int = 2
    min_score: float = 0.15
    use_rerank: bool = True
    enable_keyword_extraction: bool = True
    vector_relevance_mode: Literal["strict_0_1", "normalize_minmax"] = "strict_0_1"
    token_budget_total: int = 3500
    token_budget_entities: int = 700
    token_budget_relations: int = 900
    token_budget_chunks: int = 1900


@dataclass
class KeywordTiers:
    high_level_keywords: List[str]
    low_level_keywords: List[str]


class _KeywordPayload(BaseModel):
    high_level_keywords: List[str]
    low_level_keywords: List[str]


@dataclass
class _Evidence:
    evidence_id: str
    kind: Literal["chunk", "entity", "relation", "community"]
    content: str
    score: float
    source_segment_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized = re.sub(r"\s+", " ", self.content or "").strip().lower()
        self.content_hash = hashlib.sha1(normalized.encode("utf-8")).hexdigest()

    def dedup_key(self) -> tuple[str, str]:
        if self.source_segment_id:
            return ("source", str(self.source_segment_id))
        if self.content_hash:
            return ("content_hash", self.content_hash)
        return ("id", f"{self.kind}:{self.evidence_id}")

    def stable_sort_key(self) -> tuple[Any, ...]:
        kind_rank = {"chunk": 0, "relation": 1, "entity": 2, "community": 3}[self.kind]
        return (
            kind_rank,
            self.evidence_id,
            self.source_segment_id or "",
            self.content_hash,
        )


class GraphRetriever(BaseRetriever):
    vector_store: Optional[VectorStore] = None
    graph_store: BaseGraphStore
    config: GraphQueryConfig = Field(default_factory=GraphQueryConfig)
    embedder: Optional[Embeddings] = None
    llm: Optional[BaseChatModel] = None
    doc_store: Optional[BaseStore[str, Document]] = None
    id_key: str = "segment_id"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _STOPWORDS = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "for",
        "of",
        "to",
        "in",
        "on",
        "at",
        "is",
        "are",
        "в",
        "на",
        "и",
        "или",
        "по",
        "о",
        "что",
        "как",
    }

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        return self._run(query=query, top_k=top_k)

    async def aretrieve(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        return await self._arun(query=query, top_k=top_k)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return self._run(query=query, top_k=None)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        return await self._arun(query=query, top_k=None)

    def _run(self, query: str, top_k: Optional[int]) -> List[Document]:
        clean_query = (query or "").strip()
        if not clean_query:
            return []

        logger.info(f"Graph retrieval mode={self.config.mode} query={clean_query!r}")
        self._validate_config()
        keywords = self._extract_keywords(clean_query)

        if self.config.mode == "local":
            items = self._local(clean_query, keywords)
        elif self.config.mode == "global":
            items = self._global(clean_query, keywords)
        elif self.config.mode == "hybrid":
            local_items, global_items = self._run_local_global_parallel(clean_query, keywords)
            items = self._rrf([local_items, global_items])
        else:
            local_items, global_items = self._run_local_global_parallel(clean_query, keywords)
            hybrid_items = self._rrf([local_items, global_items])
            mix_chunks = self._vector_chunks(clean_query, self.config.top_k_chunks * 2)
            items = self._rrf([hybrid_items, mix_chunks])

        return self._finalize(clean_query, items, top_k=top_k)

    async def _arun(self, query: str, top_k: Optional[int]) -> List[Document]:
        clean_query = (query or "").strip()
        if not clean_query:
            return []

        logger.info(f"Async graph retrieval mode={self.config.mode} query={clean_query!r}")
        self._validate_config()
        keywords = await self._aextract_keywords(clean_query)

        if self.config.mode == "local":
            items = await self._alocal(clean_query, keywords)
        elif self.config.mode == "global":
            items = await self._aglobal(clean_query, keywords)
        elif self.config.mode == "hybrid":
            local_items, global_items = await asyncio.gather(
                self._alocal(clean_query, keywords),
                self._aglobal(clean_query, keywords),
            )
            items = self._rrf([local_items, global_items])
        else:
            local_items, global_items = await asyncio.gather(
                self._alocal(clean_query, keywords),
                self._aglobal(clean_query, keywords),
            )
            hybrid_items = self._rrf([local_items, global_items])
            mix_chunks = await self._avector_chunks(clean_query, self.config.top_k_chunks * 2)
            items = self._rrf([hybrid_items, mix_chunks])

        return self._finalize(clean_query, items, top_k=top_k)

    def _run_local_global_parallel(
        self, query: str, keywords: KeywordTiers
    ) -> tuple[List[_Evidence], List[_Evidence]]:
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="graph-retriever") as executor:
            local_future = executor.submit(self._local, query, keywords)
            global_future = executor.submit(self._global, query, keywords)
            local_items = local_future.result()
            global_items = global_future.result()
        return local_items, global_items

    def _validate_config(self) -> None:
        if self.config.mode not in {"local", "global", "hybrid", "mix"}:
            raise GraphConfigurationError(f"Unsupported graph mode: {self.config.mode}")
        if self.config.vector_relevance_mode not in {"strict_0_1", "normalize_minmax"}:
            raise GraphConfigurationError(
                f"Unsupported vector_relevance_mode: {self.config.vector_relevance_mode}"
            )
        if self.config.top_k_entities <= 0:
            raise GraphConfigurationError("top_k_entities must be > 0")
        if self.config.top_k_relations <= 0:
            raise GraphConfigurationError("top_k_relations must be > 0")
        if self.config.top_k_chunks <= 0:
            raise GraphConfigurationError("top_k_chunks must be > 0")
        if self.config.max_hops < 0:
            raise GraphConfigurationError("max_hops must be >= 0")
        if self.vector_store is None:
            raise GraphConfigurationError(
                "vector_store is required in strict mode for scoring and hydration"
            )

    def _extract_keywords(self, query: str) -> KeywordTiers:
        if self.config.enable_keyword_extraction:
            if self.llm is None:
                raise GraphConfigurationError(
                    "enable_keyword_extraction=True requires llm"
                )
            extractor = self._build_keyword_extractor()
            try:
                payload = extractor.invoke(self._keyword_prompt(query))
            except Exception as exc:
                raise GraphDataError(f"Structured keyword extraction failed: {exc}") from exc
            return self._keyword_tiers_from_payload(payload)

        return self._deterministic_keywords(query)

    async def _aextract_keywords(self, query: str) -> KeywordTiers:
        if self.config.enable_keyword_extraction:
            if self.llm is None:
                raise GraphConfigurationError(
                    "enable_keyword_extraction=True requires llm"
                )
            extractor = self._build_keyword_extractor()
            ainvoke = getattr(extractor, "ainvoke", None)
            if not callable(ainvoke):
                raise GraphCapabilityError(
                    "Structured keyword extractor must support ainvoke for async retrieval"
                )
            try:
                payload = await ainvoke(self._keyword_prompt(query))
            except Exception as exc:
                raise GraphDataError(f"Structured async keyword extraction failed: {exc}") from exc
            return self._keyword_tiers_from_payload(payload)

        return self._deterministic_keywords(query)

    def _build_keyword_extractor(self) -> Any:
        if self.llm is None:
            raise GraphConfigurationError("enable_keyword_extraction=True requires llm")
        builder = getattr(self.llm, "with_structured_output", None)
        if not callable(builder):
            raise GraphCapabilityError(
                "llm.with_structured_output is required when enable_keyword_extraction=True"
            )
        try:
            return builder(_KeywordPayload)
        except Exception as exc:
            raise GraphCapabilityError(f"Failed to initialize structured keyword extractor: {exc}") from exc

    def _keyword_prompt(self, query: str) -> str:
        return (
            "Extract retrieval keywords for graph retrieval.\n"
            "Return high_level_keywords as broad thematic concepts.\n"
            "Return low_level_keywords as concrete specific terms/entities.\n"
            f"Query: {query}"
        )

    def _keyword_tiers_from_payload(self, payload: Any) -> KeywordTiers:
        if not isinstance(payload, _KeywordPayload):
            raise GraphDataError(
                "Structured keyword payload must be _KeywordPayload; "
                f"got {type(payload).__name__}"
            )
        high = payload.high_level_keywords
        low = payload.low_level_keywords

        if not isinstance(high, list) or not isinstance(low, list):
            raise GraphDataError(
                "Structured keyword payload fields high_level_keywords and low_level_keywords must be lists"
            )

        high_clean = self._clean_keywords(high)
        low_clean = self._clean_keywords(low)
        if not high_clean or not low_clean:
            raise GraphDataError("Keyword tiers must be non-empty")

        return KeywordTiers(
            high_level_keywords=high_clean,
            low_level_keywords=low_clean,
        )

    def _deterministic_keywords(self, query: str) -> KeywordTiers:
        tokens = [
            token
            for token in self._tokenize(query)
            if token not in self._STOPWORDS
        ]
        if not tokens:
            tokens = [query.lower()]

        ordered: List[str] = []
        seen: set[str] = set()
        for token in tokens:
            if token in seen:
                continue
            seen.add(token)
            ordered.append(token)

        high = sorted(ordered, key=lambda t: (-len(t), t))[:2]
        low = ordered[:12]
        return KeywordTiers(high_level_keywords=high, low_level_keywords=low)

    def _clean_keywords(self, raw_keywords: List[Any]) -> List[str]:
        cleaned: List[str] = []
        seen: set[str] = set()
        for value in raw_keywords:
            token = str(value).strip().lower()
            if not token or token in seen:
                continue
            seen.add(token)
            cleaned.append(token)
        return cleaned

    def _local(self, query: str, keywords: KeywordTiers) -> List[_Evidence]:
        seed_queries = [query] + keywords.low_level_keywords
        candidates: List[tuple[GraphNode, float]] = []
        for seed_query in seed_queries:
            candidates.extend(
                self.graph_store.search_nodes_hybrid(
                    query=seed_query,
                    top_k=self.config.top_k_entities,
                    embedder=self.embedder,
                )
            )

        seed_scores = self._best_node_scores(candidates, self.config.top_k_entities)
        seed_ids = list(seed_scores.keys())
        expansion = self.graph_store.expand_subgraph(
            seed_node_ids=seed_ids,
            max_hops=self.config.max_hops,
            max_nodes=max(
                self.config.top_k_entities,
                self.config.top_k_entities * (self.config.max_hops + 2),
            ),
        )

        items = self._build_local_items(
            expansion.nodes,
            expansion.edges,
            expansion.hop_by_node,
            seed_scores,
        )

        edge_ids = [
            str(edge.properties.get("edge_id", f"{edge.source_id}|{edge.relation_type}|{edge.target_id}"))
            for edge in expansion.edges
        ]
        source_ids = self._merge_ids(
            self.graph_store.get_source_segment_ids_for_entities(seed_ids),
            self.graph_store.get_source_segment_ids_for_edges(edge_ids),
            [item.source_segment_id for item in items if item.source_segment_id],
        )
        hydrated = self._hydrate(query, source_ids, self.config.top_k_chunks)
        return items + hydrated

    async def _alocal(self, query: str, keywords: KeywordTiers) -> List[_Evidence]:
        seed_queries = [query] + keywords.low_level_keywords
        candidates: List[tuple[GraphNode, float]] = []
        for seed_query in seed_queries:
            candidates.extend(
                await self.graph_store.asearch_nodes_hybrid(
                    query=seed_query,
                    top_k=self.config.top_k_entities,
                    embedder=self.embedder,
                )
            )

        seed_scores = self._best_node_scores(candidates, self.config.top_k_entities)
        seed_ids = list(seed_scores.keys())
        expansion = await self.graph_store.aexpand_subgraph(
            seed_node_ids=seed_ids,
            max_hops=self.config.max_hops,
            max_nodes=max(
                self.config.top_k_entities,
                self.config.top_k_entities * (self.config.max_hops + 2),
            ),
        )

        items = await self._abuild_local_items(
            expansion.nodes,
            expansion.edges,
            expansion.hop_by_node,
            seed_scores,
        )

        edge_ids = [
            str(edge.properties.get("edge_id", f"{edge.source_id}|{edge.relation_type}|{edge.target_id}"))
            for edge in expansion.edges
        ]
        source_ids = self._merge_ids(
            await self.graph_store.aget_source_segment_ids_for_entities(seed_ids),
            await self.graph_store.aget_source_segment_ids_for_edges(edge_ids),
            [item.source_segment_id for item in items if item.source_segment_id],
        )
        hydrated = await self._ahydrate(query, source_ids, self.config.top_k_chunks)
        return items + hydrated

    def _global(self, query: str, keywords: KeywordTiers) -> List[_Evidence]:
        items: List[_Evidence] = []
        items.extend(self._communities(query))

        relation_queries = [query] + keywords.high_level_keywords
        edge_candidates: List[tuple[GraphEdge, float]] = []
        for relation_query in relation_queries:
            edge_candidates.extend(
                self.graph_store.search_edges_hybrid(
                    query=relation_query,
                    top_k=self.config.top_k_relations,
                    embedder=self.embedder,
                )
            )

        edge_scores = self._best_edge_scores(edge_candidates, self.config.top_k_relations)
        entity_scores: Dict[str, float] = defaultdict(float)
        relation_items: List[_Evidence] = []
        edge_ids: List[str] = []
        for edge, score in edge_scores:
            edge_id = str(edge.properties.get("edge_id", f"{edge.source_id}|{edge.relation_type}|{edge.target_id}"))
            edge_ids.append(edge_id)
            relation_items.append(
                _Evidence(
                    evidence_id=edge_id,
                    kind="relation",
                    content=self._render_relation(edge),
                    score=self._clamp(0.8 * score + 0.2 * self.graph_store.get_edge_prior(edge_id)),
                    source_segment_id=edge.source_segment_id,
                    metadata={"edge_id": edge_id},
                )
            )
            entity_scores[edge.source_id] = max(entity_scores[edge.source_id], score)
            entity_scores[edge.target_id] = max(entity_scores[edge.target_id], score)
        items.extend(relation_items)

        node_queries = [query] + keywords.high_level_keywords
        node_candidates: List[tuple[GraphNode, float]] = []
        for node_query in node_queries:
            node_candidates.extend(
                self.graph_store.search_nodes_hybrid(
                    query=node_query,
                    top_k=self.config.top_k_entities,
                    embedder=self.embedder,
                )
            )

        for node_id, score in self._best_node_scores(node_candidates, self.config.top_k_entities).items():
            entity_scores[node_id] = max(entity_scores[node_id], score)

        entity_items: List[_Evidence] = []
        entity_ids: List[str] = []
        for node_id, seed_score in sorted(entity_scores.items(), key=lambda row: (-row[1], row[0]))[: self.config.top_k_entities]:
            node = self.graph_store.get_node(node_id)
            if node is None:
                continue
            entity_ids.append(node_id)
            entity_items.append(
                _Evidence(
                    evidence_id=node_id,
                    kind="entity",
                    content=self._render_entity(node),
                    score=self._clamp(0.8 * seed_score + 0.2 * self.graph_store.get_node_prior(node_id)),
                    source_segment_id=node.source_segment_id,
                    metadata={"entity_id": node_id},
                )
            )
        items.extend(entity_items)

        source_ids = self._merge_ids(
            self.graph_store.get_source_segment_ids_for_entities(entity_ids),
            self.graph_store.get_source_segment_ids_for_edges(edge_ids),
            [item.source_segment_id for item in items if item.source_segment_id],
        )
        hydrated = self._hydrate(query, source_ids, self.config.top_k_chunks)
        return items + hydrated

    async def _aglobal(self, query: str, keywords: KeywordTiers) -> List[_Evidence]:
        items: List[_Evidence] = []
        items.extend(await self._acommunities(query))

        relation_queries = [query] + keywords.high_level_keywords
        edge_candidates: List[tuple[GraphEdge, float]] = []
        for relation_query in relation_queries:
            edge_candidates.extend(
                await self.graph_store.asearch_edges_hybrid(
                    query=relation_query,
                    top_k=self.config.top_k_relations,
                    embedder=self.embedder,
                )
            )

        edge_scores = self._best_edge_scores(edge_candidates, self.config.top_k_relations)
        entity_scores: Dict[str, float] = defaultdict(float)
        relation_items: List[_Evidence] = []
        edge_ids: List[str] = []
        edge_priors = await asyncio.gather(
            *[
                self.graph_store.aget_edge_prior(
                    str(edge.properties.get("edge_id", f"{edge.source_id}|{edge.relation_type}|{edge.target_id}"))
                )
                for edge, _ in edge_scores
            ]
        )
        for (edge, score), edge_prior in zip(edge_scores, edge_priors):
            edge_id = str(edge.properties.get("edge_id", f"{edge.source_id}|{edge.relation_type}|{edge.target_id}"))
            edge_ids.append(edge_id)
            relation_items.append(
                _Evidence(
                    evidence_id=edge_id,
                    kind="relation",
                    content=self._render_relation(edge),
                    score=self._clamp(0.8 * score + 0.2 * float(edge_prior)),
                    source_segment_id=edge.source_segment_id,
                    metadata={"edge_id": edge_id},
                )
            )
            entity_scores[edge.source_id] = max(entity_scores[edge.source_id], score)
            entity_scores[edge.target_id] = max(entity_scores[edge.target_id], score)
        items.extend(relation_items)

        node_queries = [query] + keywords.high_level_keywords
        node_candidates: List[tuple[GraphNode, float]] = []
        for node_query in node_queries:
            node_candidates.extend(
                await self.graph_store.asearch_nodes_hybrid(
                    query=node_query,
                    top_k=self.config.top_k_entities,
                    embedder=self.embedder,
                )
            )

        for node_id, score in self._best_node_scores(node_candidates, self.config.top_k_entities).items():
            entity_scores[node_id] = max(entity_scores[node_id], score)

        entity_items: List[_Evidence] = []
        entity_ids: List[str] = []
        for node_id, seed_score in sorted(entity_scores.items(), key=lambda row: (-row[1], row[0]))[: self.config.top_k_entities]:
            node = await self.graph_store.aget_node(node_id)
            if node is None:
                continue
            entity_ids.append(node_id)
            node_prior = await self.graph_store.aget_node_prior(node_id)
            entity_items.append(
                _Evidence(
                    evidence_id=node_id,
                    kind="entity",
                    content=self._render_entity(node),
                    score=self._clamp(0.8 * seed_score + 0.2 * node_prior),
                    source_segment_id=node.source_segment_id,
                    metadata={"entity_id": node_id},
                )
            )
        items.extend(entity_items)

        source_ids = self._merge_ids(
            await self.graph_store.aget_source_segment_ids_for_entities(entity_ids),
            await self.graph_store.aget_source_segment_ids_for_edges(edge_ids),
            [item.source_segment_id for item in items if item.source_segment_id],
        )
        hydrated = await self._ahydrate(query, source_ids, self.config.top_k_chunks)
        return items + hydrated

    def _build_local_items(
        self,
        nodes: List[GraphNode],
        edges: List[GraphEdge],
        hop_by_node: Dict[str, int],
        seed_scores: Dict[str, float],
    ) -> List[_Evidence]:
        items: List[_Evidence] = []
        max_weight = max((abs(edge.weight) for edge in edges), default=1.0) or 1.0

        for node in nodes:
            hop = hop_by_node.get(node.id, self.config.max_hops + 1)
            seed_score = seed_scores.get(node.id, 0.0)
            prior = self.graph_store.get_node_prior(node.id)
            decay = 1.0 if hop == 0 else 0.85 ** hop
            connectivity = 1.0 / (1.0 + hop)
            score = self._clamp(0.45 * seed_score + 0.30 * decay + 0.15 * prior + 0.10 * connectivity)
            items.append(
                _Evidence(
                    evidence_id=node.id,
                    kind="entity",
                    content=self._render_entity(node),
                    score=score,
                    source_segment_id=node.source_segment_id,
                    metadata={"entity_id": node.id},
                )
            )

        for edge in edges:
            edge_id = str(edge.properties.get("edge_id", f"{edge.source_id}|{edge.relation_type}|{edge.target_id}"))
            source_hop = hop_by_node.get(edge.source_id, self.config.max_hops + 1)
            target_hop = hop_by_node.get(edge.target_id, self.config.max_hops + 1)
            hop = min(source_hop, target_hop)
            decay = 1.0 if hop <= 0 else 0.85 ** hop
            endpoint = max(seed_scores.get(edge.source_id, 0.0), seed_scores.get(edge.target_id, 0.0))
            edge_prior = self.graph_store.get_edge_prior(edge_id)
            weight_score = abs(edge.weight) / max_weight
            score = self._clamp(0.35 * endpoint + 0.30 * decay + 0.25 * weight_score + 0.10 * edge_prior)
            items.append(
                _Evidence(
                    evidence_id=edge_id,
                    kind="relation",
                    content=self._render_relation(edge),
                    score=score,
                    source_segment_id=edge.source_segment_id,
                    metadata={"edge_id": edge_id},
                )
            )

        return items

    async def _abuild_local_items(
        self,
        nodes: List[GraphNode],
        edges: List[GraphEdge],
        hop_by_node: Dict[str, int],
        seed_scores: Dict[str, float],
    ) -> List[_Evidence]:
        items: List[_Evidence] = []
        max_weight = max((abs(edge.weight) for edge in edges), default=1.0) or 1.0

        node_priors = await asyncio.gather(
            *[self.graph_store.aget_node_prior(node.id) for node in nodes]
        )
        for node, prior in zip(nodes, node_priors):
            hop = hop_by_node.get(node.id, self.config.max_hops + 1)
            seed_score = seed_scores.get(node.id, 0.0)
            decay = 1.0 if hop == 0 else 0.85 ** hop
            connectivity = 1.0 / (1.0 + hop)
            score = self._clamp(0.45 * seed_score + 0.30 * decay + 0.15 * float(prior) + 0.10 * connectivity)
            items.append(
                _Evidence(
                    evidence_id=node.id,
                    kind="entity",
                    content=self._render_entity(node),
                    score=score,
                    source_segment_id=node.source_segment_id,
                    metadata={"entity_id": node.id},
                )
            )

        edge_ids = [
            str(edge.properties.get("edge_id", f"{edge.source_id}|{edge.relation_type}|{edge.target_id}"))
            for edge in edges
        ]
        edge_priors = await asyncio.gather(
            *[self.graph_store.aget_edge_prior(edge_id) for edge_id in edge_ids]
        )
        for edge, edge_id, edge_prior in zip(edges, edge_ids, edge_priors):
            source_hop = hop_by_node.get(edge.source_id, self.config.max_hops + 1)
            target_hop = hop_by_node.get(edge.target_id, self.config.max_hops + 1)
            hop = min(source_hop, target_hop)
            decay = 1.0 if hop <= 0 else 0.85 ** hop
            endpoint = max(seed_scores.get(edge.source_id, 0.0), seed_scores.get(edge.target_id, 0.0))
            weight_score = abs(edge.weight) / max_weight
            score = self._clamp(0.35 * endpoint + 0.30 * decay + 0.25 * weight_score + 0.10 * float(edge_prior))
            items.append(
                _Evidence(
                    evidence_id=edge_id,
                    kind="relation",
                    content=self._render_relation(edge),
                    score=score,
                    source_segment_id=edge.source_segment_id,
                    metadata={"edge_id": edge_id},
                )
            )

        return items

    def _render_entity(self, node: GraphNode) -> str:
        return f"Entity: {node.label}\nType: {node.type}\nDescription: {node.description or 'No description'}"

    def _render_relation(self, edge: GraphEdge) -> str:
        source_label = edge.properties.get("source_label", edge.source_id)
        target_label = edge.properties.get("target_label", edge.target_id)
        return f"Relation: {source_label} -[{edge.relation_type}]-> {target_label}\nWeight: {edge.weight}"

    def _communities(self, query: str) -> List[_Evidence]:
        rows = self._vector_search_scores_sync(
            query=query,
            k=self.config.top_k_relations,
            filter_={"is_community_summary": True},
        )
        items: List[_Evidence] = []
        for idx, (doc, score) in enumerate(rows):
            metadata = dict(doc.metadata or {})
            community_id = str(metadata.get("community_id", f"community_{idx}"))
            source_segment_id = metadata.get(self.id_key) or metadata.get("source_segment_id")
            items.append(
                _Evidence(
                    evidence_id=community_id,
                    kind="community",
                    content=doc.page_content,
                    score=self._clamp(float(score)),
                    source_segment_id=str(source_segment_id) if source_segment_id else None,
                    metadata={"community_id": community_id, **metadata},
                )
            )

        for row in self.graph_store.get_community_summaries(self.config.top_k_relations, query=query):
            summary = str(row.get("summary", "")).strip()
            if not summary:
                continue
            items.append(
                _Evidence(
                    evidence_id=str(row.get("community_id", "community")),
                    kind="community",
                    content=summary,
                    score=self._clamp(float(row.get("score", 0.0))),
                    source_segment_id=str(row.get("source_segment_id")) if row.get("source_segment_id") else None,
                    metadata={"community_id": row.get("community_id")},
                )
            )

        return self._dedupe(items)

    async def _acommunities(self, query: str) -> List[_Evidence]:
        rows = await self._vector_search_scores_async(
            query=query,
            k=self.config.top_k_relations,
            filter_={"is_community_summary": True},
        )
        items: List[_Evidence] = []
        for idx, (doc, score) in enumerate(rows):
            metadata = dict(doc.metadata or {})
            community_id = str(metadata.get("community_id", f"community_{idx}"))
            source_segment_id = metadata.get(self.id_key) or metadata.get("source_segment_id")
            items.append(
                _Evidence(
                    evidence_id=community_id,
                    kind="community",
                    content=doc.page_content,
                    score=self._clamp(float(score)),
                    source_segment_id=str(source_segment_id) if source_segment_id else None,
                    metadata={"community_id": community_id, **metadata},
                )
            )

        for row in await self.graph_store.aget_community_summaries(self.config.top_k_relations, query=query):
            summary = str(row.get("summary", "")).strip()
            if not summary:
                continue
            items.append(
                _Evidence(
                    evidence_id=str(row.get("community_id", "community")),
                    kind="community",
                    content=summary,
                    score=self._clamp(float(row.get("score", 0.0))),
                    source_segment_id=str(row.get("source_segment_id")) if row.get("source_segment_id") else None,
                    metadata={"community_id": row.get("community_id")},
                )
            )

        return self._dedupe(items)

    def _vector_chunks(self, query: str, top_k: int) -> List[_Evidence]:
        rows = self._vector_search_scores_sync(query=query, k=top_k, filter_=None)
        return self._docs_to_chunks(query=query, rows=rows)

    async def _avector_chunks(self, query: str, top_k: int) -> List[_Evidence]:
        rows = await self._vector_search_scores_async(query=query, k=top_k, filter_=None)
        return self._docs_to_chunks(query=query, rows=rows)

    def _hydrate(self, query: str, source_segment_ids: List[str], top_k: int) -> List[_Evidence]:
        if not source_segment_ids:
            return []

        unique_ids = self._merge_ids(source_segment_ids)
        if self.doc_store is not None:
            docs = self._hydrate_from_docstore_sync(unique_ids)
        else:
            docs = self._hydrate_from_vector_sync(unique_ids)

        return self._score_and_convert_hydrated_docs(query, unique_ids, docs, top_k)

    async def _ahydrate(self, query: str, source_segment_ids: List[str], top_k: int) -> List[_Evidence]:
        if not source_segment_ids:
            return []

        unique_ids = self._merge_ids(source_segment_ids)
        if self.doc_store is not None:
            docs = await self._hydrate_from_docstore_async(unique_ids)
        else:
            docs = await self._hydrate_from_vector_async(unique_ids)

        return self._score_and_convert_hydrated_docs(query, unique_ids, docs, top_k)

    def _score_and_convert_hydrated_docs(
        self,
        query: str,
        expected_ids: List[str],
        docs: List[Document],
        top_k: int,
    ) -> List[_Evidence]:
        expected = set(expected_ids)
        scored_rows: List[tuple[Document, float]] = []
        for doc in docs:
            if not isinstance(doc, Document):
                continue
            metadata = dict(doc.metadata or {})
            seg_id = metadata.get(self.id_key) or metadata.get("source_segment_id")
            if seg_id is not None and str(seg_id) not in expected:
                continue
            score = self._clamp(self._overlap(query, doc.page_content))
            scored_rows.append((doc, score))

        scored_rows.sort(key=lambda row: (-row[1], self._stable_doc_key(row[0])))
        return self._docs_to_chunks(query=query, rows=scored_rows[:top_k])
    def _hydrate_from_docstore_sync(self, segment_ids: List[str]) -> List[Document]:
        if self.doc_store is None:
            raise GraphConfigurationError("doc_store is required for document-store hydration")
        mget = getattr(self.doc_store, "mget", None)
        if not callable(mget):
            raise GraphCapabilityError("doc_store.mget is required")
        docs = mget(segment_ids)
        if docs is None:
            raise GraphDataError("doc_store.mget returned None")
        return [doc for doc in docs if isinstance(doc, Document)]

    async def _hydrate_from_docstore_async(self, segment_ids: List[str]) -> List[Document]:
        if self.doc_store is None:
            raise GraphConfigurationError("doc_store is required for document-store hydration")
        amget = getattr(self.doc_store, "amget", None)
        if not callable(amget):
            raise GraphCapabilityError("doc_store.amget is required for async hydration")
        docs = await amget(segment_ids)
        if docs is None:
            raise GraphDataError("doc_store.amget returned None")
        return [doc for doc in docs if isinstance(doc, Document)]

    def _hydrate_from_vector_sync(self, segment_ids: List[str]) -> List[Document]:
        if self.vector_store is None:
            raise GraphConfigurationError("vector_store is required for vector hydration")
        get_by_ids = getattr(self.vector_store, "get_by_ids", None)
        if not callable(get_by_ids):
            raise GraphCapabilityError("vector_store.get_by_ids is required when doc_store is not provided")
        docs = get_by_ids(segment_ids)
        if docs is None:
            raise GraphDataError("vector_store.get_by_ids returned None")
        return [doc for doc in docs if isinstance(doc, Document)]

    async def _hydrate_from_vector_async(self, segment_ids: List[str]) -> List[Document]:
        if self.vector_store is None:
            raise GraphConfigurationError("vector_store is required for vector hydration")
        aget_by_ids = getattr(self.vector_store, "aget_by_ids", None)
        if not callable(aget_by_ids):
            raise GraphCapabilityError("vector_store.aget_by_ids is required for async hydration")
        docs = await aget_by_ids(segment_ids)
        if docs is None:
            raise GraphDataError("vector_store.aget_by_ids returned None")
        return [doc for doc in docs if isinstance(doc, Document)]

    def _docs_to_chunks(self, query: str, rows: List[tuple[Document, float]]) -> List[_Evidence]:
        items: List[_Evidence] = []
        for idx, (doc, score) in enumerate(rows):
            metadata = dict(doc.metadata or {})
            segment_id = metadata.get(self.id_key) or metadata.get("source_segment_id")
            if segment_id is None and doc.id is not None:
                segment_id = str(doc.id)

            evidence_id = str(segment_id or f"chunk_{idx}")
            final_score = self._clamp(0.75 * float(score) + 0.25 * self._overlap(query, doc.page_content))
            items.append(
                _Evidence(
                    evidence_id=evidence_id,
                    kind="chunk",
                    content=doc.page_content,
                    score=final_score,
                    source_segment_id=str(segment_id) if segment_id else None,
                    metadata=metadata,
                )
            )

        return self._dedupe(items)

    def _finalize(self, query: str, items: List[_Evidence], top_k: Optional[int]) -> List[Document]:
        ranked = self._dedupe(items)

        if self.config.use_rerank:
            for item in ranked:
                item.score = self._clamp(0.75 * item.score + 0.25 * self._overlap(query, item.content))

        ranked = [item for item in ranked if item.score >= self.config.min_score]

        buckets = {
            "chunk": [],
            "relation": [],
            "entity": [],
            "community": [],
        }
        for item in ranked:
            buckets[item.kind].append(item)

        for kind in buckets:
            buckets[kind].sort(key=lambda item: (-item.score, item.stable_sort_key()))

        selected: List[_Evidence] = []
        selected.extend(self._pick_budget(buckets["chunk"], self.config.token_budget_chunks))
        selected.extend(self._pick_budget(buckets["relation"], self.config.token_budget_relations))
        selected.extend(self._pick_budget(buckets["entity"], self.config.token_budget_entities))

        spent = sum(self._tokens(item.content) for item in selected)
        remaining = max(self.config.token_budget_total - spent, 0)
        selected.extend(self._pick_budget(buckets["community"], remaining))

        selected = self._pick_budget(selected, self.config.token_budget_total)

        if top_k is not None:
            selected = selected[: max(0, top_k)]

        docs = [self._to_doc(item) for item in selected]
        docs.sort(
            key=lambda doc: (
                -float(doc.metadata.get("score", 0.0)),
                self._stable_doc_key(doc),
            )
        )
        return docs

    def _to_doc(self, item: _Evidence) -> Document:
        metadata = dict(item.metadata)
        metadata["retrieval_kind"] = item.kind
        metadata["score"] = float(item.score)
        metadata["graph_mode"] = self.config.mode
        if item.source_segment_id:
            metadata["source_segment_id"] = item.source_segment_id
        if item.kind == "entity":
            metadata.setdefault("entity_id", item.evidence_id)
        elif item.kind == "relation":
            metadata.setdefault("edge_id", item.evidence_id)
        elif item.kind == "community":
            metadata.setdefault("community_id", item.evidence_id)
        return Document(page_content=item.content, metadata=metadata)

    def _pick_budget(self, items: List[_Evidence], budget: int) -> List[_Evidence]:
        if budget <= 0:
            return []

        selected: List[_Evidence] = []
        used = 0
        for item in items:
            cost = self._tokens(item.content)
            if selected and (used + cost) > budget:
                continue
            selected.append(item)
            used += cost
            if used >= budget:
                break
        return selected

    def _rrf(self, rankings: List[List[_Evidence]], k: int = 60) -> List[_Evidence]:
        base_items: Dict[tuple[str, str], _Evidence] = {}
        rrf_scores: Dict[tuple[str, str], float] = defaultdict(float)

        for ranking in rankings:
            ordered = sorted(ranking, key=lambda item: (-item.score, item.stable_sort_key()))
            for rank, item in enumerate(ordered, start=1):
                key = item.dedup_key()
                rrf_scores[key] += 1.0 / (k + rank)
                current = base_items.get(key)
                if current is None or item.score > current.score:
                    base_items[key] = item

        if not base_items:
            return []

        max_rrf = max(rrf_scores.values())
        merged: List[_Evidence] = []
        for key, item in base_items.items():
            fused = rrf_scores[key] / max_rrf if max_rrf > 0 else 0.0
            item.score = self._clamp(0.55 * item.score + 0.45 * fused)
            merged.append(item)

        merged.sort(key=lambda item: (-item.score, item.stable_sort_key()))
        return merged

    def _best_node_scores(
        self, rows: List[tuple[GraphNode, float]], limit: int
    ) -> Dict[str, float]:
        scores: Dict[str, float] = defaultdict(float)
        for node, score in rows:
            scores[node.id] = max(scores[node.id], self._clamp(float(score)))

        ranked = sorted(scores.items(), key=lambda row: (-row[1], row[0]))[: max(limit, 0)]
        return {node_id: score for node_id, score in ranked}

    def _best_edge_scores(
        self, rows: List[tuple[GraphEdge, float]], limit: int
    ) -> List[tuple[GraphEdge, float]]:
        best: Dict[str, tuple[GraphEdge, float]] = {}
        for edge, score in rows:
            edge_id = str(edge.properties.get("edge_id", f"{edge.source_id}|{edge.relation_type}|{edge.target_id}"))
            value = self._clamp(float(score))
            current = best.get(edge_id)
            if current is None or value > current[1]:
                best[edge_id] = (edge, value)

        ranked = sorted(
            best.values(),
            key=lambda row: (
                -row[1],
                str(row[0].properties.get("edge_id", "")),
                row[0].source_id,
                row[0].target_id,
            ),
        )
        return ranked[: max(limit, 0)]

    def _dedupe(self, items: List[_Evidence]) -> List[_Evidence]:
        best: Dict[tuple[str, str], _Evidence] = {}
        for item in items:
            key = item.dedup_key()
            current = best.get(key)
            if current is None:
                best[key] = item
                continue
            if item.score > current.score:
                best[key] = item
                continue
            if item.score == current.score and item.stable_sort_key() < current.stable_sort_key():
                best[key] = item

        return sorted(best.values(), key=lambda item: (-item.score, item.stable_sort_key()))

    def _vector_search_scores_sync(
        self,
        query: str,
        k: int,
        filter_: Optional[Dict[str, Any]],
    ) -> List[tuple[Document, float]]:
        if self.vector_store is None:
            raise GraphConfigurationError("vector_store is required")

        scorer = getattr(self.vector_store, "similarity_search_with_relevance_scores", None)
        if not callable(scorer):
            raise GraphCapabilityError("vector_store.similarity_search_with_relevance_scores is required")

        kwargs: Dict[str, Any] = {"k": k}
        if filter_ is not None:
            kwargs["filter"] = filter_

        rows = scorer(query, **kwargs)
        if not isinstance(rows, list):
            raise GraphDataError("similarity_search_with_relevance_scores must return list")

        return self._coerce_vector_score_rows(
            rows,
            scorer_name="similarity_search_with_relevance_scores",
        )

    async def _vector_search_scores_async(
        self,
        query: str,
        k: int,
        filter_: Optional[Dict[str, Any]],
    ) -> List[tuple[Document, float]]:
        if self.vector_store is None:
            raise GraphConfigurationError("vector_store is required")

        scorer = getattr(self.vector_store, "asimilarity_search_with_relevance_scores", None)
        if not callable(scorer):
            raise GraphCapabilityError("vector_store.asimilarity_search_with_relevance_scores is required for async retrieval")

        kwargs: Dict[str, Any] = {"k": k}
        if filter_ is not None:
            kwargs["filter"] = filter_

        rows = await scorer(query, **kwargs)
        if not isinstance(rows, list):
            raise GraphDataError("asimilarity_search_with_relevance_scores must return list")

        return self._coerce_vector_score_rows(
            rows,
            scorer_name="asimilarity_search_with_relevance_scores",
        )

    def _coerce_vector_score_rows(
        self,
        rows: Any,
        *,
        scorer_name: str,
    ) -> List[tuple[Document, float]]:
        if not isinstance(rows, list):
            raise GraphDataError(f"{scorer_name} must return list")

        parsed: List[tuple[Document, float]] = []
        for row in rows:
            if not isinstance(row, tuple) or len(row) != 2:
                raise GraphDataError(
                    f"{scorer_name} rows must be tuple[Document, float]"
                )
            doc, score = row
            if not isinstance(doc, Document):
                raise GraphDataError(
                    f"{scorer_name} rows must contain Document"
                )
            try:
                parsed.append((doc, float(score)))
            except (TypeError, ValueError) as exc:
                raise GraphDataError(
                    f"{scorer_name} score must be numeric, got {score!r}"
                ) from exc

        if not parsed:
            return parsed

        min_score = min(score for _, score in parsed)
        max_score = max(score for _, score in parsed)
        out_of_range = min_score < 0.0 or max_score > 1.0
        if not out_of_range:
            return parsed

        if self.config.vector_relevance_mode == "normalize_minmax":
            return self._normalize_minmax_scores(parsed)

        raise GraphDataError(
            f"{scorer_name} returned relevance scores outside [0, 1] "
            f"(min={min_score:.6f}, max={max_score:.6f}). "
            "Set GraphQueryConfig(vector_relevance_mode='normalize_minmax') only when "
            "the backend returns raw distance-like scores."
        )

    def _normalize_minmax_scores(
        self,
        rows: List[tuple[Document, float]],
    ) -> List[tuple[Document, float]]:
        min_score = min(score for _, score in rows)
        max_score = max(score for _, score in rows)
        spread = max_score - min_score
        if spread <= 1e-12:
            # All raw scores are effectively identical; keep deterministic neutral relevance.
            return [(doc, 0.5) for doc, _ in rows]
        return [(doc, (score - min_score) / spread) for doc, score in rows]

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9\u0400-\u04FF]+", text.lower())

    def _overlap(self, query: str, text: str) -> float:
        query_tokens = {
            token
            for token in self._tokenize(query)
            if token and token not in self._STOPWORDS
        }
        text_tokens = set(self._tokenize(text))
        if not query_tokens:
            return 0.0
        return len(query_tokens.intersection(text_tokens)) / len(query_tokens)

    def _tokens(self, text: str) -> int:
        return max(1, int(0.75 * len(re.findall(r"\S+", text or ""))))

    def _stable_doc_key(self, doc: Document) -> tuple[Any, ...]:
        metadata = doc.metadata or {}
        return (
            metadata.get("retrieval_kind", ""),
            metadata.get("entity_id", ""),
            metadata.get("edge_id", ""),
            metadata.get("community_id", ""),
            metadata.get("source_segment_id", ""),
            hashlib.sha1((doc.page_content or "").encode("utf-8")).hexdigest(),
        )

    def _clamp(self, value: float) -> float:
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return float(value)

    def _merge_ids(self, *iterables: Iterable[Optional[str]]) -> List[str]:
        merged: List[str] = []
        seen: set[str] = set()
        for iterable in iterables:
            for value in iterable:
                if not value:
                    continue
                text = str(value)
                if text in seen:
                    continue
                seen.add(text)
                merged.append(text)
        return merged
