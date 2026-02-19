from enum import Enum
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.stores import BaseStore, ByteStore
from langchain_core.vectorstores import VectorStore
from pydantic import ConfigDict, Field, model_validator
from typing_extensions import override

from langchain_classic.storage._lc_store import create_kv_docstore


class SearchType(str, Enum):
    """Enumerator of the types of search to perform."""

    similarity = "similarity"
    """Similarity search."""
    similarity_score_threshold = "similarity_score_threshold"
    """Similarity search with a score threshold."""
    mmr = "mmr"
    """Maximal Marginal Relevance reranking of similarity search."""


class HydrationMode(str, Enum):
    """
    How to compose final results from vector-store children + doc-store parents.
    """

    parents_replace = "parents_replace"
    """Replace matched children with deduplicated hydrated parents."""
    children_enriched = "children_enriched"
    """Keep children, but enrich child content with hydrated parent content."""
    children_plus_parents = "children_plus_parents"
    """Return children first, then append deduplicated hydrated parents."""


class ScoredMultiVectorRetriever(BaseRetriever):
    """
    Score-aware dual-store retriever.

    Strict parent/child semantics:
    - child = retrieved hit from vector store
    - parent = document loaded from doc store by `id_key` from retrieved children

    Parent scoring is computed only from currently retrieved children
    (never from global tree/doc_store/vector_store scans).
    """

    model_config = ConfigDict(extra="forbid")

    vector_store: VectorStore
    """The underlying `VectorStore` to use to store small chunks
    and their embedding vectors"""

    byte_store: ByteStore | None = None
    """The lower-level backing storage layer for the parent documents"""

    doc_store: BaseStore[str, Document]
    """The storage interface for the parent documents"""

    id_key: str = "doc_id"

    search_kwargs: dict = Field(default_factory=dict)
    """Keyword arguments to pass to the search function."""

    search_type: SearchType = SearchType.similarity
    """Type of search to perform (similarity / mmr)"""

    score_threshold: float | None = None
    """Threshold for similarity search"""

    hydration_mode: HydrationMode = HydrationMode.parents_replace
    """Result composition mode for parent hydration."""

    enrichment_separator: str = "\n\n--- MATCHED CHILD CHUNK ---\n\n"
    """Separator used when `hydration_mode=children_enriched`."""

    @model_validator(mode="before")
    @classmethod
    def _shim_docstore(cls, values: dict) -> Any:
        if "search_threshold" in values:
            msg = "`search_threshold` is not supported. Use `score_threshold`."
            raise ValueError(msg)

        byte_store = values.get("byte_store")
        doc_store = values.get("doc_store")
        if byte_store is not None:
            doc_store = create_kv_docstore(byte_store)
        elif doc_store is None:
            msg = "You must pass a `doc_store` or `byte_store` parameter."
            raise ValueError(msg)
        values["doc_store"] = doc_store
        return values

    def _clone_document(self, doc: Document) -> Document:
        return Document(
            id=doc.id,
            page_content=doc.page_content,
            metadata=(doc.metadata or {}).copy(),
        )

    def _normalize_parent_key(self, doc: Document) -> None:
        pid = doc.metadata.get(self.id_key)
        if pid is not None:
            doc.metadata[self.id_key] = str(pid)

    def _normalize_document_id(self, doc: Document) -> None:
        if doc.id is not None:
            return

        segment_id = doc.metadata.get("segment_id")
        if segment_id is not None:
            doc.id = str(segment_id)
            return

        pid = doc.metadata.get(self.id_key)
        if pid is not None:
            doc.id = str(pid)

    def _finalize_documents(self, docs: list[Document]) -> list[Document]:
        for doc in docs:
            self._normalize_parent_key(doc)
            self._normalize_document_id(doc)
        return docs

    @override
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """Get documents relevant to a query.

        Args:
            query: String to find relevant documents for
            run_manager: The callbacks handler to use
        Returns:
            List of relevant documents.
        """
        children = self._search_children(query)
        ordered_parent_ids, parent_map = self._load_parents_for_retrieved_children(children)
        return self._compose_by_mode(
            children=children,
            ordered_parent_ids=ordered_parent_ids,
            parent_map=parent_map,
        )

    def _search_children(
        self,
        query: str,
    ) -> list[Document]:
        if self.search_type == SearchType.mmr:
            docs = self.vector_store.max_marginal_relevance_search(
                query,
                **self.search_kwargs,
            )
        elif self.search_type == SearchType.similarity_score_threshold:
            kwargs = self.search_kwargs.copy()
            if self.score_threshold is not None and "score_threshold" not in kwargs:
                kwargs["score_threshold"] = self.score_threshold
            docs_and_scores = self.vector_store.similarity_search_with_relevance_scores(
                query,
                **kwargs,
            )
            docs = self._annotate_scores_from_hits(docs_and_scores)
        else:
            docs = self.vector_store.similarity_search(query, **self.search_kwargs)
        return self._finalize_documents(docs)

    async def _asearch_children(
        self,
        query: str,
    ) -> list[Document]:
        if self.search_type == SearchType.mmr:
            docs = await self.vector_store.amax_marginal_relevance_search(
                query,
                **self.search_kwargs,
            )
        elif self.search_type == SearchType.similarity_score_threshold:
            kwargs = self.search_kwargs.copy()
            if self.score_threshold is not None and "score_threshold" not in kwargs:
                kwargs["score_threshold"] = self.score_threshold
            docs_and_scores = await self.vector_store.asimilarity_search_with_relevance_scores(
                query,
                **kwargs,
            )
            docs = self._annotate_scores_from_hits(docs_and_scores)
        else:
            docs = await self.vector_store.asimilarity_search(query, **self.search_kwargs)
        return self._finalize_documents(docs)

    def _annotate_scores_from_hits(
        self,
        docs_and_scores: list[tuple[Document, float]],
    ) -> list[Document]:
        """
        Attach per-child score + parent-level max score from retrieved children only.
        """
        parent_max: dict[str, float] = {}
        docs: list[Document] = []

        for doc, score in docs_and_scores:
            child = self._clone_document(doc)
            child.metadata["similarity_score"] = score
            self._normalize_parent_key(child)

            pid = child.metadata.get(self.id_key)
            if pid is not None:
                current = parent_max.get(pid)
                if current is None or score > current:
                    parent_max[pid] = score

            docs.append(child)

        for child in docs:
            pid = child.metadata.get(self.id_key)
            if pid is not None and pid in parent_max:
                child.metadata["max_similarity_score"] = parent_max[pid]
            elif "similarity_score" in child.metadata:
                child.metadata["max_similarity_score"] = child.metadata["similarity_score"]

        return docs

    def _collect_ordered_parent_ids(self, children: list[Document]) -> list[str]:
        ids: list[str] = []
        seen: set[str] = set()
        for child in children:
            pid = child.metadata.get(self.id_key)
            if pid is None:
                continue
            pid_str = str(pid)
            child.metadata[self.id_key] = pid_str
            if pid_str not in seen:
                seen.add(pid_str)
                ids.append(pid_str)
        return ids

    def _load_parents_for_retrieved_children(
        self,
        children: list[Document],
    ) -> tuple[list[str], dict[str, Document]]:
        ordered_ids = self._collect_ordered_parent_ids(children)
        if not ordered_ids:
            return ordered_ids, {}

        loaded = self.doc_store.mget(ordered_ids)
        parent_map: dict[str, Document] = {}
        for pid, doc in zip(ordered_ids, loaded):
            if doc is None:
                continue
            if not isinstance(doc, Document):
                continue
            parent_map[pid] = self._clone_document(doc)
        return ordered_ids, parent_map

    async def _aload_parents_for_retrieved_children(
        self,
        children: list[Document],
    ) -> tuple[list[str], dict[str, Document]]:
        ordered_ids = self._collect_ordered_parent_ids(children)
        if not ordered_ids:
            return ordered_ids, {}

        loaded = await self.doc_store.amget(ordered_ids)
        parent_map: dict[str, Document] = {}
        for pid, doc in zip(ordered_ids, loaded):
            if doc is None:
                continue
            if not isinstance(doc, Document):
                continue
            parent_map[pid] = self._clone_document(doc)
        return ordered_ids, parent_map

    def _build_parent_max_score_map(self, children: list[Document]) -> dict[str, float]:
        parent_max: dict[str, float] = {}
        for child in children:
            pid = child.metadata.get(self.id_key)
            if pid is None:
                continue
            score = child.metadata.get("similarity_score")
            if score is None:
                continue
            current = parent_max.get(pid)
            if current is None or score > current:
                parent_max[pid] = score
        return parent_max

    def _build_hydrated_parent_document(
        self,
        *,
        parent_id: str,
        parent_doc: Document,
        parent_max_score: float | None,
    ) -> Document:
        hydrated = self._clone_document(parent_doc)
        hydrated.metadata[self.id_key] = parent_id
        hydrated.metadata["hydrated_from_children"] = True
        if parent_max_score is not None:
            hydrated.metadata["similarity_score"] = parent_max_score
            hydrated.metadata["max_similarity_score"] = parent_max_score
        self._normalize_document_id(hydrated)
        if hydrated.id is None:
            hydrated.id = parent_id
        return hydrated

    def _build_enriched_child_document(
        self,
        *,
        child: Document,
        parent_id: str,
        parent_doc: Document,
    ) -> Document:
        enriched = self._clone_document(child)
        child_original = enriched.page_content
        parent_content = parent_doc.page_content
        enriched.page_content = (
            f"{parent_content}{self.enrichment_separator}{child_original}"
        )
        enriched.metadata[self.id_key] = parent_id
        enriched.metadata["parent_hydrated"] = True
        enriched.metadata["child_page_content_original"] = child_original
        enriched.metadata["parent_page_content"] = parent_content
        return enriched

    def _compose_by_mode(
        self,
        *,
        children: list[Document],
        ordered_parent_ids: list[str],
        parent_map: dict[str, Document],
    ) -> list[Document]:
        mode = self.hydration_mode
        parent_max = self._build_parent_max_score_map(children)

        if mode == HydrationMode.parents_replace:
            results: list[Document] = []
            emitted: set[str] = set()

            for child in children:
                pid = child.metadata.get(self.id_key)
                if pid is None:
                    results.append(self._clone_document(child))
                    continue
                parent_doc = parent_map.get(pid)
                if parent_doc is None:
                    results.append(self._clone_document(child))
                    continue
                if pid in emitted:
                    continue
                emitted.add(pid)
                results.append(
                    self._build_hydrated_parent_document(
                        parent_id=pid,
                        parent_doc=parent_doc,
                        parent_max_score=parent_max.get(pid),
                    )
                )
            return self._finalize_documents(results)

        if mode == HydrationMode.children_enriched:
            results: list[Document] = []
            for child in children:
                pid = child.metadata.get(self.id_key)
                if pid is None or pid not in parent_map:
                    child_copy = self._clone_document(child)
                    child_copy.metadata["parent_hydrated"] = False
                    results.append(child_copy)
                    continue
                results.append(
                    self._build_enriched_child_document(
                        child=child,
                        parent_id=pid,
                        parent_doc=parent_map[pid],
                    )
                )
            return self._finalize_documents(results)

        # children_plus_parents
        results = [self._clone_document(child) for child in children]
        for pid in ordered_parent_ids:
            parent_doc = parent_map.get(pid)
            if parent_doc is None:
                continue
            results.append(
                self._build_hydrated_parent_document(
                    parent_id=pid,
                    parent_doc=parent_doc,
                    parent_max_score=parent_max.get(pid),
                )
            )
        return self._finalize_documents(results)

    @override
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """Asynchronously get documents relevant to a query.

        Args:
            query: String to find relevant documents for
            run_manager: The callbacks handler to use
        Returns:
            List of relevant documents.
        """
        children = await self._asearch_children(query)
        ordered_parent_ids, parent_map = await self._aload_parents_for_retrieved_children(
            children
        )
        return self._compose_by_mode(
            children=children,
            ordered_parent_ids=ordered_parent_ids,
            parent_map=parent_map,
        )
