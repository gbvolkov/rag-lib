from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from rag_lib.graph.domain import GraphEdge, GraphNode
from rag_lib.graph.store import NetworkXGraphStore
from rag_lib.retrieval.graph_retriever import GraphQueryConfig, GraphRetriever


class StrictLexicalVectorStore(VectorStore):
    def __init__(self, docs: list[Document] | None = None):
        self.docs = docs or []

    @classmethod
    def from_texts(cls, texts, embedding, metadatas=None, **kwargs):
        metadatas = metadatas or [{} for _ in texts]
        ids = kwargs.get("ids", [None for _ in texts])
        docs = []
        for idx, text in enumerate(texts):
            metadata = dict(metadatas[idx] if idx < len(metadatas) else {})
            if idx < len(ids) and ids[idx] is not None:
                metadata.setdefault("segment_id", ids[idx])
            docs.append(Document(page_content=text, metadata=metadata))
        return cls(docs=docs)

    def add_texts(self, texts, metadatas=None, ids=None, **kwargs):
        metadatas = metadatas or [{} for _ in texts]
        ids = ids or [None for _ in texts]
        for idx, text in enumerate(texts):
            metadata = dict(metadatas[idx] if idx < len(metadatas) else {})
            if idx < len(ids) and ids[idx] is not None:
                metadata.setdefault("segment_id", ids[idx])
            self.docs.append(Document(page_content=text, metadata=metadata))
        return ids

    def similarity_search(self, query, k=4, **kwargs):
        rows = self.similarity_search_with_relevance_scores(query, k=k, **kwargs)
        return [doc for doc, _ in rows]

    def similarity_search_with_relevance_scores(self, query, k=4, filter=None, **kwargs):
        docs = self.docs
        if filter:
            docs = [
                doc
                for doc in docs
                if all((doc.metadata or {}).get(key) == value for key, value in filter.items())
            ]

        scored = [(doc, self._score(query, doc.page_content)) for doc in docs]
        scored.sort(
            key=lambda row: (
                -row[1],
                (row[0].metadata or {}).get("segment_id", ""),
                row[0].page_content,
            )
        )
        return scored[:k]

    async def asimilarity_search_with_relevance_scores(self, query, k=4, filter=None, **kwargs):
        return self.similarity_search_with_relevance_scores(query, k=k, filter=filter, **kwargs)

    def get_by_ids(self, ids):
        wanted = set(ids)
        return [doc for doc in self.docs if (doc.metadata or {}).get("segment_id") in wanted]

    async def aget_by_ids(self, ids):
        return self.get_by_ids(ids)

    def _score(self, query: str, text: str) -> float:
        q = set(query.lower().split())
        t = set(text.lower().split())
        if not q:
            return 0.0
        return len(q.intersection(t)) / len(q)

    @property
    def embeddings(self):
        return None


def _build_graph_store() -> NetworkXGraphStore:
    store = NetworkXGraphStore()
    store.add_node(
        GraphNode(
            id="probability",
            type="Concept",
            label="Probability Theory",
            description="Mathematics of random events",
            source_segment_id="seg_prob",
        )
    )
    store.add_node(
        GraphNode(
            id="bayes",
            type="Concept",
            label="Bayes Theorem",
            description="Posterior update rule",
            source_segment_id="seg_bayes",
        )
    )
    store.add_node(
        GraphNode(
            id="markov",
            type="Concept",
            label="Markov Chain",
            description="Memoryless random process",
            source_segment_id="seg_markov",
        )
    )
    store.add_edge(
        GraphEdge(
            source_id="probability",
            target_id="bayes",
            relation_type="INCLUDES",
            weight=1.0,
            source_segment_id="seg_rel_1",
        )
    )
    store.add_edge(
        GraphEdge(
            source_id="bayes",
            target_id="markov",
            relation_type="RELATED_TO",
            weight=0.8,
            source_segment_id="seg_rel_2",
        )
    )
    return store


def _build_vector_store() -> StrictLexicalVectorStore:
    docs = [
        Document(
            page_content="Probability theory studies random events and distributions.",
            metadata={"segment_id": "seg_prob"},
        ),
        Document(
            page_content="Bayes theorem updates probability of hypotheses.",
            metadata={"segment_id": "seg_bayes"},
        ),
        Document(
            page_content="Community summary: probability, bayesian reasoning, and statistical inference.",
            metadata={"segment_id": "seg_comm", "is_community_summary": True, "community_id": "c1"},
        ),
        Document(
            page_content="Duplicate chunk for source dedup verification.",
            metadata={"segment_id": "seg_prob"},
        ),
    ]
    return StrictLexicalVectorStore(docs=docs)


def test_local_retrieval_returns_graph_and_hydrated_chunks():
    store = _build_graph_store()
    vector_store = _build_vector_store()
    retriever = GraphRetriever(
        vector_store=vector_store,
        graph_store=store,
        config=GraphQueryConfig(mode="local", max_hops=2, top_k_chunks=4, enable_keyword_extraction=False),
    )

    results = retriever.retrieve("probability")

    assert results
    assert any(doc.metadata.get("retrieval_kind") == "entity" for doc in results)
    assert any(doc.metadata.get("retrieval_kind") == "relation" for doc in results)
    assert any(doc.metadata.get("retrieval_kind") == "chunk" for doc in results)
    assert all(doc.metadata.get("graph_mode") == "local" for doc in results)


def test_global_retrieval_includes_community_signal():
    store = _build_graph_store()
    vector_store = _build_vector_store()
    retriever = GraphRetriever(
        vector_store=vector_store,
        graph_store=store,
        config=GraphQueryConfig(mode="global", top_k_relations=8, enable_keyword_extraction=False),
    )

    results = retriever.retrieve("probability")

    assert results
    assert any(doc.metadata.get("retrieval_kind") in {"community", "relation"} for doc in results)


def test_hybrid_and_mix_modes_return_ranked_results():
    store = _build_graph_store()
    vector_store = _build_vector_store()

    hybrid = GraphRetriever(
        vector_store=vector_store,
        graph_store=store,
        config=GraphQueryConfig(mode="hybrid", enable_keyword_extraction=False),
    )
    mix = GraphRetriever(
        vector_store=vector_store,
        graph_store=store,
        config=GraphQueryConfig(mode="mix", enable_keyword_extraction=False),
    )

    hybrid_results = hybrid.retrieve("probability theory")
    mix_results = mix.retrieve("probability theory")

    assert hybrid_results
    assert mix_results
    assert all("score" in doc.metadata for doc in hybrid_results)
    assert all("score" in doc.metadata for doc in mix_results)
    assert any(doc.metadata.get("retrieval_kind") == "chunk" for doc in mix_results)


def test_max_hops_controls_reachability():
    store = NetworkXGraphStore()
    store.add_node(GraphNode(id="A", label="Alpha", type="Concept", source_segment_id="seg_a"))
    store.add_node(GraphNode(id="B", label="Beta", type="Concept", source_segment_id="seg_b"))
    store.add_node(GraphNode(id="C", label="Gamma", type="Concept", source_segment_id="seg_c"))
    store.add_edge(GraphEdge(source_id="A", target_id="B", relation_type="REL"))
    store.add_edge(GraphEdge(source_id="B", target_id="C", relation_type="REL"))

    vector_store = StrictLexicalVectorStore(
        docs=[
            Document(page_content="Alpha source", metadata={"segment_id": "seg_a"}),
            Document(page_content="Beta source", metadata={"segment_id": "seg_b"}),
            Document(page_content="Gamma source", metadata={"segment_id": "seg_c"}),
        ]
    )

    hop1 = GraphRetriever(
        vector_store=vector_store,
        graph_store=store,
        config=GraphQueryConfig(mode="local", max_hops=1, enable_keyword_extraction=False),
    )
    hop2 = GraphRetriever(
        vector_store=vector_store,
        graph_store=store,
        config=GraphQueryConfig(mode="local", max_hops=2, enable_keyword_extraction=False),
    )

    hop1_ids = {doc.metadata.get("entity_id") for doc in hop1.retrieve("Alpha")}
    hop2_ids = {doc.metadata.get("entity_id") for doc in hop2.retrieve("Alpha")}

    assert "C" not in hop1_ids
    assert "C" in hop2_ids


def test_chunk_dedup_by_source_segment_id():
    store = _build_graph_store()
    vector_store = _build_vector_store()
    retriever = GraphRetriever(
        vector_store=vector_store,
        graph_store=store,
        config=GraphQueryConfig(mode="local", top_k_chunks=6, enable_keyword_extraction=False),
    )

    results = retriever.retrieve("probability")
    chunk_ids = [
        doc.metadata.get("source_segment_id")
        for doc in results
        if doc.metadata.get("retrieval_kind") == "chunk"
    ]
    filtered = [cid for cid in chunk_ids if cid]
    assert len(filtered) == len(set(filtered))


def test_content_hash_dedup_across_different_ids():
    store = NetworkXGraphStore()
    vector_store = StrictLexicalVectorStore(
        docs=[
            Document(page_content="same normalized text", metadata={}),
            Document(page_content="same   normalized   text", metadata={}),
        ]
    )
    retriever = GraphRetriever(
        vector_store=vector_store,
        graph_store=store,
        config=GraphQueryConfig(mode="mix", top_k_chunks=10, min_score=0.0, enable_keyword_extraction=False),
    )

    results = retriever.retrieve("same")
    chunk_results = [doc for doc in results if doc.metadata.get("retrieval_kind") == "chunk"]
    assert len(chunk_results) == 1


def test_stable_tie_ordering_is_deterministic():
    store = NetworkXGraphStore()
    vector_store = StrictLexicalVectorStore(
        docs=[
            Document(page_content="topic alpha", metadata={"segment_id": "seg_a"}),
            Document(page_content="topic beta", metadata={"segment_id": "seg_b"}),
        ]
    )
    retriever = GraphRetriever(
        vector_store=vector_store,
        graph_store=store,
        config=GraphQueryConfig(mode="mix", top_k_chunks=4, min_score=0.0, enable_keyword_extraction=False),
    )

    first = [
        (doc.metadata.get("retrieval_kind"), doc.metadata.get("source_segment_id"), doc.page_content)
        for doc in retriever.retrieve("topic")
    ]
    second = [
        (doc.metadata.get("retrieval_kind"), doc.metadata.get("source_segment_id"), doc.page_content)
        for doc in retriever.retrieve("topic")
    ]

    assert first == second


@pytest.mark.asyncio
async def test_async_parity_with_sync():
    store = _build_graph_store()
    vector_store = _build_vector_store()
    retriever = GraphRetriever(
        vector_store=vector_store,
        graph_store=store,
        config=GraphQueryConfig(mode="hybrid", enable_keyword_extraction=False),
    )

    sync_results = retriever.retrieve("probability")
    async_results = await retriever.aretrieve("probability")

    sync_view = [
        (
            doc.metadata.get("retrieval_kind"),
            doc.metadata.get("entity_id") or doc.metadata.get("edge_id") or doc.metadata.get("community_id") or doc.metadata.get("source_segment_id"),
            float(doc.metadata.get("score", 0.0)),
        )
        for doc in sync_results
    ]
    async_view = [
        (
            doc.metadata.get("retrieval_kind"),
            doc.metadata.get("entity_id") or doc.metadata.get("edge_id") or doc.metadata.get("community_id") or doc.metadata.get("source_segment_id"),
            float(doc.metadata.get("score", 0.0)),
        )
        for doc in async_results
    ]

    assert sync_view == async_view
