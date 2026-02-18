from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from rag_lib.graph.domain import GraphEdge, GraphNode
from rag_lib.graph.store import NetworkXGraphStore
from rag_lib.retrieval.graph_retriever import GraphQueryConfig, GraphRetriever


class InMemoryLexicalVectorStore(VectorStore):
    def __init__(self, docs: list[Document]):
        self.docs = docs

    @classmethod
    def from_texts(cls, texts, embedding, metadatas=None, **kwargs):
        docs = []
        metadatas = metadatas or [{} for _ in texts]
        for idx, text in enumerate(texts):
            docs.append(Document(page_content=text, metadata=dict(metadatas[idx])))
        return cls(docs)

    def add_texts(self, texts, metadatas=None, ids=None, **kwargs):
        metadatas = metadatas or [{} for _ in texts]
        ids = ids or [None for _ in texts]
        for idx, text in enumerate(texts):
            metadata = dict(metadatas[idx] if idx < len(metadatas) else {})
            if idx < len(ids) and ids[idx] is not None:
                metadata.setdefault("segment_id", ids[idx])
            self.docs.append(Document(page_content=text, metadata=metadata))
        return ids

    def similarity_search(self, query, k=4, filter=None, **kwargs):
        rows = self.similarity_search_with_relevance_scores(query, k=k, filter=filter, **kwargs)
        return [doc for doc, _ in rows]

    def similarity_search_with_relevance_scores(self, query, k=4, filter=None, **kwargs):
        query_terms = set(query.lower().split())
        rows = []
        for doc in self.docs:
            metadata = doc.metadata or {}
            if filter and not all(metadata.get(key) == value for key, value in filter.items()):
                continue
            text_terms = set(doc.page_content.lower().split())
            score = len(query_terms.intersection(text_terms)) / max(len(query_terms), 1)
            rows.append((doc, score))
        rows.sort(key=lambda x: x[1], reverse=True)
        return rows[:k]

    async def asimilarity_search_with_relevance_scores(self, query, k=4, filter=None, **kwargs):
        return self.similarity_search_with_relevance_scores(query, k=k, filter=filter, **kwargs)

    def get_by_ids(self, ids):
        wanted = {str(value) for value in ids}
        return [doc for doc in self.docs if str((doc.metadata or {}).get("segment_id", "")) in wanted]

    async def aget_by_ids(self, ids):
        return self.get_by_ids(ids)

    @property
    def embeddings(self):
        return None


def _build_fixture() -> tuple[NetworkXGraphStore, InMemoryLexicalVectorStore]:
    graph_store = NetworkXGraphStore()
    graph_store.add_node(
        GraphNode(
            id="probability",
            label="Теория вероятности",
            type="Concept",
            description="Основы вероятностного моделирования",
            source_segment_id="seg_prob",
        )
    )
    graph_store.add_node(
        GraphNode(
            id="random_variable",
            label="Случайная величина",
            type="Concept",
            description="Функция, сопоставляющая исходам числа",
            source_segment_id="seg_rv",
        )
    )
    graph_store.add_edge(
        GraphEdge(
            source_id="probability",
            target_id="random_variable",
            relation_type="DEFINES",
            source_segment_id="seg_rel",
        )
    )

    vector_store = InMemoryLexicalVectorStore(
        docs=[
            Document(
                page_content="Теория вероятности рассматривает случайные события.",
                metadata={"segment_id": "seg_prob"},
            ),
            Document(
                page_content="Случайная величина описывает численные исходы эксперимента.",
                metadata={"segment_id": "seg_rv"},
            ),
            Document(
                page_content="Сводка сообщества: вероятность, случайные величины и статистика.",
                metadata={
                    "segment_id": "seg_comm",
                    "is_community_summary": True,
                    "community_id": "community_probability",
                },
            ),
        ]
    )
    return graph_store, vector_store


def test_graph_retrieval_e2e_cyrillic_queries():
    graph_store, vector_store = _build_fixture()
    retriever = GraphRetriever(
        vector_store=vector_store,
        graph_store=graph_store,
        config=GraphQueryConfig(mode="hybrid", max_hops=2, top_k_chunks=4, enable_keyword_extraction=False),
    )

    for query in ("вероятность", "Теория вероятности"):
        results = retriever.retrieve(query)
        assert results
        assert any(doc.metadata.get("retrieval_kind") in {"chunk", "entity", "relation", "community"} for doc in results)
        assert all("score" in doc.metadata for doc in results)
