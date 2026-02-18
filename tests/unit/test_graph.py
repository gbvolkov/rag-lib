from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from rag_lib.graph.domain import GraphNode, GraphEdge
from rag_lib.graph.store import NetworkXGraphStore
from rag_lib.retrieval.graph_retriever import GraphQueryConfig, GraphRetriever


class DummyStrictVectorStore(VectorStore):
    def __init__(self, docs: list[Document] | None = None):
        self.docs = docs or []

    @classmethod
    def from_texts(cls, texts, embedding, metadatas=None, **kwargs):
        metadatas = metadatas or [{} for _ in texts]
        ids = kwargs.get("ids", [None for _ in texts])
        docs: list[Document] = []
        for idx, text in enumerate(texts):
            metadata = dict(metadatas[idx] if idx < len(metadatas) else {})
            if idx < len(ids) and ids[idx] is not None:
                metadata["segment_id"] = ids[idx]
            docs.append(Document(page_content=text, metadata=metadata))
        return cls(docs=docs)

    def add_texts(self, texts, metadatas=None, ids=None, **kwargs):
        metadatas = metadatas or [{} for _ in texts]
        ids = ids or [None for _ in texts]
        for idx, text in enumerate(texts):
            metadata = dict(metadatas[idx] if idx < len(metadatas) else {})
            if idx < len(ids) and ids[idx] is not None:
                metadata["segment_id"] = ids[idx]
            self.docs.append(Document(page_content=text, metadata=metadata))
        return ids

    def similarity_search(self, query, k=4, **kwargs):
        rows = self.similarity_search_with_relevance_scores(query, k=k, **kwargs)
        return [doc for doc, _ in rows]

    def similarity_search_with_relevance_scores(self, query, k=4, **kwargs):
        return [(doc, 1.0) for doc in self.docs[:k]]

    def get_by_ids(self, ids):
        wanted = set(ids)
        return [doc for doc in self.docs if (doc.metadata or {}).get("segment_id") in wanted]

    async def asimilarity_search_with_relevance_scores(self, query, k=4, **kwargs):
        return self.similarity_search_with_relevance_scores(query, k=k, **kwargs)

    async def aget_by_ids(self, ids):
        return self.get_by_ids(ids)

    @property
    def embeddings(self):
        return None

def test_graph_store_operations():
    store = NetworkXGraphStore()
    
    node1 = GraphNode(id="A", type="concept", label="Alpha")
    node2 = GraphNode(id="B", type="concept", label="Beta")
    store.add_node(node1)
    store.add_node(node2)

    edge = GraphEdge(source_id="A", target_id="B", relation_type="LINKS_TO")
    store.add_edge(edge)
    
    retrieved = store.get_node("A")
    assert retrieved.label == "Alpha"
    
    neighbors = store.get_neighbors("A")
    assert len(neighbors) == 1
    assert neighbors[0].id == "B"

def test_graph_retriever():
    store = NetworkXGraphStore()
    store.add_node(GraphNode(id="Einstein", type="Person", label="Albert Einstein", description="Physicist", source_segment_id="seg1"))
    store.add_node(GraphNode(id="Relativity", type="Concept", label="Theory of Relativity", description="Theory", source_segment_id="seg1"))
    store.add_edge(GraphEdge(source_id="Einstein", target_id="Relativity", relation_type="DEVELOPED", source_segment_id="seg1"))
    vector_store = DummyStrictVectorStore(
        docs=[
            Document(page_content="Albert Einstein developed relativity.", metadata={"segment_id": "seg1"})
        ]
    )

    retriever = GraphRetriever(
        vector_store=vector_store,
        graph_store=store,
        config=GraphQueryConfig(mode="local", max_hops=1, enable_keyword_extraction=False),
    )

    results = retriever.invoke("Einstein")

    assert len(results) >= 1
    assert any(d.metadata.get("retrieval_kind") == "chunk" for d in results)
    assert any(d.metadata.get("source_segment_id") == "seg1" for d in results)
    assert all("graph_mode" in d.metadata for d in results)

def test_retriever_factories():
    from rag_lib.retrieval.retrievers import get_graph_retriever
    from rag_lib.retrieval.composition import create_graph_hybrid_retriever
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document
    from typing import List

    class DummyRetriever(BaseRetriever):
        def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
            return []

    store = NetworkXGraphStore()
    graph_retriever = get_graph_retriever(
        vector_store=DummyStrictVectorStore(),
        graph_store=store,
        config=GraphQueryConfig(mode="local", enable_keyword_extraction=False),
    )
    
    # Use real instance instead of MagicMock to satisfy Pydantic validation
    vector_retriever = DummyRetriever() 
    
    hybrid = create_graph_hybrid_retriever(vector_retriever, graph_retriever)
    assert hybrid is not None
    assert len(hybrid.retrievers) == 2

