import pytest
from langchain_community.chat_models import FakeListChatModel
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from rag_lib.graph.store import NetworkXGraphStore
from rag_lib.graph.domain import GraphNode, GraphEdge
from rag_lib.graph.community import CommunityDetector
from rag_lib.processors.community_summarizer import CommunitySummarizer
from rag_lib.retrieval.graph_retriever import GraphQueryConfig, GraphRetriever


class StrictDummyVectorStore(VectorStore):
    def __init__(self):
        self.docs: list[Document] = []

    @classmethod
    def from_texts(cls, texts, embedding, metadatas=None, **kwargs):
        store = cls()
        store.add_texts(texts=texts, metadatas=metadatas, ids=kwargs.get("ids"))
        return store

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
                doc for doc in docs
                if all((doc.metadata or {}).get(key) == value for key, value in filter.items())
            ]
        return [(doc, 1.0) for doc in docs[:k]]

    def get_by_ids(self, ids):
        wanted = set(ids)
        return [doc for doc in self.docs if (doc.metadata or {}).get("segment_id") in wanted]

    async def asimilarity_search_with_relevance_scores(self, query, k=4, filter=None, **kwargs):
        return self.similarity_search_with_relevance_scores(query, k=k, filter=filter, **kwargs)

    async def aget_by_ids(self, ids):
        return self.get_by_ids(ids)

    @property
    def embeddings(self):
        return None

def test_community_workflow():
    # 1. Setup Graph with 2 communities
    store = NetworkXGraphStore()
    
    # Community A: Nodes A1, A2
    store.add_node(GraphNode(id="A1", type="Concept", label="AI"))
    store.add_node(GraphNode(id="A2", type="Concept", label="Machine Learning"))
    store.add_edge(GraphEdge(source_id="A1", target_id="A2", relation_type="SUBSET"))
    
    # Community B: Nodes B1, B2
    store.add_node(GraphNode(id="B1", type="Concept", label="Biology"))
    store.add_node(GraphNode(id="B2", type="Concept", label="Genetics"))
    store.add_edge(GraphEdge(source_id="B1", target_id="B2", relation_type="SUBSET"))
    
    # 2. Detect Communities
    communities = CommunityDetector.detect(store)
    assert len(communities) == 2
    
    # 3. Summarize Communities
    # Mock LLM
    llm = FakeListChatModel(responses=[
        "Summary of AI and ML community.", 
        "Summary of Biology and Genetics community."
    ])
    
    summarizer = CommunitySummarizer(llm=llm, store=store)
    summary_segments = summarizer.summarize(communities)
    
    assert len(summary_segments) == 2
    assert summary_segments[0].metadata["is_community_summary"] is True
    assert "community_id" in summary_segments[0].metadata
    
    # 4. Mock Indexing (Store in a strict dummy VectorStore)
    vector_store = StrictDummyVectorStore()
    # Manually index (simulating Indexer)
    for seg in summary_segments:
        vector_store.add_texts([seg.content], [seg.metadata], [seg.segment_id])
        
    # 5. Global Retrieval
    retriever = GraphRetriever(
        vector_store=vector_store,
        graph_store=store,
        config=GraphQueryConfig(mode="global", enable_keyword_extraction=False),
    )
    
    # Query logic in Global Search depends on Vector Similarity.
    # Since we use DummyVectorStore that returns everything matching filter,
    # we should get both summaries back (or up to k).
    
    results = retriever.invoke("Tell me about science")
    
    assert len(results) >= 1
    assert any("Summary of" in doc.page_content for doc in results)
    assert any(
        doc.metadata["retrieval_kind"] in {"community", "chunk", "entity", "relation"}
        for doc in results
    )
