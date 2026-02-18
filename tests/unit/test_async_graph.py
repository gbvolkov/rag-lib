import pytest
import pytest_asyncio
from langchain_core.vectorstores import VectorStore
from rag_lib.graph.store import NetworkXGraphStore
from rag_lib.graph.domain import GraphNode, GraphEdge
from rag_lib.retrieval.graph_retriever import GraphQueryConfig, GraphRetriever
from langchain_core.documents import Document


class AsyncStrictVectorStore(VectorStore):
    def __init__(self, docs=None):
        self.docs = docs or []

    @classmethod
    def from_texts(cls, texts, embedding, metadatas=None, **kwargs):
        metadatas = metadatas or [{} for _ in texts]
        ids = kwargs.get("ids", [None for _ in texts])
        docs = []
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
        return [doc for doc, _ in self.similarity_search_with_relevance_scores(query, k=k, **kwargs)]

    def similarity_search_with_relevance_scores(self, query, k=4, filter=None, **kwargs):
        docs = self.docs
        if filter:
            docs = [
                doc for doc in docs
                if all((doc.metadata or {}).get(key) == value for key, value in filter.items())
            ]
        return [(doc, 1.0) for doc in docs[:k]]

    async def asimilarity_search_with_relevance_scores(self, query, k=4, filter=None, **kwargs):
        return self.similarity_search_with_relevance_scores(query, k=k, filter=filter, **kwargs)

    def get_by_ids(self, ids):
        wanted = set(ids)
        return [doc for doc in self.docs if (doc.metadata or {}).get("segment_id") in wanted]

    async def aget_by_ids(self, ids):
        return self.get_by_ids(ids)

    @property
    def embeddings(self):
        return None

@pytest.mark.asyncio
async def test_async_networkx_store_operations():
    store = NetworkXGraphStore()
    
    # Test Async Add
    node1 = GraphNode(id="n1", label="Node1", type="test")
    node2 = GraphNode(id="n2", label="Node2", type="test")
    await store.aadd_node(node1)
    await store.aadd_node(node2)
    
    edge = GraphEdge(source_id="n1", target_id="n2", relation_type="LINK")
    await store.aadd_edge(edge)
    
    # Test Async Get
    retrieved_node = await store.aget_node("n1")
    assert retrieved_node is not None
    assert retrieved_node.id == "n1"
    
    neighbors = await store.aget_neighbors("n1")
    assert len(neighbors) == 1
    assert neighbors[0].id == "n2"
    
@pytest.mark.asyncio
async def test_async_graph_retriever_local():
    store = NetworkXGraphStore()
    node = GraphNode(id="apple", label="Apple", type="Fruit", description="A red fruit", source_segment_id="seg_apple")
    await store.aadd_node(node)
    vector_store = AsyncStrictVectorStore(
        docs=[Document(page_content="Apple is a red fruit", metadata={"segment_id": "seg_apple"})]
    )
    
    retriever = GraphRetriever(
        vector_store=vector_store,
        graph_store=store,
        config=GraphQueryConfig(mode="local", max_hops=1, enable_keyword_extraction=False),
    )
    
    # Test ainvoke which triggers _aget_relevant_documents
    results = await retriever.ainvoke("apple")
    
    assert len(results) >= 1
    assert "Apple" in results[0].page_content
    assert results[0].metadata["retrieval_kind"] in {"entity", "relation", "chunk"}

@pytest.mark.asyncio
async def test_async_graph_retriever_global():
    store = NetworkXGraphStore()
    vector_store = AsyncStrictVectorStore(
        docs=[
            Document(
                page_content="Async Summary",
                metadata={"is_community_summary": True, "segment_id": "seg_summary", "community_id": "c1"},
            ),
            Document(page_content="Hydrated chunk for summary", metadata={"segment_id": "seg_summary"}),
        ]
    )
    
    retriever = GraphRetriever(
        vector_store=vector_store,
        graph_store=store,
        config=GraphQueryConfig(mode="global", enable_keyword_extraction=False),
    )
    
    results = await retriever.ainvoke("global query")
    
    assert len(results) >= 1
    assert "Async Summary" in results[0].page_content

@pytest.mark.asyncio
async def test_async_entity_extractor():
    from rag_lib.processors.entity_extractor import EntityExtractor
    from rag_lib.core.domain import Segment, SegmentType
    from langchain_community.chat_models import FakeListChatModel
    
    store = NetworkXGraphStore()
    
    # Mock LLM response in JSON format
    json_resp = """
    {
        "entities": [{"name": "AsyncEntity", "type": "Test"}],
        "relationships": []
    }
    """
    llm = FakeListChatModel(responses=[json_resp, json_resp])
    extractor = EntityExtractor(llm=llm, store=store)
    
    segments = [
        Segment(content="Seg1", type=SegmentType.TEXT),
        Segment(content="Seg2", type=SegmentType.TEXT)
    ]
    
    await extractor.aprocess_segments(segments, concurrency=2)
    
    # Check graph
    node = await store.aget_node("AsyncEntity")
    assert node is not None
    assert node.id == "AsyncEntity"
