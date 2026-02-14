import pytest
import pytest_asyncio
from rag_lib.graph.store import NetworkXGraphStore
from rag_lib.graph.domain import GraphNode, GraphEdge
from rag_lib.retrieval.graph_retriever import GraphRetriever
from langchain_core.documents import Document

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
    node = GraphNode(id="apple", label="Apple", type="Fruit", description="A red fruit")
    await store.aadd_node(node)
    
    retriever = GraphRetriever(store=store, mode="local")
    
    # Test ainvoke which triggers _aget_relevant_documents
    results = await retriever.ainvoke("apple")
    
    assert len(results) >= 1
    assert "Apple" in results[0].page_content
    assert results[0].metadata["source"] == "graph"

@pytest.mark.asyncio
async def test_async_graph_retriever_global():
    store = NetworkXGraphStore()
    
    # Mock Vector Store with async support
    from langchain_core.vectorstores import VectorStore
    class AsyncDummyVectorStore(VectorStore):
        def add_texts(self, texts, metadatas, **kwargs):
            pass
        def similarity_search(self, query, k=4, **kwargs):
            pass
        @classmethod
        def from_texts(cls, texts, embedding, metadatas=None, **kwargs):
            pass
            
        async def asimilarity_search(self, query, k=4, **kwargs):
            return [Document(page_content="Async Summary", metadata={"is_community_summary": True})]

    vector_store = AsyncDummyVectorStore()
    
    retriever = GraphRetriever(store=store, vector_store=vector_store, mode="global")
    
    results = await retriever.ainvoke("global query")
    
    assert len(results) == 1
    assert results[0].page_content == "Async Summary"

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
