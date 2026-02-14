import pytest
from unittest.mock import MagicMock
from rag_lib.graph.domain import GraphNode, GraphEdge
from rag_lib.graph.store import NetworkXGraphStore
from rag_lib.core.domain import Segment

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

    from rag_lib.retrieval.graph_retriever import GraphRetriever
    retriever = GraphRetriever(store=store)
    
    # Mocking search_nodes since it uses simple string matching in our implementation
    results = retriever.invoke("Einstein")
    
    # Expect: 
    # 1. Doc for Einstein (Node)
    # 2. Doc for Relativity (Neighbor)
    assert len(results) >= 2
    node_ids = [d.metadata["node_id"] for d in results]
    assert "Einstein" in node_ids
    assert "Relativity" in node_ids

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
    graph_retriever = get_graph_retriever(store)
    
    # Use real instance instead of MagicMock to satisfy Pydantic validation
    vector_retriever = DummyRetriever() 
    
    hybrid = create_graph_hybrid_retriever(vector_retriever, graph_retriever)
    assert hybrid is not None
    assert len(hybrid.retrievers) == 2

