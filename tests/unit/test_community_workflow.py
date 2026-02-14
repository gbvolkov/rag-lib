import pytest
import json
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage
from langchain_community.chat_models import FakeListChatModel
from rag_lib.graph.store import NetworkXGraphStore
from rag_lib.graph.domain import GraphNode, GraphEdge
from rag_lib.graph.community import CommunityDetector
from rag_lib.processors.community_summarizer import CommunitySummarizer
from rag_lib.retrieval.graph_retriever import GraphRetriever
from rag_lib.core.domain import Segment, SegmentType

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
    
    # 4. Mock Indexing (Store in a dummy VectorStore)
    from langchain_core.vectorstores import VectorStore
    class DummyVectorStore(VectorStore):
        def __init__(self):
            self.docs = []
            
        def similarity_search(self, query, k=4, filter=None, **kwargs):
            # Simulation: return all docs that match filter
            results = []
            for doc in self.docs:
                if filter:
                    match = True
                    for key, val in filter.items():
                        if doc.metadata.get(key) != val:
                            match = False
                            break
                    if match:
                        results.append(doc)
                else:
                    results.append(doc)
            return results
        
        def add_texts(self, texts, metadatas, ids):
            from langchain_core.documents import Document
            for t, m in zip(texts, metadatas):
                self.docs.append(Document(page_content=t, metadata=m))
        
        @classmethod
        def from_texts(cls, texts, embedding, metadatas=None, **kwargs):
            pass

    vector_store = DummyVectorStore()
    # Manually index (simulating Indexer)
    for seg in summary_segments:
        vector_store.add_texts([seg.content], [seg.metadata], [seg.segment_id])
        
    # 5. Global Retrieval
    retriever = GraphRetriever(store=store, vector_store=vector_store, mode="global")
    
    # Query logic in Global Search depends on Vector Similarity.
    # Since we use DummyVectorStore that returns everything matching filter,
    # we should get both summaries back (or up to k).
    
    results = retriever.invoke("Tell me about science")
    
    assert len(results) == 2
    assert "Summary of" in results[0].page_content
    assert results[0].metadata["is_community_summary"] is True
