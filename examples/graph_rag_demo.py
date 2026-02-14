import os
import sys
import asyncio
import json

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from rag_lib.core.domain import Segment, SegmentType
from rag_lib.graph.domain import GraphNode, GraphEdge
from rag_lib.graph.store import NetworkXGraphStore
from rag_lib.retrieval.graph_retriever import GraphRetriever
from rag_lib.core.indexer import Indexer
from rag_lib.processors.entity_extractor import EntityExtractor
from langchain_community.chat_models import FakeListChatModel
from rag_lib.core.logger import logger

def run_demo():
    print("--- Starting Graph RAG Verification ---")
    
    # 1. Setup Data
    segment = Segment(content="Albert Einstein developed Relativity.", segment_id="seg1", type=SegmentType.TEXT)
    
    # 2. Setup Components
    # Mock LLM response for extraction
    mock_response = json.dumps({
        "entities": [
            {"name": "Albert Einstein", "type": "Person", "description": "Physicist"},
            {"name": "Relativity", "type": "Concept", "description": "Theory"}
        ],
        "relationships": [
            {"source": "Albert Einstein", "target": "Relativity", "relation": "DEVELOPED", "weight": 1.0}
        ]
    })
    llm = FakeListChatModel(responses=[mock_response])
    
    store = NetworkXGraphStore()
    extractor = EntityExtractor(llm=llm, store=store)
    
    # Dummy VectorStore
    class DummyVectorStore:
        def add_texts(self, texts, metadatas, ids):
            print(f"VectorStore: Added {len(texts)} texts")
        def aadd_texts(self, *args, **kwargs): pass

    vector_store = DummyVectorStore()
    
    # Initialize Indexer with EntityExtractor
    indexer = Indexer(vector_store=vector_store, embeddings=None, entity_extractor=extractor)
    
    # 3. Run Integrated Indexing
    print("--- Running Indexer (Triggering Graph Extraction) ---")
    indexer.index([segment])
    
    # 4. Verify Graph Content
    print(f"Graph initialized with nodes: {', '.join([n.label for n in store.search_nodes('')])}")
    
    # 5. Verify Retrieval
    print("\n--- Testing Retrieval for 'Einstein' ---")
    retriever = GraphRetriever(store=store)
    results = retriever.invoke("Einstein")
    
    output = []
    for doc in results:
        output.append(doc.metadata['node_id'])
        
    print(f"Retrieved Node IDs: {output}")
    
    if "Albert Einstein" in output and "Relativity" in output:
        print("SUCCESS: Retrieved both the primary node and its neighbor.")
    else:
        print("FAILURE: Did not retrieve expected nodes.")

if __name__ == "__main__":
    run_demo()
