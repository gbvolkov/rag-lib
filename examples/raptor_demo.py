import sys
import os
import asyncio
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from rag_lib.core.domain import Segment, SegmentType
from rag_lib.processors.raptor import RaptorProcessor
from langchain_community.chat_models import FakeListChatModel

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

def run_demo():
    print("--- Starting RAPTOR Demo (Real LLM) ---")
    
    # Load env for API Key
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found in .env")
        return

    # 1. Setup Data
    # Create 10 leaf segments
    # Create 20 leaf segments with distinct topics to encourage clustering
    topics = ["Quantum Physics", "Machine Learning", "Italian Cuisine", "Gardening"]
    leaves = []
    for i in range(20):
        topic = topics[i % 4]
        content = f"Leaf {i} about {topic}. {topic} is a fascinating subject with many details. Content id {i}."
        leaves.append(Segment(content=content, segment_id=f"leaf-{i}", type=SegmentType.TEXT))
    print(f"Created {len(leaves)} leaf segments.")

    # 2. Setup Components
    print("Initializing OpenAI Components...")
    llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
    embeddings = OpenAIEmbeddings()
    
    
    # Initialize Processor with Real Components
    # We are now using the real ClusteringService (requires umap-learn, scikit-learn)
    processor = RaptorProcessor(
        llm=llm, 
        embeddings=embeddings, 
        max_levels=3
        # clustering_service=None -> Defaults to real ClusteringService
    )
    print("Initialized RaptorProcessor with Real LLM, Embeddings & Clustering.")

    # 3. Process
    print("Processing segments (This may take a moment due to API calls)...")
    results = processor.process_segments(leaves)
    
    # 4. Analyze Results
    print(f"Total Segments after RAPTOR: {len(results)}")
    
    summaries = [s for s in results if s.metadata.get("is_raptor_summary")]
    print(f"Generated {len(summaries)} summaries.")
    
    for s in summaries:
        level = s.metadata["raptor_level"]
        cluster = s.metadata["raptor_cluster_id"]
        children = len(s.metadata["raptor_child_ids"])
        print(f"  - Level {level} Summary (Cluster {cluster}): {s.content[:50]}... (Children: {children})")
        
    if len(results) >= 13: # 10 leaves + 3 L1 summaries + 1 L2 summary (approx)
        print("SUCCESS: Tree structure built with real LLM summaries.")
    else:
        print(f"WARNING: Unexpected number of segments: {len(results)}")

if __name__ == "__main__":
    run_demo()
