import os
import sys
import shutil
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from example_utils import _check_install, setup_environment, print_section

# Check for python-docx
try:
    import docx
except ImportError:
    _check_install("python-docx", "docx")

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Core imports
from rag_lib.core.domain import Segment, SegmentType
from rag_lib.loaders.structured import StructuredLoader
from rag_lib.processors.entity_extractor import EntityExtractor
from rag_lib.graph.store import NetworkXGraphStore
from rag_lib.core.indexer import Indexer
from rag_lib.vectors.factory import get_vector_store

def main():
    setup_environment()
    
    # 1. Configuration
    DOCS_DIR = Path(__file__).parent.parent / "docs"
    DOCX_FILE = DOCS_DIR / "Параметризованные задачи.docx" 
    COLLECTION_NAME = "graph_workflow_demo"

    print_section("1. Loading Structured Data (DOCX)")
    
    if not DOCX_FILE.exists():
        print(f"File {DOCX_FILE} not found!")
        return

    print(f"Loading {DOCX_FILE.name} using StructuredLoader...")
    # StructuredLoader automatically extracts hierarchy (H1, H2...)
    loader = StructuredLoader(str(DOCX_FILE))
    segments = loader.load()
    
    print(f"Loaded {len(segments)} segments.")
    if segments:
        print(f"Sample Segment: {segments[0].content[:100]}...")
        print(f"Metadata: {segments[0].metadata}")

    # 2. Graph Extraction
    print_section("2. Graph Extraction & Construction")
    print("Initializing NetworkXGraphStore (In-Memory)...")
    graph_store = NetworkXGraphStore()
    
    print("Initializing EntityExtractor with ChatOpenAI (gpt-3.5-turbo)...")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    extractor = EntityExtractor(llm=llm, store=graph_store)
    
    # Process a subset to save time/cost, or all?
    # Let's process the first 5 segments for the demo.
    MAX_GRAPH_SEGMENTS = 5
    print(f"Extracting entities from first {MAX_GRAPH_SEGMENTS} segments...")
    
    segments_to_process = segments[:MAX_GRAPH_SEGMENTS]
    extractor.process_segments(segments_to_process)
    
    # Check graph stats
    num_nodes = graph_store.graph.number_of_nodes()
    num_edges = graph_store.graph.number_of_edges()
    print(f"\nGraph Statistics:\n  Nodes: {num_nodes}\n  Edges: {num_edges}")
    
    # 3. Indexing (Vector + Graph built during extraction)
    # The EntityExtractor populated the graph store. 
    # Now we also want to index the segments into Vector Store for hybrid retrieval.
    # And potentially index the graph nodes if we want to search them?
    # Our Indexer handles segments. 
    # If we pass entity_extractor to Indexer, it does extraction AUTOMATCIALLY.
    # But we did it manually to show control and stats.
    # So we just index segments now.
    
    print_section("3. Indexing Segments")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Clean previous demo DB
    demo_db_path = "./chroma_demo_db_graph"
    if os.path.exists(demo_db_path):
        shutil.rmtree(demo_db_path)

    print(f"Initializing Vector Store: {COLLECTION_NAME}")
    vector_store = get_vector_store(
        provider="chroma",
        embeddings=embeddings,
        collection_name=COLLECTION_NAME
    )
    
    indexer = Indexer(vector_store=vector_store, embeddings=embeddings)
    
    print("Indexing segments...")
    indexer.index(segments, batch_size=20)
    print("Indexing complete.")

    # 4. Retrieval (Multi-Modal: Vector + Graph)
    print_section("4. Retrieval (Vector + Graph)")
    
    query = "задача" # "problem" or "task" in Russian, likely relevant
    print(f"Query: '{query}'")
    
    # A. Vector Search
    print("\n--- Vector Search Results ---")
    vector_results = vector_store.similarity_search(query, k=2)
    for i, res in enumerate(vector_results):
        print(f"[{i+1}] {res.page_content[:150]}...")
    
    # B. Graph Search
    print("\n--- Graph Search Results ---")
    # Search for nodes matching query
    # (NetworkXGraphStore has simple text search on labels)
    graph_nodes = graph_store.search_nodes(query)
    print(f"Found {len(graph_nodes)} nodes matching '{query}':")
    
    for node in graph_nodes[:3]: # Show top 3
        print(f"  Node: {node.label} ({node.type})")
        # Get neighbors
        neighbors = graph_store.get_neighbors(node.id)
        if neighbors:
            print(f"    Neighbors ({len(neighbors)}): {[n.label for n in neighbors]}")
        else:
            print("    No neighbors.")

if __name__ == "__main__":
    main()
