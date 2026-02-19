import os
import sys
import shutil
from pathlib import Path
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from example_utils import _check_install, setup_environment, print_section

# Ensure pandas for CSV/Excel just in case, though we use JSON here primarily
try:
    import pandas
except ImportError:
    _check_install("pandas")

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Core imports
from rag_lib.core.domain import Segment, SegmentType
from rag_lib.loaders.data_loaders import JsonLoader
from rag_lib.core.indexer import Indexer
from rag_lib.vectors.factory import create_vector_store

def main():
    setup_environment()
    
    # 1. Configuration
    DOCS_DIR = Path(__file__).parent.parent / "docs"
    JSON_FILE = DOCS_DIR / "QA_data.json"
    COLLECTION_NAME = "hybrid_workflow_demo"

    print_section("1. Loading Structured Data (JSON)")
    
    if not JSON_FILE.exists():
        print(f"File {JSON_FILE} not found!")
        return

    print(f"Loading {JSON_FILE.name} using JsonLoader...")
    # JsonLoader loads a list of objects as segments
    loader = JsonLoader(str(JSON_FILE), jq_schema=".")
    segments = loader.load()
    
    # Filter out empty content
    segments = [s for s in segments if s.content and len(s.content) > 10]
    
    print(f"Loaded {len(segments)} segments.")
    if segments:
        print(f"Sample Segment Content:\n{segments[0].content[:200]}...")
        # Add a dummy metadata field for filtering demo if not present
        # QA_data.json segments likely have "source" metadata from loader.
        # Let's add a "category" field based on keywords for demo purposes.
        print("Enriching with dummy categories for hybrid search demo...")
        for i, seg in enumerate(segments):
            if "error" in seg.content.lower():
                seg.metadata["category"] = "bug_report"
            elif "how" in seg.content.lower():
                seg.metadata["category"] = "howto"
            else:
                seg.metadata["category"] = "general"

    # 2. Indexing
    print_section("2. Indexing with Metadata")
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Clean previous demo DB
    demo_db_path = "./chroma_demo_db_hybrid"
    if os.path.exists(demo_db_path):
        shutil.rmtree(demo_db_path)

    print(f"Initializing Vector Store: {COLLECTION_NAME}")
    vector_store = create_vector_store(
        provider="chroma",
        embeddings=embeddings,
        collection_name=COLLECTION_NAME
    )
    
    indexer = Indexer(vector_store=vector_store, embeddings=embeddings)
    
    # Index a subset to be fast
    MAX_INDEX = 50
    print(f"Indexing first {MAX_INDEX} segments...")
    indexer.index(segments[:MAX_INDEX], batch_size=10)
    print("Indexing complete.")

    # 3. Hybrid Retrieval (Vector + Metadata Filter)
    print_section("3. Hybrid Retrieval")
    
    query = "database connection error"
    category_filter = "bug_report"
    
    print(f"Query: '{query}'")
    print(f"Filter: category == '{category_filter}'")
    
    # Chroma filter syntax: where={"field": "value"}
    results = vector_store.similarity_search(
        query, 
        k=3,
        filter={"category": category_filter}
    )
    
    print(f"\nTop {len(results)} Results:")
    for i, res in enumerate(results):
        print(f"\n[{i+1}] Category: {res.metadata.get('category')}")
        print(f"    Content: {res.page_content[:150]}...")
        print(f"    Source: {res.metadata.get('source')}")

    # 4. Hybrid Retrieval (Another Filter)
    category_filter_2 = "howto"
    print(f"\nQuery: '{query}'")
    print(f"Filter: category == '{category_filter_2}'")
    
    results_2 = vector_store.similarity_search(
        query, 
        k=3,
        filter={"category": category_filter_2}
    )
    
    print(f"\nTop {len(results_2)} Results:")
    for i, res in enumerate(results_2):
        print(f"\n[{i+1}] Category: {res.metadata.get('category')}")
        print(f"    Content: {res.page_content[:150]}...")

if __name__ == "__main__":
    main()
