import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from example_utils import setup_environment, print_section

# 1. Imports
from rag_lib.loaders.data_loaders import JsonLoader
from rag_lib.core.indexer import Indexer
from rag_lib.vectors.factory import create_vector_store
from langchain_openai import OpenAIEmbeddings

"""
E2E Example 09: JSON Hybrid Workflow

Features Tested:
1. JsonLoader: Loading JSON data with jq schema.
2. Metadata Enrichment: Manual/Automatic tagging.
3. Hybrid Retrieval: Vector Search + Metadata Filtering.

Expected Results:
- Loading:
    - Input: "docs/QA_data.json"
    - Schema: ".[]" (Array root).
    - Output: Segments for each JSON object.
    - Sample Data: Segment(content='{"question": "Bug?", "answer": "Yes"}', type=JSON)
- Indexing:
    - Input: Segments with metadata category='bug_report'.
    - Output: Indexed vectors + metadata.
- Retrieval:
    - Query: "system crash"
    - Filter: {'category': 'bug_report'}
    - Expected Result: Only matches with 'bug_report' tag.
    - Sample Output: "Crash log..." (ignoring "Feature request" docs).
"""

def main():
    setup_environment()
    print_section("09. JSON Hybrid Workflow")

    # 2. Load (with jq schema commonly used)
    json_path = Path(__file__).parent.parent / "docs" / "QA_data.json"
    print(f"Loading {json_path}...")
    
    # Assuming JsonLoader takes jq_schema or similar
    loader = JsonLoader(str(json_path), jq_schema=".[]") 
    segments = loader.load()
    
    # Enrich with metadata manually for demo
    for s in segments:
        if "bug" in s.content.lower():
            s.metadata["category"] = "bug_report"
        else:
            s.metadata["category"] = "general"

    print(f"Loaded {len(segments)} JSON items.")

    # 3. Index
    embeddings = OpenAIEmbeddings()
    vector_store = create_vector_store("chroma", embeddings, "09_json_hybrid")
    
    indexer = Indexer(vector_store, embeddings)
    indexer.index(segments)

    # 4. Retrieve (Hybrid: Vector + Metadata Filter)
    print("Hybrid Search (Vector + Metadata Filter)...")
    
    # Chroma filter syntax
    results = vector_store.similarity_search(
        "system crash", 
        k=2,
        filter={"category": "bug_report"} 
    )
    
    for r in results:
        print(f"- {r.page_content[:80]}... [Meta: {r.metadata}]")

if __name__ == "__main__":
    main()
