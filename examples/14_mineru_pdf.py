import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from example_utils import setup_environment, print_section

# 1. Imports
from rag_lib.loaders.miner_u import MinerULoader
from rag_lib.core.indexer import Indexer
from rag_lib.vectors.factory import get_vector_store
from langchain_openai import OpenAIEmbeddings

"""
E2E Example 14: MinerU PDF Workflow

Features Tested:
1. MinerULoader: High-fidelity PDF parsing (if dependencies available).
2. Fallback Handling: Graceful degradation or error reporting.

Expected Results:
- Loading:
    - Input: "docs/statement.pdf"
    - Method: MinerULoader (wraps mineru).
    - Success Case: Returns Segments with high-quality text layout.
    - Failure Case (Missing Deps): Prints "MinerU Load Failed" (Acceptable for Demo).
- Indexing & Retrieval:
    - Standard vector flow.
    - Query: "statement date"
    - Sample Output: "Date: 2025-01-01" (if loaded successfully).
"""

def main():
    setup_environment()
    print_section("14. MinerU PDF Workflow")

    # 2. Load
    pdf_path = Path(__file__).parent.parent / "docs" / "statement.pdf"
    print(f"Loading {pdf_path} using MinerU...")
    
    # MinerU requires `mineru[all]` installed and configured.
    # We use try/except so the example can still run in environments
    # where MinerU optional dependencies are not installed.
    
    try:
        loader = MinerULoader(str(pdf_path))
        segments = loader.load()
        print(f"Loaded {len(segments)} segments.")
    except (ImportError, RuntimeError, Exception) as e:
        print(f"MinerU Load Failed (Expected if extra deps missing): {e}")
        print("Continuing trace for E2E demonstration...")
        return

    # 3. Index
    embeddings = OpenAIEmbeddings()
    vector_store = get_vector_store("chroma", embeddings, "14_mineru")
    
    indexer = Indexer(vector_store, embeddings)
    indexer.index(segments)

    # 4. Retrieve
    print("Retrieving...")
    results = vector_store.similarity_search("statement date", k=1)
    if results:
        print(f"Match: {results[0].page_content[:100]}...")

if __name__ == "__main__":
    main()
