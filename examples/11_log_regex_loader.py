import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from example_utils import setup_environment, print_section

# 1. Imports
from rag_lib.loaders.regex import RegexHierarchyLoader
from rag_lib.core.indexer import Indexer
from rag_lib.vectors.factory import get_vector_store
from langchain_openai import OpenAIEmbeddings

"""
E2E Example 11: Log Regex Loader Workflow

Features Tested:
1. RegexHierarchyLoader: Parsing logs with level detection.
2. Indexer: Storing log segments.
3. VectorStore: Retrieving specific log events.

Expected Results:
- Loading:
    - Input: "docs/dummy_log.txt"
    - Pattern: r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})" (Timestamp start).
    - Output: Segments chunked by timestamp lines.
    - Sample Data: Segment(content="2023-10-27 10:00:00 [INFO] System started...", level=1)
- Indexing:
    - Input: Log Segments.
    - Output: Indexed logs.
- Retrieval:
    - Query: "Connection failed"
    - Expected Result: Log entry mentioning failure.
    - Sample Output: "2023-10-27 10:05:00 [ERROR] Connection failed..."
"""

def main():
    setup_environment()
    print_section("11. Log Regex Loader Workflow")

    # 2. Load
    log_path = Path(__file__).parent.parent / "docs" / "dummy_log.txt"
    print(f"Loading {log_path}...")
    
    # Regex for Log: Timestamp [Level] Message
    # We want to capture the whole line or segment by timestamp.
    # Pattern: ^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})
    
    loader = RegexHierarchyLoader(
        file_path=str(log_path),
        patterns=[
            (1, r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})") # Date as Level 1
        ]
    )
    segments = loader.load()
    print(f"Loaded {len(segments)} log entries.")

    # 3. Index
    embeddings = OpenAIEmbeddings()
    vector_store = get_vector_store("chroma", embeddings, "11_log_regex")
    
    indexer = Indexer(vector_store, embeddings)
    indexer.index(segments)

    # 4. Retrieve
    print("Retrieving Error logs...")
    results = vector_store.similarity_search("Connection failed", k=1)
    if results:
        print(f"Found Log: {results[0].page_content[:100]}...")

if __name__ == "__main__":
    main()
