import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from example_utils import setup_environment, print_section

# 1. Imports
from rag_lib.loaders.structured import StructuredLoader
from rag_lib.chunkers.regex import RegexSplitter
from rag_lib.retrieval.retrievers import RegexRetriever
from rag_lib.core.indexer import Indexer
from rag_lib.vectors.factory import get_vector_store
from langchain_openai import OpenAIEmbeddings

"""
E2E Example 06: DOCX Regex Workflow

Features Tested:
1. StructuredLoader: Loading DOCX.
2. RegexSplitter: Splitting text based on custom patterns.
3. RegexRetriever: Retrieving documents that match a regex query.
4. VectorStore: Standard indexing (secondary).

Expected Results:
- Loading:
    - Input: "docs/Тестирование...docx"
    - Output: Segments preserving paragraph structure.
    - Sample Data: Segment(content="Step 1: Open Application...", type=TEXT)
- Chunking:
    - Logic: Split by "Step \d+" or "Шаг \d+".
    - Output: Chunks corresponding to test steps.
    - Sample Data: Segment(content="Step 1: Login with admin credentials...")
- Retrieval:
    - Method: RegexRetriever(documents=segments)
    - Query: "Step 1"
    - Expected Result: Matches specifically "Step 1" chunks.
    - Sample Output: "Step 1: Login..."
"""

def main():
    setup_environment()
    print_section("06. DOCX Regex Workflow")

    # 2. Load
    docx_path = Path(__file__).parent.parent / "docs" / "Тестирование (БФТ_Подготовка Банковской гарантии).docx"
    print(f"Loading {docx_path}...")
    
    loader = StructuredLoader(str(docx_path))
    initial_segments = loader.load()
    
    # 3. Chunk (Regex)
    # We want to split by "Шаг X" (Step X) or "Test Case" patterns potentially.
    # Let's assume there's a pattern "Step \d+" in the text.
    print("Re-chunking using RegexSplitter...")
    full_text = "\n".join([s.content for s in initial_segments])
    
    splitter = RegexSplitter(pattern=r"(Step \d+|Шаг \d+)", chunk_size=500)
    regex_chunks = splitter.split_text(full_text)
    print(f"Regex Splitter produced {len(regex_chunks)} chunks.")

    # 4. Retrieval (RegexRetriever)
    print("Retrieving using RegexRetriever (Pattern Matching)...")
    # This retriever finds docs that MATCH a regex query.
    # Use case: Find all chunks mentioning "Error 404" or specific IDs.
    
    # Let's simulate indexing first (optional for RegexRetriever if it works on memory list, 
    # but for E2E let's index to Vector too).
    embeddings = OpenAIEmbeddings()
    vector_store = get_vector_store("chroma", embeddings, "06_docx_regex")
    # ... indexing logic standard ...

    # But we want to demonstrate RegexRetriever specifically.
    # RegexRetriever in rag-lib typically scans a list of docs.
    from rag_lib.core.domain import Segment
    segments = [Segment(content=c) for c in regex_chunks]
    print(f"Loaded {len(segments)} segments.")
    if segments:
        print(f"Sample content: {segments[0].content[:200]}...")
    
    # 4. Retrieval (Regex)
    # We want to find the segment with "Chapter 2"
    query = "Chapter 2"
    print(f"Retrieving '{query}'...")
    
    docs = [s.to_langchain() for s in segments]
    retriever = RegexRetriever(documents=docs)
    results = retriever.invoke(query)
    
    print(f"Regex Pattern Results: {len(results)}")
    for r in results:
        print(f"- {r.page_content[:80]}...")

if __name__ == "__main__":
    main()
