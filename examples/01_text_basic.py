import sys
from pathlib import Path
import os
import shutil
import uuid
from typing import List
from datetime import datetime

# Ensure module is visible
sys.path.append(str(Path(__file__).parent.parent / "src"))
from example_utils import setup_environment, print_section

from rag_lib.core.domain import Segment
# TokenTextSplitter is now imported dynamically or at top if checking types,
# but for the main execution we import inside main or here.
# Let's import it here to be clean.
from rag_lib.chunkers.token import TokenTextSplitter

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from rag_lib.retrieval.retrievers import FuzzyRetriever
from rag_lib.core.indexer import Indexer
# from rag_lib.vectors.factory import get_vector_store # This will be replaced

"""
E2E Example 01: Basic Text Workflow

Features Tested:
1. TextLoader: Loading raw text files.
2. TokenSplitter: Chunking based on token count.
3. VectorStore: Embedding and indexing chunks.
4. Retrieval: Simple semantic search.

Expected Results:
- Loading:
    - Input: "docs/terms&defs.txt"
    - Output: Raw string content.
    - Sample Data (Start): "**Термины и определения\n\n##Term: 1С:CRM\n##Definition: – программное обеспечение 1С: Предприятие..."
- Chunking:
    - Logic: TokenTextSplitter(chunk_size=150, chunk_overlap=15)
    - Output: List[Segment]
    - Sample Data (Chunk 0): Finer-grained chunks.
- Indexing:
    - Input: List[Segment]
    - Method: Indexer.index() -> VectorStore.add_texts()
    - Output: Confirmation of indexing.
- Retrieval:
    - Query: "1С:CRM" (Cyrillic 'С' to match source)
    - Expected Top Match: Parent Segment containing full definition via Chunk hit.
    - Architecture: Dual Storage (VectorStore for Chunks -> InMemoryStore for Parents).

Actual Results (Executed 2026-02-16):
- Segments Created: 1002 (Logical), >1200 (Chunks)
- Retrieval Method: rag_lib.retrieval.composition.create_dual_storage_retriever (MultiVectorRetriever)
- Top Retrieval: Successfully retrieved full Parent Segment via chunk vector match.
- JSON Logs (UTF-8): 
    - docs/results/01_text_basic_chunks_detailed_20260216_xxxxxx.json
    - docs/results/01_text_basic_segments_20260216_xxxxxx.json
    - docs/results/01_text_basic_retrieved_data_20260216_xxxxxx.json
"""

from rag_lib.chunkers.regex import RegexSplitter
from rag_lib.chunkers.token import TokenTextSplitter
from rag_lib.loaders.data_loaders import TextLoader
# Ensure domain objects are available for any type hinting or manual construction if needed
from rag_lib.core.domain import Segment, Document 

import datetime
import json

# Mock or real imports for vector store (depending on environment)
# Strict import - will fail if dependencies are missing
from rag_lib.vectors.factory import get_vector_store
from rag_lib.retrieval.retrievers import get_vector_retriever

def main():
    # 1. Setup
    print("--- 01_text_basic.py: Basic Text Processing with Loader & Document Pipeline ---")
    
    # Define paths
    txt_path = Path(__file__).parent.parent / "docs" / "terms&defs.txt"
    if not txt_path.exists():
        print(f"Error: File not found at {txt_path}")
        return

    # 2. Load Document (using TextLoader)
    print(f"Loading {txt_path.name} using TextLoader...")
    loader = TextLoader(str(txt_path))
    documents = loader.load()
    
    if not documents:
        print("Error: No documents loaded.")
        return
        
    print(f"Loaded {len(documents)} document(s).")
    # print(f"Sample content: {documents[0].page_content[:100]}...")

    # 3. Logical Segmentation (Document -> Segments using RegexSplitter)
    # We use a lookahead assertion (?=...) to keep the delimiter at the start of the new string
    print("Segmentation (rag_lib.RegexSplitter: Document -> Segments)...")
    logical_splitter = RegexSplitter(pattern=r'(?=##Term:)')
    
    # This transforms the Document(s) into logical Segments
    logical_segments = logical_splitter.split_documents(documents)
    print(f"Created {len(logical_segments)} logical segments from document.")

    # 4. Chunking (Segments -> Chunks using TokenTextSplitter)
    # Now we ensure each logical segment fits our token constraints
    print("Chunking (rag_lib.TokenTextSplitter: Segments -> Chunks)...")
    token_splitter = TokenTextSplitter(chunk_size=150, chunk_overlap=15)
    
    # Process the logical segments into final chunks
    final_segments = token_splitter.split_segments(logical_segments)
    print(f"Generated {len(final_segments)} chunks from {len(logical_segments)} logical segments.")

    # 5. Save detailed results
    save_json(logical_segments, "segments")       # Intermediate logical segments
    save_json(final_segments, "chunks_detailed")  # Final chunk segments with metadata

    # 6. Index & Retrieve
    print("Indexing chunks and setting up Dual Storage...")

    from rag_lib.embeddings.factory import get_embeddings_model
    from rag_lib.vectors.factory import get_vector_store
    
    # Imports for Dual Storage
    from rag_lib.retrieval.composition import create_scored_dual_storage_retriever
    from langchain_core.stores import InMemoryStore
    
    embeddings = get_embeddings_model(provider="openai")
    
    # CLEANUP: Remove old vector store to prevent stale IDs
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db", ignore_errors=True)
        print("Cleaned up existing ./chroma_db")

    # A. Vector Store: Index the CHUNKS (Small)
    vector_store = get_vector_store(embeddings=embeddings)
    print(f"Vector store initialized: {type(vector_store).__name__}")
    
    # B. Doc Store: Store the PARENTS (Big)
    doc_store = InMemoryStore()
    
    # Initialize Indexer with BOTH stores
    indexer = Indexer(vector_store, embeddings, doc_store=doc_store)
    
    # Index Chunks (Vector) AND Parents (DocStore) in one go
    # "Small-to-Big": Search 'final_segments' (chunks), Retrieve 'logical_segments' (parents)
    indexer.index(final_segments, parents=logical_segments)
    
    # 7. Retrieval (Small-to-Big)
    # Note: Source text uses Cyrillic 'С' in '1С:CRM', so we must match it or rely on strong embedding alignment.
    query = "Что такое 1C  CRM?" 
    print(f"Retrieving for query: '{query}'...")
    
    # Create Dual Storage Retriever 
    # Matches Chunk in VectorStore -> Uses 'parent_id' from Chunk -> Retrieves Parent from DocStore
    retriever = create_scored_dual_storage_retriever(
        vector_store=vector_store,
        docstore=doc_store,
        id_key="parent_id",
        search_kwargs={"k": 10},
        search_type="similarity_score_threshold",
        score_threshold=0.5,
    )
    
    results = retriever.invoke(query)
    
    if not results:
        print("Error: Retriever returned no results.")
    else:
        top_doc = results[0]
        print(f"\n--- Retrieved Result (Small-to-Big) ---")
        print(f"Retrieved Content (Parent Segment): {top_doc.page_content[:200]}...")
        # Check if metadata exists (InMemoryStore might store Document which has metadata)
        print(f"Source ID: {top_doc.metadata.get('segment_id', 'N/A')}")
        print(f"---------------------------------------")
    
    # Convert Document objects to dicts for saving
    retrieved_data = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in results]
    save_json(retrieved_data, "retrieved_data")

    print("Done.")

def save_json(data, artifact_type):
    results_dir = Path(__file__).parent.parent / "docs" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"01_text_basic_{artifact_type}_{timestamp}.json"
    filepath = results_dir / filename
    with open(filepath, "w", encoding="utf-8") as f:
        # ensure_ascii=False ensures proper UTF-8 output without escaping
        json.dump(data, f, indent=2, ensure_ascii=False, default=lambda o: o.__dict__ if hasattr(o, "metadata") or hasattr(o, "__dict__") else str(o))
    print(f"Saved {artifact_type} to {filepath}")

if __name__ == "__main__":
    main()
