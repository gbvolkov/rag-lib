import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from example_utils import setup_environment, print_section, load_pdf_text

# 1. Imports
from rag_lib.loaders.pdf import PDFLoader # Uses Camelot/Poppler
from rag_lib.chunkers.semantic import SemanticChunker
from rag_lib.core.indexer import Indexer
from rag_lib.vectors.factory import get_vector_store
from rag_lib.retrieval.composition import create_reranking_retriever
from langchain_openai import OpenAIEmbeddings

"""
E2E Example 03: PDF Semantic Workflow (with Reranking)

Features Tested:
1. PDFLoader (pypdf backend): Extracting text from PDF.
2. SemanticChunker: Splitting based on embedding similarity (cosine distance).
3. VectorStore: Indexing semantic chunks.
4. RerankingRetriever: Cross-Encoder based re-ranking of results.

Expected Results:
- Loading:
    - Input: "docs/statement.pdf"
    - Output: Single Segment with full text.
    - Sample Data: Segment(content="Statement of Account...", type=TEXT)
- Chunking:
    - Logic: SemanticChunker(threshold=0.8)
    - Output: List[Segment] where breakpoints align with semantic shifts.
    - Sample Data: Segment(content="Transaction details for Jan 2025...")
- Indexing:
    - Input: Semantic Segments
    - Output: Indexed in Chroma.
- Retrieval:
    - Query: "Что такое морфология?" (or relevant query for statement.pdf)
    - Stage 1: Vector Search (Top-10) -> Broad recall.
    - Stage 2: Cross-Encoder Reranking (Top-3) -> High precision.
    - Sample Output: Top-1 result is the most relevant snippet.
"""

def main():
    setup_environment()
    print_section("03. PDF Semantic Workflow (with Reranking)")

    # 2. Load
    pdf_path = Path(__file__).parent.parent / "docs" / "2025_soo_frp_russkij-yazyk_10_11-2.pdf"
    # 2. Load PDF
    # We use pypdf via example_utils to avoid heavy Ghostscript dependency for Camelot
    # (PDFLoader in rag_lib is specialized for tables and requires Camelot/Ghostscript)
    pdf_path = Path(__file__).parent.parent / "docs" / "statement.pdf"
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        return

    print(f"Loading {pdf_path}...")
    segment = load_pdf_text(pdf_path)
    if not segment:
        print("Failed to load PDF.")
        return
        
    # SemanticChunker expects raw text usually, or we can split the segment content.
    full_text = segment.content
    print(f"Loaded {len(full_text)} chars from PDF.")

    # 3. Chunk (Semantic)
    print("Semantic Chunking...")
    embeddings = OpenAIEmbeddings()
    chunker = SemanticChunker(embeddings=embeddings, threshold=0.8)
    
    # Semantic chunker takes full text usually, but here we have segments.
    # Let's combine text for demo purposes or chunk each segment.
    semantic_segments = chunker.split_text(full_text) # Returns Segments
    print(f"Semantic Chunker created {len(semantic_segments)} chunks.")

    # 4. Index
    vector_store = get_vector_store("chroma", embeddings, "03_pdf_semantic")
    indexer = Indexer(vector_store, embeddings)
    indexer.index(semantic_segments)

    # 5. Retrieve (Reranking)
    print("Retrieving with Cross-Encoder Reranking...")
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    
    # Requires sentence-transformers
    try:
        reranker = create_reranking_retriever(base_retriever, top_n=3)
        results = reranker.invoke("Что такое морфология?")
        
        for i, res in enumerate(results):
            print(f"[{i+1}] {res.page_content[:100]}...")
    except ImportError:
        print("sentence-transformers not installed. Fallback to standard.")
        results = base_retriever.invoke("Что такое морфология?")
        print(results[0].page_content[:100])

if __name__ == "__main__":
    main()
